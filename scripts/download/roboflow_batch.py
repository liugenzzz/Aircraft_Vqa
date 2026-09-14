#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量下载多个 Roboflow Universe 项目，并生成对应的数据源配置。

用来把航空缺陷的长尾小集合并起来 —— 单个几百张，合起来能顶一个中等数据集。

先到 Universe 搜索页挑项目（这几个检索页命中率最高）：
  https://universe.roboflow.com/search?q=aircraft+defect
  https://universe.roboflow.com/search?q=class%3Arivet
  https://universe.roboflow.com/search?q=class%3Acorrosion
  https://universe.roboflow.com/search?q=class%3Amissing+screw

把看中的项目按 `workspace/project` 一行一个写进清单文件（`#` 开头是注释，
`:版本号` 可选，不写就取最新）：

    # aircraft_extra.txt
    ddiisc/aircraft-skin-defects-revised-annotations-631bu
    some-workspace/rivet-inspection:3

然后：

    export ROBOFLOW_API_KEY=xxx
    python scripts/download/roboflow_batch.py --list-file aircraft_extra.txt \
        --data-root ~/data/raw

下完会生成 configs/datasets.extra.yaml，用 --config 一起喂给 ingest：

    python scripts/ingest.py --config configs/datasets.yaml \
        --config configs/datasets.extra.yaml --data-root ~/data/raw

注意：Universe 上逐个项目的授权不同，商用前必须**逐集核对**项目页的 License。
生成的配置里 commercial_ok 一律填 false，确认过再自己改。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import _bootstrap  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

ENTRY = """  - name: {name}
    adapter: coco
    root: "{root}"
    enabled: true
    license: "见 Roboflow 项目页 —— 商用前必须核对"
    commercial_ok: false
    url: https://universe.roboflow.com/{ws}/{proj}
    category: fuselage
    object_hint:
      fuselage: fuselage
    splits: [train, valid, test]
"""


def parse_list(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            spec, _, ver = line.partition(":")
            ws, _, proj = spec.partition("/")
            if not ws or not proj:
                print(f"[skip] 第 {lineno} 行格式不对（要 workspace/project）：{line}")
                continue
            out.append({"ws": ws.strip(), "proj": proj.strip(),
                        "ver": ver.strip() or "latest"})
    return out


def safe_name(proj: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", proj).strip("_").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", required=True, help="workspace/project 清单")
    ap.add_argument("--data-root", default="~/data/raw")
    ap.add_argument("--out-config", default=os.path.join(
        REPO, "configs", "datasets.extra.yaml"))
    ap.add_argument("--format", default="coco")
    args = ap.parse_args()

    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("请先设置 ROBOFLOW_API_KEY")
        return 1

    items = parse_list(args.list_file)
    if not items:
        print(f"{args.list_file} 里没有有效条目")
        return 1

    root = os.path.expanduser(args.data_root)
    ok, failed, entries = [], [], []
    for it in items:
        name = safe_name(it["proj"])
        dest = os.path.join(root, "roboflow", name)
        print(f"\n=== {it['ws']}/{it['proj']} (v{it['ver']})")
        rc = subprocess.call([
            sys.executable, os.path.join(HERE, "download_roboflow.py"),
            "--workspace", it["ws"], "--project", it["proj"],
            "--version", it["ver"], "--out", dest, "--format", args.format])
        if rc == 0 and os.path.exists(
                os.path.join(dest, "train", "_annotations.coco.json")):
            ok.append(name)
            entries.append(ENTRY.format(name=f"rf_{name}", root=dest,
                                        ws=it["ws"], proj=it["proj"]))
        else:
            failed.append(f"{it['ws']}/{it['proj']}")

    if entries:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_config)) or ".",
                    exist_ok=True)
        with open(args.out_config, "w", encoding="utf-8") as f:
            f.write("# 由 scripts/download/roboflow_batch.py 生成。\n"
                    "# 授权逐集不同，commercial_ok 一律先填 false，"
                    "核对项目页 License 后再自行放开。\n"
                    "defaults:\n  data_root: ~/data/raw\n\ndatasets:\n")
            f.write("\n".join(entries))
        print(f"\n配置已写到 {args.out_config}")
        print("接着：python scripts/ingest.py --config configs/datasets.yaml "
              f"--config {os.path.relpath(args.out_config, REPO)} "
              f"--data-root {root}")

    print(f"\n成功 {len(ok)}：{'、'.join(ok) or '无'}")
    if failed:
        print(f"失败 {len(failed)}：{'、'.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
