#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Roboflow Universe 下载 COCO 格式数据集。

航空域的几个关键集都托管在 Roboflow，需要一个免费 API Key：
  https://app.roboflow.com  -> Settings -> Roboflow API -> Private API Key

    export ROBOFLOW_API_KEY=xxxx
    python scripts/download/download_roboflow.py \
        --workspace ddiisc --project aircraft_skin_defects --version 1 \
        --out ~/data/raw/roboflow/aircraft_skin_defects

下完的目录结构（train/valid/test + _annotations.coco.json）
可直接被 configs/datasets.yaml 里 adapter: coco 的条目读取。
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

import urllib.request

TEMPLATE = ("https://api.roboflow.com/{ws}/{proj}/{ver}/coco/download"
            "?api_key={key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="coco", help="coco | yolov8 | voc")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        print("请先设置 ROBOFLOW_API_KEY（https://app.roboflow.com -> Settings -> API）")
        return 1

    url = TEMPLATE.format(ws=args.workspace, proj=args.project,
                          ver=args.version, key=key).replace("/coco/", f"/{args.format}/")
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    zpath = os.path.join(out, "_download.zip")

    print(f">> 请求 {args.workspace}/{args.project} v{args.version} ({args.format})")
    try:
        # Roboflow 先返回一个含 export.link 的 JSON，再去取真正的 zip
        import json
        with urllib.request.urlopen(url, timeout=120) as r:
            meta = json.loads(r.read().decode())
        link = meta.get("export", {}).get("link") or meta.get("link")
        if not link:
            print(f"返回里没有下载链接：{meta}")
            return 1
        print(">> 下载 zip")
        urllib.request.urlretrieve(link, zpath)
    except Exception as e:
        print(f"下载失败：{e}\n"
              f"若是网络策略限制，可在能联网的机器上手动下载后拷贝到 {out}")
        return 1

    print(">> 解压")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    os.remove(zpath)
    print(f">> 完成：{out}")
    print("   在 configs/datasets.yaml 里把对应条目的 enabled 改成 true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
