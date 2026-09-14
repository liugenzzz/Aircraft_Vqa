#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 1：各源数据集 -> 统一中间表示 (data/interim/<name>.jsonl)。

    python scripts/ingest.py --config configs/datasets.yaml --out data/interim
    python scripts/ingest.py --only visa synthetic_panel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import yaml

from aircraft_vqa.adapters import build_adapter
from aircraft_vqa.schema import write_jsonl
from aircraft_vqa.taxonomy import get_taxonomy


def resolve(spec: dict, data_root: str) -> dict:
    """展开 {data_root}；主 root 不存在时依次尝试 root_alternatives。

    同一个数据集不同人解压出来的目录名经常不一样（按类别分包下载尤其如此），
    与其让所有人去改配置，不如在这里挨个试。
    """
    spec = dict(spec)
    expand = lambda p: os.path.expanduser(str(p).replace("{data_root}", data_root))
    candidates = [spec["root"]] + list(spec.pop("root_alternatives", []) or [])
    resolved = [expand(c) for c in candidates]
    spec["root"] = next((r for r in resolved if os.path.exists(r)), resolved[0])
    spec["_tried_roots"] = resolved
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--data-root", default=None, help="覆盖 defaults.data_root")
    ap.add_argument("--only", nargs="*", default=None, help="只处理这些 name")
    ap.add_argument("--commercial-only", action="store_true",
                    help="只处理可商用的数据源")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_root = os.path.expanduser(
        args.data_root or cfg.get("defaults", {}).get("data_root", "~/data/raw"))
    tax = get_taxonomy(os.path.abspath(args.taxonomy))
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for spec in cfg["datasets"]:
        name = spec["name"]
        if args.only and name not in args.only:
            continue
        if not args.only and not spec.get("enabled", False):
            print(f"[skip] {name}: enabled=false")
            continue
        if args.commercial_only and not spec.get("commercial_ok", False):
            print(f"[skip] {name}: 非商用授权")
            continue
        spec = resolve(spec, data_root)
        tried = spec.pop("_tried_roots", [spec["root"]])
        if not os.path.exists(spec["root"]):
            print(f"[miss] {name}: 目录不存在。试过：{tried}")
            print(f"        -> 先跑 {spec.get('download', '见 docs/01_dataset_survey.md')}")
            continue
        try:
            ad = build_adapter(spec, taxonomy=tax)
        except Exception as e:
            print(f"[fail] {name}: adapter 构建失败 {e}")
            continue

        samples, c = [], Counter()
        for s in ad.iter_samples():
            samples.append(s)
            c[s.label] += 1
            for t in s.defect_types:
                c["type:" + t] += 1
        if not samples:
            print(f"[warn] {name}: 0 条样本，检查目录结构")
            continue
        out_path = os.path.join(args.out, f"{name}.jsonl")
        n = write_jsonl(out_path, samples)
        summary[name] = {"n": n, "root": spec["root"], **{k: v for k, v in c.items()}}
        print(f"[ok]   {name}: {n} 条 -> {out_path}  "
              f"(normal={c['normal']}, anomalous={c['anomalous']})")

    with open(os.path.join(args.out, "_ingest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if not summary:
        print("\n没有任何数据源被处理。请检查 configs/datasets.yaml 里的 root 路径。")
        return 1
    print(f"\n共 {sum(v['n'] for v in summary.values())} 条统一样本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
