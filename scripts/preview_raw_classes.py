#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把某个数据源里每种**原始类别名**抽几张图出来，看清它到底指什么。

接新数据集时常遇到看不懂的类名 —— Real-IAD 用的是拼音缩写（QS / ZW / HS …），
别的集有 class_3 这种。照字面猜着往 taxonomy 里填别名，等于凭空造标签，
而且错了不报错，训练完才发现模型把"缺失"当成"划伤"。

这个脚本按原始类名分组，每组导出几张**框高亮**的对照图，看一眼就知道是什么。

    python scripts/preview_raw_classes.py --only real_iad --out /tmp/raw_cls
    python scripts/preview_raw_classes.py --only real_iad --out /tmp/raw_cls --per-class 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aircraft_vqa.adapters import build_adapter          # noqa: E402
from aircraft_vqa.config import load_dataset_configs      # noqa: E402
from aircraft_vqa.taxonomy import get_taxonomy            # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def resolve(spec: dict, data_root: str) -> dict:
    spec = dict(spec)
    expand = lambda p: os.path.expanduser(str(p).replace("{data_root}", data_root))
    cands = [spec["root"]] + list(spec.pop("root_alternatives", []) or [])
    resolved = [expand(c) for c in cands]
    spec["root"] = next((r for r in resolved if os.path.exists(r)), resolved[0])
    return spec


def render(sample, defects, out_path: str) -> bool:
    """左原图右高亮，框画成品红。"""
    try:
        with Image.open(sample.image_path) as im:
            photo = im.convert("RGB")
    except Exception:
        return False
    hi = photo.copy()
    d = ImageDraw.Draw(hi)
    w, h = photo.size
    lw = max(2, int(min(w, h) * 0.006))
    for df in defects:
        if df.bbox:
            d.rectangle([int(v) for v in df.bbox], outline=(255, 0, 255), width=lw)
    pair = Image.new("RGB", (w * 2, h))
    pair.paste(photo, (0, 0))
    pair.paste(hi, (w, 0))
    if max(pair.size) > 1600:                 # 存太大没必要
        pair.thumbnail((1600, 1600))
    pair.save(out_path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--only", nargs="*", default=None, help="只看这些源")
    ap.add_argument("--out", default="/tmp/raw_classes")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--scan", type=int, default=4000, help="最多扫多少条样本")
    ap.add_argument("--unmapped-only", action="store_true",
                    help="只导出落到 other_anomaly 的类名")
    args = ap.parse_args()

    tax = get_taxonomy(os.path.abspath(args.taxonomy))
    specs, _ = load_dataset_configs(args.config or ["configs/datasets.yaml"])
    data_root = os.path.expanduser(args.data_root)
    os.makedirs(args.out, exist_ok=True)

    n_total = 0
    for spec in specs:
        name = spec["name"]
        if args.only and name not in args.only:
            continue
        if not args.only and not spec.get("enabled", False):
            continue
        spec = resolve(spec, data_root)
        if not os.path.exists(spec["root"]):
            continue

        # 原始类名 -> [(样本, 该类名的缺陷列表)]
        by_raw = defaultdict(list)
        ad = build_adapter({k: v for k, v in spec.items()
                            if not k.startswith("_")}, taxonomy=tax)
        for i, s in enumerate(ad.iter_samples()):
            if i >= args.scan:
                break
            groups = defaultdict(list)
            for d in s.defects:
                if not d.type_raw:
                    continue
                if args.unmapped_only and d.type != "other_anomaly":
                    continue
                groups[d.type_raw].append(d)
            for raw, ds in groups.items():
                if len(by_raw[raw]) < args.per_class * 4:   # 留点余量再随机挑
                    by_raw[raw].append((s, ds))
        if not by_raw:
            continue

        print(f"\n{'=' * 60}\n{name}：{len(by_raw)} 种原始类名")
        rng = random.Random(0)
        for raw in sorted(by_raw):
            mapped = tax.map_defect(raw)
            flag = "  <== 未映射" if mapped == "other_anomaly" else ""
            print(f"  {raw:24s} -> {mapped}{flag}")
            picks = by_raw[raw]
            rng.shuffle(picks)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
            for k, (s, ds) in enumerate(picks[:args.per_class]):
                out = os.path.join(args.out, f"{name}__{safe}__{k}.jpg")
                if render(s, ds, out):
                    n_total += 1

    print(f"\n共导出 {n_total} 张到 {args.out}")
    print("左原图 / 右框高亮。看明白某个类名是什么之后，把它加进 "
          "configs/taxonomy.yaml 对应类型的 aliases。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
