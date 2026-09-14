#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看一眼分割 mask 里到底有哪些像素值，好把 class_map 填对。

分割类数据集的 mask 取值各家不同：有的 0/1/2/3，有的 0/85/170/255，
有的干脆是调色板 PNG。填错 class_map 会让整个等级映射失效且不报错，
所以接新的分割数据集之前先跑这个。

    python scripts/inspect_masks.py --masks ~/data/raw/corrosion_cs/masks
    python scripts/inspect_masks.py --masks ~/data/raw/corrosion_cs/masks -n 50
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="mask 目录")
    ap.add_argument("-n", "--num", type=int, default=30, help="抽查张数")
    args = ap.parse_args()

    d = os.path.expanduser(args.masks)
    if not os.path.isdir(d):
        print(f"目录不存在：{d}")
        return 1
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    if not files:
        print(f"{d} 下没有图片文件")
        return 1
    files = files[:args.num]

    px = Counter()          # 像素值 -> 出现的像素数
    imgs = Counter()        # 像素值 -> 出现在多少张图里
    modes = Counter()
    for fn in files:
        with Image.open(os.path.join(d, fn)) as im:
            modes[im.mode] += 1
            arr = np.array(im.convert("P") if im.mode == "P" else im.convert("L"))
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, cnts):
            px[int(v)] += int(c)
            imgs[int(v)] += 1

    total = sum(px.values())
    print(f"抽查 {len(files)} 张，图像模式 {dict(modes)}\n")
    print(f"{'像素值':>8s} {'占总像素':>10s} {'出现在几张图':>14s}")
    print("-" * 36)
    for v in sorted(px):
        print(f"{v:>8d} {px[v] / total:>9.2%} {imgs[v]:>14d}")

    nz = [v for v in sorted(px) if v != 0]
    print(f"\n非零取值共 {len(nz)} 种：{nz}")
    if len(nz) > 12:
        print("取值太多，这可能不是类别索引图（也许是原图放错目录了）。")
    else:
        print("\n按占比从大到小排（面积最大的通常是最轻的等级），"
              "填进 configs/datasets.yaml 的 class_map：")
        ordered = sorted(nz, key=lambda v: -px[v])
        guess = ["good", "fair", "poor", "severe"]
        pairs = ", ".join(f"{v}: {guess[i]}" if i < len(guess) else f"{v}: ?"
                          for i, v in enumerate(ordered))
        print(f"  class_map: {{{pairs}}}")
        print("  ↑ 这只是按面积占比的猜测，**务必用 visualize 或看图核对**，"
              "等级弄反了比没有等级更糟。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
