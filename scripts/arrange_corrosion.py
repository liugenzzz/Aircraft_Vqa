#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 VT 腐蚀分级集整理成 adapter 认识的 images/ + masks/ 平铺结构。

官方 zip 解出来是这样的（目录名带空格，层级也不是我们要的）：

    Corrosion Condition State Classification/
      512x512/Train/...      512x512/Test/...
      original/Train/...     original/Test/...

Train/Test 里面原图和 mask 具体叫什么、分不分子目录，各版本不一定一致，
所以这里不写死名字，改成**看像素判断**：mask 的取值个数极少（就是那几个
等级索引），原图的取值成千上万。判出来之后按文件名主干配对，软链到

    <root>/images/<split>_<stem>.<ext>
    <root>/masks/<split>_<stem>.<ext>

带上 split 前缀是因为 Train 和 Test 里可能有重名文件，平铺后会互相覆盖。

最后会把 mask 的像素取值直方图打出来 —— configs/datasets.yaml 里
corrosion_cs_vt 的 class_map 就照着这个填。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 取值个数不超过这个数就认为是 mask（四级 + 背景，留点余量）
MASK_MAX_UNIQUE = 16


def images_in(d: str) -> list:
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    return [n for n in names if n.lower().endswith(IMG_EXT)
            and os.path.isfile(os.path.join(d, n))]


def looks_like_mask(path: str) -> bool:
    """靠取值个数区分 mask 和照片。

    灰度分级图取值就那么几个；RGB 存的 mask 三通道相等，也照样只有几个值。
    照片随便一张都是几万种取值，不会误判。
    """
    try:
        with Image.open(path) as im:
            im.draft("L", (256, 256))       # 大图别整张解码
            a = np.array(im.convert("L"))
    except Exception:
        return False
    return len(np.unique(a)) <= MASK_MAX_UNIQUE


def classify(d: str, n_probe: int = 8) -> str:
    """返回 'mask' / 'photo' / ''（空目录）。"""
    files = images_in(d)
    if not files:
        return ""
    probe = files[:: max(1, len(files) // n_probe)][:n_probe]
    votes = Counter("mask" if looks_like_mask(os.path.join(d, f)) else "photo"
                    for f in probe)
    return votes.most_common(1)[0][0]


def scan(root: str) -> dict:
    """走一遍目录树，把每个含图目录判成 mask 还是 photo。"""
    out = {}
    for dirpath, _dirnames, _files in os.walk(root):
        # images/ masks/ 是我们自己的产物，重跑时别把它们再当输入
        rel = os.path.relpath(dirpath, root)
        if rel.split(os.sep)[0] in ("images", "masks"):
            continue
        kind = classify(dirpath)
        if kind:
            out[dirpath] = kind
    return out


# mask 文件名常见的附加后缀；配对时两边都剥掉再比
MASK_SUFFIX = ("_mask", "_gt", "_label", "_seg", "_annot", "_annotation",
               "-mask", "-gt", "-label")


def stem_key(name: str) -> str:
    """文件名主干，剥掉 mask 常见后缀，供两边配对。"""
    stem = os.path.splitext(name)[0]
    low = stem.lower()
    for suf in MASK_SUFFIX:
        if low.endswith(suf):
            return stem[: -len(suf)]
    return stem


def split_of(root: str, d: str) -> str:
    """从相对路径里认出 Train/Test，认不出就用目录名兜底。"""
    parts = [p.lower() for p in os.path.relpath(d, root).split(os.sep)]
    for p in parts:
        if p in ("train", "training"):
            return "train"
        if p in ("test", "testing", "val", "valid", "validation"):
            return "test"
    return "all"


def link(src: str, dst: str, copy: bool) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/corrosion_cs",
                    help="corrosion_cs 目录（zip 就解在它下面）")
    ap.add_argument("--variant", default="auto",
                    help="用哪一份：512x512 / original / auto（优先 512x512）")
    ap.add_argument("--copy", action="store_true", help="复制而不是软链")
    ap.add_argument("--dry-run", action="store_true", help="只看不动")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"找不到 {root}")
        return 1

    print(f"扫描 {root} …")
    kinds = scan(root)
    if not kinds:
        print("一张图都没扫到 —— zip 解压了吗？")
        return 1

    for d, k in sorted(kinds.items()):
        print(f"  [{k:5s}] {os.path.relpath(d, root)}  ({len(images_in(d))} 张)")

    # 选 variant：512x512 已经是正方形小图，喂 VLM 更省，默认优先
    def variant_of(d: str) -> str:
        parts = os.path.relpath(d, root).split(os.sep)
        for p in parts:
            if p.lower() in ("512x512", "original"):
                return p.lower()
        return ""

    variants = {variant_of(d) for d in kinds} - {""}
    pick = args.variant.lower()
    if pick == "auto":
        pick = "512x512" if "512x512" in variants else (
            "original" if "original" in variants else "")
    if pick:
        kinds = {d: k for d, k in kinds.items() if variant_of(d) in (pick, "")}
        print(f"\n采用 variant: {pick}（可选 {sorted(variants)}）")

    photos = {d for d, k in kinds.items() if k == "photo"}
    masks = {d for d, k in kinds.items() if k == "mask"}
    if not photos or not masks:
        print(f"\n没能同时认出原图和 mask 目录（photo={len(photos)} mask={len(masks)}）。")
        print("上面的分类表贴给我，我按实际结构改。")
        return 1

    # 按 split 归拢，再按文件名主干配对
    mask_by_split: dict = {}
    for d in masks:
        mask_by_split.setdefault(split_of(root, d), []).append(d)

    img_out = os.path.join(root, "images")
    msk_out = os.path.join(root, "masks")
    if not args.dry_run:
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(msk_out, exist_ok=True)

    n_pair, n_orphan = 0, 0
    used_masks = []
    for pd in sorted(photos):
        sp = split_of(root, pd)
        cands = mask_by_split.get(sp) or [d for ds in mask_by_split.values() for d in ds]
        index = {}
        for md in cands:
            for f in images_in(md):
                index.setdefault(stem_key(f), os.path.join(md, f))
        for f in images_in(pd):
            stem = stem_key(f)
            mp = index.get(stem)
            if not mp:
                n_orphan += 1
                continue
            name = f"{sp}_{stem}"
            if not args.dry_run:
                link(os.path.join(pd, f), os.path.join(img_out, name + os.path.splitext(f)[1]), args.copy)
                link(mp, os.path.join(msk_out, name + os.path.splitext(mp)[1]), args.copy)
            used_masks.append(mp)
            n_pair += 1

    print(f"\n配对成功 {n_pair} 对；原图找不到对应 mask 的 {n_orphan} 张"
          + ("（dry-run，没动文件）" if args.dry_run else ""))
    if n_orphan and n_orphan > max(1, n_pair * 0.1):
        print("  ⚠ 落单比例偏高，说明两边文件名对不上（已经试过剥 "
              f"{'/'.join(MASK_SUFFIX[:4])} 这类后缀了）——"
              "把上面的分类表和两边各几个文件名贴给我")

    # mask 取值直方图 —— class_map 照这个填
    hist: Counter = Counter()
    for mp in used_masks[:: max(1, len(used_masks) // 60)][:60]:
        try:
            with Image.open(mp) as im:
                hist.update(np.unique(np.array(im.convert("L"))).tolist())
        except Exception:
            continue
    if hist:
        print("\nmask 像素取值（抽样 60 张，数字 = 出现在多少张图里）：")
        for v, c in sorted(hist.items()):
            print(f"  {v:3d}  ->  {c} 张")
        print("\n把这些取值填进 configs/datasets.yaml 的 corrosion_cs_vt.class_map，"
              "形如 {0: background, 1: good, 2: fair, 3: poor, 4: severe}。")
        print("⚠ 等级顺序（good<fair<poor<severe）填反了比不用这个数据集更糟，"
              "对照 'Corrosion Annotation Guidelines.pdf' 核实，"
              "再用 scripts/visualize.py 抽查几张。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
