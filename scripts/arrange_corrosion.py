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

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from aircraft_vqa.adapters.base import load_mask   # noqa: E402

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

    分级图取值就那么几个（等级索引）；照片随便一张都是几万种，不会误判。

    这里刻意**不**走 load_mask：那个函数会对彩色图做 VOC 调色板反查，
    而扫描阶段每张照片都要过一遍，几万种颜色查下来又慢又刷屏 ——
    判类型只需要数取值个数，数超了立刻早退。
    """
    try:
        with Image.open(path) as im:
            if im.mode == "P":
                a = np.array(im)
            elif im.mode in ("L", "1"):
                im.draft("L", (256, 256))       # JPEG 大图别整张解码
                a = np.array(im.convert("L"))
            elif im.mode.startswith("I") or im.mode == "F":
                a = np.array(im)
            else:
                im.draft("RGB", (256, 256))
                arr = np.array(im.convert("RGB"))
                if ((arr[..., 0] == arr[..., 1]).all()
                        and (arr[..., 1] == arr[..., 2]).all()):
                    a = arr[..., 0]             # 灰度图存成了 RGB
                else:
                    cols = np.unique(arr.reshape(-1, 3), axis=0)
                    return len(cols) <= MASK_MAX_UNIQUE
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


def grade_stats(img_dir: str, msk_dir: str, n_max: int = 120) -> dict:
    """统计每个类别索引底下**原图像素**长什么样，把等级定序变成可验的数。

    锈蚀从轻到重，外观是有方向的：红棕色加深（redness 升）、
    表面变暗（brightness 降）、起皮点蚀让局部更粗糙（roughness 升）。
    面积占比和出现频率则应当递减 —— 完好区域总是最大最常见。

    单看任何一项都可能被光照带偏，但四项一起同向，就不是巧合了。
    """
    acc: dict = {}
    files = images_in(msk_dir)
    files = files[:: max(1, len(files) // n_max)][:n_max]
    stem2img = {os.path.splitext(f)[0]: f for f in images_in(img_dir)}
    n_img = 0
    for fn in files:
        ip = stem2img.get(os.path.splitext(fn)[0])
        if not ip:
            continue
        try:
            with Image.open(os.path.join(img_dir, ip)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            a = load_mask(os.path.join(msk_dir, fn))
        except Exception:
            continue
        if a is None or a.shape[:2] != rgb.shape[:2]:
            continue
        n_img += 1
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        redness = (r - b) / (r + b + 1.0)
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        sat = (mx - mn) / (mx + 1.0)
        # 局部粗糙度：相邻像素灰度差，起皮/点蚀会把它抬起来
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:] = np.abs(np.diff(gray, axis=1))
        gy[1:, :] = np.abs(np.diff(gray, axis=0))
        rough = gx + gy

        for v in np.unique(a).tolist():
            sel = a == v
            n = int(sel.sum())
            if not n:
                continue
            d = acc.setdefault(int(v), {"px": 0, "imgs": 0, "redness": 0.0,
                                        "bright": 0.0, "sat": 0.0, "rough": 0.0})
            d["px"] += n
            d["imgs"] += 1
            d["redness"] += float(redness[sel].sum())
            d["bright"] += float(gray[sel].sum())
            d["sat"] += float(sat[sel].sum())
            d["rough"] += float(rough[sel].sum())
    for d in acc.values():
        for k in ("redness", "bright", "sat", "rough"):
            d[k] /= max(1, d["px"])
    return {"n_img": n_img, "by_index": acc}


def print_grade_stats(st: dict) -> None:
    acc = st["by_index"]
    if not acc:
        print("没统计到东西 —— images/ 和 masks/ 对上了吗？")
        return
    total = sum(d["px"] for d in acc.values()) or 1
    idxs = sorted(acc)
    print(f"\n各索引底下原图像素的外观统计（{st['n_img']} 张）：")
    print(f"  {'索引':>4} {'占像素':>8} {'出现图数':>8} {'红棕度':>8} "
          f"{'亮度':>7} {'饱和度':>8} {'粗糙度':>8}")
    for v in idxs:
        d = acc[v]
        print(f"  {v:>4d} {d['px'] / total:>8.2%} {d['imgs']:>8d} "
              f"{d['redness']:>8.3f} {d['bright']:>7.1f} "
              f"{d['sat']:>8.3f} {d['rough']:>8.2f}")

    fg = [v for v in idxs if v != 0]
    if len(fg) < 2:
        return

    def mono(key, want_up):
        seq = [acc[v][key] for v in fg]
        ok = all((b > a) if want_up else (b < a) for a, b in zip(seq, seq[1:]))
        return ok, seq

    checks = [("红棕度递增", "redness", True), ("亮度递减", "bright", False),
              ("粗糙度递增", "rough", True), ("面积递减", "px", False),
              ("出现图数递减", "imgs", False)]
    print("\n  索引越大 = 越严重的话，应当看到：")
    hit = 0
    for label, key, up in checks:
        ok, _ = mono(key, up)
        hit += ok
        print(f"    {'✓' if ok else '✗'} {label}")
    print(f"\n  {hit}/{len(checks)} 项支持「索引越大越严重」。")
    if hit >= 4:
        print("  -> 按 {1: fair, 2: poor, 3: severe} 填，方向是对的。")
    elif hit <= 1:
        print("  -> 方向很可能是反的，class_map 要倒过来填。")
    else:
        print("  -> 证据不够干净，务必用 --grade-preview 出的图肉眼核对一遍。")


def grade_preview(img_dir: str, msk_dir: str, out_dir: str, per_grade: int) -> int:
    """每个类别索引挑几张该索引占比最大的图，左原图右高亮，供肉眼定序。

    等级顺序是这份数据里唯一没法从文件本身推出来的东西，填反了模型会学成
    "锈得越狠越说没事"。与其让人对着 PDF 硬猜，不如直接把图摆出来。
    """
    os.makedirs(out_dir, exist_ok=True)
    by_idx: dict = {}
    for fn in images_in(msk_dir):
        a = load_mask(os.path.join(msk_dir, fn))
        if a is None:
            continue
        vals, counts = np.unique(a, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            by_idx.setdefault(int(v), []).append((c / a.size, fn))

    stem2img = {os.path.splitext(f)[0]: f for f in images_in(img_dir)}
    n_out = 0
    for idx in sorted(by_idx):
        for frac, fn in sorted(by_idx[idx], reverse=True)[:per_grade]:
            ip = stem2img.get(os.path.splitext(fn)[0])
            if not ip:
                continue
            try:
                with Image.open(os.path.join(img_dir, ip)) as im:
                    photo = np.array(im.convert("RGB"))
                a = load_mask(os.path.join(msk_dir, fn))
            except Exception:
                continue
            if a is None or a.shape[:2] != photo.shape[:2]:
                continue
            hi = photo.copy()
            sel = a == idx
            # 该索引区域压成品红，其余保持原样 —— 一眼看出标的是哪块
            hi[sel] = (0.35 * hi[sel] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
            pair = np.concatenate([photo, hi], axis=1)
            name = f"idx{idx}_{frac:.2f}_{os.path.splitext(ip)[0]}.png"
            Image.fromarray(pair).save(os.path.join(out_dir, name))
            n_out += 1
    return n_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/corrosion_cs",
                    help="corrosion_cs 目录（zip 就解在它下面）")
    ap.add_argument("--variant", default="auto",
                    help="用哪一份：512x512 / original / auto（优先 512x512）")
    ap.add_argument("--copy", action="store_true", help="复制而不是软链")
    ap.add_argument("--dry-run", action="store_true", help="只看不动")
    ap.add_argument("--grade-preview", metavar="DIR", default="",
                    help="每个类别索引导出几张预览图，用来肉眼核对等级顺序")
    ap.add_argument("--preview-per-grade", type=int, default=4)
    ap.add_argument("--grade-stats", action="store_true",
                    help="统计每个索引底下原图的外观，用数值判断等级方向")
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
    modes: Counter = Counter()
    area: Counter = Counter()
    probe = used_masks[:: max(1, len(used_masks) // 60)][:60]
    for mp in probe:
        try:
            with Image.open(mp) as im:
                modes[im.mode] += 1
            a = load_mask(mp)
        except Exception:
            continue
        if a is None:
            continue
        vals, counts = np.unique(a, return_counts=True)
        hist.update(vals.tolist())
        for v, c in zip(vals.tolist(), counts.tolist()):
            area[v] += int(c)
    if hist:
        total_px = sum(area.values()) or 1
        print(f"\nmask 存储模式：{dict(modes)}")
        print(f"mask 类别索引（抽样 {len(probe)} 张，走的是 adapter 同款 load_mask）：")
        print(f"  {'索引':>4}  {'出现在多少张':>12}  {'占总像素':>9}")
        for v, c in sorted(hist.items()):
            print(f"  {v:>4d}  {c:>12d}  {area[v] / total_px:>8.2%}")
        print("\n把这些索引填进 configs/datasets.yaml 的 corrosion_cs_vt.class_map。")
        print("⚠ 等级顺序（good<fair<poor<severe）填反了比不用这个数据集更糟，"
              "对照 'Corrosion Annotation Guidelines.pdf' 核实，"
              "再用 --grade-preview 抽查几张。")

    if args.grade_stats and not args.dry_run:
        print_grade_stats(grade_stats(img_out, msk_out))

    if args.grade_preview and not args.dry_run:
        n = grade_preview(img_out, msk_out, args.grade_preview,
                          args.preview_per_grade)
        print(f"\n已导出 {n} 张预览到 {args.grade_preview}")
        print("文件名形如 idx3_0.72_xxx.png（索引_该索引占图比例_原名），"
              "左原图右高亮。按索引从小到大看一遍，锈蚀程度应当递增；"
              "如果反了，class_map 就要倒过来填。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
