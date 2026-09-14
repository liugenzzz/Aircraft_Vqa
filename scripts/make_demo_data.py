#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""程序化合成"飞机蒙皮 + 紧固件阵列"图像，带精确 COCO 标注。

为什么需要它：公开数据里"飞机螺丝缺失"的标注只有几百框，远不够训练。
合成数据的好处是**缺陷位置天然精确**（我们知道抹掉了哪一颗），
可以无限量生成定位样本，用来打底；真实数据再往上叠域适应。

    python scripts/make_demo_data.py --out ~/data/raw/synthetic_panel -n 300

产出 COCO 目录，可直接被 configs/datasets.yaml 的 synthetic_panel 条目读取。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [
    {"id": 1, "name": "missing_fastener",  "supercategory": "defect"},
    {"id": 2, "name": "corrosion",         "supercategory": "defect"},
    {"id": 3, "name": "scratch",           "supercategory": "defect"},
    {"id": 4, "name": "dent",              "supercategory": "defect"},
    {"id": 5, "name": "paint_peel_off",    "supercategory": "defect"},
    {"id": 6, "name": "loose_fastener",    "supercategory": "defect"},
]
NAME2ID = {c["name"]: c["id"] for c in CATEGORIES}


# ------------------------------------------------------------------ 底图
# 常见蒙皮涂装：裸铝 / 白漆 / 灰底漆 / 军绿底漆
PAINTS = [(176, 178, 184), (216, 216, 214), (150, 152, 156), (128, 134, 120)]


def make_panel(w: int, h: int, rng: random.Random) -> Image.Image:
    tint = rng.choice(PAINTS)
    k = rng.uniform(0.82, 1.12)
    base = int(min(235, tint[0] * k))
    img = Image.new("RGB", (w, h), tuple(int(min(240, c * k)) for c in tint))
    d = ImageDraw.Draw(img)
    # 纵向光照梯度，模拟机身曲面反光
    for y in range(h):
        k = 1.0 - abs(y - h * rng.uniform(0.3, 0.7)) / h
        v = int(28 * k)
        d.line([(0, y), (w, y)], fill=(base + v, base + v + 3, base + v + 8))
    # 壁板缝
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            x = rng.randint(int(w * 0.15), int(w * 0.85))
            d.line([(x, 0), (x, h)], fill=(base - 30, base - 28, base - 24),
                   width=rng.randint(2, 4))
        else:
            y = rng.randint(int(h * 0.15), int(h * 0.85))
            d.line([(0, y), (w, y)], fill=(base - 30, base - 28, base - 24),
                   width=rng.randint(2, 4))
    # 反光斑：机身曲面上的高光带
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    for _ in range(rng.randint(1, 3)):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(w // 5, w // 2), rng.randint(h // 6, h // 2)
        dl.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                   fill=(255, 255, 255, rng.randint(12, 34)))
    img = img.convert("RGBA")
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(w / 12)))
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(0.4))


def add_grain_and_vignette(img: Image.Image, rng: random.Random) -> Image.Image:
    """噪点 + 暗角 —— 真实检查照片几乎都有，缺了会让模型对噪声过敏。"""
    import numpy as np
    w, h = img.size
    arr = np.asarray(img, dtype=np.int16)
    amp = rng.randint(3, 11)
    noise = np.random.default_rng(rng.getrandbits(32)).integers(
        -amp, amp + 1, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255)

    yy, xx = np.mgrid[0:h, 0:w]
    m = rng.uniform(0.06, 0.16)
    nx = (xx - w / 2.0) / (w / 2.0 * (1 + m))
    ny = (yy - h / 2.0) / (h / 2.0 * (1 + m))
    vig = np.clip(1.0 - 0.55 * (nx ** 2 + ny ** 2), 0.35, 1.0)[..., None]
    arr = (arr.astype(np.float32) * vig).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def rotate_with_boxes(img: Image.Image, anns: list, rng: random.Random) -> tuple:
    """小角度旋转，同时把标注框按旋转后的四角重新取外接矩形。"""
    deg = rng.uniform(-7.0, 7.0)
    if abs(deg) < 0.6:
        return img, anns
    w, h = img.size
    rot = img.rotate(deg, resample=Image.BICUBIC, expand=False,
                     fillcolor=(120, 122, 126))
    # 裁掉旋转产生的填充角 —— 留着的话模型会把黑角当成"有缺陷"的伪特征
    sin = abs(math.sin(math.radians(deg)))
    mx = int(math.ceil(h * sin)) + 2
    my = int(math.ceil(w * sin)) + 2
    mx = min(mx, w // 4); my = min(my, h // 4)
    rot = rot.crop((mx, my, w - mx, h - my)).resize((w, h), Image.BICUBIC)
    sx = w / float(w - 2 * mx)
    sy = h / float(h - 2 * my)

    th = math.radians(-deg)                      # PIL 逆时针为正
    cx, cy = w / 2.0, h / 2.0
    out = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        pts = [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]
        xs, ys = [], []
        for px_, py_ in pts:
            dx, dy = px_ - cx, py_ - cy
            xs.append((cx + dx * math.cos(th) - dy * math.sin(th) - mx) * sx)
            ys.append((cy + dx * math.sin(th) + dy * math.cos(th) - my) * sy)
        nx1 = max(0, min(xs)); ny1 = max(0, min(ys))
        nx2 = min(w, max(xs)); ny2 = min(h, max(ys))
        if nx2 - nx1 < 4 or ny2 - ny1 < 4:
            continue
        out.append({"category": a["category"],
                    "bbox": [int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1)],
                    "area": int((nx2 - nx1) * (ny2 - ny1))})
    return rot, out


def draw_fastener(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int,
                  rng: random.Random, protruding: bool = False) -> None:
    """画一颗铆钉/螺钉：底色 + 高光 + 阴影 + 十字槽。"""
    shade = rng.randint(-12, 12)
    body = (168 + shade, 170 + shade, 176 + shade)
    d.ellipse([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1],
              fill=(110, 112, 118))                      # 座圈阴影
    rr = r + (2 if protruding else 0)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=body,
              outline=(128, 130, 136))
    d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 200, 340,
          fill=(214, 216, 220), width=max(1, r // 3))     # 高光
    if r >= 5:                                            # 十字槽
        d.line([(cx - r + 2, cy), (cx + r - 2, cy)], fill=(120, 122, 128))
        d.line([(cx, cy - r + 2), (cx, cy + r - 2)], fill=(120, 122, 128))
    if protruding:                                        # 松动：投影拉长
        d.ellipse([cx - rr + 2, cy - rr + 3, cx + rr + 2, cy + rr + 3],
                  outline=(95, 97, 103), width=2)


def draw_hole(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    """紧固件缺失后留下的空孔。"""
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(70, 68, 66),
              outline=(52, 50, 48))
    d.arc([cx - r, cy - r, cx + r, cy + r], 20, 160, fill=(118, 114, 110), width=2)


def draw_corrosion(img: Image.Image, cx: int, cy: int, r: int,
                   rng: random.Random) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(rng.randint(30, 70)):
        a = rng.uniform(0, math.tau)
        dist = rng.uniform(0, r)
        px, py = cx + math.cos(a) * dist, cy + math.sin(a) * dist
        rad = rng.uniform(1.5, r * 0.3)
        col = (rng.randint(130, 175), rng.randint(70, 105), rng.randint(25, 55),
               rng.randint(110, 210))
        d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=col)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.1)))


def draw_scratch(d: ImageDraw.ImageDraw, x1, y1, x2, y2, rng: random.Random) -> None:
    d.line([(x1, y1), (x2, y2)], fill=(232, 232, 236), width=rng.randint(2, 4))
    d.line([(x1 + 1, y1 + 2), (x2 + 1, y2 + 2)], fill=(120, 120, 126), width=1)


def draw_dent(img: Image.Image, cx, cy, r, rng: random.Random) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(72, 74, 80, 130))
    d.ellipse([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6],
              fill=(52, 54, 60, 120))
    w_ = max(3, r // 4)
    d.arc([cx - r, cy - r, cx + r, cy + r], 25, 185, fill=(38, 40, 46, 215), width=w_)
    d.arc([cx - r, cy - r, cx + r, cy + r], 195, 355, fill=(245, 246, 250, 215), width=w_)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(1.5, r / 8))))


def draw_peel(img: Image.Image, cx, cy, r, rng: random.Random) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pts = []
    for i in range(rng.randint(6, 10)):
        a = math.tau * i / 8
        dist = r * rng.uniform(0.5, 1.0)
        pts.append((cx + math.cos(a) * dist, cy + math.sin(a) * dist))
    d.polygon(pts, fill=(205, 200, 188, 205), outline=(120, 116, 108, 235))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.6)))


# ------------------------------------------------------------------ 主流程
def gen_image(idx: int, rng: random.Random, size=(768, 576)) -> tuple:
    w, h = size
    img = make_panel(w, h, rng).convert("RGBA")
    d = ImageDraw.Draw(img)

    # 紧固件阵列
    r = rng.randint(6, 11)
    gap = rng.randint(int(r * 4.5), int(r * 7))
    rows = max(2, (h - 60) // gap)
    cols = max(3, (w - 60) // gap)
    x0 = (w - (cols - 1) * gap) // 2
    y0 = (h - (rows - 1) * gap) // 2
    jitter = max(1, r // 3)
    slots = []
    for i in range(rows):
        for j in range(cols):
            slots.append((x0 + j * gap + rng.randint(-jitter, jitter),
                          y0 + i * gap + rng.randint(-jitter, jitter)))

    anns = []
    n_def = rng.choices([0, 1, 2, 3], weights=[0.28, 0.38, 0.22, 0.12])[0]
    chosen = rng.sample(range(len(slots)), k=min(n_def, len(slots)))
    defect_slots = {}
    for si in chosen:
        defect_slots[si] = rng.choices(
            ["missing_fastener", "loose_fastener", "corrosion",
             "scratch", "dent", "paint_peel_off"],
            weights=[0.34, 0.14, 0.22, 0.12, 0.10, 0.08])[0]

    # 先画所有正常紧固件（缺失的画成孔，松动的画成外凸）
    for si, (cx, cy) in enumerate(slots):
        kind = defect_slots.get(si)
        if kind == "missing_fastener":
            draw_hole(d, cx, cy, r)
        elif kind == "loose_fastener":
            draw_fastener(d, cx, cy, r, rng, protruding=True)
        else:
            draw_fastener(d, cx, cy, r, rng)

    # 再叠加面状/线状缺陷
    for si, kind in defect_slots.items():
        cx, cy = slots[si]
        if kind == "missing_fastener":
            pad = int(r * 1.6)
            box = [cx - pad, cy - pad, cx + pad, cy + pad]
        elif kind == "loose_fastener":
            pad = int(r * 1.9)
            box = [cx - pad, cy - pad, cx + pad, cy + pad]
        elif kind == "corrosion":
            rad = rng.randint(int(r * 1.8), int(r * 3.4))
            draw_corrosion(img, cx, cy, rad, rng)
            box = [cx - rad, cy - rad, cx + rad, cy + rad]
        elif kind == "scratch":
            ln = rng.randint(int(r * 5), int(r * 12))
            a = rng.uniform(0, math.pi)
            x2, y2 = cx + math.cos(a) * ln, cy + math.sin(a) * ln
            draw_scratch(d, cx, cy, x2, y2, rng)
            box = [min(cx, x2) - 4, min(cy, y2) - 4, max(cx, x2) + 4, max(cy, y2) + 4]
        elif kind == "dent":
            rad = rng.randint(int(r * 2.0), int(r * 3.6))
            draw_dent(img, cx, cy, rad, rng)
            box = [cx - rad, cy - rad, cx + rad, cy + rad]
        else:  # paint_peel_off
            rad = rng.randint(int(r * 1.8), int(r * 3.2))
            draw_peel(img, cx, cy, rad, rng)
            box = [cx - rad, cy - rad, cx + rad, cy + rad]

        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        anns.append({"category": kind, "bbox": [x1, y1, x2 - x1, y2 - y1],
                     "area": (x2 - x1) * (y2 - y1)})

    out = img.convert("RGB")
    if rng.random() < 0.6:
        out, anns = rotate_with_boxes(out, anns, rng)
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    out = add_grain_and_vignette(out, rng)
    return out, anns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/raw/synthetic_panel")
    ap.add_argument("-n", "--num", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=576)
    args = ap.parse_args()

    root = os.path.expanduser(args.out)
    rng = random.Random(args.seed)
    splits = {"train": int(args.num * 0.8),
              "valid": int(args.num * 0.1)}
    splits["test"] = args.num - splits["train"] - splits["valid"]

    gi = 0
    for split, n in splits.items():
        sdir = os.path.join(root, split)
        os.makedirs(sdir, exist_ok=True)
        images, annotations, aid = [], [], 1
        for i in range(n):
            img, anns = gen_image(gi, rng, (args.width, args.height))
            fn = f"panel_{gi:06d}.jpg"
            img.save(os.path.join(sdir, fn), quality=92)
            images.append({"id": i, "file_name": fn,
                           "width": img.width, "height": img.height})
            for a in anns:
                annotations.append({"id": aid, "image_id": i,
                                    "category_id": NAME2ID[a["category"]],
                                    "bbox": a["bbox"], "area": a["area"],
                                    "iscrowd": 0})
                aid += 1
            gi += 1
        with open(os.path.join(sdir, "_annotations.coco.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"images": images, "annotations": annotations,
                       "categories": CATEGORIES}, f, ensure_ascii=False)
        n_anom = len({a["image_id"] for a in annotations})
        print(f"[{split}] {n} 张（含缺陷 {n_anom} 张 / 正常 {n - n_anom} 张），"
              f"{len(annotations)} 个缺陷框 -> {sdir}")
    print(f"\n完成 -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
