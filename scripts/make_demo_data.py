#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""程序化合成"飞机蒙皮 + 紧固件阵列"图像，带精确 COCO 标注。

为什么需要它：公开数据里"飞机螺丝缺失"的标注只有几百框，远不够训练。
合成数据的好处是**缺陷位置天然精确**（我们知道抹掉了哪一颗），
可以无限量生成定位样本，用来打底；真实数据再往上叠域适应。

    python scripts/make_demo_data.py --out ~/data/raw/synthetic_panel -n 300

两种场景：
  panel   蒙皮铆钉阵列（俯视）—— 紧固件缺失/松动、锈蚀、划伤、凹坑、漆层剥落、
          裂纹（从紧固件孔边起裂，符合疲劳裂纹的真实萌生位置）
  closeup 紧固件特写（侧视）—— 螺纹损伤、杆部锈蚀、头部划伤
          （螺纹在俯视图里根本看不见，必须单独出特写场景）

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
    {"id": 7, "name": "crack",             "supercategory": "defect"},
    {"id": 8, "name": "thread_damage",     "supercategory": "defect"},
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


def draw_crack(img: Image.Image, x0: float, y0: float, length: float,
               angle: float, rng: random.Random) -> list:
    """从 (x0,y0) 起裂的折线裂纹，返回外接框 [x1,y1,x2,y2]。

    真实结构裂纹绝大多数**从紧固件孔边或壁板缝起裂**（孔边应力集中），
    所以调用方传进来的起点是孔边而不是随机位置 —— 这个先验直接决定
    模型学到的是"裂纹长在孔边"还是"裂纹随便长"。
    """
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 主裂纹：分段折线，每段方向小幅抖动，越往尖端越细
    pts = [(x0, y0)]
    n_seg = rng.randint(5, 10)
    seg = length / n_seg
    a = angle
    for i in range(n_seg):
        a += rng.uniform(-0.42, 0.42)
        px, py = pts[-1]
        pts.append((px + math.cos(a) * seg, py + math.sin(a) * seg))
    for i in range(len(pts) - 1):
        t = 1.0 - i / float(len(pts))              # 尖端收细
        w = max(1, int(round(3.0 * t)))
        d.line([pts[i], pts[i + 1]], fill=(28, 26, 30, 240), width=w)
        # 裂纹两侧的亮边（金属翻边反光），没有它看起来像划线
        d.line([(pts[i][0] + 1, pts[i][1] + 1), (pts[i + 1][0] + 1, pts[i + 1][1] + 1)],
               fill=(238, 238, 242, 120), width=1)

    # 分叉：真实疲劳裂纹常在中段分叉
    if rng.random() < 0.45 and len(pts) > 4:
        k = rng.randint(2, len(pts) - 2)
        bx, by = pts[k]
        ba = a + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
        bl = length * rng.uniform(0.25, 0.5)
        bp = [(bx, by)]
        for i in range(3):
            ba += rng.uniform(-0.3, 0.3)
            px, py = bp[-1]
            bp.append((px + math.cos(ba) * bl / 3, py + math.sin(ba) * bl / 3))
        for i in range(len(bp) - 1):
            d.line([bp[i], bp[i + 1]], fill=(34, 32, 36, 225), width=1)
        pts += bp

    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.35)))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    pad = 5
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


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


# ------------------------------------------------------------------ 特写场景
def _rot_point(x, y, cx, cy, ncx, ncy, th):
    dx, dy = x - cx, y - cy
    return (ncx + dx * math.cos(th) - dy * math.sin(th),
            ncy + dx * math.sin(th) + dy * math.cos(th))


def draw_bolt(L: int, W: int, rng: random.Random, damage: str = "") -> tuple:
    """在局部坐标系里画一颗侧视螺栓（螺纹朝右），返回 (RGBA 图, 缺陷局部框, 几何)。

    俯视的铆钉阵列看不见螺纹，所以螺纹损伤必须用特写侧视图来合成。
    螺纹损伤的视觉特征是**周期性被破坏**：牙顶被搓平、牙距不规则、
    留下深色啃痕和亮色毛刺。所以正常段画成严格周期的亮牙顶+暗牙底，
    损伤段则抹掉牙顶、只留不规则啃痕 —— 两者必须一眼可分，否则等于没标。
    """
    HW = int(W * 2.1)                       # 头部宽度
    HL = int(W * 1.5)                       # 头部长度
    pad = 16
    iw, ih = L + HL + pad * 2, HW + pad * 2
    im = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cy = ih // 2
    x_head0, x_head1 = pad, pad + HL
    x_sh0, x_sh1 = x_head1, x_head1 + L

    tone = rng.randint(-14, 14)

    def shade(t: float, lo: int = 78, hi: int = 232) -> tuple:
        """t=0 边缘（暗）, t=1 中轴（亮）—— 圆柱体明暗。"""
        v = int(lo + (hi - lo) * t) + tone
        v = max(0, min(255, v))
        return (v, v + 2, v + 6)

    # 六角头：带倒角的多边形 + 圆柱明暗
    ch = HW // 5
    d.polygon([(x_head0 + ch, cy - HW // 2), (x_head1 - ch, cy - HW // 2),
               (x_head1, cy - HW // 2 + ch), (x_head1, cy + HW // 2 - ch),
               (x_head1 - ch, cy + HW // 2), (x_head0 + ch, cy + HW // 2),
               (x_head0, cy + HW // 2 - ch), (x_head0, cy - HW // 2 + ch)],
              fill=shade(0.55), outline=shade(0.12))
    for i in range(HW // 2):
        t = 1.0 - abs(i - HW // 4) / (HW / 4.0)
        yy = cy - HW // 2 + ch // 2 + i
        d.line([(x_head0 + ch // 2, yy), (x_head1 - ch // 2, yy)],
               fill=shade(0.30 + 0.55 * max(0.0, t)))

    # 光杆：逐行渐变，做出圆柱感
    half = max(2, W // 2)
    for i in range(-half, half + 1):
        t = 1.0 - abs(i) / float(half)
        d.line([(x_sh0, cy + i), (x_sh1, cy + i)], fill=shade(0.18 + 0.72 * t))

    pitch = max(5, int(W * 0.42))
    sk = max(2, pitch // 2)                 # 牙侧倾角
    dmg_x0 = dmg_x1 = None
    if damage == "thread_damage":
        span = rng.randint(int(L * 0.20), int(L * 0.40))
        lo = x_sh0 + int(L * 0.15)
        dmg_x0 = rng.randint(lo, max(lo + 1, x_sh1 - span - 2))
        dmg_x1 = dmg_x0 + span

    x = x_sh0 + pitch
    while x < x_sh1 - 2:
        if dmg_x0 is not None and dmg_x0 <= x <= dmg_x1:
            x += pitch                      # 损伤段：牙顶已被搓平，什么都不画
            continue
        # 正常牙型：暗牙底 + 紧邻的亮牙顶，严格等距
        d.line([(x, cy - half), (x - sk, cy + half)], fill=shade(0.05), width=2)
        d.line([(x + 2, cy - half), (x - sk + 2, cy + half)],
               fill=shade(1.0), width=1)
        x += pitch

    if dmg_x0 is not None:
        # 啃痕：不规则深色凹坑，横跨杆径
        for _ in range(rng.randint(4, 8)):
            gx = rng.randint(dmg_x0, dmg_x1)
            gy = cy + rng.randint(-half, half)
            gl = rng.randint(2, max(3, pitch))
            d.line([(gx, gy - gl // 2), (gx + rng.randint(-2, 2), gy + gl // 2)],
                   fill=shade(0.0, 38, 60), width=rng.randint(2, 3))
        # 毛刺：翻起的金属边，亮而突兀
        for _ in range(rng.randint(2, 4)):
            bx = rng.randint(dmg_x0, dmg_x1)
            by = cy + rng.choice([-1, 1]) * half
            d.line([(bx, by), (bx + rng.randint(2, 5), by + rng.randint(-3, 3))],
                   fill=shade(1.0, 200, 252), width=2)
        # 损伤段轮廓被啃掉一块，杆径出现缺口
        d.line([(dmg_x0 + 2, cy - half), (dmg_x1 - 2, cy - half)],
               fill=shade(0.02, 60, 86), width=2)

    im = im.filter(ImageFilter.GaussianBlur(0.35))
    box = None
    if dmg_x0 is not None:
        box = [dmg_x0 - 5, cy - half - 6, dmg_x1 + 5, cy + half + 6]
    return im, box, (x_head0, x_head1, x_sh0, x_sh1, cy, W, HW)


def gen_closeup(idx: int, rng: random.Random, size=(768, 576)) -> tuple:
    """紧固件特写场景：侧视螺栓，可注入螺纹损伤 / 锈蚀 / 划伤。"""
    w, h = size
    img = make_panel(w, h, rng).convert("RGBA")
    anns = []

    n_bolt = rng.randint(1, 3)
    for _ in range(n_bolt):
        L = rng.randint(int(w * 0.22), int(w * 0.42))
        W = rng.randint(int(L * 0.11), int(L * 0.17))
        kind = rng.choices(["", "thread_damage", "corrosion", "scratch"],
                           weights=[0.34, 0.38, 0.18, 0.10])[0]
        bolt, dmg_local, geo = draw_bolt(L, W, rng,
                                         "thread_damage" if kind == "thread_damage" else "")
        deg = rng.uniform(-35, 35)
        rot = bolt.rotate(deg, resample=Image.BICUBIC, expand=True)
        px = rng.randint(0, max(1, w - rot.width))
        py = rng.randint(0, max(1, h - rot.height))
        img.alpha_composite(rot, (px, py))

        th = math.radians(-deg)
        cx, cy_ = bolt.width / 2.0, bolt.height / 2.0
        ncx, ncy = rot.width / 2.0, rot.height / 2.0

        def to_global(bx):
            pts = [(bx[0], bx[1]), (bx[2], bx[1]), (bx[0], bx[3]), (bx[2], bx[3])]
            gs = [_rot_point(x, y, cx, cy_, ncx, ncy, th) for x, y in pts]
            xs = [g[0] + px for g in gs]; ys = [g[1] + py for g in gs]
            return [max(0, min(xs)), max(0, min(ys)),
                    min(w, max(xs)), min(h, max(ys))]

        if kind == "thread_damage" and dmg_local:
            box = to_global(dmg_local)
        elif kind == "corrosion":
            _, _, x_sh0, x_sh1, bcy, bW, _ = geo
            mid = (x_sh0 + x_sh1) / 2
            g = _rot_point(mid, bcy, cx, cy_, ncx, ncy, th)
            gx, gy = g[0] + px, g[1] + py
            rad = int(bW * rng.uniform(1.1, 1.8))
            draw_corrosion(img, gx, gy, rad, rng)
            box = [gx - rad, gy - rad, gx + rad, gy + rad]
        elif kind == "scratch":
            x_h0, x_h1, _, _, bcy, _, bHW = geo
            g = _rot_point((x_h0 + x_h1) / 2, bcy, cx, cy_, ncx, ncy, th)
            gx, gy = g[0] + px, g[1] + py
            ln = bHW
            draw_scratch(ImageDraw.Draw(img), gx - ln / 2, gy - ln / 3,
                         gx + ln / 2, gy + ln / 3, rng)
            box = [gx - ln / 2 - 3, gy - ln / 3 - 3, gx + ln / 2 + 3, gy + ln / 3 + 3]
        else:
            continue

        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        anns.append({"category": kind, "bbox": [x1, y1, x2 - x1, y2 - y1],
                     "area": (x2 - x1) * (y2 - y1)})

    out = img.convert("RGB")
    if rng.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    out = add_grain_and_vignette(out, rng)
    return out, anns


# ------------------------------------------------------------------ 成像劣化

# 真实检查照片里总有一部分是糊的、逆光的、被挡住的。这类图必须教模型说
# "无法确认、建议补拍"，而不是硬给一个框。system 里写了这条要求，
# 就得有样本把它教出来，否则那句话是空指令。
QUALITY_KINDS = ["blur", "occlusion", "overexposure", "darkness"]


def degrade(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    import numpy as np
    w, h = img.size
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(3.5, 7.0)))
    if kind == "occlusion":
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for _ in range(rng.randint(1, 2)):
            bw = rng.randint(int(w * 0.35), int(w * 0.6))
            bh = rng.randint(int(h * 0.3), int(h * 0.55))
            x = rng.randint(-bw // 4, w - bw // 2)
            y = rng.randint(-bh // 4, h - bh // 2)
            g = rng.randint(20, 55)
            d.rectangle([x, y, x + bw, y + bh], fill=(g, g, g + 4, 235))
        out = img.convert("RGBA")
        out.alpha_composite(layer.filter(ImageFilter.GaussianBlur(2.0)))
        return out.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if kind == "overexposure":
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        r = min(w, h) * rng.uniform(0.45, 0.8)
        glow = np.clip(1.0 - (((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5) / r, 0, 1)
        arr = arr + (glow[..., None] ** 1.5) * rng.uniform(150, 230)
    else:                                        # darkness
        arr = arr * rng.uniform(0.16, 0.30)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


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
             "scratch", "dent", "paint_peel_off", "crack"],
            weights=[0.28, 0.12, 0.18, 0.10, 0.08, 0.06, 0.18])[0]

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
        elif kind == "crack":
            # 从孔边起裂：起点落在紧固件外缘，方向背向圆心
            a = rng.uniform(0, math.tau)
            sx = cx + math.cos(a) * (r + 1)
            sy = cy + math.sin(a) * (r + 1)
            box = draw_crack(img, sx, sy,
                             rng.uniform(r * 4.0, r * 11.0), a, rng)
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


def _emit(root: str, scene: str, counts: dict, rng: random.Random,
          size: tuple, start: int, degraded_frac: float = 0.0) -> int:
    """生成一个场景的 COCO 数据集，返回下一个全局序号。

    degraded_frac 的比例会被做成模糊/遮挡/过曝/欠曝的"拍废了"的图，
    并在 COCO 的 image 条目里记 quality 字段，构建器据此只出"无法确认"样本。
    """
    gen = gen_closeup if scene == "closeup" else gen_image
    prefix = "bolt" if scene == "closeup" else "panel"
    gi = start
    for split, n in counts.items():
        sdir = os.path.join(root, scene, split)
        os.makedirs(sdir, exist_ok=True)
        images, annotations, aid = [], [], 1
        for i in range(n):
            img, anns = gen(gi, rng, size)
            quality = None
            if rng.random() < degraded_frac:
                quality = rng.choice(QUALITY_KINDS)
                img = degrade(img, quality, rng)
            fn = f"{prefix}_{gi:06d}.jpg"
            img.save(os.path.join(sdir, fn), quality=92)
            entry = {"id": i, "file_name": fn,
                     "width": img.width, "height": img.height}
            if quality:
                entry["quality"] = quality
            images.append(entry)
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
        from collections import Counter
        id2n = {c["id"]: c["name"] for c in CATEGORIES}
        dist = Counter(id2n[a["category_id"]] for a in annotations)
        n_anom = len({a["image_id"] for a in annotations})
        n_deg = sum(1 for im in images if im.get("quality"))
        print(f"[{scene}/{split}] {n} 张（含缺陷 {n_anom} / 正常 {n - n_anom}"
              f"{f' / 成像不佳 {n_deg}' if n_deg else ''}），{len(annotations)} 个框")
        print(f"            {dict(dist.most_common())}")
    return gi


def _split_counts(n: int) -> dict:
    tr = int(n * 0.8)
    va = int(n * 0.1)
    return {"train": tr, "valid": va, "test": n - tr - va}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/raw/synthetic")
    ap.add_argument("-n", "--num", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--scene", default="mix", choices=["mix", "panel", "closeup"],
                    help="panel=蒙皮铆钉阵列；closeup=紧固件特写（螺纹损伤只在这里出现）")
    ap.add_argument("--closeup-frac", type=float, default=0.3,
                    help="scene=mix 时特写场景的占比")
    ap.add_argument("--degraded-frac", type=float, default=0.09,
                    help="做成模糊/遮挡/过曝/欠曝的比例，用于生成"
                         "「无法确认，建议补拍」这类样本")
    args = ap.parse_args()

    root = os.path.expanduser(args.out)
    rng = random.Random(args.seed)
    size = (args.width, args.height)

    # 两种场景分目录产出：特写里是螺栓、蒙皮里是壁板，被检对象根本不是一回事，
    # 混在一个数据集里会让问答说出"这张检查口盖照片"却配一张螺栓特写。
    if args.scene == "panel":
        plan = {"panel": args.num}
    elif args.scene == "closeup":
        plan = {"closeup": args.num}
    else:
        n_close = int(args.num * args.closeup_frac)
        plan = {"panel": args.num - n_close, "closeup": n_close}

    gi = 0
    for scene, n in plan.items():
        if n <= 0:
            continue
        gi = _emit(root, scene, _split_counts(n), rng, size, gi,
                   args.degraded_frac)

    print(f"\n完成 -> {root}")
    print("   configs/datasets.yaml 里对应两个条目："
          "synthetic_panel -> {root}/panel，synthetic_closeup -> {root}/closeup"
          .replace("{root}", root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
