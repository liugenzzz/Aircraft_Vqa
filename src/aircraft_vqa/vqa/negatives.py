"""正常区域采样：给"区域指代"任务造真阴性框。"""
from __future__ import annotations

import random
from typing import Optional

from ..geometry import iou


def sample_clean_box(w: int, h: int, defect_boxes: list,
                     rng: Optional[random.Random] = None,
                     tries: int = 30, min_frac: float = 0.12,
                     max_frac: float = 0.32,
                     hard: bool = True) -> Optional[list]:
    """采一个与任何缺陷框都不相交的方框。采不到返回 None。

    `hard=True` 时**优先贴着缺陷旁边采**（近邻但不重叠）。
    纯随机采空白区会让模型学到捷径 —— 只要框落在画面边缘或大片空白就答"没有"，
    根本没去看框里的内容。难负样本必须和正样本在视觉上足够接近。
    """
    rng = rng or random

    def ok(box):
        return all(iou(box, d) == 0.0 for d in defect_boxes)

    if hard and defect_boxes:
        for _ in range(tries):
            d = rng.choice(defect_boxes)
            dw, dh = d[2] - d[0], d[3] - d[1]
            bw = max(8, int(dw * rng.uniform(0.8, 1.6)))
            bh = max(8, int(dh * rng.uniform(0.8, 1.6)))
            # 沿某个方向平移出去，留一点间隙保证不重叠
            ang = rng.uniform(0, 6.283185)
            dist = max(dw, dh) * rng.uniform(1.0, 2.2)
            import math
            cx = (d[0] + d[2]) / 2 + math.cos(ang) * dist
            cy = (d[1] + d[3]) / 2 + math.sin(ang) * dist
            x1 = int(max(0, min(w - bw, cx - bw / 2)))
            y1 = int(max(0, min(h - bh, cy - bh / 2)))
            box = [x1, y1, x1 + bw, y1 + bh]
            if ok(box):
                return box

    for _ in range(tries):
        bw = int(w * rng.uniform(min_frac, max_frac))
        bh = int(h * rng.uniform(min_frac, max_frac))
        if bw < 8 or bh < 8:
            continue
        x1 = rng.randint(0, max(0, w - bw))
        y1 = rng.randint(0, max(0, h - bh))
        box = [x1, y1, x1 + bw, y1 + bh]
        if ok(box):
            return box
    return None
