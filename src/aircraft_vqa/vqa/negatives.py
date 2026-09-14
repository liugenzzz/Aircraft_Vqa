"""正常区域采样：给"区域指代"任务造真阴性框。"""
from __future__ import annotations

import random
from typing import Optional

from ..geometry import iou


def sample_clean_box(w: int, h: int, defect_boxes: list,
                     rng: Optional[random.Random] = None,
                     tries: int = 30, min_frac: float = 0.12,
                     max_frac: float = 0.32) -> Optional[list]:
    """在图中随机采一个与任何缺陷框都不相交的方框。采不到返回 None。"""
    rng = rng or random
    for _ in range(tries):
        bw = int(w * rng.uniform(min_frac, max_frac))
        bh = int(h * rng.uniform(min_frac, max_frac))
        if bw < 8 or bh < 8:
            continue
        x1 = rng.randint(0, max(0, w - bw))
        y1 = rng.randint(0, max(0, h - bh))
        box = [x1, y1, x1 + bw, y1 + bh]
        if all(iou(box, d) == 0.0 for d in defect_boxes):
            return box
    return None
