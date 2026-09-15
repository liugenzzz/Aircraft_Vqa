"""几何工具：mask -> bbox、方位词、尺寸词、Qwen3-VL 坐标归一化。"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

# ---------------------------------------------------------------- 连通域

def _components(small: np.ndarray):
    """连通域 -> [(area, y0, y1, x0, x1)]。有 scipy 就走它，没有返回 None。

    scipy.ndimage.label 默认就是 4-连通，和下面那份手写 BFS 语义一致；
    512x512 的大片连通域上快两个数量级，真实锈蚀 mask 动辄就是这种。
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None
    lab, n = ndimage.label(small)
    if not n:
        return []
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        out.append((int(areas[i + 1]), int(sl[0].start), int(sl[0].stop) - 1,
                    int(sl[1].start), int(sl[1].stop) - 1))
    return out


def _components_bfs(small: np.ndarray):
    """没有 scipy 时的退路：手写 4-连通 BFS。"""
    sh, sw = small.shape
    visited = np.zeros((sh, sw), dtype=bool)
    comps = []
    for sy in range(sh):
        row = small[sy]
        if not row.any():
            continue
        for sx in np.nonzero(row)[0]:
            sx = int(sx)
            # visited 会被下面的 BFS 边走边改，而这一行的候选列表是循环开始
            # 前就算好的。不在这里复查，同一个连通域里每个已访问像素都会被
            # 当成新起点，各自吐出一个 1 像素的框 —— 一块完整锈斑能炸成几十处。
            if visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            y0 = y1 = sy
            x0 = x1 = int(sx)
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                if cy < y0: y0 = cy
                if cy > y1: y1 = cy
                if cx < x0: x0 = cx
                if cx > x1: x1 = cx
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            comps.append((area, y0, y1, x0, x1))
    return comps


def connected_boxes(mask: np.ndarray,
                    min_area_ratio: float = 2e-4,
                    max_components: int = 12,
                    work_size: int = 512) -> list:
    """二值 mask -> 连通域 bbox 列表（绝对像素，原图尺度）。

    为了速度，先把 mask 缩到长边 <= work_size 做连通域分析，再把坐标缩放回去；
    bbox 精度对 VQA 训练完全够用。返回按面积降序，最多 max_components 个。
    """
    m = mask.astype(bool)
    if not m.any():
        return []
    H, W = m.shape
    scale = max(H, W) / float(work_size)
    if scale > 1.0:
        ys = (np.arange(int(H / scale)) * scale).astype(int).clip(0, H - 1)
        xs = (np.arange(int(W / scale)) * scale).astype(int).clip(0, W - 1)
        small = m[np.ix_(ys, xs)]
    else:
        scale, ys, xs = 1.0, None, None
        small = m

    sh, sw = small.shape
    comps = _components(small)
    if comps is None:
        comps = _components_bfs(small)

    comps.sort(key=lambda c: -c[0])
    total_small = sh * sw
    out = []
    for area, y0, y1, x0, x1 in comps[:max_components]:
        if area / total_small < min_area_ratio:
            continue
        if scale > 1.0:
            bx0, by0 = int(xs[x0]), int(ys[y0])
            bx1 = int(xs[min(x1, len(xs) - 1)] + scale)
            by1 = int(ys[min(y1, len(ys) - 1)] + scale)
        else:
            bx0, by0, bx1, by1 = x0, y0, x1 + 1, y1 + 1
        out.append([max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)])
    if not out:  # 缺陷极小但确实存在 -> 退化为整体外接框，不要丢样本
        yy, xx = np.nonzero(m)
        out = [[int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]]
    return out


def mask_area_ratio(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    return float(m.sum()) / float(m.size) if m.size else 0.0


def bbox_area_ratio(bbox, w: int, h: int) -> float:
    if not bbox or w <= 0 or h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(w * h)


def merge_boxes(boxes: list) -> Optional[list]:
    if not boxes:
        return None
    xs1 = min(b[0] for b in boxes); ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes); ys2 = max(b[3] for b in boxes)
    return [xs1, ys1, xs2, ys2]


def iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- 语言化

_ROWS = ("上", "中", "下")
_COLS = ("左", "中", "右")


def region_word(bbox, w: int, h: int) -> str:
    """bbox 中心落在九宫格哪一格 -> '左上' / '正中' / '右下' 等。"""
    if not bbox or w <= 0 or h <= 0:
        return "画面中"
    cx = (bbox[0] + bbox[2]) / 2.0 / w
    cy = (bbox[1] + bbox[3]) / 2.0 / h
    ri = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    ci = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    if ri == 1 and ci == 1:
        return "正中"
    if ri == 1:
        return _COLS[ci] + "侧中部"
    if ci == 1:
        return "中" + _ROWS[ri] + "部"
    return _COLS[ci] + _ROWS[ri]


def size_word(area_ratio: float) -> str:
    """面积占比 -> 统一句式的范围词。

    返回的是完整短语（"范围较小"而不是"较小"），模板里直接用 {size}。
    之前"范围极小/范围较小/中等大小/范围较大"语法结构都不一样，读着别扭。

    注意：这只是 bbox 面积的机械映射，信息量有限。真正有价值的是形态描述
    （点状/条状/片状、边缘是否翻起），那需要让 VLM 真的看图，模板给不了。
    见 docs/02_pipeline.md 的"已知局限"。
    """
    if area_ratio < 0.002:
        return "范围极小"
    if area_ratio < 0.01:
        return "范围较小"
    if area_ratio < 0.05:
        return "范围中等"
    if area_ratio < 0.15:
        return "范围较大"
    return "范围很大"


def severity_from_area(base: str, area_ratio: float) -> str:
    """按缺陷占比对本体默认严重度做一档升降，让标签不至于全表一个值。"""
    order = ["minor", "major", "critical"]
    i = order.index(base) if base in order else 1
    if area_ratio >= 0.08:
        i = min(2, i + 1)
    elif area_ratio > 0 and area_ratio < 0.003:
        i = max(0, i - 1)
    return order[i]


# ---------------------------------------------------------------- Qwen 坐标

def to_qwen_box(bbox, w: int, h: int, mode: str = "norm1000") -> list:
    """绝对像素 bbox -> Qwen3-VL 训练/推理坐标。

    mode="norm1000"：Qwen3-VL 原生的 [0,1000] 归一化坐标（默认）。
    mode="abs"     ：保留绝对像素；ms-swift 等框架会自行换算成 norm1000，
                     用这个模式时务必保证导出的图片尺寸与标注一致。
    """
    x1, y1, x2, y2 = bbox
    if mode == "abs":
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]
    f = lambda v, s: int(max(0, min(1000, round(v / s * 1000))))
    return [f(x1, w), f(y1, h), f(x2, w), f(y2, h)]


def smart_resize(h: int, w: int, factor: int = 28,
                 min_pixels: int = 4 * 28 * 28,
                 max_pixels: int = 16384 * 28 * 28) -> tuple:
    """复刻 Qwen-VL 的 smart_resize：把尺寸对齐到 factor 的整数倍。

    预缩放图片时用它，可以保证训练坐标与模型实际看到的网格一致。
    """
    import math
    h_bar = max(factor, int(round(h / factor)) * factor)
    w_bar = max(factor, int(round(w / factor)) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, int(math.floor(h / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(w / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = int(math.ceil(h * beta / factor)) * factor
        w_bar = int(math.ceil(w * beta / factor)) * factor
    return h_bar, w_bar
