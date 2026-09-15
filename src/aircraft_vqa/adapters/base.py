"""Adapter 基类与公共工具。

Adapter 的唯一职责：把某个源数据集的目录结构 -> `UnifiedSample` 迭代器。
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from ..geometry import (bbox_area_ratio, connected_boxes, mask_area_ratio,
                        region_word, severity_from_area)
from ..schema import Defect, UnifiedSample
from ..taxonomy import Taxonomy, get_taxonomy

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP", ".tif", ".tiff")


def image_size(path: str) -> tuple:
    """只读文件头拿宽高，不解码像素。"""
    with Image.open(path) as im:
        return im.size  # (w, h)


def _voc_palette(n: int = 256) -> np.ndarray:
    """PASCAL VOC 的标准调色板 —— 分割数据集事实上的默认配色。"""
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    return pal


def _rgb_to_index(arr: np.ndarray) -> np.ndarray:
    """彩色 mask -> 类别索引。按 VOC 调色板反查，查不到的按出现顺序兜底。"""
    colors, inv = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
    pal = _voc_palette()
    lut = {tuple(c): i for i, c in enumerate(pal)}
    idx = np.array([lut.get(tuple(c), -1) for c in colors], dtype=np.int32)
    unknown = idx < 0
    if unknown.any():
        # 不是 VOC 配色。按颜色排序兜底给索引，但这个顺序跨图不保证一致，
        # 必须让人看见，不能默默产出一份对不上的标签。
        print(f"[load_mask] 警告：mask 里有 {int(unknown.sum())} 种颜色不在 VOC "
              f"调色板里 {[tuple(int(x) for x in c) for c in colors[unknown]][:6]}，"
              "已按颜色排序临时编号 —— 请显式提供 color_map 再用")
        nxt = int(idx.max()) + 1 if (~unknown).any() else 0
        idx[unknown] = np.arange(nxt, nxt + int(unknown.sum()))
    return idx[inv].reshape(arr.shape[:2])


def load_mask(path: str) -> Optional[np.ndarray]:
    """读分割 mask，返回**类别索引**矩阵。

    这里刻意不走 convert("L")：调色板图（P 模式）一转灰度，索引就变成了
    调色板颜色的亮度 —— VOC 配色下 1/2/3 会变成 38/75/113，class_map 静默
    失配，等级信息整批丢掉且不报错。VT 腐蚀集就是这么存的。
    """
    if not path or not os.path.exists(path):
        return None
    with Image.open(path) as im:
        if im.mode == "P":
            return np.array(im)                      # 调色板索引即类别索引
        if im.mode in ("L", "1"):
            return np.array(im.convert("L"))
        if im.mode.startswith("I") or im.mode == "F":
            return np.array(im).astype(np.int32)     # 16/32 位标签图
        arr = np.array(im.convert("RGB"))
    if (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all():
        return arr[..., 0]                           # 灰度图存成了 RGB
    return _rgb_to_index(arr)


def list_images(d: str) -> list:
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(IMG_EXT))


class BaseAdapter:
    """子类实现 `iter_samples()`。"""

    name = "base"

    def __init__(self, root: str, *, taxonomy: Optional[Taxonomy] = None,
                 license: str = "unknown", commercial_ok: bool = False,
                 object_hint: Optional[dict] = None, categories: Optional[list] = None,
                 **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.tax = taxonomy or get_taxonomy()
        self.license = license
        self.commercial_ok = commercial_ok
        self.object_hint = object_hint or {}     # 源 category -> 统一 object 名
        self.categories = categories             # None = 全部
        self.opts = kwargs

    # ---- 子类接口 -------------------------------------------------
    def iter_samples(self) -> Iterator[UnifiedSample]:  # pragma: no cover
        raise NotImplementedError

    # ---- 公共构造逻辑 ---------------------------------------------
    def _want(self, category: str) -> bool:
        return self.categories is None or category in self.categories

    def _object_for(self, category: str) -> str:
        if category in self.object_hint:
            return self.object_hint[category]
        return self.tax.map_object(category)

    def _severity(self, canon: str, area_ratio: float) -> str:
        base = self.tax.default_severity(canon)
        if not self.tax.area_scales_severity(canon):
            return base
        return severity_from_area(base, area_ratio)

    def make_sample(self, *, sample_id: str, image_path: str, label: str,
                    category: str, split: str, mask_path: Optional[str] = None,
                    raw_defect: str = "", boxes: Optional[list] = None,
                    box_labels: Optional[list] = None, meta: Optional[dict] = None
                    ) -> Optional[UnifiedSample]:
        """统一的样本组装：mask/boxes -> Defect 列表，并补齐方位、面积、严重度。"""
        try:
            w, h = image_size(image_path)
        except Exception:
            return None

        obj = self._object_for(category)
        oinfo = self.tax.object_info(obj)
        s = UnifiedSample(
            sample_id=sample_id, image_path=image_path, width=w, height=h,
            label=label, dataset=self.name, category=category,
            object_name=obj, object_zh=oinfo.get("zh", "待检部件"),
            aircraft_ctx=oinfo.get("aircraft_ctx", "待检部件"),
            split=split, mask_path=mask_path, license=self.license,
            commercial_ok=self.commercial_ok, meta=meta or {},
        )
        if label != "anomalous":
            return s

        canon_default = self.tax.map_defect(raw_defect)

        # 优先用显式给定的框（检测类数据集）
        if boxes:
            labs = box_labels or [raw_defect] * len(boxes)
            for b, lb in zip(boxes, labs):
                ct = self.tax.map_defect(lb)
                ar = bbox_area_ratio(b, w, h)
                s.defects.append(Defect(
                    type=ct, type_raw=str(lb), type_zh=self.tax.zh(ct),
                    bbox=[int(v) for v in b], area_ratio=round(ar, 6),
                    region=region_word(b, w, h), severity=self._severity(ct, ar)))
            return s

        # 其次用 mask 提连通域
        m = load_mask(mask_path) if mask_path else None
        if m is not None and m.any():
            binm = m > 0
            total_ar = mask_area_ratio(binm)
            for b in connected_boxes(binm):
                ar = bbox_area_ratio(b, w, h)
                s.defects.append(Defect(
                    type=canon_default, type_raw=raw_defect,
                    type_zh=self.tax.zh(canon_default), bbox=b,
                    area_ratio=round(ar, 6), region=region_word(b, w, h),
                    severity=self._severity(canon_default, ar)))
            if s.defects:
                s.meta["mask_area_ratio"] = round(total_ar, 6)
                return s

        # 只有图像级标签：仍然保留，供"识别"任务使用（定位任务会自动跳过）
        s.defects.append(Defect(
            type=canon_default, type_raw=raw_defect, type_zh=self.tax.zh(canon_default),
            bbox=None, severity=self.tax.default_severity(canon_default)))
        return s
