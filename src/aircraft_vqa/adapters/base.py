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


def load_mask(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    with Image.open(path) as im:
        return np.array(im.convert("L"))


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
