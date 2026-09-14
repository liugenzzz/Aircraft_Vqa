"""MVTec AD / MPDD / BTAD 等"MVTec 式"目录结构的通用 adapter。

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect>/*.png
    <root>/<category>/ground_truth/<defect>/*_mask.png
"""
from __future__ import annotations

import os
from typing import Iterator

from ..schema import UnifiedSample
from .base import BaseAdapter, list_images


class MVTecAdapter(BaseAdapter):
    name = "mvtec_ad"
    mask_suffixes = ("_mask", "")

    def _find_mask(self, gt_dir: str, stem: str) -> str:
        for suf in self.mask_suffixes:
            for ext in (".png", ".jpg", ".bmp"):
                p = os.path.join(gt_dir, stem + suf + ext)
                if os.path.exists(p):
                    return p
        return ""

    def iter_samples(self) -> Iterator[UnifiedSample]:
        if not os.path.isdir(self.root):
            return
        for cat in sorted(os.listdir(self.root)):
            cdir = os.path.join(self.root, cat)
            if not os.path.isdir(cdir) or not self._want(cat):
                continue
            for split in ("train", "test", "validation"):
                sdir = os.path.join(cdir, split)
                if not os.path.isdir(sdir):
                    continue
                for defect in sorted(os.listdir(sdir)):
                    ddir = os.path.join(sdir, defect)
                    if not os.path.isdir(ddir):
                        continue
                    is_good = defect in ("good", "ok", "normal")
                    gt_dir = os.path.join(cdir, "ground_truth", defect)
                    for fn in list_images(ddir):
                        stem = os.path.splitext(fn)[0]
                        mask = "" if is_good else self._find_mask(gt_dir, stem)
                        s = self.make_sample(
                            sample_id=f"{self.name}/{cat}/{split}/{defect}/{stem}",
                            image_path=os.path.join(ddir, fn),
                            label="normal" if is_good else "anomalous",
                            category=cat, split=split, mask_path=mask or None,
                            raw_defect="" if is_good else defect,
                            meta={"source_defect_dir": defect})
                        if s:
                            yield s


class MPDDAdapter(MVTecAdapter):
    name = "mpdd"


class MVTecLOCOAdapter(BaseAdapter):
    """MVTec LOCO AD：ground_truth 下每张图是一个目录，内含多张实例 mask。

    注意：原始发布只区分 logical_anomalies / structural_anomalies，不含
    `screw_too_long` 这类细粒度名。若在 category 目录下放了
    `defect_names.json`（{"图片stem": "screw_too_long"}），本 adapter 会优先用它。
    """

    name = "mvtec_loco"

    def iter_samples(self) -> Iterator[UnifiedSample]:
        import json
        if not os.path.isdir(self.root):
            return
        for cat in sorted(os.listdir(self.root)):
            cdir = os.path.join(self.root, cat)
            if not os.path.isdir(cdir) or not self._want(cat):
                continue
            override = {}
            ov_path = os.path.join(cdir, "defect_names.json")
            if os.path.exists(ov_path):
                with open(ov_path, encoding="utf-8") as f:
                    override = json.load(f)
            for split in ("train", "validation", "test"):
                sdir = os.path.join(cdir, split)
                if not os.path.isdir(sdir):
                    continue
                for defect in sorted(os.listdir(sdir)):
                    ddir = os.path.join(sdir, defect)
                    if not os.path.isdir(ddir):
                        continue
                    is_good = defect in ("good", "ok", "normal")
                    for fn in list_images(ddir):
                        stem = os.path.splitext(fn)[0]
                        raw = "" if is_good else override.get(stem, defect)
                        masks = []
                        gdir = os.path.join(cdir, "ground_truth", defect, stem)
                        if os.path.isdir(gdir):
                            masks = [os.path.join(gdir, m) for m in list_images(gdir)]
                        s = self.make_sample(
                            sample_id=f"{self.name}/{cat}/{split}/{defect}/{stem}",
                            image_path=os.path.join(ddir, fn),
                            label="normal" if is_good else "anomalous",
                            category=cat, split=split,
                            mask_path=masks[0] if masks else None,
                            raw_defect=raw,
                            meta={"anomaly_kind": defect, "n_instance_masks": len(masks)})
                        if s is None:
                            continue
                        # 多实例 mask：合并其余 mask 的连通域
                        if len(masks) > 1 and s.is_anomalous:
                            extra = self._extra_defects(s, masks[1:], raw)
                            s.defects.extend(extra)
                        yield s

    def _extra_defects(self, s: UnifiedSample, mask_paths: list, raw: str) -> list:
        from ..geometry import bbox_area_ratio, connected_boxes, region_word, severity_from_area
        from ..schema import Defect
        from .base import load_mask
        out = []
        ct = self.tax.map_defect(raw)
        for mp in mask_paths:
            m = load_mask(mp)
            if m is None or not m.any():
                continue
            for b in connected_boxes(m > 0):
                ar = bbox_area_ratio(b, s.width, s.height)
                out.append(Defect(type=ct, type_raw=raw, type_zh=self.tax.zh(ct), bbox=b,
                                  area_ratio=round(ar, 6), region=region_word(b, s.width, s.height),
                                  severity=severity_from_area(self.tax.default_severity(ct), ar)))
        return out
