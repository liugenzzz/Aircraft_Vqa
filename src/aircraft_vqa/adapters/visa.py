"""VisA (SPot-the-difference) adapter —— 走官方 split_csv 清单。

    <root>/split_csv/1cls.csv : object,split,label,image,mask
    <root>/<object>/Data/Images/{Normal,Anomaly}/*.JPG
    <root>/<object>/Data/Masks/Anomaly/*.png
"""
from __future__ import annotations

import csv
import os
from typing import Iterator

from ..schema import UnifiedSample
from .base import BaseAdapter


class VisAAdapter(BaseAdapter):
    name = "visa"

    def iter_samples(self) -> Iterator[UnifiedSample]:
        csv_path = os.path.join(self.root, "split_csv",
                                self.opts.get("split_csv", "1cls.csv"))
        if not os.path.exists(csv_path):
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                obj = row["object"].strip()
                if not self._want(obj):
                    continue
                img = os.path.join(self.root, row["image"].strip())
                if not os.path.exists(img):
                    continue
                mask_rel = (row.get("mask") or "").strip()
                mask = os.path.join(self.root, mask_rel) if mask_rel else None
                label = "anomalous" if row["label"].strip().lower() != "normal" else "normal"
                stem = os.path.splitext(os.path.basename(img))[0]
                # VisA 发布版未给细粒度缺陷名，只有 Anomaly；PCB 类的异常以
                # 缺件/错件/多件为主，这里按对象给一个先验 raw 名。
                raw = "" if label == "normal" else self.opts.get(
                    "defect_hint", {}).get(obj, "anomaly")
                s = self.make_sample(
                    sample_id=f"{self.name}/{obj}/{row['split'].strip()}/{stem}",
                    image_path=img, label=label, category=obj,
                    split=row["split"].strip(), mask_path=mask, raw_defect=raw,
                    meta={"row": i})
                if s:
                    yield s
