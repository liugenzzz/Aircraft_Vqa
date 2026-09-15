"""检测/分割类标注的通用 adapter：COCO(Roboflow 导出)、YOLO、掩码分割。"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterator

import numpy as np

from ..schema import UnifiedSample
from .base import BaseAdapter, IMG_EXT, list_images, load_mask


class CocoAdapter(BaseAdapter):
    """Roboflow / 通用 COCO 目录：

        <root>/<split>/_annotations.coco.json
        <root>/<split>/*.jpg
    """

    name = "coco"
    ann_names = ("_annotations.coco.json", "annotations.json", "instances.json")

    def _find_ann(self, d: str):
        for n in self.ann_names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        for n in sorted(os.listdir(d)):
            if n.endswith(".json"):
                return os.path.join(d, n)
        return None

    def iter_samples(self) -> Iterator[UnifiedSample]:
        splits = self.opts.get("splits") or ["train", "valid", "val", "test"]
        cat_name = self.opts.get("category") or os.path.basename(self.root)
        for split in splits:
            sdir = os.path.join(self.root, split)
            if not os.path.isdir(sdir):
                continue
            ann_path = self._find_ann(sdir)
            if not ann_path:
                continue
            with open(ann_path, encoding="utf-8") as f:
                coco = json.load(f)
            cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
            by_img = defaultdict(list)
            for a in coco.get("annotations", []):
                by_img[a["image_id"]].append(a)
            for im in coco.get("images", []):
                path = os.path.join(sdir, im["file_name"])
                if not os.path.exists(path):
                    continue
                anns = by_img.get(im["id"], [])
                boxes, labels = [], []
                for a in anns:
                    name = cats.get(a["category_id"], "defect")
                    if name.lower() in ("background", "none", "good", "normal", "ok"):
                        continue
                    x, y, w, h = a["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(name)
                stem = os.path.splitext(im["file_name"])[0]
                s = self.make_sample(
                    sample_id=f"{self.name}/{cat_name}/{split}/{stem}",
                    image_path=path, label="anomalous" if boxes else "normal",
                    category=cat_name, split=split, boxes=boxes or None,
                    box_labels=labels or None,
                    meta={"coco_image_id": im["id"],
                          **({"image_quality": im["quality"],
                              "degradation": im.get("degradation") or {}}
                             if im.get("quality") else {})})
                if s:
                    yield s


class YoloAdapter(BaseAdapter):
    """YOLO 目录：<root>/<split>/images/*.jpg + <root>/<split>/labels/*.txt

    类名从 <root>/data.yaml 的 names 读取。
    """

    name = "yolo"

    def _names(self) -> list:
        for n in ("data.yaml", "data.yml", "dataset.yaml"):
            p = os.path.join(self.root, n)
            if os.path.exists(p):
                import yaml
                with open(p, encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                names = d.get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                if isinstance(names, list):
                    return names
        return []

    def iter_samples(self) -> Iterator[UnifiedSample]:
        from .base import image_size
        names = self._names()
        cat_name = self.opts.get("category") or os.path.basename(self.root)
        for split in (self.opts.get("splits") or ["train", "valid", "val", "test"]):
            idir = os.path.join(self.root, split, "images")
            ldir = os.path.join(self.root, split, "labels")
            if not os.path.isdir(idir):
                continue
            for fn in list_images(idir):
                path = os.path.join(idir, fn)
                stem = os.path.splitext(fn)[0]
                lp = os.path.join(ldir, stem + ".txt")
                boxes, labels = [], []
                if os.path.exists(lp):
                    try:
                        w, h = image_size(path)
                    except Exception:
                        continue
                    for line in open(lp, encoding="utf-8"):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        ci = int(float(parts[0]))
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                      (cx + bw / 2) * w, (cy + bh / 2) * h])
                        labels.append(names[ci] if ci < len(names) else f"class_{ci}")
                s = self.make_sample(
                    sample_id=f"{self.name}/{cat_name}/{split}/{stem}",
                    image_path=path, label="anomalous" if boxes else "normal",
                    category=cat_name, split=split, boxes=boxes or None,
                    box_labels=labels or None)
                if s:
                    yield s


class MaskSegAdapter(BaseAdapter):
    """语义分割式数据（腐蚀分级等）：图像目录 + 同名 mask 目录。

    mask 像素值即类别索引，由 `class_map` 给出 {像素值: 类别名}。

    若该数据集给的是**同一种缺陷的有序等级**（VT 腐蚀集的
    good/fair/poor/severe 就是），用 `grade_type` 指明这些名字是哪种缺陷的等级：

        grade_type: corrosion
        class_map: {1: good, 2: fair, 3: poor, 4: severe}

    这样等级会写进 Defect.grade，严重度由等级给出而不是按面积估。
    不这么声明的话，"good" 这种词会被当成普通类别名去查别名表 ——
    而 "good" 在 MVTec 系里是"正常"的意思，撞车了会出错，所以必须显式声明。
    """

    name = "mask_seg"

    def iter_samples(self) -> Iterator[UnifiedSample]:
        from ..geometry import (bbox_area_ratio, connected_boxes, region_word,
                                severity_from_area)
        from ..schema import Defect
        from .base import image_size
        img_dir = os.path.join(self.root, self.opts.get("images_dir", "images"))
        mask_dir = os.path.join(self.root, self.opts.get("masks_dir", "masks"))
        class_map = {int(k): v for k, v in (self.opts.get("class_map") or {}).items()}
        grade_type = self.opts.get("grade_type") or ""
        bg = set(self.opts.get("background_values", [0]))
        cat_name = self.opts.get("category") or os.path.basename(self.root)
        split = self.opts.get("split", "train")
        if not os.path.isdir(img_dir):
            return
        for fn in list_images(img_dir):
            path = os.path.join(img_dir, fn)
            stem = os.path.splitext(fn)[0]
            mp = ""
            for ext in IMG_EXT:
                cand = os.path.join(mask_dir, stem + ext)
                if os.path.exists(cand):
                    mp = cand
                    break
            m = load_mask(mp) if mp else None
            try:
                w, h = image_size(path)
            except Exception:
                continue
            defects = []
            if m is not None:
                for val in sorted(set(np.unique(m)) - bg):
                    raw = class_map.get(int(val), f"class_{int(val)}")
                    if grade_type:
                        # 显式声明了是某种缺陷的有序等级
                        ct, grade = grade_type, raw
                    else:
                        ct = self.tax.map_defect(raw)
                        grade = raw if self.tax.grade_info(ct, raw) else ""
                    if grade and not self.tax.grade_info(ct, grade):
                        grade = ""          # 等级名不在本体里，按普通缺陷处理
                    for b in connected_boxes(m == val):
                        ar = bbox_area_ratio(b, w, h)
                        sev = (self.tax.grade_info(ct, grade).get("severity")
                               if grade else self._severity(ct, ar))
                        defects.append(Defect(
                            type=ct, type_raw=raw, type_zh=self.tax.zh(ct), bbox=b,
                            area_ratio=round(ar, 6), region=region_word(b, w, h),
                            grade=grade, severity=sev))
            s = self.make_sample(
                sample_id=f"{self.name}/{cat_name}/{split}/{stem}",
                image_path=path, label="anomalous" if defects else "normal",
                category=cat_name, split=split, mask_path=mp or None)
            if s:
                s.defects = defects
                yield s
