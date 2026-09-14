"""Real-IAD adapter（多视角）。

官方以 json 清单发布：<root>/realiad_jsons/<class>.json，条目形如
    {"image_path": ..., "mask_path": ..., "anomaly_class": "...", "category": ...}
不同发布版本字段名略有出入，这里做容错取值。
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from ..schema import UnifiedSample
from .base import BaseAdapter

_IMG_KEYS = ("image_path", "image", "img_path", "file_name")
_MASK_KEYS = ("mask_path", "mask", "gt_path", "anomaly_mask")
_CLS_KEYS = ("anomaly_class", "defect_type", "anomaly_type", "label")
_VIEW_KEYS = ("view", "camera", "view_id", "pose")


def _pick(d: dict, keys) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


class RealIADAdapter(BaseAdapter):
    name = "real_iad"

    def iter_samples(self) -> Iterator[UnifiedSample]:
        jdir = os.path.join(self.root, self.opts.get("json_dir", "realiad_jsons"))
        if not os.path.isdir(jdir):
            return
        img_root = os.path.join(self.root, self.opts.get("image_dir", "realiad_1024"))
        if not os.path.isdir(img_root):
            img_root = self.root
        for jf in sorted(f for f in os.listdir(jdir) if f.endswith(".json")):
            cat = os.path.splitext(jf)[0]
            if not self._want(cat):
                continue
            with open(os.path.join(jdir, jf), encoding="utf-8") as f:
                data = json.load(f)
            for split in ("train", "test"):
                for item in (data.get(split) or []):
                    rel = _pick(item, _IMG_KEYS)
                    if not rel:
                        continue
                    img = rel if os.path.isabs(rel) else os.path.join(img_root, rel)
                    if not os.path.exists(img):
                        img = os.path.join(img_root, cat, rel)
                        if not os.path.exists(img):
                            continue
                    mrel = _pick(item, _MASK_KEYS)
                    mask = None
                    if mrel:
                        cand = mrel if os.path.isabs(mrel) else os.path.join(img_root, mrel)
                        mask = cand if os.path.exists(cand) else None
                    raw = _pick(item, _CLS_KEYS)
                    is_ok = raw.lower() in ("ok", "good", "normal", "0", "")
                    stem = os.path.splitext(os.path.basename(img))[0]
                    s = self.make_sample(
                        sample_id=f"{self.name}/{cat}/{split}/{stem}",
                        image_path=img, label="normal" if is_ok else "anomalous",
                        category=cat, split=split, mask_path=mask,
                        raw_defect="" if is_ok else raw,
                        meta={"view": _pick(item, _VIEW_KEYS)})
                    if s:
                        yield s
