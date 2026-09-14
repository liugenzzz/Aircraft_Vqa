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

    def _diagnose(self, cat: str, data, splits: dict, img_root: str) -> None:
        """一条样本都没读出来时，把 json 实际长什么样打出来。

        Real-IAD 的元数据在不同发布版本里字段名有出入，与其让人对着 0 条
        结果猜，不如直接把顶层键、split 键、条目字段名和拼出来的第一个路径
        亮出来 —— 照着改 _IMG_KEYS 或 image_dir 就行。
        """
        top = list(data)[:8] if isinstance(data, dict) else f"list[{len(data)}]"
        print(f"[real_iad] {cat}: 读出 0 条样本，json 结构如下 ——")
        print(f"           顶层键: {top}")
        print(f"           识别到的 split: {list(splits)}")
        for split, items in list(splits.items())[:1]:
            if items:
                item = items[0]
                print(f"           条目字段: {list(item)[:12]}")
                rel = _pick(item, _IMG_KEYS)
                print(f"           取到的图片相对路径: {rel!r}")
                if rel:
                    print(f"           拼出的绝对路径: {os.path.join(img_root, rel)}")
        print(f"           图片根目录: {img_root}"
              f"（存在: {os.path.isdir(img_root)}）")
        print("           -> 路径对不上就改 configs/datasets.yaml 里的 image_dir；"
              "字段名对不上就在 adapters/realiad.py 的 _IMG_KEYS 里补一个")

    def iter_samples(self) -> Iterator[UnifiedSample]:
        jdir = os.path.join(self.root, self.opts.get("json_dir", "realiad_jsons"))
        if not os.path.isdir(jdir):
            print(f"[real_iad] 找不到元数据目录 {jdir}，"
                  f"确认 realiad_jsons.zip 已解压")
            return
        n_yield = 0
        img_root = os.path.join(self.root, self.opts.get("image_dir", "realiad_1024"))
        if not os.path.isdir(img_root):
            img_root = self.root
        for jf in sorted(f for f in os.listdir(jdir) if f.endswith(".json")):
            cat = os.path.splitext(jf)[0]
            if not self._want(cat):
                continue
            with open(os.path.join(jdir, jf), encoding="utf-8") as f:
                data = json.load(f)

            # 不同发布版本的 json 顶层键不一定叫 train/test，兜底找任意
            # "值是一串 dict" 的键当作一个 split
            if isinstance(data, list):
                splits = {"train": data}
            else:
                splits = {k: data[k] for k in ("train", "test")
                          if isinstance(data.get(k), list)}
                if not splits:
                    splits = {k: v for k, v in data.items()
                              if isinstance(v, list) and v and isinstance(v[0], dict)}

            before = n_yield
            for split, items in splits.items():
                for item in items:
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
                        n_yield += 1
                        yield s

            if n_yield == before:
                self._diagnose(cat, data, splits, img_root)
