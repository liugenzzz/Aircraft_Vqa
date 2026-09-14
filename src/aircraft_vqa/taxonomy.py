"""缺陷本体加载与原始类别名归一化。"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

import yaml

_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "taxonomy.yaml",
)


def _norm_key(s: str) -> str:
    """把 'Paint-peel-off' / 'paint peel off' 之类统一成 'paint_peel_off'。"""
    s = s.strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return re.sub(r"_+", "_", s).strip("_")


class Taxonomy:
    def __init__(self, path: Optional[str] = None):
        with open(path or _DEFAULT, encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.defect_types: dict = self.raw["defect_types"]
        self.objects: dict = self.raw["objects"]
        self.severity: dict = self.raw["severity"]

        self._alias2type: dict = {}
        for canon, spec in self.defect_types.items():
            self._alias2type[_norm_key(canon)] = canon
            for a in spec.get("aliases", []):
                self._alias2type[_norm_key(str(a))] = canon

        self._obj_alias: dict = {}
        for canon in self.objects:
            self._obj_alias[_norm_key(canon)] = canon

    # ---- 缺陷类型 -------------------------------------------------
    def map_defect(self, raw_name: str) -> str:
        """原始类别名 -> canonical 缺陷类型。未知则退化为 other_anomaly。

        匹配顺序：精确 alias -> 子串包含（长 alias 优先，避免 'crack' 抢走
        'crazing_crack' 之外的误配）。
        """
        key = _norm_key(raw_name or "")
        if not key:
            return "other_anomaly"
        if key in self._alias2type:
            return self._alias2type[key]
        for alias in sorted(self._alias2type, key=len, reverse=True):
            if len(alias) >= 4 and alias in key:
                return self._alias2type[alias]
        return "other_anomaly"

    def zh(self, defect_type: str) -> str:
        return self.defect_types.get(defect_type, {}).get("zh", defect_type)

    def en(self, defect_type: str) -> str:
        return self.defect_types.get(defect_type, {}).get("en", defect_type)

    def group(self, defect_type: str) -> str:
        return self.defect_types.get(defect_type, {}).get("group", "other")

    def action(self, defect_type: str) -> str:
        return self.defect_types.get(defect_type, {}).get("action", "记录并由持照人员进一步评估")

    def default_severity(self, defect_type: str) -> str:
        return self.defect_types.get(defect_type, {}).get("severity_default", "major")

    def severity_zh(self, sev: str) -> str:
        return self.severity.get(sev, {}).get("zh", sev)

    def severity_desc(self, sev: str) -> str:
        return self.severity.get(sev, {}).get("desc", "")

    def all_types(self) -> list:
        return list(self.defect_types.keys())

    def siblings(self, defect_type: str) -> list:
        """同族的其他缺陷类型 —— 用来造"看起来很像"的强干扰项。"""
        g = self.group(defect_type)
        return [t for t in self.defect_types
                if t != defect_type and self.group(t) == g]

    # ---- 对象 -----------------------------------------------------
    def map_object(self, raw_name: str) -> str:
        key = _norm_key(raw_name or "")
        if key in self._obj_alias:
            return self._obj_alias[key]
        for canon in sorted(self._obj_alias, key=len, reverse=True):
            if len(canon) >= 3 and canon in key:
                return self._obj_alias[canon]
        return "unknown"

    def object_info(self, obj: str) -> dict:
        return self.objects.get(obj, self.objects["unknown"])


@lru_cache(maxsize=4)
def get_taxonomy(path: Optional[str] = None) -> Taxonomy:
    return Taxonomy(path)
