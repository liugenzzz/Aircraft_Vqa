"""统一中间表示（UIR）。

所有源数据集经 adapter 归一化成 `UnifiedSample`，VQA 构建器只认这一种结构，
新增数据集时不需要改构建逻辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator, Optional

BBox = list  # [x1, y1, x2, y2]，绝对像素坐标，左上原点


@dataclass
class Defect:
    """一处缺陷实例。"""

    type: str                      # canonical 类型，见 configs/taxonomy.yaml
    type_raw: str = ""             # 源数据集里的原始类别名，便于溯源
    type_zh: str = ""
    bbox: Optional[BBox] = None    # 绝对像素坐标
    area_ratio: float = 0.0        # 缺陷面积 / 整图面积
    severity: str = "major"
    region: str = ""               # 九宫格方位词，如 "左上"
    polygon: Optional[list] = None  # 可选轮廓点 [[x,y], ...]
    score: float = 1.0             # 标注置信度（合成/伪标注时 < 1）

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}


@dataclass
class UnifiedSample:
    """一张图 + 其全部缺陷标注。"""

    sample_id: str
    image_path: str
    width: int
    height: int
    label: str                       # "normal" | "anomalous"
    dataset: str
    category: str                    # 源数据集内的类别/子集名
    object_name: str = "unknown"     # 归一化后的被检对象，见 taxonomy.objects
    object_zh: str = "待检部件"
    aircraft_ctx: str = "待检部件"
    split: str = "train"
    defects: list = field(default_factory=list)   # list[Defect]
    mask_path: Optional[str] = None
    license: str = "unknown"
    commercial_ok: bool = False
    meta: dict = field(default_factory=dict)

    # ---- 便捷属性 -------------------------------------------------
    @property
    def is_anomalous(self) -> bool:
        return self.label == "anomalous"

    @property
    def defect_types(self) -> list:
        """去重且保序的缺陷类型列表。"""
        seen, out = set(), []
        for d in self.defects:
            if d.type not in seen:
                seen.add(d.type)
                out.append(d.type)
        return out

    def localizable_defects(self) -> list:
        """有 bbox、可用于定位任务的缺陷。"""
        return [d for d in self.defects if d.bbox]

    # ---- 序列化 ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["defects"] = [x.to_dict() for x in self.defects]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedSample":
        d = dict(d)
        d["defects"] = [Defect(**x) for x in d.get("defects", [])]
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


def write_jsonl(path: str, samples: Iterable[UnifiedSample]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[UnifiedSample]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield UnifiedSample.from_dict(json.loads(line))


def dump_records(path: str, records: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_records(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _unused(*args: Any) -> None:  # pragma: no cover
    pass
