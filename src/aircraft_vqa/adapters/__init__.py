"""Adapter 注册表。datasets.yaml 里的 `adapter` 字段就是这里的 key。"""
from .base import BaseAdapter
from .detection import CocoAdapter, MaskSegAdapter, YoloAdapter
from .mvtec import MPDDAdapter, MVTecAdapter, MVTecLOCOAdapter
from .realiad import RealIADAdapter
from .visa import VisAAdapter

REGISTRY = {
    "mvtec_ad": MVTecAdapter,
    "mvtec": MVTecAdapter,
    "mpdd": MPDDAdapter,
    "mvtec_loco": MVTecLOCOAdapter,
    "visa": VisAAdapter,
    "real_iad": RealIADAdapter,
    "coco": CocoAdapter,
    "yolo": YoloAdapter,
    "mask_seg": MaskSegAdapter,
}


def build_adapter(spec: dict, taxonomy=None) -> BaseAdapter:
    """spec 取自 configs/datasets.yaml 的单个条目。"""
    spec = dict(spec)
    key = spec.pop("adapter")
    if key not in REGISTRY:
        raise KeyError(f"未知 adapter: {key}，可选 {sorted(REGISTRY)}")
    cls = REGISTRY[key]
    root = spec.pop("root")
    name = spec.pop("name", None)
    for drop in ("enabled", "notes", "url", "download", "priority"):
        spec.pop(drop, None)
    ad = cls(root, taxonomy=taxonomy, **spec)
    if name:
        ad.name = name
    return ad


__all__ = ["REGISTRY", "build_adapter", "BaseAdapter"]
