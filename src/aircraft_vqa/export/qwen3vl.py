# -*- coding: utf-8 -*-
"""导出为 Qwen3-VL 指令微调可直接吃的格式。

支持三种：
  llamafactory : LLaMA-Factory 的 sharegpt/messages 风格
  swift        : ms-swift 的 messages 风格（坐标用 abs 时由 swift 自行归一化）
  openai       : OpenAI/Qwen 官方 messages（content 为多模态 parts 列表）

坐标约定
  Qwen3-VL 原生输出 [0,1000] 归一化坐标，因此默认 coord_mode="norm1000"。
  若用 ms-swift 的 grounding 数据格式（bbox 为绝对像素、由框架换算），
  构建时用 --coord-mode abs，并确保训练时图片尺寸与标注一致。
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

IMAGE_TOKEN = "<image>"


def _img_path(path: str, image_root: Optional[str], relative: bool) -> str:
    if relative and image_root:
        try:
            return os.path.relpath(path, image_root)
        except ValueError:
            return path
    return path


def _turn_pairs(r: dict) -> list:
    """统一取出 (问, 答) 序列：多轮条目走 turns，单轮走 question/answer。"""
    if r.get("turns"):
        return [(t["question"], t["answer"]) for t in r["turns"]]
    return [(r["question"], r["answer"])]


def to_llamafactory(r: dict, image_root=None, relative=False,
                    with_system=True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    for i, (q, a) in enumerate(_turn_pairs(r)):
        # 图片只挂在第一轮，后续轮次靠上下文承接
        msgs.append({"role": "user",
                     "content": f"{IMAGE_TOKEN}{q}" if i == 0 else q})
        msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs, "images": [_img_path(r["image"], image_root, relative)]}


def to_swift(r: dict, image_root=None, relative=False, with_system=True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    for i, (q, a) in enumerate(_turn_pairs(r)):
        msgs.append({"role": "user",
                     "content": f"{IMAGE_TOKEN}{q}" if i == 0 else q})
        msgs.append({"role": "assistant", "content": a})
    out = {"messages": msgs, "images": [_img_path(r["image"], image_root, relative)]}
    if r.get("task", "").startswith("grounding") or r.get("task") == "counting":
        out["_meta"] = {"task": r["task"], "coord_mode": r.get("coord_mode")}
    return out


def to_openai(r: dict, image_root=None, relative=False, with_system=True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    for i, (q, a) in enumerate(_turn_pairs(r)):
        if i == 0:
            msgs.append({"role": "user", "content": [
                {"type": "image",
                 "image": _img_path(r["image"], image_root, relative)},
                {"type": "text", "text": q}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": q}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return {"messages": msgs}


EXPORTERS = {
    "llamafactory": to_llamafactory,
    "swift": to_swift,
    "openai": to_openai,
}


def export_records(records: Iterable[dict], out_path: str, fmt: str = "llamafactory",
                   image_root: Optional[str] = None, relative: bool = False,
                   with_system: bool = True, as_json_array: bool = False) -> int:
    """写出 jsonl（默认）或 json 数组（LLaMA-Factory 的 dataset_info 也支持）。"""
    fn = EXPORTERS[fmt]
    items = [fn(r, image_root, relative, with_system) for r in records]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if as_json_array:
            json.dump(items, f, ensure_ascii=False, indent=1)
        else:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)
