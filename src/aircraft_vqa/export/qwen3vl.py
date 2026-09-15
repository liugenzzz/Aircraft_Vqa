# -*- coding: utf-8 -*-
"""导出为 Qwen3-VL 指令微调可直接吃的格式。

支持四种：
  sharegpt     : ShareGPT 风格（conversations + from/value），LLaMA-Factory 默认吃这个
  llamafactory : messages 风格（role/content）
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


# 训练框架会忽略不认识的键，所以把溯源信息带在条目里不影响训练，
# 但做分层评测（按任务/按数据源/按缺陷类型看指标）和排错时非常关键。
META_KEYS = ("task", "family", "output_format", "dataset", "category", "split",
             "image_status", "image_defect_types", "asked_defect_types",
             "answer_defect_types", "variant", "coord_mode", "image_hw",
             "license")


def _meta(r: dict) -> dict:
    return {k: r[k] for k in META_KEYS if k in r}


def _img_path(path: str, image_root: Optional[str], relative: bool) -> str:
    if relative and image_root:
        try:
            return os.path.relpath(path, image_root)
        except ValueError:
            return path
    return path


def _images(r: dict, image_root, relative) -> list:
    """条目自带 images 时用它（多图对比），否则用单张 image。"""
    paths = r.get("images") or [r["image"]]
    return [_img_path(p, image_root, relative) for p in paths]


def _turn_pairs(r: dict) -> list:
    """统一取出 (问, 答) 序列：多轮条目走 turns，单轮走 question/answer。"""
    if r.get("turns"):
        return [(t["question"], t["answer"]) for t in r["turns"]]
    return [(r["question"], r["answer"])]


def to_llamafactory(r: dict, image_root=None, relative=False,
                    with_system=True, with_meta: bool = True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    for i, (q, a) in enumerate(_turn_pairs(r)):
        # 图片只挂在第一轮，后续轮次靠上下文承接
        msgs.append({"role": "user",
                     "content": f"{IMAGE_TOKEN}{q}" if i == 0 else q})
        msgs.append({"role": "assistant", "content": a})
    out = {"messages": msgs,
           "images": [_img_path(r["image"], image_root, relative)]}
    if with_meta:
        out["id"] = r.get("qa_id")
        out["meta"] = _meta(r)
    return out


def to_sharegpt(r: dict, image_root=None, relative=False,
                with_system=True, with_meta: bool = True) -> dict:
    """ShareGPT 格式：conversations 用 from/value，system 提到顶层。

        {"conversations": [{"from": "human", "value": "<image>问题"},
                           {"from": "gpt",   "value": "答案"}],
         "system": "...", "images": ["/abs/path.jpg"]}

    多轮就是继续追加 human/gpt 对，图片 token 只挂第一轮。
    """
    convs = []
    imgs = _images(r, image_root, relative)
    for i, (q, a) in enumerate(_turn_pairs(r)):
        # 图片 token 数量必须等于 images 的长度，多图时全挂在第一轮
        convs.append({"from": "human",
                      "value": f"{IMAGE_TOKEN * len(imgs)}{q}" if i == 0 else q})
        convs.append({"from": "gpt", "value": a})
    out = {"conversations": convs, "images": imgs}
    if with_system and r.get("system"):
        out["system"] = r["system"]
    if with_meta:
        out["id"] = r.get("qa_id")
        out["meta"] = _meta(r)
    return out


def to_swift(r: dict, image_root=None, relative=False, with_system=True,
             with_meta: bool = True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    imgs = _images(r, image_root, relative)
    for i, (q, a) in enumerate(_turn_pairs(r)):
        msgs.append({"role": "user",
                     "content": f"{IMAGE_TOKEN * len(imgs)}{q}" if i == 0 else q})
        msgs.append({"role": "assistant", "content": a})
    out = {"messages": msgs, "images": imgs}
    if with_meta:
        out["id"] = r.get("qa_id")
        out["meta"] = _meta(r)
    return out


def to_openai(r: dict, image_root=None, relative=False, with_system=True,
              with_meta: bool = True) -> dict:
    msgs = []
    if with_system and r.get("system"):
        msgs.append({"role": "system", "content": r["system"]})
    for i, (q, a) in enumerate(_turn_pairs(r)):
        if i == 0:
            parts = [{"type": "image", "image": p}
                     for p in _images(r, image_root, relative)]
            msgs.append({"role": "user",
                         "content": parts + [{"type": "text", "text": q}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": q}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    out = {"messages": msgs}
    if with_meta:
        out["id"] = r.get("qa_id")
        out["meta"] = _meta(r)
    return out


EXPORTERS = {
    "sharegpt": to_sharegpt,
    "llamafactory": to_llamafactory,
    "swift": to_swift,
    "openai": to_openai,
}


def export_records(records: Iterable[dict], out_path: str, fmt: str = "sharegpt",
                   image_root: Optional[str] = None, relative: bool = False,
                   with_system: bool = True, as_json_array: bool = False,
                   with_meta: bool = True) -> int:
    """写出 jsonl（默认）或 json 数组（LLaMA-Factory 的 dataset_info 也支持）。"""
    fn = EXPORTERS[fmt]
    items = [fn(r, image_root, relative, with_system, with_meta) for r in records]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if as_json_array:
            json.dump(items, f, ensure_ascii=False, indent=1)
        else:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)
