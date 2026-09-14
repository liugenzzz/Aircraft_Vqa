# -*- coding: utf-8 -*-
"""质量校验：宁可少，不要错。构建完必须跑一遍，不通过的条目直接丢。"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Optional


def _parse_boxes(ans: str):
    """从答案里抽出 JSON 框列表；答案可能是 '共 3 处…\n[...]' 这种混合体。"""
    m = re.search(r"\[.*\]", ans, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _check_turns(r: dict) -> list:
    """多轮条目：逐轮按其实际形态校验，并检查轮次结构。"""
    errs = []
    turns = r["turns"]
    if len(turns) < 2:
        errs.append("too_few_turns")
    for i, t in enumerate(turns):
        if not t.get("question") or not t.get("answer"):
            errs.append(f"empty_turn_{i}")
            continue
        if re.search(r"\{[a-z_]+\}", t["question"] + t["answer"]):
            errs.append(f"unfilled_placeholder_turn_{i}")
        # 带 JSON 的那一轮按定位规则查
        if "[" in t["answer"]:
            probe = dict(r, task="grounding_all", answer=t["answer"],
                         label=r.get("label"))
            probe.pop("turns", None)
            if r.get("yes_no") == "no":
                probe["task"] = "grounding_negative"
            errs += [f"turn_{i}_{e}" for e in check_record(probe)]
    return errs


def check_record(r: dict, check_image: bool = False) -> list:
    """返回问题列表，空列表 = 通过。"""
    errs = []
    if r.get("turns"):
        errs += _check_turns(r)
    if not r.get("question") or not r.get("answer"):
        errs.append("empty_qa")
    if "{" in r.get("question", "") and "}" in r.get("question", ""):
        if re.search(r"\{[a-z_]+\}", r["question"]):
            errs.append("unfilled_placeholder_q")
    if re.search(r"\{[a-z_]+\}", r.get("answer", "")):
        errs.append("unfilled_placeholder_a")

    if check_image and not os.path.exists(r.get("image", "")):
        errs.append("image_missing")

    task = r.get("task", "")
    if task.startswith("grounding") or task == "counting":
        boxes = _parse_boxes(r["answer"])
        if boxes is None:
            errs.append("answer_not_json")
        else:
            if task == "grounding_negative" and boxes:
                errs.append("negative_with_boxes")
            if task in ("grounding_single", "grounding_all") and not boxes \
                    and r.get("label") == "anomalous":
                errs.append("positive_without_boxes")
            hi = 1000 if r.get("coord_mode", "norm1000") == "norm1000" else None
            for b in boxes:
                if not isinstance(b, dict) or "bbox_2d" not in b:
                    errs.append("bad_box_item"); break
                c = b["bbox_2d"]
                if not (isinstance(c, list) and len(c) == 4):
                    errs.append("bad_box_len"); break
                if c[2] <= c[0] or c[3] <= c[1]:
                    errs.append("degenerate_box"); break
                if hi and (min(c) < 0 or max(c) > hi):
                    errs.append("box_out_of_range"); break
                if not hi:
                    w, h = r.get("width", 0), r.get("height", 0)
                    if c[0] < 0 or c[1] < 0 or c[2] > w or c[3] > h:
                        errs.append("box_out_of_image"); break
        if task == "counting":
            m = re.search(r"共发现\s*(\d+)\s*处", r["answer"])
            if m and boxes is not None and int(m.group(1)) != len(boxes):
                errs.append("count_mismatch")

    if task == "classification_mc":
        letter = (r.get("answer_letter") or "")
        if not letter or letter not in "ABCDEF":
            errs.append("bad_mc_letter")
        elif not r["answer"].startswith(letter):
            errs.append("mc_answer_letter_mismatch")
        opts = r.get("options") or []
        if len(set(opts)) != len(opts):
            errs.append("duplicate_options")

    if task == "discrimination":
        yn = r.get("yes_no")
        if yn == "no" and re.search(r"存在(异常|缺陷)|不合格", r["answer"]):
            errs.append("negative_answer_says_positive")
        if yn == "yes" and re.search(r"未(见|发现)异常|(?<!不)合格。", r["answer"]):
            errs.append("positive_answer_says_negative")
    return errs


def run_qc(records: list, check_image: bool = False) -> tuple:
    """返回 (通过的条目, 报告)。"""
    ok, bad, counter = [], [], Counter()
    for r in records:
        errs = check_record(r, check_image)
        if errs:
            counter.update(errs)
            bad.append({"qa_id": r.get("qa_id"), "task": r.get("task"), "errors": errs})
        else:
            ok.append(r)
    report = {
        "n_input": len(records), "n_pass": len(ok), "n_fail": len(bad),
        "pass_rate": round(len(ok) / len(records), 4) if records else 0.0,
        "errors": dict(counter.most_common()),
        "examples": bad[:20],
    }
    return ok, report
