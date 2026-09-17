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
                         image_status=r.get("image_status"))
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
    for field in ("question", "answer"):
        w = duplicated_word(r.get(field, "") or "")
        if w:
            errs.append(f"duplicated_word_{field[0]}:{w}")
    for t in (r.get("turns") or []):
        w = duplicated_word(t.get("value", "") or t.get("answer", "") or "")
        if w:
            errs.append(f"duplicated_word_turn:{w}")
            break

    if check_image:
        for p in (r.get("images") or [r.get("image", "")]):
            if not os.path.exists(p):
                errs.append("image_missing")
                break

    # 多图条目：图片 token 数必须和图片数一致，否则训练时直接报错
    if r.get("images"):
        n_tok = r.get("question", "").count("<image>")
        if n_tok and n_tok != len(r["images"]):
            errs.append("image_token_count_mismatch")

    task = r.get("task", "")

    # 按 output_format 自动校验答案形态。这是踩了两次坑之后加的硬闸：
    # system 说什么格式，答案就必须是什么格式。
    ofmt = r.get("output_format")
    ans = r.get("answer", "").strip()
    if ofmt == "json_only":
        if not (ans.startswith("[") and ans.endswith("]")):
            errs.append("not_pure_json")
        else:
            try:
                json.loads(ans)                 # 整串必须能解析，不能有多余字符
            except Exception:
                errs.append("not_pure_json")
    elif ofmt == "text":
        if "bbox_2d" in ans:
            errs.append("text_format_has_boxes")
    elif ofmt == "text_then_json":
        if "\n" not in ans or not ans.rsplit("\n", 1)[-1].strip().startswith("["):
            errs.append("not_text_then_json")

    # system 必须与 output_format 对应
    if ofmt and r.get("system"):
        from .vqa.templates import SYSTEM_BY_OUTPUT
        if r["system"] != SYSTEM_BY_OUTPUT.get(ofmt, r["system"]):
            errs.append("system_format_mismatch")

    # 幻觉闸门：答案断言存在的缺陷必须是图里真有的
    img_t = set(r.get("image_defect_types") or [])
    ans_t = set(r.get("answer_defect_types") or [])
    if ans_t and img_t and not ans_t <= img_t:
        errs.append("answer_asserts_absent_defect")

    # 不确定样本必须有物理依据 —— 没有劣化证据就说"看不清"，是在教模型耍赖
    if task == "uncertainty" and not (r.get("degradation") or r.get("image_quality")):
        errs.append("uncertainty_without_evidence")

    # 答案不能提问题里没出现过的缺陷名（负样本模板复用正样本变量槽的典型症状）
    if task in ("grounding_negative", "grounding_counterfactual"):
        q = r.get("question", "")
        for name in re.findall(r"[\u4e00-\u9fa5]{2,6}", r.get("answer", "")):
            if len(name) >= 3 and name not in q:
                errs.append("answer_mentions_unasked_term")
                break

    if task == "uncertainty":
        a = r.get("answer", "")
        if not re.search(r"无法确认|不足以|不可辨|补拍|重新取证", a):
            errs.append("uncertainty_without_hedge")
        if "bbox_2d" in a:
            errs.append("uncertainty_with_boxes")

    if task.startswith("grounding") or task == "counting":
        boxes = _parse_boxes(r["answer"])
        if boxes is None:
            errs.append("answer_not_json")
        else:
            if task in ("grounding_negative", "grounding_counterfactual") and boxes:
                errs.append("negative_with_boxes")
            if task in ("grounding_single", "grounding_all") and not boxes \
                    and r.get("image_status") == "anomalous":
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
            m = re.search(r"共\s*(\d+)\s*处", r["answer"])
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


# 拼接产生的叠词。size_word/severity 这类工具返回的是**完整短语**
# （"范围中等"而不是"中等"），模板或 f-string 再补一次前缀就会拼出
# "范围范围中等"。只读 stats 看不出来，只有读答案原文才发现。
DUP_WORDS = ("范围范围", "程度程度", "严重严重", "位于位于", "缺陷缺陷",
             "检查检查", "区域区域")


def duplicated_word(text: str) -> str:
    for w in DUP_WORDS:
        if w in text:
            return w
    return ""


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
