#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用大模型扩写问法库（一次性离线跑，产出 configs/question_bank.json）。

**为什么是扩问法而不是逐条改写答案**：
问法是"骨架"，只有十几个任务、每个几条种子，扩写一次几百个 token 就够；
而答案有几十万条，逐条调大模型既贵又容易把坐标和数量改坏。
扩问法的收益（句式多样性、不让模型过拟合到固定问法）几乎和逐条改写一样，
成本却差三四个数量级。所以推荐的用法是：

    问法 —— 大模型扩写（这个脚本，一次性，几十次调用）
    答案 —— 模板确定性生成（可复现、事实不会漂）
    答案润色 —— 可选，只对描述类任务开，且有事实指纹保护（vqa/llm.py）

用法：

    export LLM_API_KEY=sk-xxx
    export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    python scripts/gen_question_bank.py --model qwen-plus --per-task 20

扩写结果会逐条校验占位符与可格式化性，不合格的直接丢弃并报数。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401

from aircraft_vqa.vqa import templates as T

PROMPT = """你在为一个民航机务维修视觉检查助手准备训练数据的**问法模板**。

任务类型：{task}
任务说明：{desc}

已有的种子问法（注意里面的 {{占位符}}）：
{seeds}

请再写 {n} 条**不同句式**的问法，要求：
1. 必须是一线机务/质检人员会说的中文，可以带工卡、AMM、SRM、放行这类行话；
2. 占位符只能用这些：{allowed}；其中这些必须出现：{required}；
   占位符要原样保留大括号，不要翻译、不要替换成具体词；
3. 句式要有变化：陈述式、疑问式、祈使式都要有，长短交错；
4. 不要重复种子问法，也不要互相重复；
5. 只输出一个 JSON 数组，形如 ["问法1", "问法2"]，不要任何解释。"""

DESC = {
    "grounding_single": "定位图中指定类型的缺陷，答案是 JSON 边界框列表",
    "grounding_all": "检出并列出图中全部异常及其类型，答案是 JSON",
    "grounding_negative": "对一张正常图提出找缺陷的要求，正确答案是空列表",
    "referring_region": "给定一个坐标框，判断该区域内是否存在异常",
    "region_word": "用方位词（左上/正中等）描述缺陷在画面中的位置",
    "counting": "清点某类缺陷的数量并逐个定位",
    "discrimination": "判断整张图有无异常",
    "classification_open": "判断缺陷属于哪一类（开放式回答）",
    "description": "写一段检查记录，说明部件状况与缺陷情况",
    "severity_action": "评估缺陷严重程度并给出维修处置建议",
    "object_recognition": "识别被检部件是什么、检查时关注什么",
}

DUMMY = {"defect": "紧固件缺失", "defect_en": "missing fastener",
         "obj": "检查口盖", "ctx": "可卸口盖壁板", "box": "[10, 20, 30, 40]"}


def placeholders(text: str) -> set:
    return set(re.findall(r"\{([a-z_]+)\}", text))


def validate(task: str, q: str, seen: set) -> str:
    """返回拒收理由，空串表示通过。"""
    q = q.strip()
    if not (6 <= len(q) <= 120):
        return "长度不合适"
    if q in seen:
        return "重复"
    ph = placeholders(q)
    allowed = T.ALLOWED_PLACEHOLDERS.get(task, set())
    required = T.REQUIRED_PLACEHOLDERS.get(task, set())
    if not ph <= allowed:
        return f"用了池外占位符 {sorted(ph - allowed)}"
    if not required <= ph:
        return f"缺少必需占位符 {sorted(required - ph)}"
    try:
        q.format(**DUMMY)
    except Exception as e:
        return f"无法格式化: {e}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/question_bank.json")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""))
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要发出的 prompt，不调用大模型")
    args = ap.parse_args()

    tasks = args.tasks or list(T.QUESTIONS)
    prompts = {}
    for task in tasks:
        seeds = T.QUESTIONS.get(task, [])
        prompts[task] = PROMPT.format(
            task=task, desc=DESC.get(task, ""), n=args.per_task,
            seeds="\n".join(f"- {x}" for x in seeds),
            allowed="、".join(f"{{{p}}}" for p in sorted(
                T.ALLOWED_PLACEHOLDERS.get(task, []))) or "（无）",
            required="、".join(f"{{{p}}}" for p in sorted(
                T.REQUIRED_PLACEHOLDERS.get(task, []))) or "（无强制）")

    if args.dry_run:
        for task, p in prompts.items():
            print(f"\n{'=' * 20} {task} {'=' * 20}\n{p}")
        print(f"\n共 {len(prompts)} 次调用，每次约 "
              f"{sum(len(p) for p in prompts.values()) // len(prompts)} 字符输入。")
        return 0

    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"环境变量 {args.api_key_env} 未设置。\n"
              f"想先看看会发什么 prompt，用 --dry-run。")
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("需要 openai 包：pip install openai")
        return 1
    client = OpenAI(api_key=key, base_url=args.base_url or None)

    bank, stats = {}, {}
    for task, prompt in prompts.items():
        try:
            resp = client.chat.completions.create(
                model=args.model, temperature=args.temperature,
                messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
        except Exception as e:
            print(f"[fail] {task}: {e}")
            continue
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            print(f"[fail] {task}: 返回里没有 JSON 数组")
            continue
        try:
            cands = json.loads(m.group(0))
        except Exception as e:
            print(f"[fail] {task}: JSON 解析失败 {e}")
            continue

        seen = set(T.QUESTIONS.get(task, []))
        kept, rejected = [], {}
        for q in cands:
            if not isinstance(q, str):
                continue
            why = validate(task, q, seen)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            seen.add(q.strip())
            kept.append(q.strip())
        bank[task] = kept
        stats[task] = {"kept": len(kept), "rejected": rejected}
        print(f"[ok] {task}: 收 {len(kept)}/{len(cands)} 条"
              + (f"，拒收原因 {rejected}" if rejected else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "model": args.model,
                   "generated_at": datetime.now(timezone.utc).isoformat(),
                   "stats": stats, "questions": bank}, f,
                  ensure_ascii=False, indent=1)
    total = sum(len(v) for v in bank.values())
    print(f"\n共扩写 {total} 条问法 -> {args.out}")
    print("在 configs/build.yaml 里设置 question_bank 指向它即可生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
