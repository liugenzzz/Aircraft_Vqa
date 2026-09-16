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

扩写结果会逐条校验，不合格的直接丢弃并报数。校验分两档：
- 占位符：白名单 + 必需项 + 可 format()，任何风格都查；
- 风格（--style instruction，默认）：祈使句为主、标准书面中文、
  拒口语语气词、拒行话堆砌、拒无标点碎片。

**为什么强制指令式**：在窄领域数据上做 SFT，如果问法本身口语化、碎片化，
模型的语言层会被带偏 —— 通用对话时也变得生硬、爱省略。指令式表述离
instruct 模型原本的 SFT 分布最近，对语言能力的扰动最小。
另外强烈建议按 5~10% 混入通用指令数据（build_vqa.py 的 --mix-general），
这是防语言层退化最有效的一招。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401

from aircraft_vqa.vqa import templates as T

PROMPT = """你在为一个民航机务维修视觉检查助手准备训练数据的**指令模板**。

任务类型：{task}
任务说明：{desc}

已有的种子指令（注意里面的 {{占位符}}）：
{seeds}

请再写 {n} 条**不同表述**的指令，要求：

【风格 —— 最重要】
1. 以**祈使句为主**（"请……""标出……""列出……""判断……"），
   这是指令微调模型最熟悉的形态；疑问句最多占三成，且必须是完整规范的问句。
2. 用**标准书面中文**。不要口语语气词（呗、啊、哈、嘛、咋、呗），
   不要方言，不要网络用语，不要表情符号，不要省略主谓的短语碎片。
3. 术语要克制：民航维修术语（工卡、AMM、SRM、适航、放行）**最多出现一次**，
   而且必须用得自然。堆砌行话会让模型学到一种畸形文风。
4. 每条 10~40 字，句子结构完整，以句号、问号或分号结尾。

【内容】
5. 占位符只能用这些：{allowed}；其中这些必须出现：{required}；
   占位符要原样保留大括号，不要翻译、不要替换成具体词。
6. 表述角度要有变化：有的强调输出格式，有的强调检查对象，
   有的强调排查范围，但都要是**清晰、可执行的指令**。
7. 不要重复种子指令，也不要互相重复。

只输出一个 JSON 数组，形如 ["指令1", "指令2"]，不要任何解释。"""

# 口语/碎片化标记 —— 命中就拒收。
# 理由：在窄领域数据上做 SFT，如果问法本身是口语碎片，模型的语言层会被带偏，
# 表现为通用对话时也变得生硬、爱省略。指令式表述离 instruct 模型原本的
# SFT 分布最近，对语言能力的扰动最小。
COLLOQUIAL = ["呗", "咋", "嘛", "哈哈", "啦", "喽", "哟", "呐", "啊？",
              "这玩意", "整一下", "搞一下", "瞅", "瞧瞧", "弄一下"]
JARGON = ["工卡", "AMM", "SRM", "适航", "放行", "定检", "IPC", "NDT", "机务"]
END_PUNCT = ("。", "？", "；", "！", ")", "）")

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


def validate(task: str, q: str, seen: set, style: str = "instruction") -> str:
    """返回拒收理由，空串表示通过。"""
    q = q.strip()
    if not (6 <= len(q) <= 120):
        return "长度不合适"
    if q in seen:
        return "重复"
    if style == "instruction":
        for w in COLLOQUIAL:
            if w in q:
                return f"口语化（{w}）"
        if not q.endswith(END_PUNCT):
            return "结尾缺标点，可能是碎片"
        n_jargon = sum(q.count(w) for w in JARGON)
        if n_jargon > 1:
            return f"行话堆砌（{n_jargon} 处）"
        if q.count("，") > 3:
            return "从句过多，不像清晰指令"
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


def parse_array(text: str):
    """从模型输出里取出字符串数组，返回 (列表, 是否是从截断输出里抢救的)。

    先按完整 JSON 数组解析；失败就退而求其次，把已经写完整的字符串项捡出来。
    输出被 max_tokens 切掉时，前面那几十条其实是好的，整批丢掉太浪费 ——
    而且"一条都没有"会让人以为是模型不听话，其实只是没写完。
    """
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v, False
        except Exception:
            pass
    start = text.find("[")
    if start < 0:
        return None, False
    # 捡出所有成对引号里的内容（跳过转义引号），最后一条没闭合的自然落选
    items = re.findall(r'"((?:[^"\\]|\\.)*)"', text[start:])
    out = []
    for it in items:
        try:
            out.append(json.loads(f'"{it}"'))
        except Exception:
            continue
    return (out, True) if out else (None, False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/question_bank.json")
    ap.add_argument("--llm-config", default="configs/llm_pool.json",
                    help="模型池配置")
    ap.add_argument("--purpose", default="paraphrase",
                    help="用 llm_pool.json 里哪个用途的参数（扩问法建议高温度）")
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--style", default="instruction",
                    choices=["instruction", "loose"],
                    help="instruction=强制指令式并做风格校验（默认，"
                         "对模型语言层扰动最小）；loose=只校验占位符")
    ap.add_argument("--chunk", type=int, default=25,
                    help="每轮最多要几条。要太多的话模型后半段质量下滑，"
                         "也更容易撞上截断")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="每个任务最多问几轮 —— 风格校验会拒掉一部分，"
                         "一轮常常不够数")
    ap.add_argument("--workers", type=int, default=0,
                    help="并发路数，默认取池里可用模型数")
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

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    from aircraft_vqa.llm import LLMPool
    pool = LLMPool.from_file_or_none(args.llm_config)
    if not pool or not pool.available(args.purpose):
        print(f"模型池不可用（{args.llm_config}）。先填好配置，"
              f"再跑 scripts/llm_pool_check.py 确认连得上。\n"
              f"想先看会发什么 prompt，用 --dry-run。")
        return 1
    print(f"模型池：\n{pool.describe()}\n")

    def one_round(task: str, want: int, have: list) -> tuple:
        """要一轮，返回 (合格问法, 拒收原因计数, 用的模型)。

        把已经收下的问法一并发过去当"别再写这些"，否则多轮之间会大量重复，
        风格校验挡不住重复，只有 seen 能挡，白烧调用。
        """
        prompt = prompts[task]
        rejected_extra = {}
        # 条数必须每轮都改写成本轮真正要的 want。只在续轮改的话，第一轮
        # prompt 里写着 per_task（比如 40）而预算是按 chunk（25）算的，
        # 两边对不上，照样会截断。
        prompt = prompt.replace(f"请再写 {args.per_task} 条",
                                f"请再写 {want} 条")
        if have:
            prompt += ("\n\n【已经有了，不要重复，也不要只改动一两个字】\n"
                       + "\n".join(f"- {x}" for x in have[-40:]))
        # max_tokens 要按"要几条"算。一条中文问法约 20~25 字，JSON 的引号
        # 逗号还要占一些 —— 用途里那个固定的 512 在要 40 条时必然截断，
        # 模型写到一半被切，数组闭不上，最后报成"返回里没有 JSON 数组"，
        # 看不出是被截了。
        budget = max(512, want * 48 + 320)
        text, used, meta = pool.chat([{"role": "user", "content": prompt}],
                                     purpose=args.purpose,
                                     override={"max_tokens": budget},
                                     return_meta=True)
        if not text:
            return [], {"模型没返回": 1}, used
        cands, salvaged = parse_array(text)
        if cands is None:
            tail = text.strip()[-120:].replace("\n", " ")
            why = ("输出被 max_tokens 截断" if meta.get("truncated")
                   else "返回里没有 JSON 数组")
            return [], {f"{why}（末尾：…{tail}）": 1}, used
        if salvaged:
            rejected_extra["从截断的输出里抢救"] = \
                rejected_extra.get("从截断的输出里抢救", 0) + 1
        seen = set(T.QUESTIONS.get(task, [])) | set(have)
        kept, rejected = [], {}
        for q in cands:
            if not isinstance(q, str):
                continue
            why = validate(task, q, seen, args.style)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            seen.add(q.strip())
            kept.append(q.strip())
        rejected.update(rejected_extra)
        return kept, rejected, used

    def gen_task(task: str) -> tuple:
        """一个任务跑到够数为止 —— 风格校验会拒掉一部分，
        一轮定生死的话拿到手常常不足量，还得靠人猜着重跑。"""
        have, rejected, used = [], {}, ""
        # 一轮别要太多：要 60 条时模型后半段质量明显下滑，而且一次写太长
        # 更容易撞上截断。分几轮要，每轮还能把已有的发回去避重。
        ask = min(args.chunk, int(args.per_task * 1.6) + 2)
        for rnd in range(args.max_rounds):
            need = args.per_task - len(have)
            if need <= 0:
                break
            kept, rej, used = one_round(task, min(ask, int(need * 1.6) + 2), have)
            have.extend(kept)
            for k, v in rej.items():
                rejected[k] = rejected.get(k, 0) + v
            if not kept:
                # 颗粒无收。如果是被截断，下一轮少要点还有救；
                # 其他原因（模型不返回、压根没写数组）再试也一样，直接停。
                if any("截断" in k for k in rej):
                    ask = max(5, ask // 2)
                    continue
                break
        return task, have[:args.per_task], rejected, used

    bank, stats = {}, {}
    n_workers = max(1, min(args.workers or len(pool.available(args.purpose)),
                           len(prompts)))
    print(f"并发 {n_workers} 路，每个任务最多 {args.max_rounds} 轮，"
          f"目标每任务 {args.per_task} 条\n")
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(gen_task, t) for t in prompts]
        for fut in cf.as_completed(futs):
            task, kept, rejected, used = fut.result()
            bank[task] = kept
            stats[task] = {"kept": len(kept), "rejected": rejected}
            flag = "" if len(kept) >= args.per_task else "  <- 没到目标"
            print(f"[{'ok' if kept else 'fail'}] {task:24s} 收 {len(kept):>3d} 条"
                  f"{flag}" + (f"  拒收 {rejected}" if rejected else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "pool": args.llm_config, "style": args.style,
                   "generated_at": datetime.now(timezone.utc).isoformat(),
                   "stats": stats, "questions": bank}, f,
                  ensure_ascii=False, indent=1)
    total = sum(len(v) for v in bank.values())
    print(f"\n模型池调用统计：{json.dumps(pool.stats.summary(), ensure_ascii=False)}")
    print(f"共扩写 {total} 条问法 -> {args.out}")
    print("在 configs/build.yaml 里设置 question_bank 指向它即可生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
