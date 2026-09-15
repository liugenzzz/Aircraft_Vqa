#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大模型改写层的小批量烟测：先拿 100 条试，确认靠谱了再全量跑。

改写层的风险是**悄悄改事实** —— 把坐标、数量、缺陷类型改掉，而 loss 照样很好看。
所以这个脚本不是"跑完就完"，它会：
  1. 只对可以改写的任务取样（定位类结构化输出根本不送去改写）；
  2. 每条做事实指纹比对（数字、JSON 片段必须逐字一致）+ 重跑质检；
  3. 输出一份并排对照文件，人工过一遍；
  4. 给出回退率与各类拒收原因的统计。

    # 1) 把端点和模型填进 configs/llm_pool.json，key 走环境变量
    export LOCAL_LLM_KEY=xxx
    # 2) 先体检，确认模型通
    python scripts/llm_pool_check.py
    # 3) 跑烟测
    python scripts/llm_smoke.py --n 100
    python scripts/llm_smoke.py --n 100 --llm-config configs/llm_pool.prod.json

    python scripts/llm_smoke.py --dry-run          # 不调模型，只看会送什么进去

池里可以放多个模型，按权重路由、失败自动转移；报告会按模型分别统计采纳率，
方便看出哪个模型改写质量更好。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import yaml

from aircraft_vqa.qc import check_record
from aircraft_vqa.vqa.llm import PROTECTED_TASKS, LLMRewriter


def load_records(vqa_dir: str) -> list:
    out = []
    for fp in glob.glob(os.path.join(vqa_dir, "*.raw.jsonl")):
        with open(fp, encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def sample_balanced(records: list, n: int, rng: random.Random) -> list:
    """按任务均匀取样，别让样本全集中在最多的那个任务上。"""
    cand = [r for r in records if r["task"] not in PROTECTED_TASKS]
    by_task = {}
    for r in cand:
        by_task.setdefault(r["task"], []).append(r)
    tasks = sorted(by_task)
    if not tasks:
        return []
    per = max(1, n // len(tasks))
    out = []
    for t in tasks:
        rng.shuffle(by_task[t])
        out += by_task[t][:per]
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/vqa")
    ap.add_argument("--config", default="configs/build.yaml")
    ap.add_argument("--out", default="data/vqa/llm_smoke.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--llm-config", default=None,
                    help="模型池配置；默认取 build.yaml 里 llm.pool_config")
    ap.add_argument("--purpose", default="rewrite",
                    help="用 llm_pool.json 里哪个用途的参数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="不调模型，只打印将要送进去的条目")
    args = ap.parse_args()

    records = load_records(args.vqa)
    if not records:
        print(f"{args.vqa} 下没有 *.raw.jsonl，先跑 scripts/build_vqa.py")
        return 1
    rng = random.Random(args.seed)
    picked = sample_balanced(records, args.n, rng)
    if not picked:
        print("没有可改写的任务条目（定位类结构化输出不参与改写）")
        return 1

    print(f"从 {len(records)} 条里按任务均匀取了 {len(picked)} 条")
    print(f"任务分布：{dict(Counter(r['task'] for r in picked).most_common())}")
    print(f"受保护、不送改写的任务：{sorted(PROTECTED_TASKS)}")

    if args.dry_run:
        for r in picked[:5]:
            print(f"\n--- [{r['task']}] {r['question']}")
            print(f"    原答案：{r['answer'][:160]}")
        print(f"\n（--dry-run，未调用模型。共 {len(picked)} 条待改写）")
        return 0

    cfg = (yaml.safe_load(open(args.config, encoding="utf-8")) or {}).get("llm", {})
    pool_path = args.llm_config or cfg.get("pool_config")
    rw = LLMRewriter.from_config(pool_path, args.purpose)
    if not rw.enabled:
        print(f"模型池不可用（{pool_path}）。先填好 configs/llm_pool.json，"
              f"再跑 scripts/llm_pool_check.py 确认连得上。")
        return 1
    print(f"\n模型池：\n{rw.pool.describe()}\n")

    rows, reasons = [], Counter()
    for i, r in enumerate(picked, 1):
        before = r["answer"]
        rw.rewrite(r)
        ok = bool(r.get("rewritten"))
        if not ok:
            reasons[r.get("rewrite_rejected", "no_response")] += 1
        rows.append({
            "任务": r["task"], "输出格式": r.get("output_format"),
            "改写模型": r.get("rewritten_by"),
            "问": r["question"],
            "原答案": before,
            "改写后": r["answer"] if ok else None,
            "是否采纳": ok,
            "拒收原因": r.get("rewrite_rejected"),
            "图片": r.get("images") or r["image"],
        })
        if i % 20 == 0:
            print(f"  ...{i}/{len(picked)}，已采纳 {sum(x['是否采纳'] for x in rows)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ok = sum(x["是否采纳"] for x in rows)
    print(f"\n{'=' * 60}")
    print(f"采纳 {n_ok}/{len(rows)} ({n_ok / len(rows):.0%})")
    if reasons:
        print("拒收原因：")
        for k, v in reasons.most_common():
            tip = {"fact_drift": "改动了数字或 JSON —— 事实漂移，必须拒",
                   "qc_fail": "改完过不了质检",
                   "no_response": "模型没返回（超时/报错）"}.get(k, "")
            print(f"  {k:14s} {v:4d}  {tip}")
    by_task = Counter(x["任务"] for x in rows if x["是否采纳"])
    print(f"各任务采纳数：{dict(by_task.most_common())}")
    by_model = Counter(x["改写模型"] for x in rows if x["是否采纳"])
    if by_model:
        print(f"各模型采纳数：{dict(by_model.most_common())}")
    print(f"\n模型池调用统计：{json.dumps(rw.pool.stats.summary(), ensure_ascii=False)}")
    print(f"\n并排对照 -> {args.out}")
    print("请人工过一遍：改写后是否更自然、有没有丢信息、有没有编造。")
    print("采纳率低于 60% 就别全量跑，先调 vqa/llm.py 里的 REWRITE_SYSTEM。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
