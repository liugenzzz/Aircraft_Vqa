#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每种任务抽一条出来，看看实际长什么样。

构建完扫一眼这个文件，比看统计数字直观得多 —— 问法顺不顺、答案对不对、
导出成训练格式之后 <image> 挂没挂对，一眼就能发现。

    python scripts/preview_tasks.py --vqa data/vqa --out data/vqa/preview.jsonl
    python scripts/preview_tasks.py --dataset synthetic_panel   # 只看某个源

输出每行一条，含：任务说明 + 原始问答 + 导出后的 messages 形态。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import _bootstrap  # noqa: F401

from aircraft_vqa.export import EXPORTERS

TASK_DESC = {
    "grounding_single": ("缺陷定位", "给定缺陷类型，输出该类型所有实例的边界框"),
    "grounding_all": ("缺陷定位", "检出图中全部异常，输出框 + 类型"),
    "grounding_negative": ("缺陷定位", "对正常图提同样要求，正确答案是空列表"),
    "grounding_counterfactual": ("缺陷定位", "对有缺陷的图问一个图里不存在的类型，答空列表（更难的负样本）"),
    "referring_region": ("缺陷定位", "给定一个坐标框，判断该区域内是否存在异常（局部聚焦）"),
    "counting": ("缺陷定位", "清点某类缺陷数量并逐个定位"),
    "region_word": ("缺陷定位", "用方位词描述缺陷位置，不给坐标（语义定位）"),
    "discrimination": ("缺陷识别", "判断整图有无异常"),
    "classification_open": ("缺陷识别", "开放式缺陷类型判断"),
    "classification_mc": ("缺陷识别", "四选一，干扰项取同族缺陷"),
    "description": ("缺陷识别", "综合检查记录：部件 + 缺陷 + 位置 + 范围 + 严重度"),
    "severity_action": ("缺陷识别", "严重度评估 + 维修处置建议"),
    "object_recognition": ("缺陷识别", "被检对象识别与航空语境"),
    "grade_assessment": ("缺陷识别", "有序程度分级判定（仅带等级标注的源会产生）"),
    "uncertainty": ("缺陷识别", "成像模糊/遮挡/过曝时答无法确认并建议补拍"),
    "multi_turn": ("多轮追问", "有无 → 定位 → 处置，复刻机务问诊流程"),
    "pair_compare": ("多图对比", "正常件参考图 + 待检图 → 指出差异（机务实际就是比着看）"),
}
ORDER = list(TASK_DESC)


def pick(records: list, task: str, rng: random.Random, prefer_anomalous=True):
    cands = [r for r in records if r["task"] == task]
    if not cands:
        return None
    if prefer_anomalous and task != "grounding_negative":
        anom = [r for r in cands if r.get("image_status") == "anomalous"]
        cands = anom or cands
    # 优先挑合成源 —— 图片能对着看，缺陷类型也是真的细粒度
    syn = [r for r in cands if r["dataset"].startswith("synthetic")]
    return rng.choice(syn or cands)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/vqa")
    ap.add_argument("--out", default="data/vqa/preview.jsonl")
    ap.add_argument("--dataset", default=None, help="只从这个源里挑")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format", default="sharegpt",
                    help="sharegpt|llamafactory|swift|openai")
    ap.add_argument("--raw", action="store_true",
                    help="只输出训练文件里真实的那一行，不带任务说明等元信息")
    args = ap.parse_args()

    records = []
    for fp in glob.glob(os.path.join(args.vqa, "*.raw.jsonl")):
        with open(fp, encoding="utf-8") as f:
            records += [json.loads(l) for l in f if l.strip()]
    if not records:
        print(f"{args.vqa} 下没有 *.raw.jsonl，先跑 scripts/build_vqa.py")
        return 1
    if args.dataset:
        records = [r for r in records if r["dataset"] == args.dataset]

    export = EXPORTERS[args.format]
    rng = random.Random(args.seed)
    rows, missing = [], []
    for task in ORDER:
        r = pick(records, task, rng)
        if r is None:
            missing.append(task)
            continue
        family, desc = TASK_DESC[task]
        row = {
            "任务": task,
            "任务族": family,
            "说明": desc,
            "来源": f"{r['dataset']}/{r['category']}",
            "图片": r.get("images") or r["image"],
            "图片尺寸": [r["width"], r["height"]],
            "图像状态": r.get("image_status"),
            "图里有的缺陷": r.get("image_defect_types") or [],
            "这条问的缺陷": r.get("asked_defect_types") or [],
            "答案断言的缺陷": r.get("answer_defect_types") or [],
            "输出格式": r.get("output_format"),
            "坐标模式": r.get("coord_mode"),
            "system": r["system"],
        }
        if r.get("turns"):
            row["多轮"] = [{"问": t["question"], "答": t["answer"]}
                           for t in r["turns"]]
        else:
            row["问"] = r["question"]
            row["答"] = r["answer"]
        for k in ("options", "answer_letter", "grade", "count", "n_boxes",
                  "region_answer", "yes_no", "severity", "variant",
                  "degradation", "reference_image"):
            if k in r:
                row[k] = r[k]
        row[f"导出后（{args.format} 格式）"] = export(r)
        rows.append(export(r) if args.raw else row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(rows)}/{len(ORDER)} 种任务各抽一条 -> {args.out}")
    for task in ORDER:
        if task not in missing:
            fam, _ = TASK_DESC[task]
            print(f"  [{fam}] {task}")
    if missing:
        print(f"\n当前数据没有这些任务：{missing}")
        print("  grade_assessment 需要带有序等级标注的源（VT 腐蚀分级集）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
