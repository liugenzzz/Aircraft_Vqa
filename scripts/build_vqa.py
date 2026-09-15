#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：统一样本 -> VQA 数据集（含平衡、质检、切分、导出）。

    python scripts/build_vqa.py --interim data/interim --out data/vqa
    python scripts/build_vqa.py --total 50000 --format swift --coord-mode abs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import _bootstrap  # noqa: F401
import yaml

from aircraft_vqa.balance import diversity as st_mod_diversity
from aircraft_vqa.balance import (balance_samples, cap_class_imbalance,
                                  class_distribution, dedup, group_split,
                                  quota_sample, ratio_gap, stats)
from aircraft_vqa.export import export_records
from aircraft_vqa.qc import run_qc
from aircraft_vqa.schema import dump_records, read_jsonl
from aircraft_vqa.taxonomy import get_taxonomy
from aircraft_vqa.vqa import BuildConfig, VQABuilder
from aircraft_vqa.vqa.llm import load_rewriter


def _mix_general(train_path: str, general_path: str, ratio: float,
                 seed: int) -> None:
    """把通用指令数据按比例混进 train 导出文件。

    窄领域 SFT 最常见的副作用是语言层退化：模型答专业题很准，
    但一聊别的就生硬、爱省略。按 5~10% 混入通用指令数据是最省事的解法，
    比反复调问法风格有效得多。
    """
    import random as _r
    if not os.path.exists(general_path):
        print(f"[mix] 找不到通用数据 {general_path}，跳过")
        return
    if not (0 < ratio < 0.5):
        print(f"[mix] mix_ratio={ratio} 不在 (0, 0.5) 内，跳过")
        return
    with open(train_path, encoding="utf-8") as f:
        ours = [l for l in f if l.strip()]
    with open(general_path, encoding="utf-8") as f:
        pool = [l for l in f if l.strip()]
    if not pool:
        print(f"[mix] {general_path} 是空的，跳过")
        return
    want = int(len(ours) * ratio / (1 - ratio))
    rng = _r.Random(seed)
    picked = (rng.sample(pool, want) if want <= len(pool)
              else [rng.choice(pool) for _ in range(want)])
    if want > len(pool):
        print(f"[mix] 通用数据只有 {len(pool)} 条，需要 {want} 条，将重复采样")
    merged = ours + picked
    rng.shuffle(merged)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(merged)
    print(f"[mix] 混入通用数据 {len(picked)} 条，train 共 {len(merged)} 条"
          f"（通用占比 {len(picked) / len(merged):.1%}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--config", default="configs/build.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--out", default="data/vqa")
    ap.add_argument("--total", type=int, default=None, help="目标问答总量")
    ap.add_argument("--format", default=None,
                    help="sharegpt|llamafactory|swift|openai")
    ap.add_argument("--coord-mode", default=None, help="norm1000|abs")
    ap.add_argument("--only", nargs="*", default=None, help="只用这些数据源")
    ap.add_argument("--commercial-only", action="store_true")
    ap.add_argument("--check-images", action="store_true",
                    help="质检时逐条确认图片存在（慢但稳）")
    ap.add_argument("--llm-rewrite", action="store_true", help="启用大模型改写层")
    ap.add_argument("--mix-general", default=None,
                    help="通用指令数据 jsonl（已是导出格式），按比例混进 train，"
                         "防止窄领域微调把模型的语言层带偏")
    ap.add_argument("--mix-ratio", type=float, default=0.08,
                    help="通用数据在 train 中的占比，建议 0.05~0.10")
    ap.add_argument("--ratio-strict", action="store_true",
                    help="严格服从 task_ratio（会按最稀缺任务缩量，数据量小很多）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tax = get_taxonomy(os.path.abspath(args.taxonomy))
    os.makedirs(args.out, exist_ok=True)

    # ---- 读入统一样本 ----
    files = sorted(glob.glob(os.path.join(args.interim, "*.jsonl")))
    samples = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        if name.startswith("_") or (args.only and name not in args.only):
            continue
        part = list(read_jsonl(fp))
        if args.commercial_only:
            part = [s for s in part if s.commercial_ok]
        samples.extend(part)
        print(f"[load] {name}: {len(part)}")
    if not samples:
        print("没有可用样本，先跑 scripts/ingest.py")
        return 1

    # ---- 样本层平衡 ----
    before = Counter(s.label for s in samples)
    samples = balance_samples(samples, cfg.get("normal_per_anomalous", 1.0),
                              cfg.get("seed", 0))
    after = Counter(s.label for s in samples)
    print(f"[balance] {dict(before)} -> {dict(after)}")

    # ---- 构建问答 ----
    bcfg = BuildConfig(
        seed=cfg.get("seed", 0),
        coord_mode=args.coord_mode or cfg.get("coord_mode", "norm1000"),
        box_label=cfg.get("box_label", "zh"),
        max_qa_per_sample=cfg.get("max_qa_per_sample", 4),
        n_options=cfg.get("n_options", 4),
        skip_unknown_type_tasks=cfg.get("skip_unknown_type_tasks", True),
        active_defect_types=cfg.get("active_defect_types"),
        question_bank=cfg.get("question_bank"),
        lang=cfg.get("lang", "zh"),
        term_annotation=cfg.get("term_annotation", False),
    )
    if bcfg.active_defect_types:
        print(f"[scope] 本期缺陷范围 {len(bcfg.active_defect_types)} 类："
              f"{'、'.join(tax.zh(t) for t in bcfg.active_defect_types)}")
    n_bank = sum(len(v) for v in
                 __import__("aircraft_vqa.vqa.templates", fromlist=["x"])
                 .load_question_bank(bcfg.question_bank or "").values())
    print(f"[qbank] 扩写问法 {n_bank} 条"
          + ("" if n_bank else "（未生成，仅用种子问法；见 scripts/gen_question_bank.py）"))
    builder = VQABuilder(bcfg, tax)

    # 多图对比要一张同类正常件参考图。按 (数据源, 类别) 收集正常图，
    # 只有真的有正常图的类别才会产出 pair_compare 样本。
    from collections import defaultdict as _dd
    ref_pool = _dd(list)
    for s_ in samples:
        if not s_.is_anomalous:
            ref_pool[(s_.dataset, s_.category)].append(s_.image_path)
    ref_pool = {k: v for k, v in ref_pool.items() if len(v) >= 3}
    builder.set_reference_pool(ref_pool)
    if ref_pool:
        print(f"[pair] 参考图池：{len(ref_pool)} 个类别，"
              f"共 {sum(len(v) for v in ref_pool.values())} 张正常图")

    records = list(builder.build_many(samples))
    print(f"[build] 生成 {len(records)} 条原始问答")
    if builder.failures:
        print("[build] ⚠ 模板异常（这些任务被静默跳过了，务必查）：")
        for k, v in builder.failures.most_common(8):
            print(f"        {v:5d}x  {k}")

    records = dedup(records)
    print(f"[dedup] 保留 {len(records)} 条")

    # ---- 类别均衡：压平"类型识别"任务的长尾 ----
    cls_tasks = cfg.get("class_balance_tasks",
                        ["classification_open", "classification_mc"])
    mom = float(cfg.get("class_max_over_min", 3.0))
    if mom > 0:
        before = class_distribution(records, cls_tasks)
        records = cap_class_imbalance(records, cls_tasks, mom,
                                      seed=cfg.get("seed", 0))
        after = class_distribution(records, cls_tasks)
        if before != after:
            print(f"[class] 类型识别任务类别均衡（最多 ≤ {mom}× 最少）")
            print(f"        前 {before}")
            print(f"        后 {after}")

    # ---- 质检 ----
    records, report = run_qc(records, check_image=args.check_images)
    print(f"[qc] 通过 {report['n_pass']}/{report['n_input']} "
          f"({report['pass_rate']:.2%})  错误分布={report['errors']}")

    # ---- 配比抽样 ----
    tratio = cfg.get("task_ratio", {})
    total = args.total
    if total is None and args.ratio_strict:
        from collections import defaultdict
        by_t = defaultdict(int)
        for r in records:
            by_t[r["task"]] += 1
        w = sum(v for v in tratio.values() if v > 0) or 1.0
        total = min(int(by_t[k] / (v / w)) for k, v in tratio.items()
                    if v > 0 and by_t.get(k))
    records = quota_sample(records, tratio, total, cfg.get("seed", 0))
    print(f"[quota] 抽样后 {len(records)} 条")

    gaps = ratio_gap(records, tratio)
    short = [g for g in gaps if g["attain"] < 0.6]
    if short:
        print("[ratio] 以下任务未达目标配比 —— 通常是源数据缺少对应标注，"
              "不是构建器的问题：")
        for g in short:
            print(f"         {g['task']:22s} 实际 {g['actual']:.1%} / "
                  f"目标 {g['target']:.1%}（达成 {g['attain']:.0%}，{g['n']} 条）")
        print("         补法见 docs/01_dataset_survey.md：类型/严重度类问题需要"
              "带细粒度缺陷类别的源（MVTec AD screw、Roboflow aircraft_skin_defects 等）。")

    # ---- 可选：大模型改写 ----
    if args.llm_rewrite:
        rw = load_rewriter(cfg.get("llm", {}))
        if rw.enabled:
            n_ok = 0
            for i, r in enumerate(records):
                rw.rewrite(r)
                n_ok += int(bool(r.get("rewritten")))
                if (i + 1) % 200 == 0:
                    print(f"  ...改写 {i+1}/{len(records)}，成功 {n_ok}")
            print(f"[llm] 改写成功 {n_ok}/{len(records)}")
        else:
            print("[llm] provider=none，跳过改写")

    # ---- 切分 + 导出 ----
    splits = group_split(records, tuple(cfg.get("split_ratios", [0.95, 0.03, 0.02])),
                         cfg.get("seed", 0))
    exp = cfg.get("export", {})
    fmt = args.format or exp.get("format", "sharegpt")
    n_total = 0
    for name, recs in splits.items():
        if not recs:
            continue
        dump_records(os.path.join(args.out, f"{name}.raw.jsonl"), recs)
        n = export_records(recs, os.path.join(args.out, f"{name}.{fmt}.jsonl"),
                           fmt=fmt, with_system=exp.get("with_system", True),
                           relative=exp.get("relative_paths", False),
                           with_meta=exp.get("with_meta", True))
        n_total += n
        print(f"[export] {name}: {n} 条 -> {args.out}/{name}.{fmt}.jsonl")

    # ---- 可选：混入通用指令数据 ----
    if args.mix_general:
        _mix_general(os.path.join(args.out, f"train.{fmt}.jsonl"),
                     args.mix_general, args.mix_ratio, cfg.get("seed", 0))

    # 多样性门限：模板法最容易"换汤不换药"，构建完必须报出来
    _d = st_mod_diversity(records)
    thin = {k: v for k, v in _d["unique_questions_per_task"].items() if v < 30}
    if thin:
        print(f"[diversity] 以下任务的不同问法数不足 30（模板复读风险）：")
        for k, v in sorted(thin.items(), key=lambda x: x[1]):
            print(f"            {k:26s} {v}")
        print("            -> 跑 scripts/gen_question_bank.py 用大模型扩写问法")
    share = _d.get("top20_share_excl_empty", 0)
    print(f"[diversity] 排除空列表后，top20 答案占比 {share:.1%}"
          + ("  ⚠ 超过 5%，有模板在复读" if share > 0.05 else ""))

    st = stats(records)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": st, "qc": report, "ratio_gap": gaps,
                   "splits": {k: len(v) for k, v in splits.items()}},
                  f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(st, ensure_ascii=False, indent=2))
    print(f"\n完成：{n_total} 条问答 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
