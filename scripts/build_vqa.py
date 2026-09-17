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

# 重定向到文件时 Python 默认块缓冲，要攒够几 KB 才落盘 —— 长任务
# nohup 出去，日志会空好几分钟，看起来像没在跑。这里强制行缓冲，
# 不依赖调用方记得加 -u。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)
    except Exception:
        pass
import yaml

from aircraft_vqa.balance import diversity as st_mod_diversity
from aircraft_vqa.balance import (balance_samples, cap_class_imbalance,
                                  cap_dataset_share,
                                  class_distribution, dedup, group_split,
                                  quota_sample, ratio_gap, stats)
from aircraft_vqa.export import export_records
from aircraft_vqa.qc import run_qc
from aircraft_vqa.schema import dump_records, read_jsonl
from aircraft_vqa.taxonomy import get_taxonomy
from aircraft_vqa.vqa import BuildConfig, VQABuilder
from aircraft_vqa.vqa.llm import load_rewriter


def apply_overrides(cfg: dict, pairs) -> None:
    """--set a.b=1 形式的临时覆盖。

    调配比、改平衡系数是高频操作，每次都去改 yaml 容易忘了改回来，
    也不好在脚本里批量跑对比实验。值按 JSON 解析，解析不了就当字符串。
    """
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set 要写成 KEY=VALUE，收到：{item}")
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except Exception:
            val = raw
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
        print(f"[override] {key} = {val!r}")


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
    ap.add_argument("--llm-cache", default="",
                    help="改写断点缓存路径，默认 <out>/rewrite_cache.jsonl。"
                         "五万条要跑七到十小时，崩了重跑同一条命令即可续上")
    ap.add_argument("--llm-workers", type=int, default=0,
                    help="改写并发路数，默认按池里各模型的 concurrency 之和")
    ap.add_argument("--mix-general", default=None,
                    help="通用指令数据 jsonl（已是导出格式），按比例混进 train，"
                         "防止窄领域微调把模型的语言层带偏")
    ap.add_argument("--mix-ratio", type=float, default=0.08,
                    help="通用数据在 train 中的占比，建议 0.05~0.10")
    ap.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    help="临时覆盖 build.yaml 里的任意配置，可给多次。"
                         "支持点号路径与 JSON 值，例如："
                         "--set normal_per_anomalous=1.5 "
                         "--set task_ratio.grounding_single=0.2 "
                         "--set active_defect_types='[\"crack\",\"corrosion\"]'")
    ap.add_argument("--ratio-strict", action="store_true",
                    help="严格服从 task_ratio（会按最稀缺任务缩量，数据量小很多）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    apply_overrides(cfg, args.set)
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

    # ---- 数据源限流（在正负平衡之前）----
    max_share = cfg.get("max_share_per_dataset", 0) or 0
    ds_weights = cfg.get("dataset_weights") or None
    if max_share or ds_weights:
        samples, rep = cap_dataset_share(samples, max_share, ds_weights,
                                         cfg.get("seed", 0))
        # rep 里除了各源的计数，还可能有 "_note" 这种说明项（字符串）。
        # 之前直接当 dict 取，源数少于 4 个时 25% 上限无解、note 一写就崩。
        note = rep.get("_note")
        counts = {k: v for k, v in rep.items() if isinstance(v, dict)}
        tot = sum(r["after"] for r in counts.values()) or 1
        print(f"[source] 限流后 {tot} 条"
              + (f"（单源上限 {max_share:.0%}）" if max_share else ""))
        if note:
            print(f"         注：{note}")
        for ds, r in sorted(counts.items(), key=lambda kv: -kv[1]["after"]):
            mark = "  <- 限流" if r["after"] < r["before"] else (
                "  <- 过采样" if r["after"] > r["before"] else "")
            print(f"         {ds:24s} {r['before']:>6d} -> "
                  f"{r['after']:>6d}  {r['after'] / tot:>6.1%}{mark}")

    # ---- 样本层平衡 ----
    before = Counter(s.label for s in samples)
    samples = balance_samples(samples, cfg.get("normal_per_anomalous", 1.0),
                              cfg.get("seed", 0))
    after = Counter(s.label for s in samples)
    print(f"[balance] {dict(before)} -> {dict(after)}")
    # 限流可能把某一侧削掉太多，导致目标正负比根本达不到。
    # "有无判定"的指标是在这个先验下测的，达不到必须说出来，不能默默跑过去。
    want = cfg.get("normal_per_anomalous", 1.0)
    n_a = after.get("anomalous", 0)
    if want > 0 and n_a:
        got = after.get("normal", 0) / n_a
        if abs(got - want) > 0.1 * want:
            print(f"  ⚠ 实际正/异 = {got:.2f}:1，没达到设定的 {want:.2f}:1"
                  f"（正常图只剩 {after.get('normal', 0)} 张，不够配 {n_a} 张异常图）。"
                  "\n    '有无判定'的指标是在这个先验下测的，换了先验就不可比。"
                  "\n    要么调低 max_share_per_dataset 的力度，"
                  "要么把 normal_per_anomalous 调成实际值。")

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
    _T = __import__("aircraft_vqa.vqa.templates", fromlist=["x"])
    _bank = _T.load_question_bank(bcfg.question_bank or "")
    n_bank = sum(len(v) for v in _bank.values())
    if n_bank:
        thin = [t for t, v in _bank.items() if len(v) < 10]
        print(f"[qbank] 扩写问法 {n_bank} 条，覆盖 {len(_bank)} 个任务"
              + (f"；其中 {len(thin)} 个任务不足 10 条：{thin}" if thin else ""))
    else:
        # "0 条"有好几种原因，混为一谈会让人往错的方向查。
        # 真实踩到：补跑单个任务时整份覆盖了问法库，剩下两个空列表。
        path = bcfg.question_bank or ""
        if not path:
            why = "configs/build.yaml 里没配 question_bank"
        elif not os.path.exists(path):
            why = f"{path} 不存在"
        elif _bank:
            why = (f"{path} 里 {len(_bank)} 个任务全是空列表 —— "
                   "多半是只跑了部分任务把整份库覆盖了，看看有没有 "
                   f"{path}.bak 可以还原")
        else:
            why = f"{path} 里没有 questions 字段或为空"
        print(f"[qbank] 扩写问法 0 条，仅用种子问法 —— {why}")
        print("        生成：python scripts/gen_question_bank.py --per-task 40")
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
    TYPE_TASKS = list(cls_tasks)
    cls_before = cls_after = cls_dropped = 0
    if mom > 0:
        before = class_distribution(records, cls_tasks)
        records = cap_class_imbalance(
            records, cls_tasks, mom, seed=cfg.get("seed", 0),
            max_oversample=float(cfg.get("class_max_oversample", 3.0)))
        after = class_distribution(records, cls_tasks)
        cls_before = sum(before.values())
        cls_after = sum(after.values())
        n_os = sum(1 for r in records if r.get("oversampled"))
        cls_dropped = max(0, cls_before - (cls_after - n_os))
        if before != after:
            print(f"[class] 类型识别任务类别均衡"
                  f"（头部 ≤ {mom}× 中位数，长尾重复采样顶到 中位数/{mom}）")
            print(f"        前 {before}")
            print(f"        后 {after}")
            ratio_b = (max(before.values()) / min(before.values())
                       if before and min(before.values()) else 0)
            ratio_a = (max(after.values()) / min(after.values())
                       if after and min(after.values()) else 0)
            print(f"        最多/最少 {ratio_b:.1f}× -> {ratio_a:.1f}×；"
                  f"真实数据保留 {1 - cls_dropped / max(1, cls_before):.0%}，"
                  f"重复采样 {n_os} 条")

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
        # 别一律甩给源数据：类别均衡那一步自己也会砍掉大量类型题，
        # 分不清是"源数据没有"还是"我们自己扔了"，就会去补错地方。
        by_cap = {g["task"] for g in short} & set(TYPE_TASKS)
        print("[ratio] 以下任务未达目标配比：")
        for g in short:
            why = "  <- 类别均衡砍掉的" if (g["task"] in by_cap and cls_dropped) else ""
            print(f"         {g['task']:22s} 实际 {g['actual']:.1%} / "
                  f"目标 {g['target']:.1%}（达成 {g['attain']:.0%}，{g['n']} 条）{why}")
        if by_cap and cls_dropped:
            print(f"         类别均衡这一步把类型题从 {cls_before} 条压到 "
                  f"{cls_after} 条（丢 {cls_dropped / cls_before:.0%}）——"
                  "先看是不是 class_max_over_min 收得太紧，再去补数据。")
        print("         源数据侧的补法见 docs/01_dataset_survey.md：类型/严重度类"
              "问题需要带细粒度缺陷类别的源"
              "（MVTec AD screw、Roboflow aircraft_skin_defects 等）。")

    # ---- 可选：大模型改写 ----
    if args.llm_rewrite:
        rw = load_rewriter(cfg.get("llm", {}))
        if rw.enabled:
            cache = args.llm_cache or os.path.join(args.out, "rewrite_cache.jsonl")
            os.makedirs(args.out, exist_ok=True)
            print(f"[llm] 断点缓存 -> {cache}（崩了重跑同一条命令即可续上）")
            st = rw.rewrite_many(records, workers=args.llm_workers,
                                 cache_path=cache)
            if st.get("n_cache_hit"):
                print(f"       本次命中缓存 {st['n_cache_hit']} 条")
            print(f"[llm] {st['workers']} 路并发，送出 {st['n_sent']} 条"
                  f"（跳过受保护任务 {st['n_skipped_protected']} 条），"
                  f"改写成功 {st['n_rewritten']}，用时 {st['seconds'] / 60:.1f} 分钟")
            if st["rejected"]:
                # 被拒的说明改写动了事实或过不了质检，回退到模板答案。
                # 这个比例高就说明 prompt 或模型不合适，别闷头跑完。
                print(f"       回退 {sum(st['rejected'].values())} 条："
                      f"{st['rejected']}")
        else:
            print("[llm] provider=none，跳过改写")

    # ---- 切分 + 导出 ----
    splits = group_split(records, tuple(cfg.get("split_ratios", [0.95, 0.03, 0.02])),
                         cfg.get("seed", 0))
    # 划分是按"图片的连通分量"整组切的（pair_compare 会把两张图连起来），
    # 组粒度比单个样本粗，小数据集上 val/test 可能被整组甩空。
    # 空的验证集看起来和"训练很顺"一模一样，必须说出来。
    for name in ("val", "test"):
        n = len(splits.get(name) or [])
        if n == 0:
            print(f"  ⚠ {name} 划分是空的 —— 样本太少，整组都落进了 train。"
                  "这样评估无从谈起，加大数据量或调 split_ratios。")
        elif n < 50:
            print(f"  ⚠ {name} 划分只有 {n} 条，样本量不足以说明问题。")
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

    # system 随数据集一起落盘：训练时它进 prompt，**推理时必须用同一套**，
    # 否则模型的行为会漂（尤其是"只输出 JSON、不附加解释"这条契约）。
    from aircraft_vqa.vqa.templates import OUTPUT_FORMAT, SYSTEM_BY_OUTPUT
    with open(os.path.join(args.out, "system_prompts.json"), "w",
              encoding="utf-8") as f:
        json.dump({"note": "训练用的 system。推理时按任务的 output_format "
                           "取同一条，不要换措辞。",
                   "by_output_format": SYSTEM_BY_OUTPUT,
                   "task_to_output_format": OUTPUT_FORMAT},
                  f, ensure_ascii=False, indent=2)
    print(f"[export] system 提示词 -> {args.out}/system_prompts.json")

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
