# -*- coding: utf-8 -*-
"""配比与切分。

两层平衡：
1. 样本层：控制正常图 / 异常图的比例（公开 IAD 数据集普遍 9:1 偏正常，
   直接全用会把模型训成"什么都说没问题"）。
2. 问答层：按目标任务配比抽样，保证定位/识别两族以及各子任务不失衡。

切分按 sample_id 分组做，保证同一张图的所有问答只出现在同一个 split，
避免验证集泄漏。
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Optional


def balance_samples(samples: list, normal_per_anomalous: float = 1.0,
                    seed: int = 0) -> list:
    """按比例下采样正常样本。normal_per_anomalous <= 0 表示不限制。"""
    if normal_per_anomalous <= 0:
        return samples
    rng = random.Random(seed)
    anom = [s for s in samples if s.is_anomalous]
    norm = [s for s in samples if not s.is_anomalous]
    keep = int(len(anom) * normal_per_anomalous)
    if len(norm) > keep:
        rng.shuffle(norm)
        norm = norm[:keep]
    out = anom + norm
    rng.shuffle(out)
    return out


def auto_total(by_task: dict, ratio: dict, quantile: float = 0.6) -> int:
    """不给 total 时，推一个"既不浪费又不太失衡"的总量。

    取 have/ratio 的加权分位数而不是最小值：用最小值会被供给最少的那个任务
    卡死（实测会白白丢掉 80%+ 的数据），而超出供给的任务会由 quota_sample
    的重分配逻辑兜住，只是占比略微上浮。
    """
    caps = sorted((len(by_task[k]) / r, r) for k, r in ratio.items() if r > 0)
    if not caps:
        return 0
    acc, half = 0.0, sum(r for _, r in caps) * quantile
    for cap, r in caps:
        acc += r
        if acc >= half:
            return max(1, int(cap))
    return max(1, int(caps[-1][0]))


def quota_sample(records: list, task_ratio: dict, total: Optional[int] = None,
                 seed: int = 0) -> list:
    """按 task_ratio 给定的目标占比抽样。

    某个任务供给不足时，把缺口按剩余任务的相对权重重新分配，
    所以最终总量会尽量贴近 total，而不是直接缩水。
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    for v in by_task.values():
        rng.shuffle(v)

    ratio = {k: v for k, v in task_ratio.items() if v > 0 and k in by_task}
    if not ratio:
        return records
    tot_w = sum(ratio.values())
    ratio = {k: v / tot_w for k, v in ratio.items()}

    if total is None:
        total = auto_total(by_task, ratio)

    out, remaining, pending = [], total, dict(ratio)
    for _ in range(4):                       # 最多重分配 4 轮
        if remaining <= 0 or not pending:
            break
        w = sum(pending.values())
        short = {}
        for k, r in list(pending.items()):
            want = int(round(remaining * r / w))
            have = len(by_task[k])
            take = min(want, have)
            out.extend(by_task[k][:take])
            by_task[k] = by_task[k][take:]
            if take < want:
                short[k] = True
        remaining = total - len(out)
        pending = {k: v for k, v in pending.items() if k not in short and by_task[k]}
    rng.shuffle(out)
    return out


def ratio_gap(records: list, task_ratio: dict) -> list:
    """对比实际占比与目标占比，返回偏离明细（按缺口从大到小）。

    偏离往往不是 bug 而是数据供给不足的信号 —— 比如源数据集只有
    "有无异常"两级标签时，"缺陷类型"类问题天然造不出来。
    """
    if not records:
        return []
    tot = len(records)
    actual = Counter(r["task"] for r in records)
    w = sum(v for v in task_ratio.values() if v > 0) or 1.0
    rows = []
    for task, v in task_ratio.items():
        if v <= 0:
            continue
        target = v / w
        got = actual.get(task, 0) / tot
        rows.append({"task": task, "n": actual.get(task, 0),
                     "actual": round(got, 4), "target": round(target, 4),
                     "attain": round(got / target, 3) if target else 0.0})
    return sorted(rows, key=lambda r: r["attain"])


def cap_class_imbalance(records: list, tasks, max_over_min: float = 3.0,
                        key: str = "target_type", seed: int = 0) -> list:
    """压平"类型识别"类任务的类别长尾。

    `异常类别识别准确率` 这类指标通常按**宏平均**算（每类等权），
    类别样本量差 4~5 倍时，样本最少的那几类学不动，宏平均会被直接拖死。
    这里把最多的类下采样到 max_over_min × 最少类，其他任务不受影响。
    """
    rng = random.Random(seed)
    tasks = set(tasks)
    target, other = [], []
    for r in records:
        (target if r.get("task") in tasks and r.get(key) else other).append(r)
    if not target:
        return records

    by_cls = defaultdict(list)
    for r in target:
        by_cls[r[key]].append(r)
    n_min = min(len(v) for v in by_cls.values())
    cap = max(1, int(n_min * max_over_min))

    kept = []
    for cls, rows in by_cls.items():
        if len(rows) > cap:
            rng.shuffle(rows)
            rows = rows[:cap]
        kept.extend(rows)
    rng.shuffle(kept)
    return other + kept


def class_distribution(records: list, tasks, key: str = "target_type") -> dict:
    tasks = set(tasks)
    return dict(Counter(r[key] for r in records
                        if r.get("task") in tasks and r.get(key)).most_common())


def dedup(records: list) -> list:
    """按 (image, question, answer) 去重 —— 模板随机可能撞车。"""
    seen, out = set(), []
    for r in records:
        key = hashlib.md5(
            f"{r['image']}|{r['question']}|{r['answer']}".encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def group_split(records: list, ratios=(0.95, 0.03, 0.02), seed: int = 0) -> dict:
    """按 sample_id 哈希分组切分，同一张图不跨 split。"""
    tr, va, te = ratios
    out = {"train": [], "val": [], "test": []}
    for r in records:
        h = int(hashlib.md5(f"{seed}:{r['sample_id']}".encode()).hexdigest()[:8], 16)
        p = (h % 10000) / 10000.0
        if p < tr:
            out["train"].append(r)
        elif p < tr + va:
            out["val"].append(r)
        else:
            out["test"].append(r)
    return out


def diversity(records: list, top_k: int = 20) -> dict:
    """多样性体检：问法够不够花、答案是不是大段复读。

    模板法最容易出的问题是"换汤不换药" —— 统计上看着量很大，
    实际几十个模板复读几万遍。distinct-2 和高频答案占比能一眼看出来。
    """
    import re as _re
    by_task_q = defaultdict(set)
    q_all, a_all = [], []
    for r in records:
        by_task_q[r["task"]].add(r["question"])
        q_all.append(r["question"])
        a_all.append(r["answer"])

    def distinct_n(texts: list, n: int = 2) -> float:
        grams, total = set(), 0
        for t in texts:
            toks = _re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9_]+", t)
            for i in range(len(toks) - n + 1):
                grams.add(tuple(toks[i:i + n]))
                total += 1
        return round(len(grams) / total, 4) if total else 0.0

    ans_counter = Counter(a_all)
    top = ans_counter.most_common(top_k)
    # "[]" 是负样本的**正确答案**，不是模板复读，单独算一档才看得清真实复读率
    free = [a for a in a_all if a.strip() != "[]"]
    free_counter = Counter(free)
    free_top = free_counter.most_common(top_k)
    return {
        "n_empty_answer": len(a_all) - len(free),
        "unique_questions_per_task": {k: len(v) for k, v in
                                      sorted(by_task_q.items())},
        "distinct_2_question": distinct_n(q_all, 2),
        "distinct_2_answer": distinct_n(a_all, 2),
        "unique_answer_ratio": round(len(ans_counter) / len(a_all), 4)
        if a_all else 0.0,
        f"top{top_k}_answer_share": round(
            sum(c for _, c in top) / len(a_all), 4) if a_all else 0.0,
        # 真正该看的是这个：排除 [] 之后，高频答案还占多少
        f"top{top_k}_share_excl_empty": round(
            sum(c for _, c in free_top) / len(free), 4) if free else 0.0,
        "unique_answer_ratio_excl_empty": round(
            len(free_counter) / len(free), 4) if free else 0.0,
        "top5_answers": [{"n": c, "text": t[:70]} for t, c in free_top[:5]],
    }


def negative_breakdown(records: list) -> dict:
    """两类负样本要分开统计 —— 它们考的是不同能力。

    "这图没问题"（正常图）和"这图有问题但不是你问的那个"（反事实）
    混在一起看占比，会掩盖其中一类不足。
    """
    empty = [r for r in records if r.get("n_boxes") == 0
             or r.get("count") == 0]
    normal_img = [r for r in empty if r.get("image_status") == "normal"]
    counterfactual = [r for r in empty if r.get("image_status") == "anomalous"]
    n = len(records) or 1
    return {
        "total_negative": len(empty),
        "negative_share": round(len(empty) / n, 4),
        "on_normal_image": len(normal_img),
        "on_normal_share": round(len(normal_img) / n, 4),
        "counterfactual": len(counterfactual),
        "counterfactual_share": round(len(counterfactual) / n, 4),
    }


def bbox_edge_stats(records: list) -> dict:
    """框贴边率。大量 0 / 1000 说明合成时缺陷被贴到了图像边缘。"""
    import json as _json
    import re as _re
    edge = total = 0
    for r in records:
        m = _re.search(r"\[.*\]", r.get("answer", ""), _re.S)
        if not m:
            continue
        try:
            items = _json.loads(m.group(0))
        except Exception:
            continue
        hi = 1000 if r.get("coord_mode", "norm1000") == "norm1000" else None
        for it in items:
            if not isinstance(it, dict) or "bbox_2d" not in it:
                continue
            c = it["bbox_2d"]
            total += 1
            if hi and (min(c) <= 1 or max(c) >= hi - 1):
                edge += 1
    return {"n_boxes": total, "touching_edge": edge,
            "edge_ratio": round(edge / total, 4) if total else 0.0}


def stats(records: list) -> dict:
    return {
        "total": len(records),
        "by_family": dict(Counter(r["family"] for r in records)),
        "by_task": dict(Counter(r["task"] for r in records).most_common()),
        "by_dataset": dict(Counter(r["dataset"] for r in records).most_common()),
        "by_image_status": dict(Counter(r.get("image_status") for r in records)),
        "by_image_defect_type": dict(Counter(
            t for r in records
            for t in (r.get("image_defect_types") or [])).most_common()),
        "by_asked_defect_type": dict(Counter(
            t for r in records
            for t in (r.get("asked_defect_types") or [])).most_common()),
        "n_images": len({r["image"] for r in records}),
        "commercial_ok": dict(Counter(str(r.get("commercial_ok")) for r in records)),
        "by_output_format": dict(Counter(r.get("output_format", "?")
                                         for r in records)),
        "diversity": diversity(records),
        "negatives": negative_breakdown(records),
        "bbox_edge": bbox_edge_stats(records),
    }
