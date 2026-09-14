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


def stats(records: list) -> dict:
    return {
        "total": len(records),
        "by_family": dict(Counter(r["family"] for r in records)),
        "by_task": dict(Counter(r["task"] for r in records).most_common()),
        "by_dataset": dict(Counter(r["dataset"] for r in records).most_common()),
        "by_label": dict(Counter(r["label"] for r in records)),
        "by_defect_type": dict(Counter(
            t for r in records for t in (r.get("defect_types") or [])).most_common()),
        "n_images": len({r["image"] for r in records}),
        "commercial_ok": dict(Counter(str(r.get("commercial_ok")) for r in records)),
    }
