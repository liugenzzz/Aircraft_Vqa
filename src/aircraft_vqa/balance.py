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


def cap_dataset_share(samples: list, max_share: float = 0.0,
                      weights: Optional[dict] = None, seed: int = 0) -> tuple:
    """限制单个数据源占样本总量的比例，避免某一家把域先验带偏。

    真实情形：Real-IAD 两万条占了全库 53%，而它拍的是灯箱里的端子排、
    电路板。指标要的是**飞机**蒙皮上的缺陷识别，让一半训练数据是工业
    小件，视觉先验会跟着跑偏。

    下采样时按 normal/anomalous 分层，保持该源自身的正负比例不变 ——
    整体下采样会把正负比也一起改掉，那是另一个旋钮该管的事。

    weights 给每个源一个乘数（>1 过采样，<1 下采样），在限流之前生效，
    用来把航空域的小样本源顶上来。

    返回 (新样本列表, 每个源的 before/after 计数)。
    """
    rng = random.Random(seed)
    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s.dataset].append(s)

    report = {k: {"before": len(v)} for k, v in by_ds.items()}

    # 1) 显式权重：过采样用重复，下采样用抽取
    if weights:
        for ds, w in weights.items():
            pool = by_ds.get(ds)
            if not pool or w is None or w == 1:
                continue
            if w <= 0:
                by_ds[ds] = []
                continue
            target = int(round(len(pool) * w))
            if target <= len(pool):
                picked = list(pool)
                rng.shuffle(picked)
                by_ds[ds] = picked[:target]
            else:
                extra = []
                while len(pool) + len(extra) < target:
                    extra.append(rng.choice(pool))
                by_ds[ds] = pool + extra

    # 2) 份额上限
    #
    # 用"注水"求一个统一的封顶值 L：每个源保留 min(自身条数, L)，
    # 使最大的那个占比不超过 max_share。
    #
    # 不能一轮轮地"按当前总量算目标再砍" —— 被砍的源自己也在分母里，
    # 砍完总量变小、份额又超标，于是越砍越少；只有两个源时一路收敛到 0。
    # 而且砍了 A 会改变 B 的分母，逐个处理必然互相算错。
    #
    # L/sum(min(s_j, L)) 关于 L 单调不减，二分即可。
    if max_share and 0 < max_share < 1:
        sizes = {ds: len(v) for ds, v in by_ds.items() if v}
        n = len(sizes)
        if n:
            # n 个源不可能人人都低于 1/n，低于这个值的上限无解，按均分处理
            share = max(max_share, 1.0 / n)
            if share > max_share + 1e-9:
                report["_note"] = (
                    f"max_share={max_share:.0%} 对 {n} 个源无解"
                    f"（至少 1/{n}={1.0 / n:.0%}），已按 {share:.0%} 处理")

            def ok(L: int) -> bool:
                tot = sum(min(v, L) for v in sizes.values())
                return tot > 0 and min(max(sizes.values()), L) <= tot * share

            lo, hi = 1, max(sizes.values())
            if ok(hi):
                lo = hi                      # 本来就没人超限
            else:
                while lo < hi:               # 找满足条件的最大 L
                    mid = (lo + hi + 1) // 2
                    if ok(mid):
                        lo = mid
                    else:
                        hi = mid - 1
            for ds, pool in list(by_ds.items()):
                if len(pool) <= lo:
                    continue
                anom = [x for x in pool if x.is_anomalous]
                norm = [x for x in pool if not x.is_anomalous]
                rng.shuffle(anom)
                rng.shuffle(norm)
                # 按原正负比例分层截断，别把正负比也一起改了
                r = len(anom) / len(pool)
                n_a = min(len(anom), int(round(lo * r)))
                by_ds[ds] = anom[:n_a] + norm[:max(0, lo - n_a)]

    out = []
    for ds, v in by_ds.items():
        report[ds]["after"] = len(v)
        out.extend(v)
    for k, r in report.items():
        if isinstance(r, dict):          # "_note" 之类的说明项不是计数
            r.setdefault("after", 0)
    rng.shuffle(out)
    return out, report


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
                        key: str = "target_type", seed: int = 0,
                        max_oversample: float = 3.0) -> list:
    """压平"类型识别"类任务的类别长尾。

    `异常类别识别准确率` 这类指标通常按**宏平均**算（每类等权），
    类别样本量差 4~5 倍时，样本最少的那几类学不动，宏平均会被直接拖死。
    做法是双向收拢：头部类下采样到 max_over_min × 中位数，长尾类重复采样
    顶到 中位数 / max_over_min（复制倍数不超过 max_oversample，
    重复同一条问答会助长记忆而不是泛化）。其他任务不受影响。
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
    counts = sorted(len(v) for v in by_cls.values())

    # 基准取**中位数**而不是最少类。
    #
    # 按最少类算是致命的：真实数据里 paint_peeling 只有 91 条，
    # cap = 91x3 = 273，于是 dent(1757) 和 crack(1662) 被砍到 273，
    # 整个类型识别任务 5470 -> 1852，白扔 66%。一个长尾类就能把所有类
    # 一起拖下水，而且扔掉的恰恰是标注最好的那些。
    #
    # 中位数为基准：长尾类保持原样（本来就不多，不会撑爆宏平均），
    # 头部类压到一个合理量级，比例仍然受控，但不会被最稀有的那类绑架。
    mid = counts[len(counts) // 2]
    cap = max(1, int(mid * max_over_min))
    floor = max(1, int(cap / max_over_min))

    kept = []
    for cls, rows in by_cls.items():
        if len(rows) > cap:
            rng.shuffle(rows)
            rows = rows[:cap]
        elif len(rows) < floor and max_oversample > 1:
            # 长尾类**顶上来**，而不是把头部砍下去。
            # 但重复同一条问答会助长记忆而不是泛化，所以复制倍数封顶。
            want = min(floor, int(len(rows) * max_oversample))
            extra = [dict(rows[rng.randrange(len(rows))]) for _ in
                     range(max(0, want - len(rows)))]
            for i, e in enumerate(extra):
                e["qa_id"] = f"{e.get('qa_id', '')}#os{i}"   # 去重时别被当成同一条
                e["oversampled"] = True
            rows = rows + extra
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
    """按**图片的连通分量**切分，同一张图绝不跨 split。

    只按 sample_id 哈希是不够的：pair_compare 会从参考图池里额外取一张
    正常件图，那张图属于另一个 sample。于是 A 的图在 train 里当主体，
    同时在 test 里给 B 当参考图 —— 模型训练时见过这张图，测试集就是漏的。
    实测确实发生了（train 与 test 共用 2 张图）。

    做法：把"同一条记录里出现的图片"并到一个组，整组一起分。
    """
    tr, va, te = ratios

    # 并查集：同一条记录里的图片必须同组
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def keys_of(r):
        """这条记录牵涉到的所有 key：主 sample_id + 它用到的每张图。"""
        ks = [f"sid:{r['sample_id']}"]
        ks += [f"img:{p}" for p in (r.get("images") or [])
               if isinstance(p, str)]
        if isinstance(r.get("image"), str):
            ks.append(f"img:{r['image']}")
        return ks

    for r in records:
        ks = keys_of(r)
        for k in ks[1:]:
            union(ks[0], k)

    out = {"train": [], "val": [], "test": []}
    for r in records:
        root = find(keys_of(r)[0])
        h = int(hashlib.md5(f"{seed}:{root}".encode()).hexdigest()[:8], 16)
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
