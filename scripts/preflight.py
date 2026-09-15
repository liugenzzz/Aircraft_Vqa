#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据体检：在正式构建前，先看清每个源到底被读成了什么。

adapter 读不到东西、读错类型、mask 没对上，这些问题在 ingest 的一行汇总里
看不出来。这个脚本对每个数据源抽样，把**实际读到的结构**摊开：
样本数、正负比、缺陷类型分布、有多少框、mask 覆盖率、分辨率范围、
以及类型映射到 other_anomaly 的原始类别名（这些是本体里缺别名的）。

    python scripts/preflight.py --data-root ~/data/raw
    python scripts/preflight.py --data-root ~/data/raw --only mvtec_ad_screw
    python scripts/preflight.py --data-root ~/data/raw --sample 300
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401
import yaml

from aircraft_vqa.adapters import build_adapter
from aircraft_vqa.config import load_dataset_configs
from aircraft_vqa.taxonomy import get_taxonomy


# 本来就没有细粒度语义的泛称，落到 other_anomaly 是正确行为
GENERIC_RAW = {"anomaly", "defect", "damage", "bad", "abnormal", "ng",
               "combined", "misc", "unknown"}


def resolve(spec: dict, data_root: str) -> dict:
    spec = dict(spec)
    expand = lambda p: os.path.expanduser(str(p).replace("{data_root}", data_root))
    cands = [spec["root"]] + list(spec.pop("root_alternatives", []) or [])
    resolved = [expand(c) for c in cands]
    spec["root"] = next((r for r in resolved if os.path.exists(r)), resolved[0])
    spec["_tried"] = resolved
    return spec


def inspect(name: str, spec: dict, tax, limit: int, seed: int = 0) -> dict:
    """蓄水池抽样：顺序取前 N 条会严重误导 —— 很多数据集把训练集（全正常）
    排在前面，前 500 条会显示"异常率 0%"。"""
    ad = build_adapter({k: v for k, v in spec.items() if not k.startswith("_")},
                       taxonomy=tax)
    rng = random.Random(seed)
    pool, total = [], 0
    for s in ad.iter_samples():
        total += 1
        if len(pool) < limit:
            pool.append(s)
        else:
            j = rng.randrange(total)
            if j < limit:
                pool[j] = s

    rep = {"n": 0, "total": total, "normal": 0, "anomalous": 0, "with_box": 0,
           "n_boxes": 0, "types": Counter(), "raw_unmapped": Counter(),
           "cats": Counter(), "sizes": [], "grades": Counter(),
           "objects": Counter()}
    for s in pool:
        rep["n"] += 1
        rep["normal" if not s.is_anomalous else "anomalous"] += 1
        rep["cats"][s.category] += 1
        rep["objects"][s.object_name] += 1
        rep["sizes"].append((s.width, s.height))
        loc = s.localizable_defects()
        if loc:
            rep["with_box"] += 1
            rep["n_boxes"] += len(loc)
        for d in s.defects:
            rep["types"][d.type] += 1
            if d.grade:
                rep["grades"][d.grade] += 1
            # 映射不上的原始类别名 —— 这些该在 taxonomy 的 aliases 里补。
            # anomaly/defect/damage 这类泛称本来就没有细粒度语义，
            # 落到 other_anomaly 是正确行为，不算问题。
            if (d.type == "other_anomaly" and d.type_raw
                    and d.type_raw.lower() not in GENERIC_RAW):
                rep["raw_unmapped"][d.type_raw] += 1
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--data-root", default="~/data/raw")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=500,
                    help="每个源最多抽查多少条（0 = 全量）")
    ap.add_argument("--all", action="store_true",
                    help="连 enabled: false 的源也一起查")
    args = ap.parse_args()

    tax = get_taxonomy(os.path.abspath(args.taxonomy))
    specs, _ = load_dataset_configs(args.config or ["configs/datasets.yaml"])
    data_root = os.path.expanduser(args.data_root)
    limit = args.sample or 10 ** 9

    problems, ok_n = [], 0
    for spec in specs:
        name = spec["name"]
        if args.only and name not in args.only:
            continue
        if not args.only and not args.all and not spec.get("enabled", False):
            continue
        spec = resolve(spec, data_root)
        print(f"\n{'=' * 68}\n{name}   [{spec['adapter']}]  {spec['root']}")
        if not os.path.exists(spec["root"]):
            print(f"  ✗ 目录不存在。试过：{spec['_tried']}")
            problems.append(f"{name}: 目录不存在")
            continue
        try:
            rep = inspect(name, spec, tax, limit)
        except Exception as e:
            print(f"  ✗ adapter 报错：{type(e).__name__}: {e}")
            problems.append(f"{name}: adapter 报错 {e}")
            continue

        if rep["n"] == 0:
            print("  ✗ 读到 0 条样本 —— 目录结构或配置对不上")
            problems.append(f"{name}: 0 条样本")
            continue

        ok_n += 1
        n, a = rep["n"], rep["anomalous"]
        tag = f"（全库 {rep['total']}，随机抽样 {n}）" if rep["total"] > n else ""
        print(f"  样本 {rep['total']}{tag}  "
              f"正常 {rep['normal']} / 异常 {a}（异常率 {a / n:.1%}）")
        if a:
            cov = rep["with_box"] / a
            print(f"  异常图中带框的 {rep['with_box']} ({cov:.1%})，"
                  f"共 {rep['n_boxes']} 个框，平均 {rep['n_boxes'] / a:.2f} 个/图")
            if cov < 0.5:
                print("    ⚠ 过半异常图没有框 —— mask/标注可能没读到，"
                      "这类图只能用于有无判定，不能做定位")
                problems.append(f"{name}: 异常图带框率仅 {cov:.0%}")
        ws = [w for w, _ in rep["sizes"]]
        hs = [h for _, h in rep["sizes"]]
        print(f"  分辨率 W {min(ws)}~{max(ws)}  H {min(hs)}~{max(hs)}")
        print(f"  类别 {dict(rep['cats'].most_common(6))}")
        print(f"  对象 {dict(rep['objects'])}")
        print(f"  缺陷类型 {dict(rep['types'].most_common(10))}")
        if rep["grades"]:
            print(f"  有序等级 {dict(rep['grades'])}")
        if rep["raw_unmapped"]:
            print(f"  ⚠ 映射不到本体的原始类别名（建议补进 taxonomy 的 aliases）：")
            for k, v in rep["raw_unmapped"].most_common(10):
                print(f"      {k}  ×{v}")
            problems.append(f"{name}: {len(rep['raw_unmapped'])} 个类别名未映射")
        oa = rep["types"].get("other_anomaly", 0)
        tot = sum(rep["types"].values())
        if tot and oa / tot > 0.5:
            print(f"    ⚠ {oa / tot:.0%} 的缺陷落到 other_anomaly —— "
                  f"这个源基本只能做定位与有无判定，出不了类型题")

    print(f"\n{'=' * 68}\n可用数据源 {ok_n} 个")
    if problems:
        print(f"需要处理的问题 {len(problems)} 条：")
        for p in problems:
            print(f"  · {p}")
        return 1
    print("没有发现问题，可以跑 ingest 了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
