#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看改写断点缓存：跑到一半就能知道改得好不好，不用等全部跑完。

行数只说明它在动，说明不了质量。这里直接统计采纳率和拒收原因 ——
采纳率塌了就该停下来调 prompt，而不是等七小时跑完再发现。

    python scripts/inspect_rewrite_cache.py --cache data/vqa/rewrite_cache.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/vqa/rewrite_cache.jsonl")
    ap.add_argument("--show", type=int, default=5, help="抽几条改写结果看看")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        print(f"找不到 {args.cache}")
        return 1

    n, ok, bad = 0, 0, collections.Counter()
    by_model = collections.Counter()
    samples, lens = [], []
    broken = 0
    for line in open(args.cache, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            broken += 1          # 还在写的最后一行
            continue
        n += 1
        if d.get("a"):
            ok += 1
            by_model[d.get("m") or "?"] += 1
            lens.append(len(d["a"]))
            if len(samples) < args.show:
                samples.append(d["a"])
        else:
            bad[d.get("why") or "no_response"] += 1

    if not n:
        print("缓存里还没有完整的行，再等等")
        return 0

    rate = ok / n
    print(f"已完成 {n} 条{'（末尾 1 行还在写）' if broken else ''}")
    print(f"采纳 {ok}（{rate:.1%}）  回退 {n - ok}")
    if bad:
        print(f"回退原因：{dict(bad)}")
        print("  fact_drift = 改写动了数字/JSON，事实校验拦下并回退到模板答案")
        print("  qc_fail    = 改写后过不了质检")
    if by_model:
        print(f"各副本采纳数：{dict(by_model)}")
    if lens:
        lens.sort()
        print(f"改写后长度：中位数 {lens[len(lens) // 2]}，"
              f"最短 {lens[0]}，最长 {lens[-1]}")

    print()
    if rate >= 0.8:
        print("采纳率正常，继续跑。")
    elif rate >= 0.6:
        print("⚠ 采纳率偏低。看下面的样例，多半是 prompt 要调；"
              "现在停掉调完再跑，已完成的不会重来。")
    else:
        print("⚠⚠ 采纳率过低，建议停掉（Ctrl-C 或 pkill -f build_vqa）。"
              "断点缓存还在，调完 vqa/llm.py 的 REWRITE_SYSTEM 接着跑。")

    if samples:
        print("\n改写结果抽样：")
        for s in samples:
            print(f"  · {s[:100]}")
        print("\n（缓存只存改写后的文本；原文对照要等跑完看 "
              "train.sharegpt.jsonl 里的 answer_template 字段）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
