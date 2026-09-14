#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽检工具：把 VQA 条目连同答案里的框画出来，人工过一眼。

构建完**一定要看几十张**，模板对不对、框准不准，肉眼两分钟就能发现。

    python scripts/visualize.py --vqa data/vqa/train.raw.jsonl -n 12 --out /tmp/vis
    python scripts/visualize.py --vqa data/vqa/train.raw.jsonl --task grounding_all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import _bootstrap  # noqa: F401
from PIL import Image, ImageDraw

COLORS = [(255, 64, 64), (64, 200, 64), (64, 160, 255), (255, 176, 32),
          (200, 64, 255), (32, 220, 220)]


def parse_boxes(ans: str):
    m = re.search(r"\[.*\]", ans, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [b for b in data if isinstance(b, dict) and "bbox_2d" in b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa", default="data/vqa/train.raw.jsonl")
    ap.add_argument("--out", default="data/vis")
    ap.add_argument("-n", "--num", type=int, default=12)
    ap.add_argument("--task", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.vqa, encoding="utf-8") if l.strip()]
    if args.task:
        recs = [r for r in recs if r["task"] == args.task]
    if args.dataset:
        recs = [r for r in recs if r["dataset"] == args.dataset]
    if not recs:
        print("没有匹配的条目")
        return 1
    random.Random(args.seed).shuffle(recs)
    recs = recs[:args.num]

    os.makedirs(args.out, exist_ok=True)
    index = []
    for i, r in enumerate(recs):
        if not os.path.exists(r["image"]):
            print(f"[miss] {r['image']}")
            continue
        im = Image.open(r["image"]).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for j, b in enumerate(parse_boxes(r["answer"])):
            x1, y1, x2, y2 = b["bbox_2d"]
            if r.get("coord_mode", "norm1000") == "norm1000":
                x1, x2 = x1 / 1000 * W, x2 / 1000 * W
                y1, y2 = y1 / 1000 * H, y2 / 1000 * H
            c = COLORS[j % len(COLORS)]
            d.rectangle([x1, y1, x2, y2], outline=c, width=3)
            d.text((x1 + 3, max(0, y1 - 12)), str(b.get("label", "")), fill=c)
        fn = f"{i:03d}_{r['task']}.jpg"
        im.save(os.path.join(args.out, fn), quality=90)
        index.append({"file": fn, "task": r["task"], "dataset": r["dataset"],
                      "question": r["question"], "answer": r["answer"]})
        print(f"[{i:03d}] {r['task']:20s} {r['question'][:46]}")

    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n{len(index)} 张 -> {args.out}（问答文本见 index.json）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
