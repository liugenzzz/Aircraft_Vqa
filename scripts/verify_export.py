#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""出厂检查：直接验最终导出的 ShareGPT 文件，而不是构建过程中的统计。

stats 看的是"我们以为生成了什么"，这里看的是"文件里实际躺着什么"。
两者不一致过好几次了，所以训练前跑这一遍，红了就别开训。

    python scripts/verify_export.py --dir data/vqa
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import _bootstrap  # noqa: F401

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aircraft_vqa.vqa import templates as T  # noqa: E402

PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def check(path: str, sysmap: dict) -> tuple:
    """返回 (行数, 问题列表, 每个任务的图片集合)."""
    bad, n = [], 0
    imgs_by_sample = {}
    seen_fmt = collections.Counter()
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        tag = f"{os.path.basename(path)}:{ln}"
        try:
            r = json.loads(line)
        except Exception as e:
            bad.append(f"{tag} JSON 解析失败: {e}")
            continue

        conv = r.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            bad.append(f"{tag} conversations 结构不对")
            continue
        if conv[0].get("from") != "human" or conv[-1].get("from") != "gpt":
            bad.append(f"{tag} 首尾角色不对：{[c.get('from') for c in conv]}")
        for i, c in enumerate(conv):
            want = "human" if i % 2 == 0 else "gpt"
            if c.get("from") != want:
                bad.append(f"{tag} 第 {i} 轮角色应为 {want}")
                break

        imgs = r.get("images") or []
        n_tok = sum(c.get("value", "").count("<image>") for c in conv)
        if n_tok != len(imgs):
            bad.append(f"{tag} <image> 数 {n_tok} != images 数 {len(imgs)}")
        for p in imgs:
            if not os.path.exists(p):
                bad.append(f"{tag} 图片不存在：{p}")
                break

        sysp = r.get("system", "")
        if not sysp:
            bad.append(f"{tag} 缺 system")
        elif sysp not in sysmap:
            bad.append(f"{tag} system 不在已知的四种里")

        for c in conv:
            v = c.get("value", "")
            m = PLACEHOLDER.search(v)
            if m:
                bad.append(f"{tag} 有没填的占位符 {m.group(0)}：{v[:60]}")
                break

        ans = conv[-1].get("value", "")
        if sysp == T.SYSTEM_JSON_ONLY:
            seen_fmt["json_only"] += 1
            try:
                arr = json.loads(ans)
            except Exception:
                bad.append(f"{tag} json_only 的答案不是合法 JSON：{ans[:60]}")
            else:
                if not isinstance(arr, list):
                    bad.append(f"{tag} json_only 的答案不是数组")
                for box in arr if isinstance(arr, list) else []:
                    b = (box or {}).get("bbox_2d")
                    if (not isinstance(b, list) or len(b) != 4
                            or not all(isinstance(x, int) for x in b)):
                        bad.append(f"{tag} bbox_2d 格式不对：{b}")
                        break
                    if not all(0 <= x <= 1000 for x in b):
                        bad.append(f"{tag} bbox_2d 超出 0-1000：{b}")
                        break
                    if b[0] >= b[2] or b[1] >= b[3]:
                        bad.append(f"{tag} bbox_2d 左上角不在右下角左上方：{b}")
                        break
        if not ans.strip() and sysp != T.SYSTEM_JSON_ONLY:
            bad.append(f"{tag} 答案为空")

        # 图片按行收，不依赖 meta.sample_id —— 导出时关掉 with_meta 的话
        # 这里会一条都收不到，划分泄漏检查就静默失效，而"没报错"看起来
        # 和"没问题"一模一样。
        imgs_by_sample.setdefault(
            (r.get("meta") or {}).get("sample_id") or f"line{ln}",
            set()).update(imgs)
    return n, bad, imgs_by_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/vqa")
    ap.add_argument("--max-show", type=int, default=12)
    args = ap.parse_args()

    sysmap = {v: k for k, v in T.SYSTEM_BY_OUTPUT.items()}
    splits, all_bad = {}, []
    for name in ("train", "val", "test"):
        p = os.path.join(args.dir, f"{name}.sharegpt.jsonl")
        if not os.path.exists(p):
            print(f"  ✗ 缺少 {p}")
            all_bad.append(f"缺少 {p}")
            continue
        n, bad, by_sample = check(p, sysmap)
        splits[name] = (n, by_sample)
        print(f"  {name:6s} {n:>7} 条   问题 {len(bad)}")
        all_bad += bad

    # 划分之间不能共用图片，否则验证集是漏的
    imgsets = {}
    for name, (_, by_sample) in splits.items():
        s = set()
        for v in by_sample.values():
            s |= v
        imgsets[name] = s
    if not any(imgsets.values()):
        all_bad.append("一张图片路径都没读到，划分泄漏检查没能真正执行")
    names = list(imgsets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = imgsets[names[i]] & imgsets[names[j]]
            if common:
                all_bad.append(
                    f"{names[i]} 与 {names[j]} 共用 {len(common)} 张图 —— "
                    f"验证集是漏的，例：{list(common)[:2]}")

    print()
    if all_bad:
        print(f"发现 {len(all_bad)} 个问题：")
        for b in all_bad[:args.max_show]:
            print(f"  · {b}")
        if len(all_bad) > args.max_show:
            print(f"  …… 还有 {len(all_bad) - args.max_show} 个")
        return 1
    total = sum(n for n, _ in splits.values())
    print(f"出厂检查通过：{total} 条，结构 / 占位符 / 坐标 / system / "
          "划分隔离全部正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
