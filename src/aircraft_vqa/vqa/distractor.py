"""多选题干扰项生成。

好的干扰项要"像但不是"：优先同族缺陷（紧固件族内部互为干扰），
其次跨族，最后才用兜底项，保证题目有区分度而不是送分题。
"""
from __future__ import annotations

import random
from typing import Optional

from ..taxonomy import Taxonomy

LETTERS = "ABCDEF"


def make_options(tax: Taxonomy, correct_types: list, n_options: int = 4,
                 rng: Optional[random.Random] = None) -> tuple:
    """返回 (选项文本列表, 正确答案字母)。选项文本为中文缺陷名。"""
    rng = rng or random
    correct = correct_types[0]
    correct_zh = tax.zh(correct)
    exclude = set(correct_types)

    pool_near = [t for t in tax.siblings(correct) if t not in exclude]
    pool_far = [t for t in tax.all_types()
                if t not in exclude and t not in pool_near and t != "other_anomaly"]
    rng.shuffle(pool_near)
    rng.shuffle(pool_far)

    need = n_options - 1
    distract = (pool_near + pool_far)[:need]
    if len(distract) < need:                       # 极端情况兜底
        distract += ["other_anomaly"] * (need - len(distract))

    opts = [correct_zh] + [tax.zh(t) for t in distract]
    # 去重后可能不足，补齐
    seen, uniq = set(), []
    for o in opts:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    for t in pool_far:
        if len(uniq) >= n_options:
            break
        if tax.zh(t) not in seen:
            seen.add(tax.zh(t))
            uniq.append(tax.zh(t))
    opts = uniq[:n_options]

    rng.shuffle(opts)
    idx = opts.index(correct_zh)
    return opts, LETTERS[idx]


def format_options(opts: list) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
