"""多选题干扰项生成。

好的干扰项要"像但不是"：优先同族缺陷（紧固件族内部互为干扰），
其次跨族，最后才用兜底项，保证题目有区分度而不是送分题。
"""
from __future__ import annotations

import random
from typing import Optional

from ..taxonomy import Taxonomy

LETTERS = "ABCDEF"

# 视觉上容易混淆的缺陷对 —— 干扰项优先从这里取，题目才有区分度。
# 不加这个的话会出现"panel 图里拿螺纹损伤当干扰项"，不看图都能排掉。
CONFUSABLE = [
    {"scratch", "crack"},                       # 细长的亮痕 vs 暗痕
    {"corrosion", "paint_peeling"},             # 都是片状变色
    {"fastener_missing", "fastener_loose"},     # 都在紧固件位置
    {"dent", "corrosion"},                      # 都有明暗渐变
    {"thread_damage", "fastener_wrong_spec"},   # 都在螺纹段
    {"breakage", "crack"},
]

# 部件类型 -> 该部件上可能出现的缺陷。不在这个池里的当干扰项属于送分。
PART_DEFECTS = {
    "fastener": {"fastener_missing", "fastener_loose", "thread_damage",
                 "fastener_wrong_spec", "corrosion", "scratch", "breakage"},
    "fastener_kit": {"fastener_missing", "fastener_wrong_spec", "extra_part",
                     "corrosion", "thread_damage"},
    "structure": {"fastener_missing", "fastener_loose", "crack", "dent",
                  "corrosion", "scratch", "paint_peeling", "breakage",
                  "misalignment"},
    "avionics": {"fastener_missing", "misalignment", "extra_part", "corrosion",
                 "breakage", "contamination"},
}


def part_pool(role: str) -> Optional[set]:
    """该部件上可能出现的缺陷集合；不认识的部件返回 None（不做限制）。"""
    return PART_DEFECTS.get(role)


def make_options(tax: Taxonomy, correct_types: list, n_options: int = 4,
                 rng: Optional[random.Random] = None,
                 active: Optional[set] = None,
                 part_role: Optional[str] = None) -> tuple:
    """返回 (选项文本列表, 正确答案字母)。选项文本为中文缺陷名。

    `active` 给定时，干扰项只从本期训练范围内的缺陷类型里取 —— 否则选项里会
    冒出范围外的词（比如只训 8 类却出现"多余件"），等于凭空教了一个不训练的
    标签，模型既没见过它的样子，也不知道该不该选。
    """
    rng = rng or random
    correct = correct_types[0]
    correct_zh = tax.zh(correct)
    exclude = set(correct_types)

    part = part_pool(part_role) if part_role else None

    def usable(t: str) -> bool:
        return (t not in exclude and t != "other_anomaly"
                and (active is None or t in active)
                # 干扰项必须是该部件上**可能出现**的缺陷。拿螺纹损伤去干扰
                # 一张蒙皮图，不看图都能排掉，题目就没有区分度了。
                and (part is None or t in part))

    # 一级：视觉易混对（划伤↔裂纹、腐蚀↔漆层剥落…）
    confuse = set()
    for pair in CONFUSABLE:
        if correct in pair:
            confuse |= pair
    pool_conf = [t for t in confuse if usable(t)]
    # 二级：同族
    pool_near = [t for t in tax.siblings(correct)
                 if usable(t) and t not in pool_conf]
    # 三级：其余
    pool_far = [t for t in tax.all_types()
                if usable(t) and t not in pool_conf and t not in pool_near]
    rng.shuffle(pool_conf)
    rng.shuffle(pool_near)
    rng.shuffle(pool_far)

    need = n_options - 1
    distract = (pool_conf + pool_near + pool_far)[:need]
    n_options = min(n_options, len(distract) + 1)  # 可选类型不够就少出几个选项

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
