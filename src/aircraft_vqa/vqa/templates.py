# -*- coding: utf-8 -*-
"""问题/回答模板库（中文为主，保留少量英文以增强双语鲁棒性）。

设计原则
1. 同一任务给多套问法，避免模型过拟合到单一句式；
2. 问法全部采用航线维修口吻（"按 AMM 要求…""请判定是否可放行"），
   让模型学到的是维修场景语气，而不是通用 VQA 语气；
3. 定位类答案统一走 JSON，便于训练后直接解析。
"""

SYSTEM_PROMPT = (
    "你是一名民航机务维修视觉检查助手，擅长在飞机蒙皮、壁板、紧固件等结构件的检查照片中"
    "发现并定位螺丝/铆钉缺失、松动、锈蚀、裂纹、凹坑、划伤、漆层剥落等异常。"
    "回答要准确、简洁、可执行；没有把握的地方要明确说明，不要臆测。"
)

SYSTEM_PROMPT_GROUNDING = (
    "你是一名民航机务维修视觉检查助手。请严格按照用户要求的 JSON 格式输出目标框，"
    "不要输出多余解释。图中若不存在被问及的目标，输出空列表 []。"
)

# ---------------------------------------------------------------- 定位

Q_GROUNDING_SINGLE = [
    "请在图中定位所有{defect}的位置，用 JSON 输出边界框。",
    "这张{obj}检查照片中存在{defect}，请将其框出，以 JSON 格式给出 bbox_2d。",
    "按 JSON 格式标出图中{defect}所在的区域。",
    "Locate every {defect_en} in this image and return the bounding boxes in JSON.",
    "请框出该{obj}上{defect}的具体位置，以 JSON 输出。",
]

Q_GROUNDING_ALL = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "对该{ctx}做一次外观检查，把所有发现的缺陷连同类型一起框出来（JSON 格式）。",
    "逐一定位图中的所有异常并标注类型，按 JSON 返回。",
    "Detect all defects in this image and output bbox_2d with labels in JSON.",
]

Q_GROUNDING_NEGATIVE = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "请框出图中所有存在{defect}的位置（JSON 格式）；若不存在，返回空列表。",
    "对该{ctx}做外观检查，用 JSON 标出所有缺陷区域。",
]

A_GROUNDING_EMPTY = [
    "[]\n本图未发现异常，无需标注。",
    "[]\n该{obj}外观检查未见{defect}或其他可见缺陷。",
    "[]",
]

Q_REGION_YESNO = [
    "图中坐标 {box} 框出的区域，是否存在异常？",
    "请判断 {box} 这个范围内的{obj}有没有缺陷。",
    "请重点检查 {box} 区域，判断该处是否需要处理。",
]

A_REGION_POSITIVE = [
    "是。{box} 区域内存在{defect}，位于画面{region}，缺陷范围{size}。",
    "该区域存在异常：{defect}（{region}，{size}），建议按{severity}等级处理。",
]

A_REGION_NEGATIVE = [
    "否。{box} 区域内未见异常，该处{obj}表面完好。",
    "该区域未发现缺陷，外观正常。",
]

Q_REGION_WORD = [
    "图中的{defect}大致位于画面的哪个方位？",
    "请描述{defect}在图中的位置。",
]

A_REGION_WORD = [
    "{defect}位于画面{region}，缺陷范围{size}。",
    "在画面{region}可以看到{defect}，占比{size}。",
]

Q_COUNT = [
    "图中一共有几处{defect}？请先给出数量，再用 JSON 列出每一处的位置。",
    "请清点图中{defect}的数量，并逐个框出其位置。",
]

A_COUNT = "共发现 {n} 处{defect}。\n{json}"

# ---------------------------------------------------------------- 识别

Q_DISCRIMINATION = [
    "这张{obj}检查照片中是否存在异常？",
    "请按航线检查标准判断该{ctx}的外观是否合格。",
    "请判断该{obj}是否存在需要记录的缺陷。",
]

A_DISCRIMINATION_POS = [
    "存在异常。可见{defect_list}，位于画面{region}。",
    "不合格。该{obj}上发现{defect_list}，需要记录并进一步评估。",
    "是，存在缺陷：{defect_list}（{region}）。",
]

A_DISCRIMINATION_NEG = [
    "未发现异常。该{obj}表面完好，紧固件齐全，无锈蚀、裂纹或明显损伤。",
    "合格。外观检查未见缺陷。",
    "否，本图未见异常。",
]

Q_CLASSIFY_OPEN = [
    "图中的缺陷属于哪一类？",
    "请判断该{obj}上出现的异常属于什么类型。",
    "请判断该损伤应归入哪一种缺陷类型。",
]

A_CLASSIFY_OPEN = [
    "属于{defect}（{defect_en}）。{evidence}",
    "该异常是{defect}。{evidence}",
]

Q_CLASSIFY_MC = "图中该{obj}存在的缺陷类型是以下哪一种？\n{options}"

Q_DESCRIBE = [
    "请描述这张{obj}检查照片中的缺陷情况。",
    "请写一段简短的检查记录，说明该{ctx}的外观状况。",
    "详细说明图中异常的类型、位置和范围。",
]

A_DESCRIBE_POS = (
    "被检部件为{obj}（{ctx}）。检查发现{n}处异常：{items}。"
    "综合判定严重程度为{severity}（{severity_desc}）。"
)

A_DESCRIBE_NEG = (
    "被检部件为{obj}（{ctx}）。外观检查未见{negative_aspects}，状态正常，可放行。"
)

Q_SEVERITY = [
    "这处缺陷的严重程度如何？应该怎么处理？",
    "请评估该异常对结构完整性的影响并给出处置建议。",
    "请按维修手册说明该缺陷的处理方式。",
]

A_SEVERITY = (
    "严重程度：{severity}。{severity_desc}\n"
    "缺陷类型：{defect}，位于画面{region}，范围{size}。\n"
    "处置建议：{action}"
)

Q_OBJECT = [
    "图中被检查的是什么部件？",
    "请识别这张照片里的对象。",
    "请说明该结构在飞机上的位置，以及检查时的关注重点。",
]

A_OBJECT = [
    "图中是{obj}，在飞机上对应{ctx}。检查时重点关注{focus}。",
    "被检对象为{obj}（{ctx}）。",
]

OBJECT_FOCUS = {
    "fastener": "紧固件是否齐全、有无松动退出、头部是否变形或锈蚀",
    "fastener_kit": "各规格紧固件数量是否与清单一致、有无混装错装",
    "structure": "蒙皮有无裂纹、凹坑、腐蚀及漆层剥落，铆接排是否完好",
    "avionics": "元器件有无缺件、虚焊、烧蚀及接插件是否到位",
    "unknown": "表面完整性与连接可靠性",
}

NEGATIVE_ASPECTS = [
    "紧固件缺失或松动",
    "锈蚀与腐蚀痕迹",
    "裂纹、凹坑等结构损伤",
    "漆层剥落或划伤",
]


# ---------------------------------------------------------------- 多轮

# 机务实际的问法是追问式的：先问有没有问题，再问在哪，最后问怎么处理。
# 单轮问答学不到这个流程，也学不到"承接上文指代"（"那它严重吗"里的"它"）。
Q_MT_T1 = [
    "请检查这张{obj}照片，判断是否存在异常。",
    "请判断该{ctx}的外观检查是否合格。",
]
Q_MT_T2_POS = [
    "请给出该缺陷的位置，用 JSON 输出边界框。",
    "请把它框出来，以 JSON 格式给出 bbox_2d。",
    "请标出具体位置，输出 bbox_2d。",
]
Q_MT_T2_NEG = [
    "请再确认一遍，把所有{defect}的位置用 JSON 框出来。",
    "请复查该图，标出所有存在{defect}的区域（JSON 格式）。",
]
Q_MT_T3_POS = [
    "该缺陷的严重程度如何，应如何处理？",
    "请说明该缺陷需要立即处理，还是可以推迟到下次定检。",
    "请按维修手册给出该缺陷的处置方案。",
]
# 正常图的多轮到"复查后确认无异常"就结束了。再追加一轮"请确认结论以便签署"
# 属于流程套话，不含任何视觉信息，只会稀释训练信号。


# ---------------------------------------------------------------- 问法池

# 任务 -> 种子问法。大模型扩写出来的问法会并进这个池子（见
# scripts/gen_question_bank.py 与 load_question_bank）。
QUESTIONS = {
    "grounding_single": Q_GROUNDING_SINGLE,
    "grounding_all": Q_GROUNDING_ALL,
    "grounding_negative": Q_GROUNDING_NEGATIVE,
    "referring_region": Q_REGION_YESNO,
    "region_word": Q_REGION_WORD,
    "counting": Q_COUNT,
    "discrimination": Q_DISCRIMINATION,
    "classification_open": Q_CLASSIFY_OPEN,
    "description": Q_DESCRIBE,
    "severity_action": Q_SEVERITY,
    "object_recognition": Q_OBJECT,
    "multi_turn": Q_MT_T1,
}

# 每个任务允许出现的占位符 —— 扩写出来的问法只要用了池外的占位符，
# 或者漏掉了必需的占位符，就会被拒收（format 时会直接 KeyError）。
ALLOWED_PLACEHOLDERS = {
    "grounding_single": {"defect", "defect_en", "obj", "ctx"},
    "grounding_all": {"obj", "ctx"},
    "grounding_negative": {"defect", "obj", "ctx"},
    "referring_region": {"box", "obj", "ctx"},
    "region_word": {"defect", "obj", "ctx"},
    "counting": {"defect", "obj", "ctx"},
    "discrimination": {"obj", "ctx"},
    "classification_open": {"obj", "ctx"},
    "description": {"obj", "ctx"},
    "severity_action": {"obj", "ctx"},
    "object_recognition": {"obj", "ctx"},
    "multi_turn": {"obj", "ctx"},
}
REQUIRED_PLACEHOLDERS = {
    "grounding_single": {"defect"},      # 不点名缺陷类型就没法"单目标定位"
    "referring_region": {"box"},         # 不给框就不成其为"区域指代"
    "region_word": {"defect"},
    "counting": {"defect"},
}


def load_question_bank(path: str) -> dict:
    """读取大模型扩写出的问法库，并入种子池。文件不存在就返回空。"""
    import json
    import os
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        bank = json.load(f)
    return bank.get("questions", bank)


def merged_questions(bank: dict = None) -> dict:
    """种子问法 + 扩写问法，按任务合并去重。"""
    out = {k: list(v) for k, v in QUESTIONS.items()}
    for task, extra in (bank or {}).items():
        if task not in out:
            continue
        seen = set(out[task])
        for q in extra:
            if q not in seen:
                seen.add(q)
                out[task].append(q)
    return out
