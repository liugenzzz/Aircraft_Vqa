# -*- coding: utf-8 -*-
"""问题/回答模板库。

三层职责分离
  system    角色 + 输出契约 + 坐标约定，**按任务族绑定**，不是全局一把梭
  user      任务本身，同一任务准备大量改写变体
  assistant 答案；答案的多样性比问法的多样性更重要

几条硬约束（都是踩过坑之后加的）
1. 定位族的答案**只能是 JSON**，一个字的解释都不能带 —— system 里写了
   "不附加任何解释"，答案却跟一句说明，等于亲手教模型不遵循 system。
2. 负样本的答案**不能提问题里没出现过的缺陷名** —— 模板变量泄漏会让模型
   学到莫名其妙的关联。
3. 不写没有视觉证据的"依据是……"，那是在教模型说套话装作有推理。
4. 不下适航结论。严重度是按缺陷类型的规则映射出来的，不是从图上判断的，
   更不来自任何真实手册条款；措辞一律降为"建议方向"并带限定。
"""

# ---------------------------------------------------------------- system

# 定位族：契约要写死，坐标约定要明确
SYSTEM_GROUNDING = (
    "你是一名民航机务维修视觉检查助手。\n"
    "输出边界框时使用 JSON 数组，每项形如 "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "缺陷类型"}。\n'
    "坐标已归一化到 0-1000 区间，原点在图像左上角。\n"
    "图中不存在被问及的目标时，输出 []，不附加任何解释。"
)

# 识别族：可以有文字，但要求给出可核对的事实
SYSTEM_INSPECT = (
    "你是一名民航机务维修视觉检查助手，负责在飞机蒙皮、壁板、紧固件等结构件的"
    "检查照片中发现并描述异常。\n"
    "回答只陈述图像中可核对的事实：缺陷类型、位置、数量、范围。\n"
    "图像模糊、遮挡或过曝导致无法确认时，直接说明无法确认并建议补拍，不要臆测。"
)

# 多轮：额外强调承接上文
SYSTEM_DIALOG = (
    "你是一名民航机务维修视觉检查助手。用户会围绕同一张检查照片连续追问，"
    "后续提问中的指代（如「它」「该缺陷」）均指向前文已确认的对象。\n"
    "回答只陈述图像中可核对的事实；无法确认时直接说明，不要臆测。"
)

SYSTEM_BY_FAMILY = {
    "localization": SYSTEM_GROUNDING,
    "recognition": SYSTEM_INSPECT,
    "dialog": SYSTEM_DIALOG,
}

# 处置建议的统一限定语 —— 严重度来自类型规则而非图像判读，必须说清楚
ADVISORY_SUFFIX = (
    "（以上为基于外观的初步判断，实际处置须以该机型现行 AMM/SRM 条款"
    "和持照人员的现场评估为准。）"
)

# ---------------------------------------------------------------- 定位

Q_GROUNDING_SINGLE = [
    "请在图中定位所有{defect}的位置，用 JSON 输出边界框。",
    "按 JSON 格式标出图中{defect}所在的区域。",
    "请框出该{obj}上{defect}的具体位置，以 JSON 输出。",
    "检查该{ctx}，把属于{defect}的区域用 JSON 框出来。",
    "图中是否存在{defect}？存在的话用 JSON 给出边界框。",
    "请逐一定位{defect}，按 JSON 返回 bbox_2d。",
]

# 前提直接给死的问法。**必须与反事实负样本成对出现**，否则模型会学到
# "用户说有就一定有"，部署时随口一问就幻觉出一个框。
Q_GROUNDING_PRESUPPOSE = [
    "这张{obj}检查照片中存在{defect}，请将其框出，以 JSON 格式给出 bbox_2d。",
    "该{ctx}上有{defect}，把位置标出来（JSON）。",
    "图里那处{defect}在哪？用 JSON 给框。",
]

Q_GROUNDING_ALL = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "对该{ctx}做一次外观检查，把所有发现的缺陷连同类型一起框出来（JSON 格式）。",
    "逐一定位图中的所有异常并标注类型，按 JSON 返回。",
    "把这张图里能看到的缺陷都框出来，标上类型，JSON 格式。",
    "请输出该{obj}上全部缺陷的 bbox_2d 与类型。",
]

# 正常图的"找缺陷"问法。注意：不含 {defect}，避免答案里出现问题没提过的类型。
Q_GROUNDING_NEGATIVE = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "对该{ctx}做外观检查，用 JSON 标出所有缺陷区域。",
    "把这张图里的缺陷都框出来，JSON 格式。",
    "请输出该{obj}上全部异常的 bbox_2d。",
]

Q_REGION_YESNO = [
    "图中坐标 {box} 框出的区域，是否存在异常？",
    "请判断 {box} 这个范围内的{obj}有没有缺陷。",
    "请重点检查 {box} 区域，判断该处是否需要处理。",
    "{box} 这块看一下，有没有问题？",
]

A_REGION_POSITIVE = [
    "{box} 区域内存在{defect}，位于画面{region}，缺陷范围{size}。",
    "该区域存在异常：{defect}（{region}，{size}）。",
    "有。{box} 内可见{defect}。",
]

A_REGION_NEGATIVE = [
    "{box} 区域内未见异常，该处{obj}表面完好。",
    "该区域未发现缺陷，外观正常。",
    "没有。这块范围内{obj}状态正常。",
]

Q_REGION_WORD = [
    "图中的{defect}大致位于画面的哪个方位？",
    "请描述{defect}在图中的位置。",
    "{defect}在哪个位置？用方位描述即可，不用给坐标。",
]

A_REGION_WORD = [
    "{defect}位于画面{region}，缺陷范围{size}。",
    "在画面{region}可以看到{defect}。",
    "画面{region}那一处就是{defect}。",
]

Q_COUNT = [
    "图中一共有几处{defect}？请先给出数量，再用 JSON 列出每一处的位置。",
    "请清点图中{defect}的数量，并逐个框出其位置。",
    "数一下这张图里有几处{defect}，把每一处都标出来。",
]

A_COUNT = "共 {n} 处{defect}。\n{json}"
A_COUNT_ZERO = ["本图未发现{defect}。\n[]", "0 处。图中不存在{defect}。\n[]"]

# ---------------------------------------------------------------- 识别

Q_DISCRIMINATION = [
    "这张{obj}检查照片中是否存在异常？",
    "请按航线检查标准判断该{ctx}的外观是否合格。",
    "请判断该{obj}是否存在需要记录的缺陷。",
    "这块{obj}有没有问题？",
    "帮我看下这张图，外观正常吗？",
]

# 答案里不能出现裸的"是/否" —— 问法池里"是否合格"与"有没有异常"极性相反。
A_DISCRIMINATION_POS = [
    "存在异常。可见{defect_list}{where}。",
    "不合格。该{obj}上发现{defect_list}{where}，需要记录并进一步评估。",
    "该{obj}存在缺陷：{defect_list}{where}，本项检查不通过。",
]

A_DISCRIMINATION_NEG = [
    "未发现异常。该{obj}表面完好，紧固件齐全，无锈蚀、裂纹或明显损伤。",
    "检查合格，外观未见缺陷。",
    "该{obj}未见异常，本项检查通过。",
    "没问题，这块{obj}状态正常。",
]

Q_CLASSIFY_OPEN = [
    "图中的缺陷属于哪一类？",
    "请判断该{obj}上出现的异常属于什么类型。",
    "请判断该损伤应归入哪一种缺陷类型。",
    "这是什么类型的缺陷？",
]

# 不写"依据是……可见相应特征"这种空话；只陈述能核对的事实。
A_CLASSIFY_OPEN = [
    "属于{defect}{evidence}",
    "该异常是{defect}{evidence}",
    "{defect}{evidence}",
]

Q_CLASSIFY_MC = "图中该{obj}存在的缺陷类型是以下哪一种？\n{options}"

Q_DESCRIBE = [
    "请描述这张{obj}检查照片中的缺陷情况。",
    "请写一段简短的检查记录，说明该{ctx}的外观状况。",
    "详细说明图中异常的类型、位置和范围。",
    "说说这张图的情况。",
]

A_DESCRIBE_POS = (
    "被检部件为{obj}（{ctx}）。检查发现 {n} 处异常：{items}。"
)
A_DESCRIBE_NEG = (
    "被检部件为{obj}（{ctx}）。外观检查未见{negative_aspects}，状态正常。"
)

Q_SEVERITY = [
    "这处缺陷的严重程度如何？应该怎么处理？",
    "请评估该异常对结构完整性的影响并给出处置建议。",
    "请按维修手册说明该缺陷的处理方式。",
    "这个要紧吗？大概怎么处理？",
]

# 严重度是按缺陷类型的规则映射出来的，不是从图上判断的 —— 措辞必须是
# "通常按…处理"这种建议口吻，且带限定语，不能写成适航结论。
A_SEVERITY = (
    "缺陷类型：{defect}，位于画面{region}，范围{size}。\n"
    "严重程度：{severity}（{severity_desc}）。\n"
    "建议处置方向：{action}\n" + ADVISORY_SUFFIX
)

Q_OBJECT = [
    "图中被检查的是什么部件？",
    "请识别这张照片里的对象。",
    "请说明该结构在飞机上的位置，以及检查时的关注重点。",
]

A_OBJECT = [
    "图中是{obj}，在飞机上对应{ctx}。检查时重点关注{focus}。",
    "被检对象为{obj}（{ctx}）。",
    "这是{obj}。",
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

# ---------------------------------------------------------------- 不确定

# system 里写了"无法确认时直接说明"，就必须有样本把这件事教出来，
# 否则那句话是空指令。这类样本由合成器刻意生成的模糊/遮挡/过曝图提供。
Q_UNCERTAIN = [
    "图中是否存在{defect}？",
    "请检查该{obj}，判断有没有{defect}。",
    "这张照片上能看出{defect}吗？",
    "请框出图中的{defect}（JSON 格式）。",
]

# 补救措施要跟成因对上 —— 遮挡了却让人"在良好光照下补拍"是答非所问。
A_UNCERTAIN = [
    "该图{reason}，无法确认是否存在{defect}，建议{fix}后复查。",
    "成像条件不足（{reason}），本图不足以做出判定，建议{fix}。",
    "{reason}，该区域细节不可辨，无法给出结论，需{fix}重新取证。",
]

UNCERTAIN_REASON = {
    "blur": "整体离焦模糊",
    "occlusion": "关键区域被遮挡",
    "overexposure": "局部过曝、细节丢失",
    "darkness": "曝光不足、画面过暗",
}

UNCERTAIN_FIX = {
    "blur": "稳定持机、重新对焦后补拍",
    "occlusion": "移开遮挡物或换一个角度补拍",
    "overexposure": "避开强反光方向、换角度补拍",
    "darkness": "补充照明后重拍",
}

# ---------------------------------------------------------------- 分级

Q_GRADE = [
    "请判断图中{defect}的程度等级。",
    "该{obj}上的{defect}发展到什么程度了？请给出等级判定。",
    "请按腐蚀状态标准评定图中{defect}的等级，并说明判定依据。",
    "这处{defect}算严重吗？到哪一级了？",
]

A_GRADE = (
    "等级判定：{grade_zh}。\n"
    "外观特征：{grade_desc}；位于画面{region}，范围{size}。\n"
    "建议处置方向：{action}\n" + ADVISORY_SUFFIX
)

# ---------------------------------------------------------------- 多轮

Q_MT_T1 = [
    "请检查这张{obj}照片，判断是否存在异常。",
    "请判断该{ctx}的外观检查是否合格。",
    "先看下这张图，有问题吗？",
]
Q_MT_T2_POS = [
    "请给出该缺陷的位置，用 JSON 输出边界框。",
    "请把它框出来，以 JSON 格式给出 bbox_2d。",
    "请标出具体位置，输出 bbox_2d。",
    "在哪？给个框。",
]
Q_MT_T2_NEG = [
    "请再确认一遍，把所有缺陷的位置用 JSON 框出来。",
    "请复查该图，标出所有异常区域（JSON 格式）。",
    "再仔细看一遍，真的没有吗？有的话框出来。",
]
Q_MT_T3_POS = [
    "该缺陷的严重程度如何，应如何处理？",
    "请说明该缺陷需要立即处理，还是可以推迟到下次定检。",
    "请给出该缺陷的处置方向。",
    "那这个要紧吗？",
]

# ---------------------------------------------------------------- 问法池

QUESTIONS = {
    "grounding_single": Q_GROUNDING_SINGLE + Q_GROUNDING_PRESUPPOSE,
    "grounding_all": Q_GROUNDING_ALL,
    "grounding_negative": Q_GROUNDING_NEGATIVE,
    "grounding_counterfactual": Q_GROUNDING_SINGLE + Q_GROUNDING_PRESUPPOSE,
    "referring_region": Q_REGION_YESNO,
    "region_word": Q_REGION_WORD,
    "counting": Q_COUNT,
    "discrimination": Q_DISCRIMINATION,
    "classification_open": Q_CLASSIFY_OPEN,
    "description": Q_DESCRIBE,
    "severity_action": Q_SEVERITY,
    "object_recognition": Q_OBJECT,
    "grade_assessment": Q_GRADE,
    "uncertainty": Q_UNCERTAIN,
    "multi_turn": Q_MT_T1,
}

ALLOWED_PLACEHOLDERS = {
    "grounding_single": {"defect", "defect_en", "obj", "ctx"},
    "grounding_all": {"obj", "ctx"},
    "grounding_negative": {"obj", "ctx"},
    "grounding_counterfactual": {"defect", "defect_en", "obj", "ctx"},
    "referring_region": {"box", "obj", "ctx"},
    "region_word": {"defect", "obj", "ctx"},
    "counting": {"defect", "obj", "ctx"},
    "discrimination": {"obj", "ctx"},
    "classification_open": {"obj", "ctx"},
    "description": {"obj", "ctx"},
    "severity_action": {"obj", "ctx"},
    "object_recognition": {"obj", "ctx"},
    "grade_assessment": {"defect", "obj", "ctx"},
    "uncertainty": {"defect", "obj", "ctx"},
    "multi_turn": {"obj", "ctx"},
}
REQUIRED_PLACEHOLDERS = {
    "grade_assessment": {"defect"},
    "grounding_single": {"defect"},
    "grounding_counterfactual": {"defect"},
    "referring_region": {"box"},
    "region_word": {"defect"},
    "counting": {"defect"},
    "uncertainty": {"defect"},
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
