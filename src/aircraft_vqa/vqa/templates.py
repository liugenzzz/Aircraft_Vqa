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

# ！system 按**输出格式**绑定，不按任务语义族绑定。
#
# 踩过两次的坑：第一次全局一把梭，第二次按 localization/recognition 分。
# 第二次仍然错，因为 localization 里混着两种输出格式 —— grounding_* 输出纯
# JSON，而 referring_region/region_word 输出自然语言。结果 4 个定位任务里
# 3 个违反自己的 system。契约是关于**输出长什么样**的，跟任务语义无关。

SYSTEM_JSON_ONLY = (
    "你是一名民航机务维修视觉检查助手。\n"
    "输出边界框时使用 JSON 数组，每项形如 "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "缺陷类型"}。\n'
    "坐标已归一化到 0-1000 区间，原点在图像左上角。\n"
    "只输出该 JSON 数组，不附加任何解释。图中不存在被问及的目标时，输出 []。"
)

SYSTEM_TEXT = (
    "你是一名民航机务维修视觉检查助手，负责在飞机蒙皮、壁板、紧固件等结构件的"
    "检查照片中发现并描述异常。\n"
    "回答只陈述图像中可核对的事实：缺陷类型、位置、数量、范围。\n"
    "图像模糊、遮挡或过曝导致无法确认时，直接说明无法确认并建议如何补拍，"
    "不要臆测。"
)

SYSTEM_TEXT_THEN_JSON = (
    "你是一名民航机务维修视觉检查助手。\n"
    "回答分两部分：先用一句话给出结论（如数量），再另起一行给出 JSON 数组，"
    '每项形如 {"bbox_2d": [x1, y1, x2, y2], "label": "缺陷类型"}。\n'
    "坐标已归一化到 0-1000 区间，原点在图像左上角。\n"
    "没有符合条件的目标时，JSON 部分输出 []。"
)

SYSTEM_DIALOG = (
    "你是一名民航机务维修视觉检查助手。用户会围绕同一张检查照片连续追问。\n"
    "需要给坐标时输出 JSON 数组，每项形如 "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "缺陷类型"}，坐标归一化到 0-1000；'
    "其余问题用简短自然语言回答。\n"
    "回答只陈述图像中可核对的事实；无法确认时直接说明，不要臆测。"
)

# 输出格式 -> system。meta 里会带 output_format，质检据此自动校验答案形态。
SYSTEM_BY_OUTPUT = {
    "json_only": SYSTEM_JSON_ONLY,
    "text": SYSTEM_TEXT,
    "text_then_json": SYSTEM_TEXT_THEN_JSON,
    "dialog": SYSTEM_DIALOG,
}

# 任务 -> 输出格式。新增任务时必须在这里登记，否则构建器会报错。
OUTPUT_FORMAT = {
    "grounding_single": "json_only",
    "grounding_all": "json_only",
    "grounding_negative": "json_only",
    "grounding_counterfactual": "json_only",
    "counting": "text_then_json",
    "referring_region": "text",
    "region_word": "text",
    "discrimination": "text",
    "classification_open": "text",
    "classification_mc": "text",
    "description": "text",
    "severity_action": "text",
    "object_recognition": "text",
    "grade_assessment": "text",
    "uncertainty": "text",
    "pair_compare": "text",
    "multi_turn": "dialog",
}

# 处置建议的限定语。**准备多个变体，且只在一部分样本上出现** ——
# 每条都带同一句，模型会当成机械后缀复读，白占 token。
ADVISORY_VARIANTS = [
    "（以上为基于外观的初步判断，实际处置须以该机型现行 AMM/SRM 条款和"
    "持照人员的现场评估为准。）",
    "以上仅为外观初判，具体限度请查该机型 SRM 相应章节。",
    "最终判定以持照人员现场检查和适用手册为准。",
    "该结论基于单张照片，建议结合实物复查后再定。",
    "具体可接受限度以该机型现行手册为准。",
    "注：仅凭外观照片无法替代手册规定的检测方法。",
]
ADVISORY_RATE = 0.5      # 只在一半样本上带

# ---------------------------------------------------------------- 定位

Q_GROUNDING_SINGLE = [
    "请在图中定位所有{defect}的位置，用 JSON 输出边界框。",
    "按 JSON 格式标出图中{defect}所在的区域。",
    "请框出该{obj}上{defect}的具体位置，以 JSON 输出。",
    "检查该{ctx}，把属于{defect}的区域用 JSON 框出来。",
    "图中是否存在{defect}？存在的话用 JSON 给出边界框。",
    "请逐一定位{defect}，按 JSON 返回 bbox_2d。",
    "请只标注{defect}这一类，其他类型忽略，JSON 输出。",
    "请给出该{obj}上{defect}的全部检出框。",
    "请以 JSON 表示{defect}的位置。",
    "针对{defect}做一次定位，输出 bbox_2d 数组。",
    "该{ctx}上的{defect}在哪些位置？JSON 格式回答。",
    "请检出图中{defect}并给出归一化坐标。",
]

# 前提直接给死的问法。**必须与反事实负样本成对出现**，否则模型会学到
# "用户说有就一定有"，部署时随口一问就幻觉出一个框。
Q_GROUNDING_PRESUPPOSE = [
    "这张{obj}检查照片中存在{defect}，请将其框出，以 JSON 格式给出 bbox_2d。",
    "该{ctx}上存在{defect}，请标出其位置（JSON）。",
    "图中存在{defect}，请用 JSON 给出其边界框。",
]

Q_GROUNDING_ALL = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "对该{ctx}做一次外观检查，把所有发现的缺陷连同类型一起框出来（JSON 格式）。",
    "逐一定位图中的所有异常并标注类型，按 JSON 返回。",
    "请框出图中可见的全部缺陷并标注类型，JSON 格式。",
    "请输出该{obj}上全部缺陷的 bbox_2d 与类型。",
    "全面排查这张{obj}照片，所有异常点都要框出来并注明类型（JSON）。",
    "请以 JSON 数组形式给出该{ctx}上每一处缺陷的位置和类别。",
    "这张图需要做全面检查，把发现的问题逐一定位并分类，JSON 输出。",
    "请标注出图中所有需要记录的缺陷，含位置框与类型。",
    "对该{obj}逐处排查，输出所有缺陷的边界框与名称。",
    "请框出该{ctx}上所有不合格之处，并注明问题类型。",
    "请给出这张图上全部异常的检测结果，JSON 格式。",
    "请完整排查该{obj}照片，标出全部缺陷。",
    "请按 JSON 输出该{obj}的全部缺陷检出结果，每项含 bbox_2d 与 label。",
]

# 正常图的"找缺陷"问法。注意：不含 {defect}，避免答案里出现问题没提过的类型。
Q_GROUNDING_NEGATIVE = [
    "请检查这张{obj}照片，列出全部异常区域及其类型，用 JSON 输出。",
    "对该{ctx}做外观检查，用 JSON 标出所有缺陷区域。",
    "请框出图中全部缺陷，JSON 格式。",
    "请输出该{obj}上全部异常的 bbox_2d。",
    "全面排查这张{obj}照片，所有异常点都要框出来并注明类型（JSON）。",
    "请以 JSON 数组形式给出该{ctx}上每一处缺陷的位置和类别。",
    "请标注出图中所有需要记录的缺陷，含位置框与类型。",
    "对该{obj}逐处排查，输出所有缺陷的边界框与名称。",
    "请框出该{ctx}上所有不合格之处，并注明问题类型。",
    "请完整排查该{obj}照片，标出全部缺陷。",
]

Q_REGION_YESNO = [
    "图中坐标 {box} 框出的区域，是否存在异常？",
    "请判断 {box} 范围内的{obj}是否存在缺陷。",
    "请重点检查 {box} 区域，判断该处是否需要处理。",
    "请检查 {box} 区域，说明其中是否存在缺陷。",
]

A_REGION_POSITIVE = [
    "{box} 区域内存在{defect}，位于画面{region}，缺陷范围{size}。",
    "该区域存在异常：{defect}（{region}，{size}）。",
    "有。{box} 内可见{defect}。",
]

# 每条都必须带 {box} 或 {obj}。没有占位符的模板会在成千上万条样本里
# 一字不差地重复 —— 实测 "该区域未发现缺陷，外观正常。" 一句就出现 2066 次，
# 直接把 top20 答案占比顶上去，模型学到的是复读而不是判读。
A_REGION_NEGATIVE = [
    "{box} 区域内未见异常，该处{obj}表面完好。",
    "{box} 范围内未发现缺陷，外观正常。",
    "没有。这块范围内{obj}状态正常。",
    "{box} 内该处{obj}表面连续，无缺陷特征。",
    "查看 {box}，未见异常。",
    "{box} 这一处{obj}检查通过，无需记录。",
    "该框内{obj}表面完整，没有发现问题。",
    "{box} 区域外观正常，{obj}本体未见损伤。",
    "框选范围内{obj}状况良好，无异常。",
    "{box} 未见缺陷迹象，该处{obj}可继续使用。",
]

Q_REGION_WORD = [
    "图中的{defect}大致位于画面的哪个方位？",
    "请描述{defect}在图中的位置。",
    "{defect}在哪个位置？用方位描述即可，不用给坐标。",
    "请说明{defect}出现在这张{obj}照片的哪一块区域。",
    "用方位词描述一下{defect}的所在。",
    "请说明{defect}在图中的大致方位。",
    "请指出{defect}在该{ctx}上的分布位置。",
    "请不用坐标，描述{defect}位于画面哪一侧。",
    "请说明该处{defect}偏向画面的哪个方向。",
    "请用上下左右描述{defect}的位置。",
]

A_REGION_WORD = [
    "{defect}位于画面{region}，{size}。",
    "在画面{region}可以看到{defect}。",
    "画面{region}那一处就是{defect}。",
]

Q_COUNT = [
    "图中一共有几处{defect}？请先给出数量，再用 JSON 列出每一处的位置。",
    "请清点图中{defect}的数量，并逐个框出其位置。",
    "请统计图中{defect}的处数，并逐一标出。",
    "该{obj}上有多少处{defect}？给出数量和各自的框。",
    "请统计{defect}的处数，并输出每处的 bbox_2d。",
    "请先给出{defect}的数量，再给出各自位置。",
    "请清点该{ctx}上{defect}的总数并逐一定位。",
    "图里{defect}共计多少处？逐个标注。",
    "先数量后位置，说明该{obj}上{defect}的情况。",
    "请给出{defect}的计数结果与坐标清单。",
]

A_COUNT = "共 {n} 处{defect}。\n{json}"
A_COUNT_ZERO = ["本图未发现{defect}。\n[]", "0 处。图中不存在{defect}。\n[]"]

# ---------------------------------------------------------------- 识别

Q_DISCRIMINATION = [
    "这张{obj}检查照片中是否存在异常？",
    "请按航线检查标准判断该{ctx}的外观是否合格。",
    "请判断该{obj}是否存在需要记录的缺陷。",
    "请判断该{obj}是否存在可见缺陷。",
    "请给出该{obj}的外观检查结论。",
    "该{ctx}本次目视检查是否通过？",
    "请给出这张{obj}照片的检查结论：有异常还是无异常。",
    "请判断该{obj}上是否有需要开具工卡的缺陷。",
    "请判断图中{ctx}的状态是否正常。",
    "请说明该{obj}的外观状况，并指出是否有需要处理之处。",
    "请对该{obj}做一次快速判定：合格或不合格。",
    "图中{ctx}有无可见损伤？",
    "请判断该件可否直接装复，还是需先行处理。",
    "请说明该{obj}当前是否存在外观缺陷。",
]

# 答案里不能出现裸的"是/否" —— 问法池里"是否合格"与"有没有异常"极性相反。
# {where} 由构建器拼成"，共 N 处：裂纹位于画面右下，腐蚀锈蚀位于画面左下"
# 这种一一对应的形式。"可见裂纹、腐蚀锈蚀，位于右下、左下"让人对不上号。
A_DISCRIMINATION_POS = [
    "存在异常。可见{defect_list}{where}。",
    "不合格。该{obj}上发现{defect_list}{where}，需要记录并进一步评估。",
    "该{obj}存在缺陷：{defect_list}{where}，本项检查不通过。",
]

A_DISCRIMINATION_NEG = [
    "未发现异常。该{obj}表面完好，紧固件齐全，无锈蚀、裂纹或明显损伤。",
    "该{obj}检查合格，外观未见缺陷。",
    "该{obj}未见异常，本项检查通过。",
    "没问题，这块{obj}状态正常。",
    "外观正常。{obj}表面无可见损伤，连接部位完好。",
    "该{ctx}目视检查未见问题，状态良好。",
    "这张图没发现缺陷，{obj}表面平整、无异色。",
    "该{obj}检查结论为合格：未见裂纹、腐蚀或紧固件异常。",
    "该{obj}各处完好，本次检查无异常记录。",
    "未见任何需要记录的缺陷，{ctx}状态正常。",
    "看下来是好的，{obj}没有明显问题。",
    "无异常。该{obj}表面与紧固部位均未见损伤迹象。",
]

# 图中只有一种缺陷时用这组
Q_CLASSIFY_OPEN = [
    "图中的缺陷属于哪一类？",
    "请判断该{obj}上出现的异常属于什么类型。",
    "请判断该损伤应归入哪一种缺陷类型。",
    "请给出图中缺陷的类型名称。",
    "该{ctx}上的问题应该按哪种缺陷分类记录？",
    "请给出图中异常的缺陷类别。",
    "请说明该处损伤属于哪种缺陷。",
    "请识别该{obj}上缺陷的具体类型。",
    "请判定图中缺陷所属的类别。",
    "请判定该异常的缺陷类别并说明位置。",
    "这张{obj}照片上的问题属于什么性质的缺陷？",
    "请把图中缺陷归到对应的类别里。",
]
# 图中有多种缺陷时按位置点名，否则"图中的缺陷"指代不唯一
Q_CLASSIFY_NAMED = [
    "画面{region}那一处缺陷属于哪一类？",
    "请判定位于画面{region}的异常属于什么缺陷类型。",
    "请给出画面{region}处缺陷的类别名称。",
    "图中{region}区域的损伤应归入哪一种缺陷？",
    "请识别画面{region}那处异常的缺陷类型。",
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
    "请说明这张图的检查情况。",
    "请为该{obj}写一条目视检查记录。",
    "请描述该{ctx}照片上可见的情况。",
    "请概括该{obj}的外观检查结果。",
    "请按「部件—发现—位置」的顺序记录这张图的检查结果。",
    "请用一段话说明图中{ctx}的状况。",
    "请把该{obj}照片的检查情况整理成文字记录。",
    "请描述图中可见的全部异常及其分布。",
    "简要说明该{ctx}的表面状况。",
]

A_DESCRIBE_POS = (
    "被检部件为{obj}（{ctx}）。检查发现 {n} 处异常：{items}。"
)
A_DESCRIBE_NEG = (
    "被检部件为{obj}（{ctx}）。外观检查未见{negative_aspects}，状态正常。"
)

# 图中只有一种缺陷时用这组（指代唯一）
Q_SEVERITY_ONE = [
    "这处缺陷的严重程度如何？应该怎么处理？",
    "请评估该异常对结构完整性的影响并给出处置建议。",
    "请按维修手册说明该缺陷的处理方式。",
    "请判定该缺陷的等级，并给出下一步动作。",
    "请说明该问题的跟进方式。",
    "请说明该异常的处置方向与紧急程度。",
    "请说明该缺陷需要立即处理还是可以排期，以及如何修理。",
    "请给出该缺陷的严重程度判定及建议措施。",
    "请评估这处损伤，并说明需要采取什么措施。",
    "请评估该缺陷的影响，并说明后续措施。",
    "请给出等级判定和处理建议。",
    "请说明该处的处置方式。",
]
# 图中有多种缺陷时必须点名，否则"该缺陷"指代不唯一
Q_SEVERITY_NAMED = [
    "请评估图中{defect}的严重程度，并给出处置建议。",
    "针对画面{region}的{defect}，请说明其等级与处理方式。",
    "请说明{defect}这一处的严重程度与后续措施。",
    "请单独评估{defect}，给出等级判定与建议。",
    "图中{defect}要立即处理还是可以排期？请说明修理方式。",
    "请按维修手册说明{defect}的处理方式。",
]

# 严重度是按缺陷类型的规则映射出来的，不是从图上判断的 —— 措辞必须是
# "通常按…处理"这种建议口吻，且带限定语，不能写成适航结论。
# 三种组织方式：三段式 / 连贯段落 / 先结论后依据。
# 全都用同一个三段式模板的话，模型学到的是句式而不是内容。
A_SEVERITY = [
    ("缺陷类型：{defect}，位于画面{region}，{size}。\n"
     "严重程度：{severity}（{severity_desc}）。\n"
     "建议处置方向：{action}{advisory}"),
    ("画面{region}这处{defect}{size}，{severity_desc}，"
     "严重程度算{severity}。处理上{action}。{advisory}"),
    ("按{severity}处理。画面{region}这处是{defect}，{size}；"
     "该类缺陷{severity_desc}。具体做法：{action}{advisory}"),
]

Q_OBJECT = [
    "图中被检查的是什么部件？",
    "请识别这张照片里的对象。",
    "请说明该结构在飞机上的位置，以及检查时的关注重点。",
    "请说明这张照片的拍摄对象。",
    "请判断图中的被检对象，并说明它在飞机上属于哪部分。",
    "请说明该件属于飞机的哪个部位。",
    "请识别该部件类型，并指出检查时要重点看什么。",
    "请识别图中对象，并说明检查时的注意事项。",
    "请说明被检对象及其在机上的功能。",
    "这张图属于哪一类检查对象？",
    "请给出部件识别结果。",
    "请给出该件的名称与对应的检查要点。",
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
    "请判断该{ctx}上有没有{defect}。",
    "请判断该图中{defect}的有无。",
    "请确认图中{defect}的有无。",
    "该{obj}是否出现了{defect}？",
    "请判断这张照片是否足以认定{defect}。",
    "请给出关于{defect}的检查结论。",
    "请说明图中{defect}的检查情况。",
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
    "请按腐蚀状态标准评定图中{defect}的等级。",
    "请判定该处{defect}的严重等级。",
    "请给出该{ctx}上{defect}的分级结论。",
    "请说明{defect}当前发展到哪个阶段。",
    "请评定图中{defect}的状态等级并说明外观特征。",
    "请给出该{defect}所属的等级。",
    "请按分级标准判定该{obj}上{defect}的程度。",
    "请评定{defect}的程度等级。",
]

A_GRADE = [
    ("等级判定：{grade_zh}。\n"
     "外观特征：{grade_desc}；位于画面{region}，{size}。\n"
     "建议处置方向：{action}{advisory}"),
    ("画面{region}这处已经到{grade_zh}了 —— {grade_desc}，{size}。"
     "{action}。{advisory}"),
    ("{grade_desc}，{size}，因此判为{grade_zh}。建议{action}{advisory}"),
]

# ---------------------------------------------------------------- 多轮

Q_MT_T1 = [
    "请检查这张{obj}照片，判断是否存在异常。",
    "请判断该{ctx}的外观检查是否合格。",
    "请先对这张图做一次整体判定。",
    "请对该{obj}照片给出初步判定：是否存在异常。",
    "请先做一次整体判定，该{ctx}是否正常。",
    "请判断该{obj}是否有需要跟进的问题。",
    "请给出该{obj}的首轮检查结论。",
    "请说明该{ctx}的当前状态。",
    "请先整体检查一遍，说明该图是否存在缺陷。",
    "请判断图中{obj}是否存在可见缺陷。",
]
Q_MT_T2_POS = [
    "请给出该缺陷的位置，用 JSON 输出边界框。",
    "请把它框出来，以 JSON 格式给出 bbox_2d。",
    "请标出具体位置，输出 bbox_2d。",
    "请给出该缺陷的边界框。",
]
Q_MT_T2_NEG = [
    "请再确认一遍，把所有缺陷的位置用 JSON 框出来。",
    "请复查该图，标出所有异常区域（JSON 格式）。",
    "请再复查一遍，若存在缺陷请框出。",
]
# 前文只有一处缺陷时用这组（指代唯一）
Q_MT_T3_ONE = [
    "该缺陷的严重程度如何，应如何处理？",
    "请说明该缺陷需要立即处理，还是可以推迟到下次定检。",
    "请给出该缺陷的处置方向。",
    "该缺陷是否需要立即处理？",
]
# 前文有多处缺陷时必须点名，否则"该缺陷"指代不唯一 ——
# 用模糊指代配只讲其中一个的答案，等于教模型遇到歧义就默认挑第一个、
# 并静默丢掉其余的。
Q_MT_T3_MANY = [
    "其中的{defect}严重程度如何，应如何处理？",
    "请先说明{defect}这一处的严重程度与处理方式。",
    "请单独说明{defect}的处置方向。",
]

# ---------------------------------------------------------------- 多图对比

# 机务实际就是拿正常件比对着看。MMAD 与 Anomaly-OV 都强调这个能力。
Q_PAIR_COMPARE = [
    "第一张是同型号的正常件参考图，第二张是待检件。请对比两图，"
    "指出待检件上与参考图不一致的地方。",
    "对照参考图（第一张）检查第二张，说明差异出现在哪里、属于什么问题。",
    "请说明后一张相比前一张存在哪些不一致之处。",
    "前一张是标准状态的{obj}，后一张是待检的。请找出差异。",
    "请以第一张为基准，说明第二张{ctx}上多出了什么问题。",
    "请比对这两张照片，指出待检件的不同之处。",
    "第一张正常、第二张待检，请指出第二张的异常之处。",
    "对比参考图，说明待检{obj}存在哪些偏差。",
    "请并排比对两图，说明后一张存在的问题。",
    "请用第一张作参照，逐处指出第二张{ctx}的差异区域及性质。",
    "参考图在前、待检图在后，请给出比对结论。",
    "请说明这两张{obj}照片的区别，并判断后一张是否需要处理。",
]

A_PAIR_COMPARE_POS = [
    "与参考图相比，待检件上{items}。",
    "差异在于：{items}。其余部位与参考图一致。",
    "对比可见{items}；参考图对应位置无此现象。",
]
# 同上：每条都带 {obj} 或 {ctx}，否则几千条样本共用一句。
A_PAIR_COMPARE_NEG = [
    "两图未见实质差异，待检{obj}状态与参考图一致。",
    "对比下来没有发现异常，待检{obj}与参考图相符。",
    "逐处比对后未发现偏差，待检{obj}符合参考状态。",
    "待检{obj}与参考图一致，没有需要记录的差异。",
    "两张图对应部位都吻合，待检{obj}未见异常。",
    "比对结果为一致，待检{obj}无额外问题。",
    "参照参考图逐处核对，该{obj}未发现偏差。",
    "该{obj}与参考状态相符，本项比对通过。",
    "两图对照，{obj}的外观特征一致，无新增缺陷。",
    "比对未见差异，{ctx}状态正常。",
    "以参考图为基准核对，待检{obj}各处均一致。",
    "该{obj}与参考图无可见区别，检查通过。",
    "对照下来，{ctx}未出现参考图上没有的特征。",
    "两图一致，待检{obj}无需开具工卡。",
    "比对完成，{obj}外观与参考状态相同，未见异常。",
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
    "classification_open": Q_CLASSIFY_OPEN + Q_CLASSIFY_NAMED,
    "description": Q_DESCRIBE,
    "severity_action": Q_SEVERITY_ONE + Q_SEVERITY_NAMED,
    "object_recognition": Q_OBJECT,
    "grade_assessment": Q_GRADE,
    "uncertainty": Q_UNCERTAIN,
    "multi_turn": Q_MT_T1,
    "pair_compare": Q_PAIR_COMPARE,
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
    "classification_open": {"obj", "ctx", "region"},
    "description": {"obj", "ctx"},
    "severity_action": {"obj", "ctx", "defect", "region"},
    "object_recognition": {"obj", "ctx"},
    "grade_assessment": {"defect", "obj", "ctx"},
    "uncertainty": {"defect", "obj", "ctx"},
    "multi_turn": {"obj", "ctx"},
    "pair_compare": {"obj", "ctx"},
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
