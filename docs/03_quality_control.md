# 质检与验收

## 1. 自动质检（`src/aircraft_vqa/qc.py`）

每条问答在导出前都要过一遍 `check_record()`，不通过直接丢。检查项：

| 错误码 | 含义 |
|---|---|
| `unfilled_placeholder_q/a` | 模板占位符没被替换（`{obj}` 漏进最终文本）|
| `answer_not_json` | 定位任务的答案里解析不出 JSON |
| `negative_with_boxes` | 正常图的答案里却有框 —— **最危险的一类，会直接教模型幻觉** |
| `positive_without_boxes` | 异常图的定位答案是空列表 |
| `degenerate_box` | x2 ≤ x1 或 y2 ≤ y1 |
| `box_out_of_range` | norm1000 模式下坐标超出 [0,1000] |
| `box_out_of_image` | abs 模式下坐标超出图片尺寸 |
| `count_mismatch` | "共发现 N 处"与实际框数对不上 |
| `mc_answer_letter_mismatch` | 选择题答案字母与选项对不上 |
| `duplicate_options` | 选择题出现重复选项 |
| `negative_answer_says_positive` | 判定为"无异常"但答案文字说有 |
| `positive_answer_says_negative` | 反之 |
| `image_missing` | 图片文件不存在（需 `--check-images`）|
| `not_pure_json` | `json_only` 任务的答案不是纯 JSON（整串必须能 `json.loads`）|
| `not_text_then_json` | `text_then_json` 任务没有「一句话 + 换行 + JSON」结构 |
| `text_format_has_boxes` | `text` 任务的答案里出现了 `bbox_2d` |
| `system_format_mismatch` | system 与该任务登记的输出格式不匹配 |
| `answer_asserts_absent_defect` | 答案断言的缺陷不在图里（幻觉数据）|
| `uncertainty_without_evidence` | 说"看不清"却没有劣化证据 —— 等于教模型耍赖 |
| `image_token_count_mismatch` | 多图条目的 `<image>` 数量与 `images` 长度不符 |
| `answer_mentions_unasked_term` | 答案提到问题里没出现过的缺陷名（模板变量泄漏）|
| `uncertainty_without_hedge` | 不确定样本的答案没给出"无法确认/建议补拍" |
| `uncertainty_with_boxes` | 不确定样本却输出了框 |
| `too_few_turns` | 多轮条目轮次少于 2 |
| `turn_N_*` | 多轮里第 N 轮的问题（前缀后面跟上表任一错误码）|

质检报告写进 `data/vqa/stats.json` 的 `qc` 字段，含错误分布和前 20 个失败样例。

**通过率低于 95% 就别往下走**，先看错误分布定位原因。

## 1b. 多样性体检（`stats.json` 的 `diversity` 段）

模板法最容易出的问题是"换汤不换药"：统计上量很大，实际几十个模板复读几万遍。
构建完看这几个数：

| 指标 | 含义 | 建议阈值 |
|---|---|---|
| `unique_questions_per_task` | 每个任务实际用到的不同问法数 | **≥ 30**，不足就跑 `gen_question_bank.py` 扩写 |
| `distinct_2_question` / `distinct_2_answer` | 二元组多样性 | 越高越好，答案低于 0.05 要警惕 |
| `unique_answer_ratio` | 不重复答案占比 | 越高越好 |
| `top20_answer_share` | 最高频的 20 条答案占总量比例 | **≤ 5%**，超了说明有模板在复读，必须重写 |

`negatives` 段把两类负样本分开统计 —— "这图没问题"（正常图）和
"这图有问题但不是你问的那个"（反事实）考的是不同能力，混在一起看占比
会掩盖其中一类不足。

`bbox_edge` 段统计框贴边率。大量 0 / 1000 说明合成时缺陷被贴到了图像边缘，
或者坐标换算有 clip 问题。

## 2. 人工抽检（必做）

自动质检查不出"框是准的但标的类型不对"这类问题。构建完至少看 30~50 张：

```bash
python scripts/visualize.py --vqa data/vqa/train.raw.jsonl -n 40 --out data/vis
python scripts/visualize.py --vqa data/vqa/train.raw.jsonl --task grounding_all
python scripts/visualize.py --vqa data/vqa/train.raw.jsonl --dataset synthetic_panel
```

框会画在图上，问答原文在 `data/vis/index.json`。重点看三件事：

1. **框对不对**——尤其换了 `coord_mode` 之后；
2. **类型对不对**——taxonomy 的 alias 映射有没有张冠李戴；
3. **正常图有没有被塞进框**。

## 3. 已知的坑

### 3.1 不要按条目随机切分
同一张图会同时出现在训练集和验证集，验证分数虚高。本仓库的 `group_split`
按 `sample_id` 哈希分组，已经规避。

### 3.2 不要给没有细粒度标签的数据编类型
VisA 发布版只有 `Anomaly` / `Normal`，没有缺陷类型。把它的异常统一标成
"紧固件缺失"看起来能凑数，实际是在教模型错误关联。本仓库的做法是
落到 `other_anomaly`，并让 `skip_unknown_type_tasks` 自动跳过类型类问题。

### 3.3 MVTec 系不能商用
MVTec AD / LOCO 是 CC BY-NC-SA 4.0。用 `--commercial-only` 可以一键排除所有
非商用源，授权信息从 adapter 一路带到最终条目里。

### 3.4 合成数据只能打底
`scripts/make_demo_data.py` 生成的蒙皮铆钉阵列，纹理和光照都比真实照片
规整得多。它的价值是**缺陷位置绝对精确**、可无限量生成，适合给定位能力打底；
**域真实度不够**，必须再叠真实航空数据做域适应，不要指望只靠它。

### 3.4b 问法不能口语化，也不能指代不明
两件事，都踩过：

**口语化**：项目要求问法偏指令性（保护模型的语言层）。扩模板时很容易把
"帮我看下这张图""这个要紧吗"这类加回来 —— 已加回归测试
`test_no_colloquial_questions_in_any_pool` 锁死。

**指代不明**：图里有多处缺陷时，"这个要紧吗""图中的缺陷属于哪一类"都没有
唯一先行词。用模糊指代配只讲其中一个的答案，是在教模型遇到歧义就默认挑第一个。
现在 `severity_action` / `classification_open` / `multi_turn` 都按
"图里有几种缺陷"分两组问法：单一缺陷可用通用指代，多缺陷必须点名
（点缺陷名或点方位）。

### 3.5 正常样本别丢太多
`normal_per_anomalous` 调到 0.3 以下时，模型会倾向于"看到什么都报缺陷"。
默认 1.0（正负 1:1）是个安全起点。

### 3.6 system 要按输出格式绑，不是按任务语义绑
这个坑踩了两次。第一次全局一套 system；第二次按 localization/recognition 分 ——
**还是错**，因为 localization 里混着纯 JSON 输出和自然语言输出两种任务，
4 个定位任务里 3 个违反自己的 system。

契约是关于"输出长什么样"的。现在按 `json_only` / `text_then_json` / `text` /
`dialog` 分，每个任务在 `templates.OUTPUT_FORMAT` 登记，漏登记直接报错。
质检按 `output_format` 自动校验答案形态。

### 3.6b 模板异常不要静默吞掉
构建器里曾有 `try: fn() except: rec = None`，结果模板里一个 `.format` 用错，
`grade_assessment` 和 `severity_action` 两整类任务凭空消失都没人发现。
现在异常计入 `builder.failures` 并在构建日志里打印，`strict: true` 可直接抛。
**跑大批量之前先用 strict 跑一遍小样本。**

### 3.7 负样本别复用正样本的变量槽
问的是"列出全部异常"，答的是"未见**划伤**或其他可见缺陷" —— "划伤"从哪来的？
问题里根本没提。这是负样本模板复用了正样本的 `{defect}` 槽。
现在负样本问法池不含 `{defect}`，答案也只是 `[]`，质检里
`answer_mentions_unasked_term` 兜底。

### 3.8 不要写没有视觉证据的"依据"
"依据是画面右下处可见相应特征" —— 这句话是空的，在教模型说套话装作有推理。
要么别做推理型答案，要么就让 VLM 真的看图产出形态描述（见 02 文档 14.1）。

### 3.9 不要下适航结论
严重度是按缺陷类型的规则映射出来的，不是从图上判断的，更不来自任何真实手册条款；
底图还可能是合成图。在合成图上训练出自信的适航判定，是会出事的。
现在所有处置类答案都带限定语，且不出现"不影响适航""可放行"这类断言。

### 3.10 "看不清"必须有物理依据
没有劣化证据就答"无法确认"，是在教模型随机耍赖 —— 该给答案的时候说看不清，
在机务场景是负向能力，比没有这类样本更糟。
合成器把劣化参数（`{type: "occlusion", ratio: 0.42}`）写进标注一路带到 meta，
质检里 `uncertainty_without_evidence` 卡死没有证据的条目。
接真实数据时用 Laplacian 方差、过曝像素占比这类指标先筛。

### 3.10b 螺纹损伤看不见的话等于没标
螺纹在俯视的铆钉阵列里根本不可见，所以 `thread_damage` 只在**特写场景**
（`gen_closeup`，侧视螺栓）里生成。同理，两种场景的被检对象一个是壁板、
一个是螺栓，必须分成两个数据集条目，否则问答会说出"这张检查口盖照片"
却配一张螺栓特写。

### 3.11 裂纹要从孔边起裂
真实疲劳裂纹绝大多数从紧固件孔边或壁板缝萌生（应力集中）。合成时如果让
裂纹随机长在蒙皮中间，模型学到的关联就是错的。`draw_crack` 的起点由调用方
传入孔边坐标，不是随机位置。

### 3.12 改写层可能悄悄改数字
所以 `vqa/llm.py` 做了事实指纹比对（数字 + JSON 片段必须逐字一致），
并且定位类答案根本不送去改写。自己改这一层时别把这两道保险拆了。

## 4. 验收清单

- [ ] `qc.pass_rate` ≥ 0.95
- [ ] 人工抽检 ≥ 40 张，框位与类型无明显错误
- [ ] `grounding_negative` 占比 ≥ 8%
- [ ] 正负样本比在 1:1 ~ 1:2 之间
- [ ] `by_dataset` 里航空域数据占比 ≥ 30%（不足就对该层过采样）
- [ ] train/val/test 无图片级重叠（`group_split` 保证，可用 sample_id 交集复核）
- [ ] 若要商业交付：`commercial_ok` 全为 True
- [ ] 8 类缺陷每类样本数 ≥ 300（`stats.json` 的 `by_defect_type`）
- [ ] 正负样本 1:1（二分类指标的前提，偏了指标就失真）
- [ ] 类型识别任务里最多类 ≤ 3× 最少类（宏平均指标的前提）
- [ ] 多轮条目里，正常图那一轮的框输出确实是 `[]`
- [ ] 启用了问法扩写的话，抽查 20 条：应为规范的指令式书面中文，
      不含口语语气词，术语不堆砌
- [ ] 已按 5~10% 混入通用指令数据（`--mix-general`），防语言层退化
- [ ] 负样本合计 ≥ 12%，其中**反事实负样本**（有缺陷的图上问不存在的类型）不低于 6%
- [ ] `counting` 覆盖 0 / 1 / 多三种边界
- [ ] 选择题正确答案的 A/B/C/D 分布不偏（最多不超过最少的 2.2 倍）
- [ ] 抽查 `severity_action` 答案：是否都带了"以上为初步判断…以 AMM/SRM 为准"的限定语
- [ ] **留了一份真实照片测试集**（合成图上的指标不作数）
