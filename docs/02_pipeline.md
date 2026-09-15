# 数据方案与流水线设计

## 1. 为什么要有"统一中间表示"这一层

源数据集的组织方式五花八门：MVTec 是目录即标签、VisA 是一张 CSV 清单、
Roboflow 导出 COCO、腐蚀集是语义分割 mask、Real-IAD 是多视角 JSON。
如果每个数据集都直接写一套"图片 → 问答"的逻辑，加第 5 个数据集时就会失控。

所以中间插一层 `UnifiedSample`（见 `src/aircraft_vqa/schema.py`）：

```
源数据集 ──adapter──▶ UnifiedSample ──builder──▶ VQA 条目 ──exporter──▶ 训练格式
```

- **加新数据集**：只写一个 adapter（几十行），问答逻辑一行不用改；
- **改问题体系**：只改 builder/templates，所有数据集同步生效；
- **换训练框架**：只改 exporter。

`UnifiedSample` 的关键字段：

| 字段 | 说明 |
|---|---|
| `label` | `normal` / `anomalous` |
| `defects[]` | 每处缺陷：canonical 类型、bbox（绝对像素）、面积占比、九宫格方位、严重度 |
| `object_name` / `aircraft_ctx` | 被检对象及其**航空语境**（如 `screw` → "蒙皮壁板紧固螺钉"）|
| `license` / `commercial_ok` | 授权信息一路带到最终条目，支持 `--commercial-only` 一键过滤 |

## 2. 缺陷本体：把各家类别名收敛到一套航空语义

`configs/taxonomy.yaml` 定义 14 个 canonical 缺陷类型，每个带 `aliases` 列表。
源数据的 `Missing-head`、`missing_pushpin`、`missing_screw` 统统映射到
`fastener_missing`；`scratch_head` / `scratch_neck` / `scuff` 映射到 `scratch`。

映射是"精确 alias → 长 alias 子串"两级匹配，匹配不上退化为 `other_anomaly`
（**不会瞎猜**）。每个类型还带了 `action` 字段，即维修处置建议，
"缺陷影响分析"类问答直接取用，保证话术专业且一致。

加新数据集时通常只需要在 `aliases` 里追加几个词。

## 3. 有序程度分级

有些数据集给的不是"有/无"而是**有序的程度标签**。VT 腐蚀集的
`good / fair / poor / severe` 就是按 AASHTO 与桥梁检查员手册标的腐蚀状态，
把它们塌成一个"腐蚀锈蚀"等于把最有价值的信息扔了。

`configs/taxonomy.yaml` 的 `grades` 段把等级映射成程度词 + 严重度 + 处置方案：

| 等级 | 程度词 | 严重度 | 处置 |
|---|---|---|---|
| good | 轻微锈蚀 | minor | 清洁表面并记录，按计划监控 |
| fair | 点蚀起皮 | major | 除锈至基材，测点蚀深度后补涂 |
| poor | 层状剥落 | major | 机械除锈并评估剩余壁厚 |
| severe | 截面损失 | **critical** | 立即停用，测剩余截面并按 SRM 换件或补强 |

带等级的样本会：

1. **严重度由等级给出**，不再按面积占比估（面积大不等于程度深）；
2. 触发额外的 `grade_assessment` 任务（"发展到什么程度了？"），
   答案含等级判定 + 判定依据 + 处置方案；
3. 让 `classification_open` / `description` / `severity_action` 的答案带上程度词
   （"腐蚀锈蚀（层状剥落）"而不是笼统的"腐蚀锈蚀"）。

数据源侧用 `grade_type` **显式声明**这些名字是哪种缺陷的等级：

```yaml
grade_type: corrosion
class_map: {1: good, 2: fair, 3: poor, 4: severe}
```

必须显式声明是因为 `good` 在 MVTec 系里是"正常"的意思，撞车了会出错。

> 接新的分割数据集前先跑 `python scripts/inspect_masks.py --masks <mask目录>`
> 看 mask 的真实像素取值。填错 class_map 不会报错，但整个等级映射会静默失效；
> **等级顺序弄反了比没有等级更糟**。

## 4. 本期缺陷范围（8 类）

本体里定义了 14 类，但**本期只做 8 类**（`configs/build.yaml` 的
`active_defect_types`）：

| 族 | 类型 | 严重度 | 主要数据来源 |
|---|---|---|---|
| 紧固件 | `fastener_missing` 紧固件缺失 | major | 合成蒙皮 + Roboflow `Missing-head` |
| | `fastener_loose` 紧固件松动 | major | 合成蒙皮（螺栓外凸 + 投影拉长） |
| | `thread_damage` 螺纹损伤 | major | **合成特写** + MVTec AD `screw` |
| 结构 | `crack` 裂纹 | **critical** | **合成蒙皮** + Roboflow `Crack` |
| | `dent` 凹坑 | major | 合成蒙皮 + UTS |
| 表面 | `corrosion` 腐蚀锈蚀 | major | 合成 + VT 腐蚀分级 |
| | `scratch` 划伤 | minor | 合成 + Roboflow |
| | `paint_peeling` 漆层剥落 | minor | 合成 + Roboflow |

不在名单里的类型会**降级成 `other_anomaly`**而不是丢弃：框依然是真的，
定位和"有无异常"照常使用，只是不再声称知道它具体属于哪一类。
想全做就把 `active_defect_types` 置空。

## 5. system 按「输出格式」绑定

三层职责分开：

| 层 | 只负责 |
|---|---|
| system | 角色 + 输出契约 + 坐标约定，**按输出格式绑定** |
| user | 任务本身，同一任务备大量改写变体 |
| assistant | 答案；答案的多样性比问法的多样性更重要 |

**这里连着踩了两个坑，值得记下来。**

第一次：全局一把梭，一套 system 管所有任务 —— 定位类答案带解释，违反契约。

第二次：改成按任务语义族（localization / recognition）绑 —— **还是错**。
因为 `localization` 里混着两种输出格式：`grounding_*` 输出纯 JSON，
而 `referring_region` / `region_word` 输出自然语言。结果 4 个定位任务里
3 个违反自己的 system。

**契约是关于"输出长什么样"的，跟任务语义无关。** 现在按输出格式分三套
（`templates.SYSTEM_BY_OUTPUT`）：

| 输出格式 | 任务 | 契约 |
|---|---|---|
| `json_only` | `grounding_single/all/negative/counterfactual` | 只输出 JSON 数组，不附加任何解释 |
| `text_then_json` | `counting` | 先一句话结论，另起一行给 JSON |
| `text` | `referring_region` / `region_word` / `discrimination` / `classification_*` / `description` / `severity_action` / `grade_assessment` / `uncertainty` / `pair_compare` | 自然语言，不出现 bbox |
| `dialog` | `multi_turn` | 需要坐标时给 JSON，其余自然语言 |

每个任务必须在 `templates.OUTPUT_FORMAT` 里登记，**漏登记会直接报错**，
不会悄悄用错 system。条目的 `meta.output_format` 带着这个标记，
质检据此**自动校验答案形态**：`json_only` 的答案必须整串 `json.loads()` 成功，
`text` 的答案不得出现 `bbox_2d`。

## 6. 两大任务族与 15 个子任务

对标 MMAD 的 7 子任务体系（见调研报告 E1），按项目需求收敛成两族：

### 缺陷定位 (localization)

| 任务 | 问 | 答 | 作用 |
|---|---|---|---|
| `grounding_single` | 定位指定类型的缺陷 | JSON 框列表 | 指令跟随 + 精确定位 |
| `grounding_all` | 列出全部异常 | JSON 框 + 类型 | 全图检测 |
| `grounding_negative` | 对**正常图**问同样的话 | `[]` | 抑制幻觉 |
| `grounding_counterfactual` | 对**有缺陷的图**问一个图里不存在的类型 | `[]` | **更难的负样本**，见下 |
| `referring_region` | 给定框问该区域是否异常 | 是/否 + 说明 | 区域级聚焦 |
| `counting` | 有几处某类缺陷 | 数量 + 逐个框 | 计数一致性 |
| `region_word` | 缺陷在画面哪个方位 | 九宫格方位词 | 语义定位（不靠坐标）|

### 多轮追问 (dialog)

| 任务 | 说明 |
|---|---|
| `multi_turn` | 有无 → 定位 → 处置的三轮追问 |

机务实际是追问式工作流："这有问题吗？"→"在哪？"→"严重吗，怎么处理？"。
单轮问答学不到**上文承接**（"那它严重吗"里的"它"指什么）。
异常图走三轮（有无 → 定位 → 处置）；正常图走两轮，第二轮是
"请再确认一遍，把所有 XX 的位置用 JSON 框出来" → `[]`，
这是对话语境下的负样本，比单轮负样本更难也更有用。到这里就结束 ——
再追加一轮"请确认结论以便签署"属于流程套话，不含视觉信息，只会稀释训练信号。
图片 token 只挂在第一轮，后续轮次靠上下文。

多轮的第一轮同时计入"缺陷识别准确率"这项指标的训练量。

### 缺陷识别 (recognition)

| 任务 | 说明 |
|---|---|
| `discrimination` | 有无异常（正负样本均衡）|
| `classification_open` | 开放式缺陷类型 |
| `classification_mc` | 四选一，干扰项优先取**同族**缺陷（紧固件族内部互扰），保证有区分度 |
| `description` | 部件 + 缺陷 + 位置 + 范围 + 严重度的综合记录 |
| `severity_action` | 严重度判定 + 处置建议（取自本体的 `action`）|
| `object_recognition` | 被检对象及其航空语境、检查关注点 |
| `grade_assessment` | 有序程度分级判定（见上一节），只在源数据带等级时触发 |
| `uncertainty` | 成像模糊/遮挡/过曝时答"无法确认，建议补拍" |

### meta 的三个缺陷字段必须分开

`defect_types` 一个字段混着三件事，会让所有分层评测失效。现在拆成：

```json
"image_defect_types":  ["fastener_missing"],   // 图里客观有什么
"asked_defect_types":  ["fastener_loose"],     // 这条问的是什么
"answer_defect_types": []                      // 答案断言存在的是什么
```

有了这三个，**幻觉校验可以完全自动化**：
`answer_defect_types ⊆ image_defect_types` 必须成立，违反的就是幻觉数据，
质检里 `answer_asserts_absent_defect` 直接拦下。

另外原来的 `label: "anomalous"` 改名为 `image_status` ——
它和 bbox 里的 `"label": "凹坑"` 同名不同义，下游脚本极易搞混。

反事实变体（问一个不存在的类型）在 `meta.variant` 里标 `counterfactual`，
这样统计"计数任务准确率"时不会把"反事实拒答"混进来污染任务分布。

### 三类关键负样本

**① 反事实负样本（`grounding_counterfactual`）**
问法池里有把前提写死的问法——"这张照片中存在紧固件缺失，请将其框出"。
如果这类问法只配正样本，模型会学到"用户说有就一定有"，
部署时用户随口一问它就幻觉出一个框。

所以必须成对造：**同一批问法**（包括诱导性的那些）+ 图里没有该缺陷 + 答案 `[]`。
而且是在**有缺陷的图**上问一个不存在的类型 —— 这比拿一张完全正常的图问难得多。
两类负样本合计占 14%（正常图 6% + 反事实 8%）。

**② 难负样本区域（`referring_region`）**
给框问"这块有没有问题"时，负样本的框**贴着缺陷旁边采**（近邻但不重叠），
不是随机采空白区。随机采会让模型学到捷径：框落在画面边缘或大片空白就答"没有"，
根本没去看框里的内容。实测难负样本到缺陷中心的平均距离只有随机采样的 **43%**。

**③ 不确定样本（`uncertainty`）**
system 里写了"无法确认时直接说明，不要臆测"，就必须有样本把这件事教出来，
否则那句话是空指令 —— 模型照样对糊图给出自信的框。

**但"看不清"必须有物理依据，否则是在教模型耍赖** —— 该给答案的时候说看不清，
在机务场景是负向能力，比没有这类样本更糟。所以：

- 合成器按 `--degraded-frac` 主动施加模糊/遮挡/过曝/欠曝，
  并把**劣化参数写进标注**（`degradation: {type: "occlusion", ratio: 0.42}`），
  一路带到 VQA 的 meta 里 —— 这张图真的被遮挡了是**可审计的事实**；
- 只有带劣化证据的样本才允许生成 uncertainty，
  质检里 `uncertainty_without_evidence` 把关；
- 劣化图**只出"无法确认"样本**，不再产出任何定位答案；
- **优先问图里确实存在的缺陷**：在劣化图上问一个根本不存在的类型，
  "本来就没有"和"被挡住看不见"是混淆的，答"无法确认"说不清是哪种；
- 补救措施与成因匹配（遮挡→移开或换角度，欠曝→补充照明）。

接真实数据时没有合成参数可用，应该用 Laplacian 方差、局部对比度、
过曝像素占比这类指标筛出真正成像差的图，再打标记。

### 多图对比（`pair_compare`）

机务实际的做法就是拿正常件比对着看，MMAD 和 Anomaly-OV 都强调这个能力。
构建时按 `(数据源, 类别)` 收集正常图组成参考图池，产出
`[正常参考图, 待检图] → 指出差异区域` 的两图样本。
导出时 `images` 有两个元素，`<image>` token 数量与之匹配（质检会校验）。
没有参考图池的数据源不会产出这类样本。

### 计数要覆盖边界

`counting` 有 25% 的概率问一个图里不存在的类型，答"本图未发现 X。\n[]"。
只喂"1 处"的样本，模型学不会真数数。

**`grounding_negative` 为什么重要**：公开 IAD 数据里正常图占 90%，但如果不显式
训练"正常图就该回空列表"，模型会倾向于在任何图上都框出点什么。这类样本在
`configs/build.yaml` 里目标占比 10%，别调低。

**`skip_unknown_type_tasks`**：源数据只有"有无异常"两级标签时（VisA 就是），
构建器会自动跳过"这是什么缺陷"类问题，而不是编一个类型出来。

## 7. 坐标约定（Qwen3-VL）

Qwen3-VL 原生使用 **[0, 1000] 归一化坐标**，输出形如：

```json
[{"bbox_2d": [x1, y1, x2, y2], "label": "紧固件缺失"}]
```

- 默认 `coord_mode: norm1000`，构建时就把绝对像素换算好，训练/推理一致；
- 若用 **ms-swift 的 grounding 数据格式**（框架自己把绝对坐标换算成 norm1000），
  改成 `--coord-mode abs`，并确保训练时喂的图片尺寸与标注一致；
- `geometry.smart_resize()` 复刻了 Qwen-VL 的尺寸对齐（28 的整数倍），
  需要预缩放图片时用它，能保证坐标与模型实际看到的网格对齐。

问题是英文问法时，框里的 `label` 自动切成英文，避免中英混搭
（`VQABuilder._is_en`）。

## 8. 配比如何对标验收指标

项目的验收要求是：

- 功能：识别螺丝缺失、锈蚀等异常；具备对**局部结构的视觉聚焦与语义定位**能力
- 性能：**缺陷识别准确率 ≥ 90%**；**异常类别识别准确率 ≥ 88%**

`configs/build.yaml` 的 `task_ratio` 就是按这个排的：

| 验收项 | 对应任务 | 配比 |
|---|---|---|
| 缺陷识别准确率 ≥90% | `discrimination` + `multi_turn` 第一轮 | 16% + 8% |
| 异常类别识别准确率 ≥88% | `classification_open` + `classification_mc` | 11% + 9% |
| 局部结构视觉聚焦 | `referring_region`（给框问该区域）、`grounding_single` | 5% + 16% |
| 语义定位 | `region_word`（方位词）、`grounding_all` | 2% + 12% |
| 抑制幻觉（保准确率的下限） | `grounding_negative` | 9% |
| 辅助能力 | `description` / `severity_action` / `object_recognition` | 10% |

两个和指标直接相关的平衡策略：

1. **正负 1:1**（`normal_per_anomalous: 1.0`）。"缺陷识别准确率"是二分类指标，
   如果训练集正负 9:1，模型全答"无异常"就能拿 90%，指标失真。改这个值会让
   指标不可比，非必要不要动。
2. **类别均衡**（`class_max_over_min: 3.0`）。"异常类别识别准确率"通常按
   宏平均算（每类等权），类别样本量差 5 倍以上时长尾类学不动，宏平均会被
   直接拖死。构建时把类型识别任务里最多的类下采到最少类的 3 倍以内，
   其他任务不受影响（定位任务不需要类别均衡，反而需要真实分布）。

构建日志会打印均衡前后的类别分布，例如：

```
[class] 类型识别任务类别均衡（最多 ≤ 3.0× 最少）
        前 {'corrosion': 168, ..., 'paint_peeling': 28}
        后 {'scratch': 84, 'corrosion': 84, ..., 'paint_peeling': 28}
```

> 指标怎么测是模型侧的事，这里只负责把数据配比排到能支撑这两个指标。

## 9. 两层平衡

1. **样本层**：`normal_per_anomalous`（默认 1.0）下采样正常图。
   VisA 原始正负比约 9:1，直接全用会把模型训成"什么都说没问题"。
2. **问答层**：`task_ratio` 目标配比抽样。某任务供给不足时，缺口按剩余任务
   权重重分配，而不是让总量被最稀缺的任务卡死。

**配比达不成会显式报出来**，例如：

```
[ratio] 以下任务未达目标配比 —— 通常是源数据缺少对应标注，不是构建器的问题：
         classification_open    实际 2.3% / 目标 8.0%（达成 29%，115 条）
```

这是数据供给信号：想补齐分类类问答，就得接入带细粒度缺陷类别的源
（MVTec AD screw、Roboflow aircraft_skin_defects）。
需要严格服从配比时加 `--ratio-strict`（数据量会小很多）。

## 10. 切分：按图分组，不按条目

`group_split` 用 `sample_id` 的哈希决定 split，同一张图的所有问答只会落在
同一个 split。按条目随机切会导致**同一张图既在训练集又在验证集**，
验证分数虚高，是这类数据集最常见的坑。

## 11. 大模型该用在哪一层

大模型有两个可用的切入点，成本差三四个数量级，**优先做第一个**：

### 11.1 扩问法（推荐，一次性几十次调用）

问法是"骨架"：13 个任务 × 每个几条种子。让大模型把它扩成每任务 20~30 条
不同句式，一次性跑完就固化成 `configs/question_bank.json`，之后构建多少条
数据都不再调模型。

```bash
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python scripts/gen_question_bank.py --model qwen-plus --per-task 25
python scripts/gen_question_bank.py --dry-run    # 先看会发什么 prompt
```

扩写结果**逐条校验**后才入库：占位符必须落在该任务的白名单内、必需占位符
不能少（比如 `grounding_single` 没有 `{defect}` 就不成其为"单目标定位"）、
能成功 `format()`、不与已有重复。不合格的直接丢弃并报数。

收益：句式多样性上来了，模型不会过拟合到"请在图中定位所有 X"这一种问法。
成本：十几次调用，几分钟。

**扩写强制走指令式（`--style instruction`，默认）**，这是为了保护模型的语言层：
在窄领域数据上做 SFT，如果问法本身口语化、碎片化，模型的语言能力会被带偏 ——
表现为通用对话时也变得生硬、爱省略。指令式表述离 instruct 模型原本的 SFT 分布
最近，扰动最小。校验会拒收：

- 口语语气词（呗、咋、嘛、整一下、瞅……）
- 结尾无标点的短语碎片
- 行话堆砌（工卡/AMM/SRM/适航/放行 这类术语出现超过 1 次）
- 从句过多（逗号超过 3 个），不像清晰指令

仓库里的**种子问法本身也按这个标准收紧过**（有一条回归测试盯着，
防止以后有人加回口语化模板）。

### 11.2 润色答案（可选，逐条调用）

`vqa/llm.py` 把模板答案改写得更像一线机务口吻，但有三道保险：

1. **保护名单**：`grounding_*` / `counting` / `classification_mc` / 多轮里带
   JSON 的轮次，答案是结构化输出，**根本不送去改写**；
2. **事实指纹比对**：改写前后所有数字和 JSON 片段必须逐字一致，否则回退；
3. **改写后重跑质检**：不通过就回退模板答案。

只建议对 `description` / `severity_action` 这类叙述性任务开启，
而且开之前先小批量跑 200 条人工看一眼回退率。

离线环境 `provider: none`，整条流水线照常工作。

### 11.3 防语言层退化：混入通用指令数据（最有效的一招）

比在问法风格上反复打磨更管用的是直接混通用数据：

```bash
python scripts/build_vqa.py --mix-general path/to/general_sft.jsonl --mix-ratio 0.08
```

通用数据文件是**已经导出成目标格式的 jsonl**（每行一个 `{"messages": [...]}`，
有没有 `images` 都行，LLaMA-Factory 支持纯文本与多模态混训）。建议 5~10%。
来源可以是 Anomaly-Instruct-125k 的抽样（见调研报告 E2）、通用中文指令集，
或者你自己业务里已有的对话数据。

模板答案准确但机械。`vqa/llm.py` 用一个辅助大模型（OpenAI 兼容接口，
DashScope 也走这个）把答案改写得更像一线机务的口吻，但有三道保险：

1. **保护名单**：`grounding_*` / `counting` / `classification_mc` 的答案是结构化
   输出，**根本不送去改写**；
2. **事实指纹比对**：改写前后所有数字和 JSON 片段必须完全一致，否则回退；
3. **改写后重跑质检**：不通过就回退模板答案。

离线环境 `provider: none`，整条流水线照常工作。

## 12. 语言策略

训练数据**统一用中文**（`configs/build.yaml` 的 `lang: zh`）：

- 问法：只用中文模板。仓库里原本有几条英文问法（约占 3%），
  但这点占比不足以真的教会双语，却会引入中英混杂的不一致，所以默认过滤掉。
  确实需要模型听懂英文指令时改成 `lang: bilingual`，并把英文占比提到 20% 以上
  才有意义。
- bbox 的 `label`：跟着问法语言走，中文问就是中文标签（`box_label: zh`）。
- **保留的英文**：`classification_open` 的答案里会带术语括注，
  形如"属于漆层剥落（paint peeling）"。这是刻意保留的 ——
  维修手册（AMM/SRM）和工卡里的缺陷术语本来就是英文，
  中文答案带一个英文括注符合实际文档习惯，也方便和手册对照。
  不想要的话把 `templates.py` 的 `A_CLASSIFY_OPEN` 里 `（{defect_en}）` 去掉即可。

选择题的干扰项只从 `active_defect_types` 里取 —— 否则选项里会冒出本期不训练
的类型（比如只训 8 类却出现"多余件"），等于凭空教一个模型没见过的标签。
可选类型不足时少出几个选项，而不是塞兜底项凑数。

## 13. 训练对接

```bash
# ShareGPT（默认）—— LLaMA-Factory 直接吃
python scripts/build_vqa.py

# 其他格式
python scripts/build_vqa.py --format llamafactory     # messages 风格
python scripts/build_vqa.py --format swift --coord-mode abs
python scripts/build_vqa.py --format openai
```

ShareGPT 导出条目形如：

```json
{"conversations": [
   {"from": "human", "value": "<image>请在图中定位所有紧固件缺失的位置，用 JSON 输出边界框。"},
   {"from": "gpt",   "value": "[{\"bbox_2d\": [412, 533, 465, 601], \"label\": \"紧固件缺失\"}]"}],
 "system": "你是一名民航机务维修视觉检查助手…",
 "images": ["/abs/path/panel_000123.jpg"]}
```

多轮就是继续追加 human/gpt 对，`<image>` 只挂第一轮。

定位任务用的是更严格的 grounding system prompt（要求只输出 JSON、
无目标时输出 `[]`），识别任务用通用 system prompt。


## 14. 已知局限与后续工作

下面这些是模板法**结构性的**短板，改模板改不掉，必须靠别的手段。
写在这里免得以后忘了。

### 14.1 答案是 label → text 的确定性映射（最大的问题）

同一类缺陷的 `severity_action` 答案几乎逐字相同。模型只要认出"裂纹"两个字
就能背出整段话，**根本不需要看图**。loss 会降得很漂亮，视觉能力零增长。

缓解手段，按性价比排：

1. **降低这类任务的权重**（已做：`severity_action` 3%、`description` 6%）；
2. **开答案改写层**（`vqa/llm.py`），用本地大模型在**保持事实不变**的前提下
   重写句式、信息顺序、详略程度，改完做事实指纹比对 + 重跑质检，不一致的丢弃。
   这一层默认关着，做正式训练集前应该打开；
3. **升级到"让强 VLM 真的看图"**（ShareGPT4V 范式）。现在的做法本质是
   LLaVA 范式 —— 从标注反推答案，图像信息只经过 bbox 这一个瓶颈。
   升级方向是把图喂给本地 VLM 生成真实的视觉描述（缺陷形态、边缘、色差、
   纹理断裂），再和 GT 融合、做一致性校验后入库。

降幻觉的具体技巧（AG-VAS）：对每个异常样本同时给 VLM 三个视觉输入 ——
**带 bbox 的原图 + 叠了 GT mask 的原图 + 一张对应的正常参考图**。
这个 trick 能显著降低生成描述时的幻觉。本仓库的 mask 和正常图都是现成的。

### 14.2 面积档位词信息量低

"范围极小/较小/中等/较大/很大"是 bbox 面积占比算出来的五档词（句式已统一），
对模型信息量有限。真正有用的是形态描述（点状/条状/片状、边缘是否翻起），
但那需要 14.1 里的 VLM 看图，模板给不了。

### 14.2b 严重度仍然是类型规则映射

`severity_action` 的等级来自缺陷类型的规则表，图像里的实际严重程度
（裂纹长度、腐蚀深度）没有参与。模型学到的还是"看到裂纹两个字就说严重"。
已做的缓解：离散事件型缺陷不按面积浮动（避免"缺一颗螺丝=轻微"这种错），
面状缺陷按面积浮动，带有序等级的源（VT 腐蚀集）直接用等级。
根治同样要靠 14.1 的 VLM 看图。

### 14.3 合成数据的 domain gap

底图是程序化合成的，纹理和光照都比真实照片规整。
**必须留一份真实照片测试集**来验证是否真的迁移得动 ——
在合成图上测出来的指标不作数。

### 14.4 与公开 benchmark 对齐

任务体系和 MMAD（ICLR 2025）的 7 子任务高度重合：
anomaly discrimination / defect classification / defect localization /
defect description / defect analysis / object classification / object analysis。
建议在元数据里加一个 `mmad_task` 字段做映射，这样模型以后能挂到
公开 benchmark 上报数。MMAD 整合的源数据（MVTec-AD、VisA、MVTec-LOCO、
GoodsAD）我们已经用了前三个。
