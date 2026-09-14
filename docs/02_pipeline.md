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

## 3. 本期缺陷范围（8 类）

本体里定义了 14 类，但**本期只做 8 类**（`configs/build.yaml` 的
`active_defect_types`）：

| 族 | 类型 | 严重度 | 主要数据来源 |
|---|---|---|---|
| 紧固件 | `fastener_missing` 紧固件缺失 | major | 合成蒙皮 + Roboflow `Missing-head` |
| | `fastener_loose` 紧固件松动 | major | 合成蒙皮 + Bolt-Rotation |
| | `thread_damage` 螺纹损伤 | major | **合成特写** + MVTec AD `screw` |
| 结构 | `crack` 裂纹 | **critical** | **合成蒙皮** + Roboflow `Crack` |
| | `dent` 凹坑 | major | 合成蒙皮 + UTS |
| 表面 | `corrosion` 腐蚀锈蚀 | major | 合成 + VT 腐蚀分级 |
| | `scratch` 划伤 | minor | 合成 + Roboflow |
| | `paint_peeling` 漆层剥落 | minor | 合成 + Roboflow |

不在名单里的类型会**降级成 `other_anomaly`**而不是丢弃：框依然是真的，
定位和"有无异常"照常使用，只是不再声称知道它具体属于哪一类。
想全做就把 `active_defect_types` 置空。

## 4. 两大任务族与 13 个子任务

对标 MMAD 的 7 子任务体系（见调研报告 E1），按项目需求收敛成两族：

### 缺陷定位 (localization)

| 任务 | 问 | 答 | 作用 |
|---|---|---|---|
| `grounding_single` | 定位指定类型的缺陷 | JSON 框列表 | 指令跟随 + 精确定位 |
| `grounding_all` | 列出全部异常 | JSON 框 + 类型 | 全图检测 |
| `grounding_negative` | 对**正常图**问同样的话 | `[]` | **抑制幻觉**，必须留足 |
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

**`grounding_negative` 为什么重要**：公开 IAD 数据里正常图占 90%，但如果不显式
训练"正常图就该回空列表"，模型会倾向于在任何图上都框出点什么。这类样本在
`configs/build.yaml` 里目标占比 10%，别调低。

**`skip_unknown_type_tasks`**：源数据只有"有无异常"两级标签时（VisA 就是），
构建器会自动跳过"这是什么缺陷"类问题，而不是编一个类型出来。

## 5. 坐标约定（Qwen3-VL）

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

## 6. 配比如何对标验收指标

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

## 7. 两层平衡

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

## 8. 切分：按图分组，不按条目

`group_split` 用 `sample_id` 的哈希决定 split，同一张图的所有问答只会落在
同一个 split。按条目随机切会导致**同一张图既在训练集又在验证集**，
验证分数虚高，是这类数据集最常见的坑。

## 9. 大模型该用在哪一层

大模型有两个可用的切入点，成本差三四个数量级，**优先做第一个**：

### 9.1 扩问法（推荐，一次性几十次调用）

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

### 9.2 润色答案（可选，逐条调用）

`vqa/llm.py` 把模板答案改写得更像一线机务口吻，但有三道保险：

1. **保护名单**：`grounding_*` / `counting` / `classification_mc` / 多轮里带
   JSON 的轮次，答案是结构化输出，**根本不送去改写**；
2. **事实指纹比对**：改写前后所有数字和 JSON 片段必须逐字一致，否则回退；
3. **改写后重跑质检**：不通过就回退模板答案。

只建议对 `description` / `severity_action` 这类叙述性任务开启，
而且开之前先小批量跑 200 条人工看一眼回退率。

离线环境 `provider: none`，整条流水线照常工作。

### 9.3 防语言层退化：混入通用指令数据（最有效的一招）

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

## 10. 训练对接

```bash
# LLaMA-Factory（默认格式）
python scripts/build_vqa.py --format llamafactory

# ms-swift
python scripts/build_vqa.py --format swift --coord-mode abs
```

导出条目形如：

```json
{"messages": [
   {"role": "system", "content": "你是一名民航机务维修视觉检查助手…"},
   {"role": "user", "content": "<image>请在图中定位所有紧固件缺失的位置，用 JSON 输出边界框。"},
   {"role": "assistant", "content": "[{\"bbox_2d\": [412, 533, 465, 601], \"label\": \"紧固件缺失\"}]"}],
 "images": ["/abs/path/panel_000123.jpg"]}
```

定位任务用的是更严格的 grounding system prompt（要求只输出 JSON、
无目标时输出 `[]`），识别任务用通用 system prompt。
