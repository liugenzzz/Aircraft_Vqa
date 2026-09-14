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

## 3. 两大任务族与 12 个子任务

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

## 4. 坐标约定（Qwen3-VL）

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

## 5. 两层平衡

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

## 6. 切分：按图分组，不按条目

`group_split` 用 `sample_id` 的哈希决定 split，同一张图的所有问答只会落在
同一个 split。按条目随机切会导致**同一张图既在训练集又在验证集**，
验证分数虚高，是这类数据集最常见的坑。

## 7. 大模型改写层（可选）

模板答案准确但机械。`vqa/llm.py` 用一个辅助大模型（OpenAI 兼容接口，
DashScope 也走这个）把答案改写得更像一线机务的口吻，但有三道保险：

1. **保护名单**：`grounding_*` / `counting` / `classification_mc` 的答案是结构化
   输出，**根本不送去改写**；
2. **事实指纹比对**：改写前后所有数字和 JSON 片段必须完全一致，否则回退；
3. **改写后重跑质检**：不通过就回退模板答案。

离线环境 `provider: none`，整条流水线照常工作。

## 8. 训练对接

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
