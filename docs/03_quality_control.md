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
| `too_few_turns` | 多轮条目轮次少于 2 |
| `turn_N_*` | 多轮里第 N 轮的问题（前缀后面跟上表任一错误码）|

质检报告写进 `data/vqa/stats.json` 的 `qc` 字段，含错误分布和前 20 个失败样例。

**通过率低于 95% 就别往下走**，先看错误分布定位原因。

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

### 3.5 正常样本别丢太多
`normal_per_anomalous` 调到 0.3 以下时，模型会倾向于"看到什么都报缺陷"。
默认 1.0（正负 1:1）是个安全起点。

### 3.6 螺纹损伤看不见的话等于没标
螺纹在俯视的铆钉阵列里根本不可见，所以 `thread_damage` 只在**特写场景**
（`gen_closeup`，侧视螺栓）里生成。同理，两种场景的被检对象一个是壁板、
一个是螺栓，必须分成两个数据集条目，否则问答会说出"这张检查口盖照片"
却配一张螺栓特写。

### 3.7 裂纹要从孔边起裂
真实疲劳裂纹绝大多数从紧固件孔边或壁板缝萌生（应力集中）。合成时如果让
裂纹随机长在蒙皮中间，模型学到的关联就是错的。`draw_crack` 的起点由调用方
传入孔边坐标，不是随机位置。

### 3.8 改写层可能悄悄改数字
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
- [ ] 多轮条目里，正常图那一轮的框输出确实是 `[]`
- [ ] 启用了问法扩写的话，抽查 20 条：应为规范的指令式书面中文，
      不含口语语气词，术语不堆砌
- [ ] 已按 5~10% 混入通用指令数据（`--mix-general`），防语言层退化
