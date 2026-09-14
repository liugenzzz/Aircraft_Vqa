# Aircraft VQA — 飞机螺丝及结构件异常检测多模态数据集构建

面向民航维修（MRO）外观检查场景，把公开异常检测数据集批量转换成
**Qwen3-VL-8B-Instruct 可直接训练的中文指令数据**。

覆盖两类能力：

- **缺陷定位**：螺丝/铆钉缺失、锈蚀、裂纹等异常的边界框输出、区域指代、计数
- **缺陷识别**：有无判定、类型分类、缺陷描述、严重度评估与维修处置建议

## 快速开始

```bash
pip install -r requirements.txt

# 1) 拿数据：VisA 免申请直下（CC BY 4.0），合成数据本地生成
bash scripts/download/download_visa.sh ~/data/raw
python scripts/make_demo_data.py --out ~/data/raw/synthetic_panel -n 2000

# 2) 归一化成统一中间表示
python scripts/ingest.py --data-root ~/data/raw --out data/interim

# 3) 构建 VQA（含平衡、质检、切分、导出）
python scripts/build_vqa.py --interim data/interim --out data/vqa

# 4) 人工抽检（必做）
python scripts/visualize.py --vqa data/vqa/train.raw.jsonl -n 40 --out data/vis
```

产出：

```
data/vqa/
  train.llamafactory.jsonl   # 可直接喂 LLaMA-Factory / ms-swift
  val.llamafactory.jsonl
  test.llamafactory.jsonl
  train.raw.jsonl            # 带完整元信息，用于抽检与二次加工
  stats.json                 # 分布统计 + 质检报告 + 配比达成情况
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/01_dataset_survey.md](docs/01_dataset_survey.md) | **数据源调研**：20+ 个候选数据集的规模/标注/授权/适配度，三层拼装方案与配比建议 |
| [docs/02_pipeline.md](docs/02_pipeline.md) | 方案设计：统一中间表示、缺陷本体、12 个子任务、Qwen3-VL 坐标约定、两层平衡 |
| [docs/03_quality_control.md](docs/03_quality_control.md) | 质检项、人工抽检方法、已知的坑、验收清单 |
| [scripts/download/README.md](scripts/download/README.md) | 各数据集的获取方式与授权 |

## 数据源一句话结论

公开数据里**没有**现成的"飞机紧固件异常"大规模数据集，可行路线是三层拼装：

1. **航空真实层**（域外观）：Roboflow `aircraft_skin_defects`（含 `Missing-head`，
   唯一直接命中"螺丝缺失"的航空标注）、UTS `aircraft-defect-detection`（9,352 张）
2. **紧固件语义层**（细粒度）：MVTec AD `screw`、MVTec LOCO `screw_bag`
   （缺件/数量错/规格错）、NPU-BOLT、Bolt-Rotation（唯一可量化"松动"）、Real-IAD
3. **缺陷形态层**（锈蚀/裂纹）：VT Corrosion CS（4 级锈蚀像素分级）、VisA、NEU-DET

再复用 MMAD 的 7 子任务题型体系，混入 Anomaly-Instruct-125k 防遗忘。
详见 [调研报告](docs/01_dataset_survey.md)。

## 加一个新数据集

只需要两步，问答逻辑一行不用改：

1. 在 `src/aircraft_vqa/adapters/` 写一个 adapter（若是 COCO/YOLO/MVTec/掩码分割
   这四种常见结构，直接复用现成的，不用写代码）；
2. 在 `configs/datasets.yaml` 加一个条目，填 `root` / `license` / `object_hint`。

源数据的类别名不认识？在 `configs/taxonomy.yaml` 对应类型的 `aliases` 里加一个词。

## 项目结构

```
configs/
  datasets.yaml     数据源清单（含授权与下载方式）
  taxonomy.yaml     缺陷本体：14 类缺陷 + 对象 + 严重度 + 维修处置建议
  build.yaml        构建配置：坐标模式、任务配比、平衡策略
src/aircraft_vqa/
  schema.py         统一中间表示 UnifiedSample
  taxonomy.py       类别名归一化
  geometry.py       mask→bbox、方位词、Qwen 坐标换算、smart_resize
  adapters/         MVTec / LOCO / VisA / Real-IAD / COCO / YOLO / 掩码分割
  vqa/              模板库、构建器、干扰项、负样本采样、大模型改写层
  export/           LLaMA-Factory / ms-swift / OpenAI 三种导出格式
  balance.py        两层平衡、分组切分、配比达成度
  qc.py             13 项自动质检
scripts/            download/ ingest / build_vqa / visualize / make_demo_data
tests/              30 个回归测试
```

## 授权提醒

MVTec AD 与 MVTec LOCO 是 **CC BY-NC-SA 4.0，禁止商用**。授权信息从 adapter
一路带到最终条目，商业交付时用 `--commercial-only` 一键排除所有非商用源。

```bash
python scripts/ingest.py --commercial-only
python scripts/build_vqa.py --commercial-only
```
