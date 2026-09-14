# 飞机螺丝 / 结构件异常检测 —— 多模态 VQA 数据源调研报告

> 目标：为 Qwen3-VL-8B-Instruct 构建 **缺陷定位（grounding）** + **缺陷识别（recognition）** 两类指令数据。
> 场景：飞机维修（MRO）外观检查 —— 螺丝/铆钉缺失、松动、锈蚀、划伤、裂纹、凹坑、漆层剥落等。

## 0. 一句话结论

公开数据里 **没有** 一个"飞机紧固件异常"的大规模现成集。可行路线是 **三层拼装**：

| 层 | 作用 | 代表数据 | 规模量级 |
|---|---|---|---|
| L1 航空真实层 | 提供**域外观**（蒙皮、铆钉排、油污、反光） | Roboflow aircraft_skin_defects 系列、Aircraft_Fuselage_DET2023 | 1k ~ 1.5w 图 |
| L2 紧固件语义层 | 提供**螺丝/螺栓的缺失、数量、螺纹损伤**等细粒度语义与像素级 mask | MVTec AD `screw`、MVTec LOCO `screw_bag`、NPU-BOLT、Bolt-Rotation、Real-IAD、MPDD | 1w ~ 15w 图 |
| L3 缺陷形态层 | 提供**锈蚀分级、裂纹、划伤**的像素级标注与严重度分级 | VT Corrosion CS、UMaine Corrosion Rating、NEU-DET、GC10-DET、Severstal、VisA | 2w+ 图 |

再叠加 **L4：已有的多模态指令数据**（MMAD / Anomaly-Instruct-125k）复用其**题型体系**并混入训练防遗忘。

---

## 1. A 类 —— 航空领域直接相关（优先级最高）

### A1. Roboflow Universe · `DDIISc/aircraft_skin_defects` ★★★★★
- **为什么最重要**：类别里**直接含 `Missing-head`（铆钉 / 紧固件 / 螺丝头缺失）**，这是需求里"螺丝缺失"唯一能直接对上的公开标注。
- 类别：`Crack` / `Dent` / `Scratch` / `Paint-peel-off` / `Missing-head`
- 规模：原版 372 张（Hawker Hunter 276 / HT2 70 / Pushpak 76）；同作者后续版本 `aircraft-skin-defects-new-dataset` 1115 张、`aircraft-skin-defects-classification-new-dataset` 4557 张。
- 标注：目标检测 bbox（COCO / YOLO / VOC 可导出）
- 获取：Roboflow API（需免费 API Key），`roboflow` pip 包或 REST 下载
- 局限：规模小、来自博物馆/教学机，光照单一 → **只能当"域样式种子"，不足以单独训练**
- 链接：https://universe.roboflow.com/ddiisc/aircraft_skin_defects

### A2. Roboflow Universe · `University of Technology Sydney/aircraft-defect-detection` ★★★★
- 规模：**9,352 张**，是目前能拿到的最大航空缺陷检测集
- 类别：dent / leak / rupture 等（偏机身宏观损伤，紧固件粒度弱）
- 用途：L1 域外观 + "结构件异常"这一支的主力
- 链接：https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection

### A3. `Aircraft_Fuselage_DET2023`（IEEE DataPort）★★★★
- 规模：**5,601 张**机身缺陷，4 类，多光照环境实拍
- 获取：IEEE DataPort，需账号（部分条目对订阅者开放）
- 链接：https://ieee-dataport.org/documents/aircraftfuselagedet2023-aircraft-fuselage-defect-detection-dataset

### A4. Roboflow 主题检索（补充长尾）
- `class:corrosion`、`class:rivet`、`class:missing screw` 三个检索页下挂着几十个小集，单个几百张，合并后可达数千张。
- 链接：https://universe.roboflow.com/search?q=class%3Acorrosion ，https://universe.roboflow.com/search?q=class%3Arivet+gun ，https://universe.roboflow.com/search?q=class:missing+screw

### A5. 发动机孔探（borescope）损伤
- 论文级数据（1,050 标注 / 11 类损伤：crack、nick/dent、burn…），**基本不公开**。列出仅供后续商务获取参考。

---

## 2. B 类 —— 紧固件 / 螺丝专用（细粒度语义主力）

### B1. MVTec AD · `screw` 子集 ★★★★★
- 规模：320 训练（全正常）+ 160 测试；**像素级 ground-truth mask**
- 缺陷类型：`manipulated_front`（头部被撬动/错位）、`scratch_head`、`scratch_neck`、`thread_side`、`thread_top`
- **价值**：这是"单颗螺丝的细粒度损伤 + 精确定位"的金标准，直接支撑 **缺陷定位任务**（mask→bbox→归一化坐标）与"螺纹损伤/头部划伤"识别。
- 授权：**CC BY-NC-SA 4.0（禁止商用）** ← 商业落地前必须替换或自采
- 链接：https://www.mvtec.com/company/research/datasets/mvtec-ad

### B2. MVTec LOCO AD · `screw_bag` 子集 ★★★★★
- 规模：761 张（该类）；全集 3,651 张 / 5 类
- **逻辑异常**：一袋标准含「2 垫圈 + 2 螺母 + 1 长螺钉 + 1 短螺钉」，异常含 **缺件（missing）**、`screw_too_long`、`screw_too_short`、`1_very_short_screw`、数量错误
- **价值**：全网**最贴近"螺丝缺失/数量不对/规格错误"语义**的公开标注 → 直接生成"少了几颗？""哪一颗规格不对？"这类问答，这正是 MRO 检查的核心问法
- 同时含 structural_anomaly（结构异常）与 logical_anomaly（逻辑异常）两套标签 + 分割标注
- 授权：**CC BY-NC-SA 4.0（禁止商用）**
- 链接：https://www.mvtec.com/company/research/datasets/mvtec-loco

### B3. NPU-BOLT ★★★
- 规模：337 张自然场景螺栓图；4 类：`blur bolt` / `bolt head` / `bolt nut` / `bolt side`
- 价值：自然背景下的**螺栓检测与指代定位**（"左起第三颗螺栓"），补 MVTec 白底摆拍的短板
- 获取：Kaggle 公开
- 链接：https://arxiv.org/pdf/2205.11191

### B4. Bolt Rotation Dataset（Data in Brief, 2025）★★★★
- 规模：1,100+ 张，实验室受控拍摄，**标注了螺栓旋转角度偏差**、相机角度、焦距、高度
- **价值**：公开数据里**唯一能量化"松动 (looseness)"**的来源 —— 松动在视觉上就是相对初始标记的转角偏差。可生成"该螺栓相对基准转动了约 N 度，判定为松动"的问答。
- 链接：https://www.sciencedirect.com/science/article/pii/S2352340925005153

### B5. Real-IAD ★★★★
- 规模：**150K 张**（99,721 正常 / 51,329 异常），30 类工业件（金属/塑料/木/陶瓷），**每件 5 个视角**，2K~5K 超高分辨率，**像素级 mask + 图像级 + 样本级三层标注**
- 价值：① 规模最大；② **多视角** → 训练"对局部结构的视觉聚焦"与"换个角度再确认"的能力；③ 含大量金属机加/紧固类零件
- 获取：需在官方页面**签署协议申请**
- 链接：https://arxiv.org/abs/2403.12580

### B6. MPDD（Metal Parts Defect Detection）★★★
- 金属喷漆零件，6 类，像素 mask，多光照/多角度/多距离拍摄 → 金属反光鲁棒性

### B7. VisA（SPot-the-difference, ECCV'22）★★★★
- 规模：**10,821 张**（9,621 正常 / 1,200 异常），12 个对象，像素级 mask
- 与本项目的关联：`pcb1~pcb4` 含**缺件 / 错位 / 多件**等"结构性缺失"语义，形态上与"少了一颗螺丝"高度同构
- **授权 CC BY 4.0（可商用）**，且托管在 **AWS Registry of Open Data，可直接 wget，无需申请** ← 工程冷启动首选
- 链接：https://registry.opendata.aws/visa/ ，https://github.com/amazon-science/spot-diff

---

## 3. C 类 —— 腐蚀 / 锈蚀（"锈蚀"这一支的主力）

### C1. Corrosion Condition State Semantic Segmentation Dataset（Virginia Tech）★★★★★
- 规模：440 张精细标注（396 train / 44 test），512×512
- **4 级腐蚀等级像素分割**：`good` / `fair` / `poor` / `severe`，标准依据 AASHTO 与 BIRM 桥检规范
- **价值**：**唯一带"严重度分级 + 像素定位"的公开锈蚀集** → 直接支撑"锈蚀到什么程度？需要打磨还是更换？"这类维修处置问答
- 链接：https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663 ，代码 https://github.com/beric7/corrosion_cs_classification

### C2. Corrosion Condition Rating Database（University of Maine）★★★★
- 规模：514 张 RGB + 像素级标注（.json/.txt），**无人机 + 单反采集**，3 级严重度
- 价值：无人机视角与飞机外表检查的**巡检视角高度相似**
- 链接：https://digitalcommons.library.umaine.edu/student_work/30/

### C3. Roboflow `class:corrosion` 系列 ★★★
- 数十个小集，含金属件锈蚀、螺栓锈蚀，bbox 为主，用于增广多样性

---

## 4. D 类 —— 金属表面通用缺陷（预训练 / 增广）

| 数据集 | 规模 | 类别 | 说明 |
|---|---|---|---|
| NEU-DET | 1,800 张 200×200 | 6 类（划痕、麻点、裂纹、氧化铁皮、夹杂、斑块） | 经典热轧钢带；"氧化/麻点"可映射锈蚀前期 |
| GC10-DET | 3,570 张 | 10 类 | NEU-DET 的升级替代 |
| Severstal Steel Defect | 12k+ | 4 类，像素 mask | Kaggle，规模大 |
| DAGM 2007 | 合成纹理 | 10 类 | 纹理型缺陷 |
| KolektorSDD / Magnetic-Tile | 千级 | 裂纹类 | 细裂纹分割 |

用途：**只做视觉底座增广**，不进入航空语义主集，防止把"钢带缺陷"的表述污染到航空话术里。

---

## 5. E 类 —— 可直接复用的多模态指令数据（省一半工作量）

### E1. MMAD（ICLR 2025）★★★★★
- **8,366 张工业图 / 39,672 道多选题 / 38 个品类 / 244 种缺陷类型**，源数据来自 MVTec AD、MVTec LOCO、VisA、GoodsAD
- 定义了工业检测的 **7 个子任务**，这套体系可以**直接照搬**到航空场景：
  1. `Anomaly Discrimination` 有无异常
  2. `Defect Classification` 缺陷分类
  3. `Defect Localization` 缺陷定位
  4. `Defect Description` 缺陷描述
  5. `Defect Analysis` 缺陷影响分析
  6. `Object Classification` 对象识别
  7. `Object Analysis` 对象构成/功能分析
- 构造方法也值得照搬：先让 GPT-4V 生成 caption → 基于 caption 按子任务定义生成带干扰项的多选题 → 人工抽检过滤
- 参考基线：GPT-4o 平均准确率 74.9%（说明这任务对通用大模型仍然很难，**值得微调**）
- 链接：https://arxiv.org/abs/2410.09453 ，https://github.com/jam-cc/MMAD

### E2. Anomaly-Instruct-125k / Anomaly-OV（CVPR 2025）★★★★
- **首个视觉异常检测指令微调数据集**，12.5 万条，含 in-the-wild / industrial / medical / 3D 多视角 四类
- 构造方法：类别名 + 异常类型 → **在图上画出异常框** → 连同短描述喂 GPT-4o 生成细粒度描述（**这个"先画框再描述"的技巧强烈建议复用**，能显著降低幻觉）
- 用途：**按 10~20% 比例混入我们的训练集**，防止模型只会答飞机题、丢失通用异常推理能力
- 链接：https://arxiv.org/html/2502.07601v1 ，https://github.com/honda-research-institute/Anomaly-OneVision ，https://xujiacong.github.io/Anomaly-OV/

---

## 6. 推荐组合与配比

### 阶段一：冷启动（2~3 天可跑通）
纯公开、免申请、可商用的数据先把流水线跑穿：
- **VisA**（AWS 直下，CC BY 4.0）→ 定位 + 识别基础题
- **合成数据**（本仓库 `scripts/make_demo_data.py`）→ 蒙皮 + 铆钉阵列，可控生成"缺失/锈蚀/划伤"

### 阶段二：语义补齐
- MVTec AD `screw` + MVTec LOCO `screw_bag` → 螺丝细粒度 & 缺失/计数语义
- VT Corrosion CS + UMaine → 锈蚀分级
- NPU-BOLT + Bolt-Rotation → 自然场景螺栓 & 松动

### 阶段三：航空域对齐
- Roboflow aircraft_skin_defects（含 Missing-head）+ UTS aircraft-defect-detection + Aircraft_Fuselage_DET2023
- **对这一层做过采样（3~5x）**，因为它决定最终域表现

### 建议 VQA 配比（总量 15~30 万条）

| 任务 | 占比 | 说明 |
|---|---|---|
| 缺陷定位 - 单目标 grounding | 20% | 给缺陷名→出框 |
| 缺陷定位 - 全图检测（列出全部异常） | 15% | 出 JSON 列表 |
| 缺陷定位 - **正常图空列表** | **10%** | **抑制幻觉的关键，必须留足** |
| 缺陷定位 - 区域指代 / 计数 | 10% | "左起第 3 颗螺钉"、"共几颗缺失" |
| 缺陷识别 - 有无判定 | 10% | 正负样本 1:1 |
| 缺陷识别 - 类型分类（开放+多选） | 15% | |
| 缺陷识别 - 描述 + 严重度 + 处置建议 | 15% | 走大模型生成 |
| 通用能力保持（Anomaly-Instruct 抽样） | 5% | 防遗忘 |

---

## 7. 授权合规提醒（务必先看）

| 数据集 | 授权 | 能否商用 |
|---|---|---|
| MVTec AD / MVTec LOCO AD | CC BY-NC-SA 4.0 | ❌ **仅研究** |
| VisA | CC BY 4.0（仓库声明） | ✅ 需署名 |
| Real-IAD | 需签署协议申请 | ⚠️ 看协议 |
| VT Corrosion CS | 学术开放 | ⚠️ 需确认 |
| Roboflow Universe | 逐集不同，多为 CC BY 4.0 / MIT | ⚠️ **逐集核对** |
| NEU-DET / GC10-DET | 学术开放 | ⚠️ 研究为主 |

> 若最终模型要商业交付：**MVTec 系只用于做方法验证和消融**，正式训练集需用 VisA + Roboflow(CC BY) + 自采 + 合成数据替换。本仓库 `configs/datasets.yaml` 里对每个源都标了 `license` 和 `commercial_ok` 字段，构建时可用 `--commercial-only` 一键过滤。

---

## 8. 公开数据补不上的缺口 → 必须自建

| 缺口 | 原因 | 建议做法 |
|---|---|---|
| 飞机**螺丝缺失**大批量样本 | 仅 Roboflow `Missing-head` 几百框 | **合成**：在正常蒙皮图上对紧固件做 inpaint 抹除，自动得到精确 mask（本仓库 demo 脚本给了可运行雏形） |
| 螺丝**松动** | 公开只有实验室 Bolt-Rotation | 自采：同一紧固件拧松不同角度连拍；或用 3D 渲染 |
| 飞机件**锈蚀分级** | 现有锈蚀集都是桥梁/钢结构 | 迁移：把 VT 锈蚀纹理贴到飞机蒙皮 ROI 上做 Poisson 融合 |
| 真实**维修话术** | 全无 | 用维修手册（AMM/SRM）术语构建 prompt 词表，喂给大模型生成时强制使用 |
