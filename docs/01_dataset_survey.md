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
- 规模：**原始数据只有 372 张**（Hawker Hunter 276 / HT2 70 / Pushpak 76）。
- **⚠ 版本选择是这个数据集最大的坑**：Roboflow 上有 20+ 个版本，`latest`（v23）是
  「single class defect」—— 所有缺陷合并成一类，`Missing-head` 直接没了。
  张数超过 372 的版本全是增广或裁剪的衍生品：

  | 版本 | 张数 | 实质 | 能不能用 |
  |---|---|---|---|
  | v1/v5/v6/v7/v22/v23 | 372~796 | 单类 | ✗ 丢类型 |
  | v8 | 14304 | 单类 + 增广 | ✗ |
  | v10~v14 | 5673 | isolated object（裁剪成小块）| ✗ 定位任务没了 |
  | v18 | 2996 | 3 类 + 灰度翻转增广 | △ 只有 3 类 |
  | v19 | 372 | 4 类 灰度 全图 无增广 | △ 少一类 |
  | **v20** | **372** | **5 类 灰度 全图 无增广** | **← 用这个** |
  | v21 | 3379 | 5 类 isolated 灰度 | ✗ 裁剪 |

  选版本看三点：**多类**（保住细粒度标注）、**全图**（裁剪版做不了定位）、
  **无增广**（增广副本不带新信息，还可能跨 split 泄漏）。
  v20 的代价是灰度，但这五类（裂纹/凹坑/划伤/漆层剥落/缺件）对颜色依赖不强，可接受。
  本仓库的 `configs`/下载脚本已把版本钉死为 v20。
- 标注：目标检测 bbox（COCO / YOLO / VOC 可导出）
- 获取：Roboflow API（需免费 API Key），`roboflow` pip 包或 REST 下载
- 局限：规模小、来自博物馆/教学机，光照单一 → **只能当"域样式种子"，不足以单独训练**
- 链接：https://universe.roboflow.com/ddiisc/aircraft_skin_defects

### A2. Roboflow Universe · `University of Technology Sydney/aircraft-defect-detection` ★★★★
- 版本：**v3「No Nulls」6,803 张**（每张都有标注）；v2 是 25,736 张但含大量未标注图。
  取 v3 的理由：**未标注 ≠ 确认无缺陷**，拿来当正样本有风险；正常图我们从
  VisA / 合成 / MVTec 的 train 集里取，不缺这一块。已钉死为 v3。
- 类别：dent / leak / rupture 等（偏机身宏观损伤，紧固件粒度弱）
- 用途：L1 域外观 + "结构件异常"这一支的主力
- 链接：https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection

### A3. `Aircraft_Fuselage_DET2023`（IEEE DataPort）★★★★
- 规模：**5,601 张**，4 类机身表面缺陷，相机在不同光照环境下拍机身不同部位 —— 这正是 MVTec 这类摆拍数据给不了的
- 包内结构：一个 `Aircraft_Fuselage_DET2023` 文件夹，前三个子目录是同一批图的 **COCO / VOC / YOLO** 三种标注，第四个是**无标注图像池**。无标注部分不是凑数 —— 原论文做的就是半监督（动态注意力 + 类别自适应伪标签分配），作者刻意留的，想做半监督或自训练可以直接用
- 获取门槛：IEEE DataPort 的条目分开放获取与订阅者专享两种，**登录后才看得到按钮**。多数高校图书馆有 IEEE 机构订阅，走校园网 IP 或图书馆远程访问（VPN / CARSI）进去多半能直接下；卡住就问图书馆的电子资源咨询
- 引用要求：页面明确要求引作者那篇《A Semi-Supervised Aircraft Fuselage Defect Detection Network with Dynamic Attention and Class-aware Adaptive Pseudo-Label Assignment》
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

### B3. NPU-BOLT ★★ —— 需另写 adapter
- 规模：337 张自然场景螺栓图；4 类：`blur bolt` / `bolt head` / `bolt nut` / `bolt side`
- 价值：自然背景下的**螺栓检测与指代定位**（"左起第三颗螺栓"）、计数，补 MVTec 白底摆拍的短板
- **注意**：它标的是"螺栓这个物体"，不是缺陷。直接走 `adapter: coco` 会把**每一颗正常螺栓都当成一处缺陷**。要用必须单独写个把框当对象框而非缺陷框的 adapter，本仓库暂未提供，所以 `enabled: false`
- 获取：Kaggle 公开
- 链接：https://arxiv.org/pdf/2205.11191

### B4. Bolt Rotation Dataset（Data in Brief, 2025）★★ —— 暂不建议接入
- 规模：**1,112 张**实拍。自制装置上五颗 M20 螺栓、其中三颗逐步逆时针旋转，三脚架 + 单反、四个焦距、多个机位，每个旋转角度都有精确测量值
- **取数有坑**：同一批作者在 Figshare 上还有一个名字很像的《A dataset depicting simulated bolt rotation》（9.88 GB，仿真的）。**务必从论文 Data Availability 一节点链接过去**，直接搜名字容易拿错
- **价值在标签不在图像**：公开数据里确实找不到第二个把"松动程度"量化成连续角度的
- **但不建议现在接**：单一装置、实验室受控光照 —— 跟 MVTec 是同一类短板，视觉域几乎迁不到真实航空场景。本仓库的合成特写场景已经能覆盖松动的外观（`loose_fastener`，螺栓外凸 + 投影拉长）。只有当需要教模型"松动是连续量而非二值状态"这个概念时才值得接，而且必须配合 NPU-BOLT 这类真实背景做域适应，论文里也得这么写清定位，否则一眼看出是拿实验室数据充实景
- DOI: 10.1016/j.dib.2025.111788

### B5. Real-IAD ★★★★★
- 规模：**151,050 张**（训练集 36,465 张纯正常，测试集 114,585 张混合），30 类工业件，**每件 5 个视角**，像素级 mask + 图像级 + 样本级三层标注
- **最大价值在于按视角单独标注**：同一个有缺陷的物体，从看不见缺陷的那个角度拍的那张图，**标签本身就是 `good`**。也就是说"单视角结论不可靠、必须换角度确认"这件事，数据里天然编码好了监督信号，不用自己造 —— 这正对项目里"对局部结构的视觉聚焦"这项要求。
  V-AUROC（单视角）与 S-AUROC（五视角聚合取最大）的差值，就是量化这个能力最直接的证据。
- 获取：**HuggingFace 组织页 `Real-IAD`**，需账号 + 同意条款（填姓名/单位/用途），部分仓库人工审核，可能等一两天
- **体量别被 200GB 吓到**：按分辨率分包（`realiad_256/512/1024/raw`）+ 按物体一个 ZIP。取 512 版加三五个金属/紧固类，几 GB 就能起步。元数据另有 `realiad_jsons`（基础）、`_sv`（单视角）、`_fuiad`（含噪训练的 FUIAD 设定）三套
- 授权：**CC BY-NC-SA 4.0，禁止商用**（与 MVTec 同）
- 现成 loader：anomalib 有 `RealIAD(root=..., category=..., resolution=...)` 的 datamodule
- 延伸：**Real-IAD D³**（CVPR 2025），20 类连接器件，RGB + 光度立体伪 3D + 微米级点云，后续想走多模态可以看
- 链接：https://huggingface.co/Real-IAD ，论文 https://arxiv.org/abs/2403.12580

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
