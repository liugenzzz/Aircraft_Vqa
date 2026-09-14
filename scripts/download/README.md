# 数据获取

**一条命令搞定能自动下的部分，剩下的给你一份待办清单：**

```bash
python scripts/download/download_all.py --data-root ~/data/raw
```

设上 Roboflow 的 key，那两个航空集也会自动下：

```bash
export ROBOFLOW_API_KEY=xxx        # https://app.roboflow.com → Settings → Roboflow API
```

> **Key 不要写进任何文件、也不要提交进仓库**，只用环境变量。
> 想长期生效就写进 `~/.bashrc` 或 `~/.zshrc`。

加 `--enable-config`，下好之后会自动把 `configs/datasets.yaml` 里对应条目的
`enabled` 改成 `true`，省得手改（只动那一行，注释和格式都保留）：

```bash
python scripts/download/download_all.py --data-root ~/data/raw --enable-config
```

其他用法：

```bash
python scripts/download/download_all.py --list    # 只看清单和状态，不下载
python scripts/download/download_all.py --check   # 手动下完后体检，确认放对位置
python scripts/download/download_all.py --only visa synthetic
python scripts/download/download_all.py --n-synth 8000   # 合成数据张数
```

## 四种获取方式

| 方式 | 含义 | 涉及数据集 |
|---|---|---|
| 直链自动下 | 脚本直接拉 | VisA（1.9 GB，CC BY 4.0 可商用）、VT 腐蚀分级集 |
| 本地生成 | 不需要网络 | 合成蒙皮阵列 + 紧固件特写（8 类缺陷全覆盖） |
| 需凭据 | 有直链但要 API Key | Roboflow 航空集 ×2 |
| 需人工获取 | 必须同意条款或提交申请 | MVTec AD、MVTec LOCO、Real-IAD、Aircraft_Fuselage_DET2023 |

## 调研过但不接入的

| 数据集 | 为什么不接 |
|---|---|
| NPU-BOLT | 标的是"螺栓这个物体"而非缺陷，走 `adapter: coco` 会把每颗**正常**螺栓当成一处缺陷。要用得单独写个把框当对象框的 adapter |
| Bolt-Rotation | 单一装置 + 实验室受控光照，视觉域迁不到真实航空场景；松动的外观合成特写场景已能覆盖。只有需要"松动是连续量"这个概念监督时才值得接 |

理由详见 [docs/01_dataset_survey.md](../../docs/01_dataset_survey.md)，那里保留了完整调研记录。

## 需要你手动下的那几个

脚本会把它们写进 `<data_root>/MANUAL_DOWNLOADS.md`，每条都写清楚了
**去哪下、下完放到哪个绝对路径、怎么解压、然后改哪个配置项**。
按里面的步骤放好文件之后跑：

```bash
python scripts/download/download_all.py --data-root ~/data/raw --check
```

都显示"已有"之后，把 `configs/datasets.yaml` 里对应条目的 `enabled` 改成 `true`。

### Roboflow 下不动时怎么办

脚本会把 HTTP 错误翻译成具体原因：

| 现象 | 原因与处理 |
|---|---|
| 401 / 403 | 用错了 key。要的是 **Private API Key**，不是 Publishable Key；且该账号要能打开对应项目页 |
| 404 | workspace / project 拼错了，或版本号不存在。这两段就是项目页 URL 里 `universe.roboflow.com/` 后面那两截；用 `--list-versions` 看有哪些版本 |
| 返回里没有下载链接 | Roboflow 还在后台打包，等 1~2 分钟重跑同一条命令 |
| 429 | 限流，等几分钟 |

实在不行就走浏览器：打开项目页 → **Download Dataset** → 选 **COCO** →
下到本地解压到脚本提示的那个目录，效果一样。

优先级：

1. **Roboflow `ddiisc/aircraft_skin_defects`** —— 含 `Missing-head`，唯一直接
   命中"螺丝缺失"的航空标注（有 API Key 就自动下）
2. **MVTec AD `screw`** —— 螺纹损伤的真实数据金标准（**禁止商用**）
3. **Real-IAD** —— 取 512 版 + 三五个类就行，别下 raw 全量（**禁止商用**）
4. **Aircraft_Fuselage_DET2023** —— 可选。IEEE DataPort 上可能要付费，
   下不到不影响流水线（航空层还有 UTS 9,352 张顶着）

## 航空域数据不够时怎么补

航空真实层决定最终域表现，这一层越厚越好。Fuselage2023 拿不到的话：

```bash
# 1) 到 Universe 搜索页挑项目，一行一个写进清单
#    https://universe.roboflow.com/search?q=aircraft+defect
#    https://universe.roboflow.com/search?q=class%3Arivet
#    https://universe.roboflow.com/search?q=class%3Acorrosion
cat > aircraft_extra.txt <<'EOF'
ddiisc/aircraft-skin-defects-revised-annotations-631bu
# workspace/project，:版本号可选，# 开头是注释
EOF

# 2) 批量下并生成配置
export ROBOFLOW_API_KEY=xxx
python scripts/download/roboflow_batch.py --list-file aircraft_extra.txt \
    --data-root ~/data/raw

# 3) 多配置一起喂给 ingest
python scripts/ingest.py --config configs/datasets.yaml \
    --config configs/datasets.extra.yaml --data-root ~/data/raw
```

单个几百张，合并十来个就能顶一个中等数据集。

> Universe 上**逐个项目的授权都不同**，生成的配置里 `commercial_ok` 一律
> 填 `false`，核对过项目页的 License 再自己放开。

## 单独的脚本

| 脚本 | 说明 |
|---|---|
| `download_all.py` | 一站式，推荐用这个 |
| `download_visa.sh` | 只下 VisA |
| `download_roboflow.py` | 下任意一个 Roboflow Universe 项目；版本默认 `latest` 自动取最新，`--list-versions` 可先看有哪些版本 |
| `download_mvtec_ad.sh` / `download_mvtec_loco.sh` | 打印 MVTec 的人工获取步骤 |

## 下完之后

```bash
python scripts/ingest.py --data-root ~/data/raw     # 归一化
python scripts/build_vqa.py --out data/vqa          # 造 VQA
```
