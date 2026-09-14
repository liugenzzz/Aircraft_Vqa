# 数据获取

**一条命令搞定能自动下的部分，剩下的给你一份待办清单：**

```bash
python scripts/download/download_all.py --data-root ~/data/raw
```

有凭据的话一起设上，Roboflow 和 Kaggle 也会自动下：

```bash
export ROBOFLOW_API_KEY=xxx        # https://app.roboflow.com → Settings → Roboflow API
# Kaggle：把 kaggle.json 放到 ~/.kaggle/，或设 KAGGLE_USERNAME / KAGGLE_KEY
pip install kaggle
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
| 直链自动下 | 脚本直接拉 | VisA（1.9 GB，CC BY 4.0 可商用） |
| 本地生成 | 不需要网络 | 合成蒙皮阵列 + 紧固件特写（8 类缺陷全覆盖） |
| 需凭据 | 有直链但要 API Key | Roboflow 航空集 ×2、NPU-BOLT(Kaggle) |
| 需人工获取 | 必须同意条款或提交申请 | MVTec AD、MVTec LOCO、VT 腐蚀集、Real-IAD、IEEE DataPort、Bolt-Rotation |

## 需要你手动下的那几个

脚本会把它们写进 `<data_root>/MANUAL_DOWNLOADS.md`，每条都写清楚了
**去哪下、下完放到哪个绝对路径、怎么解压、然后改哪个配置项**。
按里面的步骤放好文件之后跑：

```bash
python scripts/download/download_all.py --data-root ~/data/raw --check
```

都显示"已有"之后，把 `configs/datasets.yaml` 里对应条目的 `enabled` 改成 `true`。

优先级上，这三个最值得花时间弄：

1. **Roboflow `ddiisc/aircraft_skin_defects`** —— 含 `Missing-head`，唯一直接
   命中"螺丝缺失"的航空标注（有 API Key 就自动下）
2. **MVTec AD `screw`** —— 螺纹损伤的真实数据金标准（**禁止商用**）
3. **VT 腐蚀分级集** —— 唯一带锈蚀严重度分级的公开集

## 单独的脚本

| 脚本 | 说明 |
|---|---|
| `download_all.py` | 一站式，推荐用这个 |
| `download_visa.sh` | 只下 VisA |
| `download_roboflow.py` | 下任意一个 Roboflow Universe 项目 |
| `download_mvtec_ad.sh` / `download_mvtec_loco.sh` | 打印 MVTec 的人工获取步骤 |

## 下完之后

```bash
python scripts/ingest.py --data-root ~/data/raw     # 归一化
python scripts/build_vqa.py --out data/vqa          # 造 VQA
```
