# 数据下载说明

| 脚本 | 数据集 | 是否免申请 | 授权 |
|---|---|---|---|
| `download_visa.sh` | VisA (10,821 图) | ✅ 直下 | CC BY 4.0（可商用） |
| `download_mvtec_ad.sh` | MVTec AD screw/metal_nut | ❌ 需同意条款 | CC BY-NC-SA 4.0（禁商用） |
| `download_mvtec_loco.sh` | MVTec LOCO screw_bag | ❌ 需同意条款 | CC BY-NC-SA 4.0（禁商用） |
| `download_roboflow.py` | 航空蒙皮缺陷等 | ⚠️ 需免费 API Key | 逐集不同 |
| `../make_demo_data.py` | 合成蒙皮铆钉阵列 | ✅ 本地生成 | 自有 |

其余需要人工申请的（Real-IAD 协议、IEEE DataPort、VT 腐蚀集、Kaggle NPU-BOLT）
见 `docs/01_dataset_survey.md` 对应条目的链接。

下完之后：

```bash
python scripts/ingest.py --data-root ~/data/raw     # 归一化
python scripts/build_vqa.py --out data/vqa          # 造 VQA
```
