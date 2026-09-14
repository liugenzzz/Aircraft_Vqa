#!/usr/bin/env bash
# MVTec LOCO AD —— CC BY-NC-SA 4.0。screw_bag 子集是"螺丝缺失/数量错/规格错"
# 语义最贴近的公开来源。
set -euo pipefail
OUT="${1:-$HOME/data/raw}"
cat <<'TXT'
同样需要先在官网接受条款：
  1. https://www.mvtec.com/company/research/datasets/mvtec-loco/downloads
  2. 下载 mvtec_loco_anomaly_detection.tar.xz 并解压到 <OUT>/mvtec_loco_anomaly_detection
  3. configs/datasets.yaml 里把 mvtec_loco_screwbag 的 enabled 改成 true

可选增强：原始发布只区分 logical_anomalies / structural_anomalies。
若想拿到 screw_too_long / missing_nut 这类细粒度类型，在
  <OUT>/mvtec_loco_anomaly_detection/screw_bag/defect_names.json
放一个 {"图片stem": "细粒度类型名"} 的映射，adapter 会优先采用它，
构建器就能生成"哪一颗规格不对"这类高价值问答。
TXT
echo "目标目录：$OUT/mvtec_loco_anomaly_detection"
