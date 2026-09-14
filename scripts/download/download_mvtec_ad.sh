#!/usr/bin/env bash
# MVTec AD —— CC BY-NC-SA 4.0（禁止商用），需在官网同意条款后获取直链。
# 本项目主要用 screw / metal_nut 两个子集（螺纹损伤、头部划伤，含像素 mask）。
set -euo pipefail
OUT="${1:-$HOME/data/raw}"
mkdir -p "$OUT"
cat <<'TXT'
MVTec 要求先在官网接受授权条款，无法用固定直链无人值守下载。

步骤：
  1. 打开 https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
  2. 填写表单并同意 CC BY-NC-SA 4.0（仅限非商业研究用途）
  3. 下载 mvtec_anomaly_detection.tar.xz
  4. 放到本目录并解压：
       tar -xf mvtec_anomaly_detection.tar.xz -C <OUT>/mvtec_anomaly_detection
  5. configs/datasets.yaml 里把 mvtec_ad_screw 的 enabled 改成 true

提醒：该数据禁止商用。若模型要商业交付，只把它用于方法验证与消融。
TXT
echo "目标目录：$OUT/mvtec_anomaly_detection"
