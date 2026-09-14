#!/usr/bin/env bash
# VisA (SPot-the-difference, ECCV'22) —— CC BY 4.0，AWS 开放数据，免申请直下。
# 10,821 张 / 12 个对象 / 像素级 mask，约 1.9GB。
set -euo pipefail
OUT="${1:-$HOME/data/raw}"
mkdir -p "$OUT"
URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
echo ">> 下载 VisA 到 $OUT (~1.9GB)"
curl -L --retry 5 --retry-delay 3 -C - -o "$OUT/VisA.tar" "$URL"
echo ">> 解压"
mkdir -p "$OUT/VisA"
tar -xf "$OUT/VisA.tar" -C "$OUT/VisA"
echo ">> 完成：$OUT/VisA"
echo "   configs/datasets.yaml 里 visa.root 应为 {data_root}/VisA"
