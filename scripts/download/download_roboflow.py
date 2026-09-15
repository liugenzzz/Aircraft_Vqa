#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Roboflow Universe 下载数据集（默认 COCO 格式）。

需要一个免费的 Private API Key：
  https://app.roboflow.com → Settings → Roboflow API → Private API Key

    export ROBOFLOW_API_KEY=xxxx
    python scripts/download/download_roboflow.py \
        --workspace ddiisc --project aircraft_skin_defects \
        --out ~/data/raw/roboflow/aircraft_skin_defects

版本号默认 latest —— 会先查项目元信息再挑最大的那个版本，
不用你去项目页上翻。也可以 --version 3 指定。
--list-versions 只列版本不下载。

下完的目录（train/valid/test + _annotations.coco.json）可直接被
configs/datasets.yaml 里 adapter: coco 的条目读取。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

API = "https://api.roboflow.com"


def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "aircraft-vqa/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _explain(e: Exception, ws: str, proj: str) -> str:
    """把 Roboflow 的 HTTP 错误翻译成人能看懂的下一步。"""
    if isinstance(e, urllib.error.HTTPError):
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code in (401, 403):
            return ("API Key 无效或没有这个项目的权限。确认用的是 "
                    "Private API Key（不是 Publishable Key），"
                    f"且这个 key 所属账号能打开 universe.roboflow.com/{ws}/{proj}。")
        if e.code == 404:
            return (f"找不到 {ws}/{proj} 或指定的版本。请到项目页核对 workspace "
                    f"和 project 的拼写（就是 URL 里 universe.roboflow.com/ 后面那两段），"
                    f"并用 --list-versions 看有哪些版本。")
        if e.code == 429:
            return "触发限流，等几分钟再试。"
        return f"HTTP {e.code}：{body}"
    return str(e)


def list_versions(ws: str, proj: str, key: str) -> list:
    meta = _get_json(f"{API}/{ws}/{proj}?api_key={key}")
    vers = meta.get("versions") or []
    out = []
    for v in vers:
        # id 形如 "ddiisc/aircraft_skin_defects/3"
        vid = str(v.get("id", "")).rsplit("/", 1)[-1]
        if vid.isdigit():
            out.append({"version": int(vid), "name": v.get("name", ""),
                        "images": v.get("images"), "created": v.get("created")})
    return sorted(out, key=lambda x: x["version"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", default="latest",
                    help="版本号，或 latest（默认，自动挑最大的）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="coco", help="coco | yolov8 | voc")
    ap.add_argument("--list-versions", action="store_true", help="只列版本，不下载")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        print("请先设置环境变量 ROBOFLOW_API_KEY\n"
              "  获取：https://app.roboflow.com → Settings → Roboflow API → Private API Key\n"
              "  设置：export ROBOFLOW_API_KEY=你的key")
        return 1

    ws, proj = args.workspace, args.project

    # ---- 确定版本 ----
    version = args.version
    if args.list_versions or version == "latest":
        try:
            vers = list_versions(ws, proj, key)
        except Exception as e:
            print(f"查询项目版本失败：{_explain(e, ws, proj)}")
            return 1
        if not vers:
            print(f"{ws}/{proj} 下没有任何已生成的版本。"
                  f"请到项目页点 Generate 生成一个版本后再下。")
            return 1
        print(f">> {ws}/{proj} 可用版本：")
        for v in vers:
            print(f"     v{v['version']:<4} {v.get('images', '?')} 张  {v['name']}")
        if args.list_versions:
            return 0
        version = str(vers[-1]["version"])
        print(f"\n>> ⚠ 未指定版本，将使用最新的 v{version}")
        print(">>   注意：Roboflow 的「最新」往往不是「最合适」。同一项目的不同版本")
        print(">>   可能是单类/多类、全图/裁剪、有增广/无增广，差别很大。")
        print(">>   选版本看三点：① 多类（保住细粒度标注）② 全图（裁剪版做不了定位）")
        print(">>   ③ 无增广（增广副本不带新信息，还可能跨 split 泄漏）")
        print(">>   确定要哪个版本就用 --version <n> 指定。\n")

    # ---- 取导出链接 ----
    url = f"{API}/{ws}/{proj}/{version}/{args.format}?api_key={key}"
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    zpath = os.path.join(out, "_download.zip")

    print(f">> 请求导出 {ws}/{proj} v{version}（{args.format}）")
    try:
        meta = _get_json(url, timeout=180)
    except Exception as e:
        print(f"请求失败：{_explain(e, ws, proj)}")
        return 1

    link = (meta.get("export") or {}).get("link") or meta.get("link")
    if not link:
        print(f"返回里没有下载链接。原始返回：{json.dumps(meta, ensure_ascii=False)[:400]}\n"
              f"若提示还在打包（generating），等 1~2 分钟重跑本命令即可。")
        return 1

    print(">> 下载 zip")
    try:
        urllib.request.urlretrieve(link, zpath)
    except Exception as e:
        print(f"下载失败：{e}\n"
              f"兜底办法：在浏览器打开 universe.roboflow.com/{ws}/{proj}，"
              f"点 Download Dataset → 选 COCO → 下到本地后解压到：\n  {out}")
        return 1

    print(">> 解压")
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    except Exception as e:
        print(f"解压失败：{e}（下到的可能是一个错误页而不是 zip）")
        return 1
    os.remove(zpath)

    got = [d for d in ("train", "valid", "test")
           if os.path.exists(os.path.join(out, d, "_annotations.coco.json"))]
    print(f">> 完成：{out}")
    print(f"   含标注的划分：{got or '（没找到 _annotations.coco.json，检查导出格式是否为 COCO）'}")

    # 下完立刻把类别和数量亮出来 —— 版本选错时这里一眼能看出来
    # （比如只有一个 "defect" 类，说明拿到的是 single class 版本）
    for d in got:
        with open(os.path.join(out, d, "_annotations.coco.json"),
                  encoding="utf-8") as f:
            coco = json.load(f)
        names = [c["name"] for c in coco.get("categories", [])
                 if c["name"].lower() not in ("background", "none")]
        print(f"   {d}: {len(coco.get('images', []))} 张，"
              f"{len(coco.get('annotations', []))} 个标注框")
        print(f"        类别 {names}")
        if len(names) <= 1:
            print("        ⚠ 只有一个类别 —— 很可能拿到了 single class 版本，"
                  "细粒度缺陷类型全丢了。用 --list-versions 换一个多类版本。")
    print("   接着把 configs/datasets.yaml 里对应条目的 enabled 改成 true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
