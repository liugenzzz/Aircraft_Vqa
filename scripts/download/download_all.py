#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一站式取数：能自动下的直接下，不能自动下的生成一份待办清单。

    python scripts/download/download_all.py --data-root ~/data/raw          # 下能下的
    python scripts/download/download_all.py --list                          # 只看清单
    python scripts/download/download_all.py --only visa synthetic
    python scripts/download/download_all.py --data-root ~/data/raw --check  # 只体检

四种获取方式：
  auto     直链，脚本直接下（VisA）
  local    本地生成，不需要网络（合成数据）
  keyed    有直链但要 API Key / 凭据（Roboflow、Kaggle），设了环境变量就自动下
  manual   必须人工同意条款或申请（MVTec、Real-IAD、IEEE DataPort）

manual 的会写进 <data_root>/../MANUAL_DOWNLOADS.md，里面写清了：去哪下、
下完放到哪个路径、怎么解压、然后改哪个配置项。你照着放好文件，
再跑一次本脚本的 --check 就能确认放对没有。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------- 数据源登记
# dest 是相对 data_root 的目录；probe 是判断"已经下好了"的标志性子路径。
SOURCES = [
    {
        "name": "visa",
        "mode": "auto",
        "zh": "VisA（10,821 图 / 12 类 / 像素级 mask）",
        "use": "定位能力主力；PCB 缺件语义与'少一颗螺丝'同构",
        "dest": "VisA",
        "probe": "split_csv/1cls.csv",
        "size": "1.9 GB",
        "license": "CC BY 4.0（可商用，需署名）",
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "page": "https://registry.opendata.aws/visa/",
        "archive": "VisA.tar",
        "config_key": "visa",
        "config_names": ["visa"],
    },
    {
        "name": "synthetic",
        "mode": "local",
        "zh": "合成蒙皮铆钉阵列 + 紧固件特写",
        "use": "8 类缺陷全覆盖，缺陷位置绝对精确，给定位能力打底",
        "dest": "synthetic",
        "probe": "panel/train/_annotations.coco.json",
        "size": "4000 张约 600MB",
        "license": "自有合成数据（可商用）",
        "cmd": [sys.executable, os.path.join(REPO, "scripts", "make_demo_data.py"),
                "--out", "{dest}", "-n", "{n_synth}"],
        "config_key": "synthetic_panel / synthetic_closeup",
        "config_names": ["synthetic_panel", "synthetic_closeup"],
    },
    {
        "name": "aircraft_skin_defects",
        "mode": "keyed",
        "zh": "Roboflow · DDIISc aircraft_skin_defects",
        "use": "★ 类别含 Missing-head（紧固件缺失），唯一直接命中需求的航空标注",
        "dest": "roboflow/aircraft_skin_defects",
        "probe": "train/_annotations.coco.json",
        "size": "约 100 MB",
        "license": "见 Roboflow 项目页（多为 CC BY 4.0）",
        "env": "ROBOFLOW_API_KEY",
        "env_how": "https://app.roboflow.com → Settings → Roboflow API → Private API Key",
        "page": "https://universe.roboflow.com/ddiisc/aircraft_skin_defects",
        "rf": {"workspace": "ddiisc", "project": "aircraft_skin_defects",
               "version": "latest"},
        "config_key": "aircraft_skin_defects",
        "config_names": ["aircraft_skin_defects"],
    },
    {
        "name": "uts_aircraft_defect",
        "mode": "keyed",
        "zh": "Roboflow · UTS aircraft-defect-detection（9,352 图）",
        "use": "目前能拿到的最大航空缺陷检测集，航空域外观主力",
        "dest": "roboflow/aircraft-defect-detection",
        "probe": "train/_annotations.coco.json",
        "size": "约 1 GB",
        "license": "见 Roboflow 项目页",
        "env": "ROBOFLOW_API_KEY",
        "env_how": "同上",
        "page": "https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection",
        "rf": {"workspace": "university-of-technology-sydney-21uto",
               "project": "aircraft-defect-detection", "version": "latest"},
        "config_key": "uts_aircraft_defect",
        "config_names": ["uts_aircraft_defect"],
    },
    {
        "name": "npu_bolt",
        "mode": "keyed",
        "rec": "暂不建议 —— 标的是螺栓这个物体，不是缺陷",
        "zh": "NPU-BOLT（337 图，自然场景螺栓 4 类）",
        "use": "只适合做计数与指代定位；现有 coco adapter 会把每颗正常螺栓"
               "当成一处缺陷，需要单独写 adapter",
        "dest": "NPU-BOLT",
        "probe": "train/_annotations.coco.json",
        "size": "约 200 MB",
        "license": "学术开放",
        "env": "KAGGLE_USERNAME + KAGGLE_KEY",
        "env_how": "https://www.kaggle.com/settings → Create New Token，"
                   "把 kaggle.json 放到 ~/.kaggle/ 或设这两个环境变量",
        "kaggle": "xiaoqian0/npu-bolt-dataset",
        "page": "https://arxiv.org/pdf/2205.11191",
        "config_key": "npu_bolt",
        "config_names": [],
        "note": "Kaggle 上的数据集 slug 可能随作者调整，下不到就按 page 里的论文找最新链接。",
    },
    {
        "name": "mvtec_ad",
        "mode": "manual",
        "zh": "MVTec AD（screw / metal_nut 子集）",
        "use": "★ 螺纹损伤、头部划伤 + 像素级 mask，缺陷定位的金标准",
        "dest": "mvtec_anomaly_detection",
        "probe": "screw/train/good",
        "size": "4.9 GB（全集）",
        "license": "CC BY-NC-SA 4.0 —— 禁止商用，只能用于研究",
        "page": "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads",
        "steps": [
            "打开上面的下载页，填表并同意 CC BY-NC-SA 4.0 条款",
            "下载 mvtec_anomaly_detection.tar.xz",
            "mkdir -p {full_dest} && tar -xf mvtec_anomaly_detection.tar.xz -C {full_dest}",
            "确认 {full_dest}/screw/train/good 下有图",
        ],
        "config_key": "mvtec_ad_screw",
        "config_names": ["mvtec_ad_screw"],
    },
    {
        "name": "mvtec_loco",
        "mode": "manual",
        "zh": "MVTec LOCO AD（screw_bag 子集）",
        "use": "★ 缺件 / 数量错 / 规格错 —— 最贴近'螺丝缺失'语义的公开标注",
        "dest": "mvtec_loco_anomaly_detection",
        "probe": "screw_bag/train/good",
        "size": "6.5 GB（全集）",
        "license": "CC BY-NC-SA 4.0 —— 禁止商用",
        "page": "https://www.mvtec.com/company/research/datasets/mvtec-loco/downloads",
        "steps": [
            "打开下载页，填表并同意条款",
            "下载 mvtec_loco_anomaly_detection.tar.xz",
            "mkdir -p {full_dest} && tar -xf mvtec_loco_anomaly_detection.tar.xz -C {full_dest}",
            "可选：在 {full_dest}/screw_bag/ 放一个 defect_names.json"
            "（{{\"图片stem\": \"screw_too_long\"}}），能拿到细粒度缺陷类型",
        ],
        "config_key": "mvtec_loco_screwbag",
        "config_names": ["mvtec_loco_screwbag"],
    },
    {
        "name": "corrosion_cs_vt",
        "mode": "auto",
        "zh": "Virginia Tech 腐蚀分级分割集（440 图 / 4 级）",
        "use": "★ 唯一带'锈蚀严重度分级 + 像素定位'的公开集",
        "dest": "corrosion_cs",
        "probe": "images",
        "size": "约 300 MB",
        "license": "学术开放，商用需确认",
        "page": "https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663",
        "url": "https://data.lib.vt.edu/ndownloader/articles/16624663/versions/1",
        "archive": "corrosion_cs.zip",
        "note": "figshare 的直链模式（ndownloader）没在本仓库验证过；"
                "下不动就按下面的步骤走浏览器，效果一样。",
        "steps": [
            "在上面的页面点 Download all，拿到 zip（完全公开，不用填表）",
            "解压后把原图目录改名为 images/、标注 mask 目录改名为 masks/，"
            "最终形如 {full_dest}/images/*.jpg 与 {full_dest}/masks/*.png",
            "核对 mask 的像素取值："
            "python scripts/inspect_masks.py --masks {full_dest}/masks",
            "把实际取值填进 configs/datasets.yaml 里 corrosion_cs_vt 的 class_map。"
            "该条目已设 grade_type: corrosion —— good/fair/poor/severe 会被当作"
            "腐蚀的**有序等级**，映射成程度词（轻微锈蚀/点蚀起皮/层状剥落/截面损失）"
            "并直接决定严重度与处置方案",
            "等级顺序弄反了比没有等级更糟，务必用 scripts/visualize.py 抽查几张核对",
        ],
        "config_key": "corrosion_cs_vt",
        "config_names": ["corrosion_cs_vt"],
    },
    {
        "name": "real_iad",
        "mode": "manual",
        "rec": "建议（取 512 + 几个类即可，别下全量）",
        "zh": "Real-IAD（151,050 图 / 30 类 / 每件 5 视角）",
        "use": "★ 每个视角单独标注 —— 缺陷看不见的那个角度标签就是 good，"
               "'换角度确认'这件事的监督信号是现成的",
        "dest": "Real-IAD",
        "probe": "realiad_jsons",
        "size": "raw 全量 200GB；取 512 版 + 5 个类只要几 GB",
        "license": "CC BY-NC-SA 4.0 —— 禁止商用（与 MVTec 同）",
        "page": "https://huggingface.co/Real-IAD",
        "steps": [
            "注册 HuggingFace 账号，到组织页 Real-IAD 申请访问："
            "填姓名/单位/用途并同意 CC BY-NC-SA 4.0；部分仓库人工审核，可能等一两天",
            "**按分辨率和物体分包，不要下 realiad_raw**："
            "先拿 realiad_512（或 256）里三五个金属/紧固类物体的 ZIP，几 GB 就能起步",
            "元数据包 realiad_jsons 必须一起下（另有 _sv 单视角、_fuiad 含噪设定两套变体，"
            "先用基础版）",
            "解压成 {full_dest}/realiad_jsons/*.json 与 {full_dest}/realiad_512/<类名>/",
            "在 configs/datasets.yaml 的 real_iad 条目里把 image_dir 改成 realiad_512，"
            "并用 categories 限定你实际下了的那几个类",
        ],
        "note": "训练集 36,465 张纯正常图，测试集 114,585 张混合。"
                "同一物体不同视角标签不同，这正是'单视角结论不可靠'的天然监督。"
                "另有 Real-IAD D³（CVPR 2025，RGB + 光度立体 + 微米级点云），"
                "后续想走多模态可以看。",
        "config_key": "real_iad",
        "config_names": ["real_iad"],
    },
    {
        "name": "aircraft_fuselage_det2023",
        "mode": "manual",
        "rec": "推荐（航空域真实数据，别的集给不了）",
        "zh": "Aircraft_Fuselage_DET2023（5,601 图机身缺陷）",
        "use": "★ 不同光照下实拍机身不同部位的四类表面缺陷，"
               "外加一个无标注池可做半监督",
        "dest": "roboflow/aircraft_fuselage_det2023",
        "probe": "train",
        "size": "未公布",
        "license": "IEEE DataPort 条款；引用需写作者那篇半监督论文",
        "page": "https://ieee-dataport.org/documents/aircraftfuselagedet2023-aircraft-fuselage-defect-detection-dataset",
        "steps": [
            "IEEE DataPort 的条目分开放获取与订阅者专享两种，登录后才看得到按钮。"
            "多数高校图书馆有 IEEE 机构订阅 —— 走校园网 IP 或图书馆远程访问"
            "（VPN / CARSI）进去试；卡住就直接问图书馆的电子资源咨询，比自己折腾快",
            "下载包里是一个 Aircraft_Fuselage_DET2023 文件夹，"
            "前三个子目录是同一批图的 COCO / VOC / YOLO 三种标注，第四个是无标注图像池",
            "**用 COCO 那份**，整理成 {full_dest}/train|valid|test/_annotations.coco.json；"
            "无标注池先放一边（要做半监督或自训练时再用，作者刻意留的）",
            "在 configs/datasets.yaml 里照着 aircraft_skin_defects 条目新增一个"
            " adapter: coco 的条目指向 {full_dest}",
            "引用写《A Semi-Supervised Aircraft Fuselage Defect Detection Network with "
            "Dynamic Attention and Class-aware Adaptive Pseudo-Label Assignment》，"
            "页面上明确要求",
        ],
        "config_key": "（需自行新增条目）",
    },
    {
        "name": "bolt_rotation",
        "mode": "manual",
        "rec": "暂不建议 —— 单一装置 + 实验室光照，视觉域迁不过去",
        "zh": "Bolt Rotation Dataset（1,112 图，带转角标注）",
        "use": "价值在标签不在图像：公开数据里唯一把松动量化成连续角度的",
        "dest": "bolt_rotation",
        "probe": "images",
        "size": "约 500 MB",
        "license": "Data in Brief 开放获取（DOI 10.1016/j.dib.2025.111788）",
        "page": "https://www.sciencedirect.com/science/article/pii/S2352340925005153",
        "note": "自制装置上五颗 M20 螺栓、三颗逐步逆时针旋转，单反 + 四个焦距、"
                "多机位，实验室受控光照。跟 MVTec 是同一类短板：视觉域几乎迁不到"
                "真实航空场景。只适合当'松动是连续量而非二值状态'的概念监督源，"
                "别指望拿它训出能用的检测器 —— 合成特写场景已经能覆盖松动的外观。",
        "steps": [
            "**务必从论文的 Data Availability 一节点链接过去**：同一批作者在 Figshare 上"
            "还有一个名字很像的《A dataset depicting simulated bolt rotation》（9.88 GB，"
            "是仿真的），直接搜名字容易拿错。要的是实拍那 1,112 张",
            "解压到 {full_dest}",
            "转角标注是 CSV，需要写一个 adapter 把角度阈值映射成 fastener_loose；"
            "本仓库暂未提供",
        ],
        "config_key": "（需自行新增条目 + adapter）",
    },
]

MODE_ZH = {"auto": "直链自动下", "local": "本地生成", "keyed": "需凭据",
           "manual": "需人工获取"}


# ---------------------------------------------------------------- 工具
def _present(root: str, src: dict) -> bool:
    p = os.path.join(root, src["dest"], src.get("probe", ""))
    return os.path.exists(p)


def _w(text: str) -> int:
    """显示宽度：中文算 2 列。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度截断并右填充。"""
    out = ""
    for c in text:
        if _w(out + c) > width:
            break
        out += c
    return out + " " * (width - _w(out))


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _download(url: str, out: str) -> bool:
    """带断点续传的下载；curl 可用就用 curl，否则退回 urllib。"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
               "-C", "-", "-o", out, url]
        return subprocess.call(cmd) == 0
    try:
        urllib.request.urlretrieve(url, out)
        return True
    except Exception as e:
        print(f"  下载失败：{e}")
        return False


def _extract(archive: str, dest: str) -> bool:
    os.makedirs(dest, exist_ok=True)
    try:
        if archive.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz")):
            import tarfile
            with tarfile.open(archive) as t:
                t.extractall(dest)
        elif archive.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            print(f"  不认识的压缩格式：{archive}")
            return False
        return True
    except Exception as e:
        print(f"  解压失败：{e}")
        return False


# ---------------------------------------------------------------- 各模式
def do_auto(root: str, src: dict, keep: bool) -> bool:
    dest = os.path.join(root, src["dest"])
    archive = os.path.join(root, src["archive"])
    print(f"  下载 {src['size']} -> {archive}")
    # curl -C - 会自动续传，所以重复调用是安全的
    if not _download(src["url"], archive):
        return False
    print(f"  解压 -> {dest}")
    if not _extract(archive, dest):
        return False
    if not keep:
        os.remove(archive)
    if _present(root, src):
        return True
    # 压缩包里的目录结构和 adapter 期望的对不上，说清楚差在哪，别只报一个失败
    print(f"  解压完成，但没找到期望的 {src['probe']}")
    try:
        top = sorted(os.listdir(dest))[:12]
        print(f"  解压出来的顶层内容：{top}")
    except OSError:
        pass
    print(f"  需要整理成：{os.path.join(dest, src['probe'])}")
    return False


def do_local(root: str, src: dict, n_synth: int) -> bool:
    dest = os.path.join(root, src["dest"])
    cmd = [c.format(dest=dest, n_synth=str(n_synth)) for c in src["cmd"]]
    print(f"  本地生成 -> {dest}")
    return subprocess.call(cmd) == 0 and _present(root, src)


def do_roboflow(root: str, src: dict) -> bool:
    dest = os.path.join(root, src["dest"])
    cmd = [sys.executable, os.path.join(HERE, "download_roboflow.py"),
           "--workspace", src["rf"]["workspace"],
           "--project", src["rf"]["project"],
           "--version", src["rf"]["version"], "--out", dest]
    return subprocess.call(cmd) == 0 and _present(root, src)


def do_kaggle(root: str, src: dict) -> bool:
    dest = os.path.join(root, src["dest"])
    if not shutil.which("kaggle"):
        print("  未安装 kaggle CLI：pip install kaggle")
        return False
    os.makedirs(dest, exist_ok=True)
    rc = subprocess.call(["kaggle", "datasets", "download", "-d", src["kaggle"],
                          "-p", dest, "--unzip"])
    if rc != 0:
        print(f"  kaggle 下载失败（slug 可能已变，见 {src.get('page')}）")
        return False
    return True


# ---------------------------------------------------------------- 清单
def write_manual_manifest(root: str, srcs: list, out_path: str) -> None:
    lines = [
        "# 需要手动获取的数据集",
        "",
        "这些数据集要么需要先同意授权条款，要么需要提交申请，脚本没法无人值守下载。",
        "请按下面每一条的步骤下好，放到指定路径，然后跑：",
        "",
        "```bash",
        f"python scripts/download/download_all.py --data-root {root} --check",
        "```",
        "",
        "确认都变成 ✅ 之后，再把 `configs/datasets.yaml` 里对应条目的",
        "`enabled` 改成 `true`，然后重新跑 ingest 与 build。",
        "",
        "---",
        "",
    ]
    for s in srcs:
        full = os.path.join(root, s["dest"])
        lines += [
            f"## {s['zh']}",
            "",
            f"- **建议**：{s.get('rec', '推荐')}",
            f"- **用途**：{s['use']}",
            f"- **体量**：{s.get('size', '未知')}",
            f"- **授权**：{s['license']}",
            f"- **下载页**：{s['page']}",
            f"- **放到**：`{full}`",
            f"- **对应配置项**：`{s['config_key']}`",
            "",
            "步骤：",
            "",
        ]
        for i, st in enumerate(s.get("steps", []), 1):
            lines.append(f"{i}. {st.format(full_dest=full)}")
        if s.get("note"):
            lines += ["", f"> {s['note']}"]
        lines += ["", "---", ""]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def enable_in_config(config_path: str, names: list) -> list:
    """把 datasets.yaml 里这些条目的 enabled 改成 true。

    只动匹配到的 `- name:` 块里的那一行 enabled，其他内容一字不改
    （所以不走 yaml.dump，避免把注释和格式全洗掉）。
    """
    if not names or not os.path.exists(config_path):
        return []
    with open(config_path, encoding="utf-8") as f:
        lines = f.readlines()

    changed, cur = [], None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("- name:"):
            cur = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("enabled:") and cur in names:
            if "false" in stripped:
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}enabled: true\n"
                changed.append(cur)
    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def print_table(root: str, srcs: list) -> None:
    print("\n" + _pad("数据集", 40) + _pad("获取方式", 12)
          + _pad("状态", 8) + _pad("体量", 18) + _pad("建议", 34) + "用途")
    print("-" * 160)
    for s in srcs:
        ok = _present(root, s)
        print(_pad(s["zh"], 40) + _pad(MODE_ZH[s["mode"]], 12)
              + _pad("已有" if ok else "缺", 8)
              + _pad(s.get("size", "-"), 18)
              + _pad(s.get("rec", "推荐"), 34) + _pad(s["use"], 60))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/data/raw")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true", help="只列清单，不下载")
    ap.add_argument("--check", action="store_true", help="只体检已有数据")
    ap.add_argument("--n-synth", type=int, default=4000, help="合成数据张数")
    ap.add_argument("--keep-archive", action="store_true", help="解压后保留压缩包")
    ap.add_argument("--enable-config", action="store_true",
                    help="下好之后自动把 configs/datasets.yaml 里对应条目的 "
                         "enabled 改成 true")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "datasets.yaml"))
    ap.add_argument("--manifest", default=None,
                    help="手动清单输出路径（默认 <data_root>/MANUAL_DOWNLOADS.md）")
    args = ap.parse_args()

    root = os.path.expanduser(args.data_root)
    os.makedirs(root, exist_ok=True)
    srcs = [s for s in SOURCES if not args.only or s["name"] in args.only]
    manifest = args.manifest or os.path.join(root, "MANUAL_DOWNLOADS.md")

    if args.list or args.check:
        print_table(root, srcs)
        if args.check and args.enable_config:
            ready = [n for s_ in srcs if _present(root, s_)
                     for n in s_.get("config_names", [])]
            done_cfg = enable_in_config(args.config, ready)
            if done_cfg:
                print(f"\n已在 {args.config} 里启用：{'、'.join(done_cfg)}")
        manual = [s for s in srcs if s["mode"] == "manual" and not _present(root, s)]
        if manual:
            write_manual_manifest(root, manual, manifest)
            print(f"\n还差 {len(manual)} 个需要人工获取，清单已写到：\n  {manifest}")
        else:
            print("\n人工获取的部分都齐了。")
        return 0

    done, failed, manual = [], [], []
    for s in srcs:
        print(f"\n=== {s['zh']}  [{MODE_ZH[s['mode']]}]")
        if _present(root, s):
            print("  已存在，跳过")
            done.append(s["name"])
            continue

        if s["mode"] == "auto":
            ok = do_auto(root, s, args.keep_archive)
            if not ok and s.get("steps"):
                print("  自动下载没成功，转为人工获取")
                manual.append(s)
                continue
        elif s["mode"] == "local":
            ok = do_local(root, s, args.n_synth)
        elif s["mode"] == "keyed":
            envs = [e.strip() for e in s["env"].replace("+", ",").split(",")]
            missing = [e for e in envs if not os.environ.get(e)]
            if missing:
                print(f"  缺凭据 {missing}，跳过。获取方式：{s['env_how']}")
                failed.append(s["name"])
                continue
            ok = do_roboflow(root, s) if "rf" in s else do_kaggle(root, s)
        else:
            print(f"  需人工获取：{s['page']}")
            manual.append(s)
            continue

        (done if ok else failed).append(s["name"])
        print("  ✅ 完成" if ok else "  ❌ 未完成")

    if manual:
        write_manual_manifest(root, manual, manifest)

    if args.enable_config:
        ready = [n for s_ in srcs if _present(root, s_)
                 for n in s_.get("config_names", [])]
        done_cfg = enable_in_config(args.config, ready)
        if done_cfg:
            print(f"\n已在 {args.config} 里启用：{'、'.join(done_cfg)}")

    print("\n" + "=" * 60)
    print(f"已就绪 {len(done)}：{'、'.join(done) or '无'}")
    if failed:
        print(f"未成功 {len(failed)}：{'、'.join(failed)}（多半是缺凭据，见上面提示）")
    if manual:
        print(f"需人工获取 {len(manual)} 个，清单：{manifest}")
        for s in manual:
            print(f"  · {s['zh']}\n      {s['page']}\n      放到 "
                  f"{os.path.join(root, s['dest'])}")
    print("\n下一步：")
    print(f"  python scripts/ingest.py --data-root {root}")
    print("  python scripts/build_vqa.py --out data/vqa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
