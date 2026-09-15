# -*- coding: utf-8 -*-
"""照官方目录结构造等价夹具，用来在没有真实数据时验证 adapter。

MVTec AD / LOCO / Real-IAD 这三个 adapter 是照公开文档写的，
没跑过真实数据。夹具把目录结构、命名规则、mask 组织方式都复刻一遍，
结构性 bug 能在这里暴露，不用等下完 10GB 才发现读不出来。
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image


def _img(path: str, size=(256, 256), color=(120, 120, 120)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _mask(path: str, size=(256, 256), boxes=((60, 60, 120, 120),)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = np.zeros((size[1], size[0]), np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2, x1:x2] = 255
    Image.fromarray(m).save(path)


def make_mvtec_ad(root: str) -> str:
    """MVTec AD 按类别分包解压后的结构。

        <root>/screw/train/good/000.png
        <root>/screw/test/good/000.png
        <root>/screw/test/scratch_head/000.png
        <root>/screw/ground_truth/scratch_head/000_mask.png
    """
    for cat, defects in (("screw", ["scratch_head", "thread_top",
                                    "manipulated_front"]),
                         ("metal_nut", ["scratch", "bent"])):
        for i in range(3):
            _img(f"{root}/{cat}/train/good/{i:03d}.png")
        for i in range(2):
            _img(f"{root}/{cat}/test/good/{i:03d}.png")
        for d in defects:
            for i in range(2):
                _img(f"{root}/{cat}/test/{d}/{i:03d}.png")
                _mask(f"{root}/{cat}/ground_truth/{d}/{i:03d}_mask.png")
    return root


def make_mvtec_loco(root: str) -> str:
    """MVTec LOCO 的结构：ground_truth 下每张图是一个**目录**，内含多张实例 mask。

    这是全仓库最容易写错的一处 —— 别的数据集都是一图一 mask。
    """
    cat = "screw_bag"
    for i in range(3):
        _img(f"{root}/{cat}/train/good/{i:03d}.png")
    _img(f"{root}/{cat}/validation/good/000.png")
    _img(f"{root}/{cat}/test/good/000.png")
    for kind, n_inst in (("logical_anomalies", 2), ("structural_anomalies", 1)):
        for i in range(2):
            _img(f"{root}/{cat}/test/{kind}/{i:03d}.png")
            for k in range(n_inst):           # 每张图一个目录，里面 N 张 mask
                _mask(f"{root}/{cat}/ground_truth/{kind}/{i:03d}/{k:03d}.png",
                      boxes=((40 + k * 70, 40, 90 + k * 70, 90),))
    # 可选的细粒度类型映射
    with open(f"{root}/{cat}/defect_names.json", "w", encoding="utf-8") as f:
        json.dump({"000": "screw_too_long", "001": "missing_nut"}, f)
    return root


def make_real_iad(root: str, shape: str = "train_test") -> str:
    """Real-IAD：realiad_jsons/<类>.json + realiad_512/<类>/...

    不同发布版本的 json 顶层键和字段名有出入，shape 参数覆盖三种形态。
    """
    cat = "switch"
    imgs = {
        "ok": f"{cat}/C1/0001_OK.jpg",
        "ng": f"{cat}/C2/0002_NG.jpg",
        "ng2": f"{cat}/C3/0003_NG.jpg",      # 同一物体的另一视角
    }
    for rel in imgs.values():
        _img(f"{root}/realiad_512/{rel}")
    _mask(f"{root}/realiad_512/{cat}/C2/0002_NG_mask.png")

    ok = {"image_path": imgs["ok"], "anomaly_class": "OK", "view": "C1",
          "category": cat}
    ng = {"image_path": imgs["ng"], "mask_path": f"{cat}/C2/0002_NG_mask.png",
          "anomaly_class": "scratch", "view": "C2", "category": cat}
    ng2 = {"image_path": imgs["ng2"], "anomaly_class": "OK", "view": "C3",
           "category": cat}          # 有缺陷的物体、但这个角度看不见 -> 标 OK

    if shape == "no_cat_prefix":
        # 有的发布版本里 json 的相对路径不含类别名，要靠
        # <img_root>/<cat>/<rel> 兜底才找得到。图片和 mask 必须同时兜底 ——
        # 只兜图片不兜 mask，样本数看着正常，掩码却全部落空。
        strip = lambda p: p.split("/", 1)[1]
        ok = dict(ok, image_path=strip(ok["image_path"]))
        ng = dict(ng, image_path=strip(ng["image_path"]),
                  mask_path=strip(ng["mask_path"]))
        ng2 = dict(ng2, image_path=strip(ng2["image_path"]))
        data = {"train": [ok], "test": [ng, ng2]}
    elif shape == "train_test":
        data = {"train": [ok], "test": [ng, ng2]}
    elif shape == "other_keys":      # 顶层键不叫 train/test
        data = {"meta": {"version": 1}, "samples": [ok, ng, ng2]}
    else:                            # 裸列表
        data = [ok, ng, ng2]

    os.makedirs(f"{root}/realiad_jsons", exist_ok=True)
    with open(f"{root}/realiad_jsons/{cat}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    return root


def make_corrosion_cs(root: str, values=(1, 2, 3, 4)) -> str:
    """VT 腐蚀集整理后的结构：images/ + masks/，mask 像素值即等级。"""
    for i, v in enumerate(values):
        _img(f"{root}/images/corr{i}.jpg", size=(512, 512))
        m = np.zeros((512, 512), np.uint8)
        m[100:300, 100 + i * 20:320 + i * 20] = v
        os.makedirs(f"{root}/masks", exist_ok=True)
        Image.fromarray(m).save(f"{root}/masks/corr{i}.png")
    return root


def make_coco(root: str, splits=("train", "valid", "test")) -> str:
    """Roboflow COCO 导出结构。"""
    cats = [{"id": 1, "name": "Missing-head"}, {"id": 2, "name": "Crack"},
            {"id": 3, "name": "Dent"}, {"id": 4, "name": "Paint-peel-off"}]
    for sp in splits:
        images, anns, aid = [], [], 1
        for i in range(len(cats) + 1):
            fn = f"img{i}.jpg"
            _img(f"{root}/{sp}/{fn}", size=(640, 480))
            images.append({"id": i, "file_name": fn, "width": 640, "height": 480})
            if i > 0:                     # 第 0 张是无标注的正常图
                anns.append({"id": aid, "image_id": i,
                             "category_id": cats[i - 1]["id"],   # 每类都覆盖到
                             "bbox": [50 * i, 40 * i, 60, 60], "area": 3600,
                             "iscrowd": 0})
                aid += 1
        with open(f"{root}/{sp}/_annotations.coco.json", "w",
                  encoding="utf-8") as f:
            json.dump({"images": images, "annotations": anns,
                       "categories": cats}, f)
    return root
