# -*- coding: utf-8 -*-
"""对着各数据集的官方目录结构验证 adapter。

这几个 adapter 是照公开文档写的，没跑过真实数据。夹具复刻目录结构、
命名规则与 mask 组织方式，结构性 bug 在这里暴露，不用等下完 10GB。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from make_fixtures import (make_coco, make_corrosion_cs, make_mvtec_ad,
                           make_mvtec_loco, make_real_iad)

from aircraft_vqa.adapters import build_adapter
from aircraft_vqa.taxonomy import get_taxonomy

TAX = get_taxonomy(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "configs", "taxonomy.yaml"))


def _run(spec, tax=TAX):
    return list(build_adapter(spec, taxonomy=tax).iter_samples())


# ------------------------------------------------------------------ MVTec AD
def test_mvtec_ad_reads_categories_defects_and_masks(tmp_path):
    root = make_mvtec_ad(str(tmp_path / "MVTec_AD"))
    got = _run({"adapter": "mvtec_ad", "root": root, "name": "mvtec_ad_screw",
                "license": "CC BY-NC-SA 4.0", "commercial_ok": False,
                "categories": ["screw", "metal_nut"],
                "object_hint": {"screw": "screw", "metal_nut": "nut"}})
    assert len(got) == (3 + 2 + 3 * 2) + (3 + 2 + 2 * 2)      # 两个类别
    anom = [s for s in got if s.is_anomalous]
    assert len(anom) == 3 * 2 + 2 * 2
    # 缺陷目录名要正确映射到 canonical 类型
    types = {d.type for s in anom for d in s.defects}
    assert {"scratch", "thread_damage"} <= types
    # mask 必须被读出来变成框
    assert all(s.localizable_defects() for s in anom), "ground_truth mask 没读到"
    assert {s.object_name for s in got} == {"screw", "nut"}


def test_mvtec_ad_good_folder_is_not_a_defect(tmp_path):
    root = make_mvtec_ad(str(tmp_path / "m"))
    got = _run({"adapter": "mvtec_ad", "root": root, "name": "m",
                "license": "x", "commercial_ok": False, "categories": ["screw"]})
    for s in got:
        if "/good/" in s.sample_id:
            assert not s.is_anomalous and not s.defects


# ------------------------------------------------------------------ LOCO
def test_mvtec_loco_reads_per_image_mask_directories(tmp_path):
    """LOCO 的 ground_truth 下每张图是一个目录、内含多张实例 mask ——
    别的数据集都是一图一 mask，这里最容易写错。"""
    root = make_mvtec_loco(str(tmp_path / "MVTec_LOCO_AD"))
    got = _run({"adapter": "mvtec_loco", "root": root,
                "name": "mvtec_loco_screwbag", "license": "CC BY-NC-SA 4.0",
                "commercial_ok": False, "categories": ["screw_bag"],
                "object_hint": {"screw_bag": "screw_bag"}})
    assert len(got) == 3 + 1 + 1 + 2 + 2
    logical = [s for s in got if "logical_anomalies" in s.sample_id]
    assert len(logical) == 2
    # 每张 logical 图有 2 个实例 mask -> 至少 2 个框
    for s in logical:
        assert len(s.localizable_defects()) >= 2, \
            f"多实例 mask 没全读到：{[d.bbox for d in s.defects]}"
        assert s.meta["n_instance_masks"] == 2
    struct = [s for s in got if "structural_anomalies" in s.sample_id]
    assert all(len(x.localizable_defects()) >= 1 for x in struct)


def test_mvtec_loco_uses_defect_names_override(tmp_path):
    """原始发布只有 logical/structural，放 defect_names.json 能拿到细粒度类型。"""
    root = make_mvtec_loco(str(tmp_path / "m"))
    got = _run({"adapter": "mvtec_loco", "root": root, "name": "m",
                "license": "x", "commercial_ok": False,
                "categories": ["screw_bag"]})
    types = {d.type for s in got if s.is_anomalous for d in s.defects}
    assert "fastener_wrong_spec" in types, f"defect_names.json 没生效：{types}"
    assert "fastener_missing" in types


def test_mvtec_loco_validation_split_is_read(tmp_path):
    root = make_mvtec_loco(str(tmp_path / "m"))
    got = _run({"adapter": "mvtec_loco", "root": root, "name": "m",
                "license": "x", "commercial_ok": False})
    assert any(s.split == "validation" for s in got)


# ------------------------------------------------------------------ Real-IAD
@pytest.mark.parametrize("shape", ["train_test", "other_keys", "bare_list"])
def test_real_iad_handles_json_shape_variants(tmp_path, shape):
    root = make_real_iad(str(tmp_path / f"Real-IAD-{shape}"), shape)
    got = _run({"adapter": "real_iad", "root": root, "name": "real_iad",
                "license": "CC BY-NC-SA 4.0", "commercial_ok": False,
                "json_dir": "realiad_jsons", "image_dir": "realiad_512",
                "categories": ["switch"]})
    assert len(got) == 3, f"{shape} 形态没读全"
    assert {s.meta.get("view") for s in got} == {"C1", "C2", "C3"}


def test_real_iad_preserves_per_view_labels(tmp_path):
    """Real-IAD 最有价值的特性：同一物体换个角度看不见缺陷，标签就是 OK。
    这条监督信号不能在 adapter 层被抹平。"""
    root = make_real_iad(str(tmp_path / "Real-IAD"))
    got = _run({"adapter": "real_iad", "root": root, "name": "real_iad",
                "license": "x", "commercial_ok": False,
                "json_dir": "realiad_jsons", "image_dir": "realiad_512"})
    by_view = {s.meta["view"]: s.label for s in got}
    assert by_view == {"C1": "normal", "C2": "anomalous", "C3": "normal"}


def test_real_iad_reads_mask_into_boxes(tmp_path):
    root = make_real_iad(str(tmp_path / "Real-IAD"))
    got = _run({"adapter": "real_iad", "root": root, "name": "real_iad",
                "license": "x", "commercial_ok": False,
                "json_dir": "realiad_jsons", "image_dir": "realiad_512"})
    ng = next(s for s in got if s.is_anomalous)
    assert ng.localizable_defects(), "mask_path 没被读成框"
    assert ng.defects[0].type == "scratch"


# ------------------------------------------------------------------ 腐蚀分级
def test_corrosion_grade_mapping(tmp_path):
    root = make_corrosion_cs(str(tmp_path / "corrosion_cs"))
    got = _run({"adapter": "mask_seg", "root": root, "name": "corrosion_cs_vt",
                "license": "学术开放", "commercial_ok": False,
                "category": "metal_part", "images_dir": "images",
                "masks_dir": "masks", "background_values": [0],
                "grade_type": "corrosion",
                "class_map": {1: "good", 2: "fair", 3: "poor", 4: "severe"}})
    grades = {d.grade for s in got for d in s.defects}
    assert grades == {"good", "fair", "poor", "severe"}
    sev = {d.grade: d.severity for s in got for d in s.defects}
    assert sev["good"] == "minor" and sev["severe"] == "critical"


def test_corrosion_wrong_class_map_is_visible(tmp_path):
    """class_map 填错不会报错，但等级会失效 —— 必须能从数据看出来。"""
    root = make_corrosion_cs(str(tmp_path / "c"), values=(10, 20, 30, 40))
    got = _run({"adapter": "mask_seg", "root": root, "name": "c",
                "license": "x", "commercial_ok": False, "category": "metal_part",
                "images_dir": "images", "masks_dir": "masks",
                "background_values": [0], "grade_type": "corrosion",
                "class_map": {1: "good", 2: "fair"}})
    assert all(d.grade == "" for s in got for d in s.defects), \
        "像素值对不上时不该硬凑等级"


# ------------------------------------------------------------------ COCO
def test_coco_adapter_handles_unannotated_images_as_normal(tmp_path):
    root = make_coco(str(tmp_path / "rf"))
    got = _run({"adapter": "coco", "root": root, "name": "aircraft_skin_defects",
                "license": "x", "commercial_ok": False, "category": "fuselage",
                "object_hint": {"fuselage": "fuselage"},
                "splits": ["train", "valid", "test"]})
    assert len(got) == 15
    normal = [s for s in got if not s.is_anomalous]
    assert len(normal) == 3          # 每个 split 里第 0 张无标注
    types = {d.type for s in got for d in s.defects}
    assert "fastener_missing" in types      # Missing-head 要映射对
    assert {"crack", "dent", "paint_peeling"} & types
