# -*- coding: utf-8 -*-
"""对着各数据集的官方目录结构验证 adapter。

这几个 adapter 是照公开文档写的，没跑过真实数据。夹具复刻目录结构、
命名规则与 mask 组织方式，结构性 bug 在这里暴露，不用等下完 10GB。
"""
import json
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


# ---------------------------------------------------------------- Roboflow 占位超类
def test_coco_drops_roboflow_placeholder_supercategory(tmp_path):
    """Roboflow 导出首位那行占位超类不是缺陷类型，必须丢掉。

    真实踩到的坑：UTS 那份导出里这行叫 "-dents-leaks-ruptures-other"，
    归一化后子串匹配会认成 dent；aircraft_skin_defects 那行更糟，
    "CRACK-DENT-ETC-..." 会被认成 crack —— 凭空造出一个 critical 缺陷。
    """
    from aircraft_vqa.adapters.detection import CocoAdapter
    from PIL import Image

    root = tmp_path / "rf"
    split = root / "train"
    split.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(split / "a.jpg")
    coco = {
        "categories": [
            {"id": 0, "name": "-dents-leaks-ruptures-other", "supercategory": "none"},
            {"id": 1, "name": "Dent", "supercategory": "-dents-leaks-ruptures-other"},
        ],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 64, 64]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [4, 4, 10, 10]},
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    samples = list(CocoAdapter(str(root), splits=["train"],
                                    category="fuselage").iter_samples())
    assert len(samples) == 1
    raw = [d.type_raw for d in samples[0].defects]
    assert raw == ["Dent"], raw
    assert [d.type for d in samples[0].defects] == ["dent"]


def test_coco_keeps_plain_coco_categories(tmp_path):
    """普通 COCO 没有"超类指回自己"这个结构，不能被误伤。"""
    from aircraft_vqa.adapters.detection import CocoAdapter
    from PIL import Image

    root = tmp_path / "plain"
    split = root / "train"
    split.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(split / "a.jpg")
    coco = {
        "categories": [{"id": 1, "name": "crack", "supercategory": "none"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [4, 4, 10, 10]}],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    samples = list(CocoAdapter(str(root), splits=["train"],
                                    category="fuselage").iter_samples())
    assert [d.type_raw for d in samples[0].defects] == ["crack"]


def test_fastener_damage_is_not_other_anomaly():
    """UTS 的 "Fastener Damage" 曾经落到 other_anomaly，白扔一批紧固件框。"""
    from aircraft_vqa.taxonomy import get_taxonomy
    tax = get_taxonomy()
    assert tax.map_defect("Fastener Damage") == "fastener_damage"
    assert tax.group("fastener_damage") == "fastener"
    # 粗粒度兜底类，不能冒充明确的螺纹滑牙
    assert tax.map_defect("thread_side") == "thread_damage"


# ---------------------------------------------------------------- 人工放好的压缩包
def _dl():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_all", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts", "download", "download_all.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_unpack_local_extracts_manually_placed_archives(tmp_path):
    """用户已经把 screw.tar.xz 放进 MVTec_AD/ 了，只是没解压。

    以前只看 probe 路径，会把这种情况报成"还没下"，让人以为要重下一遍。
    """
    import tarfile
    dl = _dl()
    root = tmp_path
    dest = root / "MVTec_AD"
    (dest).mkdir()
    stage = tmp_path / "stage" / "screw" / "train" / "good"
    stage.mkdir(parents=True)
    (stage / "000.png").write_bytes(b"x")
    with tarfile.open(dest / "screw.tar.xz", "w:xz") as t:
        t.add(tmp_path / "stage" / "screw", arcname="screw")

    src = {"name": "mvtec_ad", "dest": "MVTec_AD", "probe": "screw/train/good"}
    assert not dl._present(str(root), src)
    assert dl.do_unpack_local(str(root), src) is True
    assert dl._present(str(root), src)


def test_unpack_local_reaches_one_level_down(tmp_path):
    """Real-IAD 的类别包在 realiad_512/ 里，不在 dest 第一层。"""
    import zipfile
    dl = _dl()
    root = tmp_path
    dest = root / "Real-IAD"
    (dest / "realiad_512").mkdir(parents=True)
    with zipfile.ZipFile(dest / "realiad_jsons.zip", "w") as z:
        z.writestr("realiad_jsons/terminalblock.json", "{}")
    with zipfile.ZipFile(dest / "realiad_512" / "terminalblock.zip", "w") as z:
        z.writestr("terminalblock/OK/x.jpg", "x")

    src = {"name": "real_iad", "dest": "Real-IAD", "probe": "realiad_jsons"}
    assert dl.do_unpack_local(str(root), src) is True
    assert (dest / "realiad_jsons" / "terminalblock.json").exists()
    # 类别包必须就地解压，不能倒进 Real-IAD/ 根目录
    assert (dest / "realiad_512" / "terminalblock" / "OK" / "x.jpg").exists()


def test_unpack_local_no_archives_is_false(tmp_path):
    dl = _dl()
    (tmp_path / "MVTec_AD").mkdir()
    assert dl.do_unpack_local(
        str(tmp_path), {"dest": "MVTec_AD", "probe": "screw/train/good"}) is False


def test_figshare_api_file_list_is_parsed(monkeypatch):
    """figshare 整包直链 403 时，要能从 API 拿到逐文件的 download_url。"""
    import io as _io
    import urllib.request
    dl = _dl()
    payload = json.dumps({"files": [
        {"name": "images.zip", "download_url": "https://x/ndownloader/files/1",
         "size": 10},
        {"name": "masks.zip", "download_url": "https://x/ndownloader/files/2",
         "size": 20},
        {"name": "readme.txt"},          # 缺 download_url，必须跳过
    ]}).encode()

    class _R:
        def read(self, *a):
            return payload
        def __enter__(self):
            return _io.BytesIO(payload)
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _R())
    files = dl._figshare_files("https://x/api/articles/1")
    assert [f["name"] for f in files] == ["images.zip", "masks.zip"]
    assert files[0]["url"].endswith("/files/1")


def test_figshare_api_failure_falls_back(monkeypatch):
    """API 不通不能炸，要退回整包直链那条路。"""
    import urllib.request
    dl = _dl()

    def _boom(*a, **k):
        raise OSError("blocked")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert dl._figshare_files("https://x/api/articles/1") == []
