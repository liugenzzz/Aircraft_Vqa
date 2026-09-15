# -*- coding: utf-8 -*-
"""本机状态（哪些源已下好）与入库模板必须分开存。

真实踩到的坑：download_all --enable-config 直接改 configs/datasets.yaml，
于是"下数据"这个动作本身在改仓库文件，每次 git pull 都撞
"local changes would be overwritten by merge"。
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import yaml

from aircraft_vqa.config import (load_dataset_configs, local_path_for,
                                 merge_dataset_specs, set_enabled)

TEMPLATE = {"defaults": {"data_root": "~/data/raw"}, "datasets": [
    {"name": "a", "adapter": "coco", "root": "/x/a", "enabled": False,
     "class_map": {1: "fair"}},
    {"name": "b", "adapter": "mvtec_ad", "root": "/x/b", "enabled": False},
]}


def _write(tmp_path):
    p = tmp_path / "datasets.yaml"
    io.open(p, "w", encoding="utf-8").write(yaml.safe_dump(TEMPLATE, sort_keys=False))
    return str(p)


def test_local_overlay_is_shallow_merged_not_replaced():
    """本地那份只写 {name, enabled}，不能把整条覆盖掉。"""
    out = merge_dataset_specs([TEMPLATE["datasets"], [{"name": "a", "enabled": True}]])
    a = [d for d in out if d["name"] == "a"][0]
    assert a["enabled"] is True
    assert a["adapter"] == "coco"          # 这些字段必须还在
    assert a["root"] == "/x/a"
    assert a["class_map"] == {1: "fair"}


def test_set_enabled_writes_local_and_leaves_template_untouched(tmp_path):
    cfg = _write(tmp_path)
    before = io.open(cfg, encoding="utf-8").read()

    assert set_enabled(cfg, ["a"]) == ["a"]

    assert io.open(cfg, encoding="utf-8").read() == before, "入库模板被改动了"
    lp = local_path_for(cfg)
    assert os.path.exists(lp)
    loc = yaml.safe_load(io.open(lp, encoding="utf-8"))
    assert loc["datasets"] == [{"name": "a", "enabled": True}]


def test_local_overlay_takes_effect_on_load(tmp_path):
    cfg = _write(tmp_path)
    specs, root = load_dataset_configs([cfg])
    assert [d["enabled"] for d in specs] == [False, False]
    assert root == "~/data/raw"

    set_enabled(cfg, ["a", "b"])
    specs, _ = load_dataset_configs([cfg])
    assert {d["name"]: d["enabled"] for d in specs} == {"a": True, "b": True}
    # 模板里的字段依然完整
    assert [d for d in specs if d["name"] == "a"][0]["class_map"] == {1: "fair"}


def test_set_enabled_is_idempotent(tmp_path):
    cfg = _write(tmp_path)
    assert set_enabled(cfg, ["a"]) == ["a"]
    assert set_enabled(cfg, ["a"]) == []          # 已经是 true，不重复写
    loc = yaml.safe_load(io.open(local_path_for(cfg), encoding="utf-8"))
    assert len(loc["datasets"]) == 1              # 不能追加重复条目


def test_load_without_local_is_unchanged(tmp_path):
    cfg = _write(tmp_path)
    set_enabled(cfg, ["a"])
    specs, _ = load_dataset_configs([cfg], use_local=False)
    assert all(d["enabled"] is False for d in specs)


def test_local_overlay_is_gitignored():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ig = io.open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
    assert "configs/*.local.yaml" in ig


def test_download_all_enable_does_not_touch_tracked_config(tmp_path):
    """download_all 的 enable_in_config 必须落到本地那份。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_all", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "scripts", "download", "download_all.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cfg = _write(tmp_path)
    before = io.open(cfg, encoding="utf-8").read()
    assert m.enable_in_config(cfg, ["b"]) == ["b"]
    assert io.open(cfg, encoding="utf-8").read() == before
    assert os.path.exists(local_path_for(cfg))
