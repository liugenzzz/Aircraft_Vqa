# -*- coding: utf-8 -*-
"""原始类名预览：导不出东西时必须交代原因。

真实踩到的坑：--scan 默认只看前 4000 条，而很多数据集把全正常的 train
排在前面（Real-IAD 每类约 5000 张，train 段全是 OK），截断后一个缺陷都
碰不到，脚本只甩一句"共导出 0 张"，看不出是数据没有、路径不对，还是
过滤条件把东西滤光了。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from make_fixtures import make_real_iad


def _load():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))   # 脚本靠 _bootstrap 找包
    spec = importlib.util.spec_from_file_location(
        "preview_raw_classes",
        os.path.join(root, "scripts", "preview_raw_classes.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _setup(tmp_path, code="HS"):
    make_real_iad(str(tmp_path / "raw" / "Real-IAD"))
    jp = tmp_path / "raw" / "Real-IAD" / "realiad_jsons" / "switch.json"
    d = json.loads(jp.read_text(encoding="utf-8"))
    d["test"][0]["anomaly_class"] = code
    jp.write_text(json.dumps(d), encoding="utf-8")

    cfg = tmp_path / "datasets.yaml"
    io.open(cfg, "w", encoding="utf-8").write(yaml.safe_dump({
        "defaults": {"data_root": str(tmp_path / "raw")},
        "datasets": [{"name": "real_iad", "adapter": "real_iad",
                      "root": str(tmp_path / "raw" / "Real-IAD"),
                      "image_dir": "realiad_512", "json_dir": "realiad_jsons",
                      "categories": ["switch"], "enabled": True}],
    }, allow_unicode=True, sort_keys=False))
    return cfg


def _run(m, cfg, tmp_path, *extra):
    argv = sys.argv
    sys.argv = ["x", "--config", str(cfg), "--data-root",
                str(tmp_path / "raw"), "--only", "real_iad",
                "--out", str(tmp_path / "out"), *extra]
    try:
        return m.main()
    finally:
        sys.argv = argv


def test_scan_defaults_to_everything(tmp_path, capsys):
    """默认不能截断 —— 截断正是上次一张都导不出来的原因。"""
    m = _load()
    cfg = _setup(tmp_path)
    capsys.readouterr()
    _run(m, cfg, tmp_path)
    out = capsys.readouterr().out
    assert "扫了 3 条样本" in out, out
    assert len(list((tmp_path / "out").iterdir())) == 1


def test_empty_result_explains_filter(tmp_path, capsys):
    """类名都映射好了却加了 --unmapped-only，要说清是过滤条件滤光的，
    并把扫到的类名连同去向一起列出来。"""
    m = _load()
    cfg = _setup(tmp_path, code="HS")        # HS 已映射成 scratch
    capsys.readouterr()
    _run(m, cfg, tmp_path, "--unmapped-only")
    out = capsys.readouterr().out
    assert "全部映射好了" in out, out
    assert "HS" in out and "scratch" in out, out
    assert "去掉 --unmapped-only" in out, out


def test_unmapped_code_is_exported(tmp_path, capsys):
    """没映射的类名要真能导出来看。"""
    m = _load()
    cfg = _setup(tmp_path, code="ZZ")        # 本体里没有的码
    capsys.readouterr()
    _run(m, cfg, tmp_path, "--unmapped-only")
    out = capsys.readouterr().out
    assert "ZZ" in out and "未映射" in out, out
    files = [p.name for p in (tmp_path / "out").iterdir()]
    assert any("ZZ" in f for f in files), files


def test_no_annotations_at_all_is_explained(tmp_path, capsys):
    """一个带类名的缺陷都没有时，要指向掩码路径而不是干瞪眼。"""
    m = _load()
    cfg = _setup(tmp_path)
    # 把掩码删掉，制造"有样本没缺陷"
    for p in (tmp_path / "raw" / "Real-IAD" / "realiad_512" / "switch").rglob("*_mask.png"):
        p.unlink()
    jp = tmp_path / "raw" / "Real-IAD" / "realiad_jsons" / "switch.json"
    d = json.loads(jp.read_text(encoding="utf-8"))
    for item in d["test"]:
        item.pop("mask_path", None)
        item["anomaly_class"] = "OK"
    jp.write_text(json.dumps(d), encoding="utf-8")
    capsys.readouterr()
    _run(m, cfg, tmp_path)
    out = capsys.readouterr().out
    assert "一个带类名的缺陷都没扫到" in out, out
    assert "preflight" in out, out
