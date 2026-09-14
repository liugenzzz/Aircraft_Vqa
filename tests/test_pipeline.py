# -*- coding: utf-8 -*-
"""流水线关键不变量的回归测试：pytest tests/ -q"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from PIL import Image

from aircraft_vqa.balance import auto_total, dedup, group_split, quota_sample
from aircraft_vqa.geometry import (connected_boxes, region_word, to_qwen_box,
                                   smart_resize)
from aircraft_vqa.qc import check_record, run_qc
from aircraft_vqa.schema import Defect, UnifiedSample
from aircraft_vqa.taxonomy import get_taxonomy
from aircraft_vqa.vqa import BuildConfig, VQABuilder

TAX = get_taxonomy(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "configs", "taxonomy.yaml"))


# ------------------------------------------------------------------ taxonomy
@pytest.mark.parametrize("raw,expect", [
    ("Missing-head", "fastener_missing"),
    ("missing_pushpin", "fastener_missing"),
    ("scratch_head", "scratch"),
    ("thread_top", "thread_damage"),
    ("screw_too_long", "fastener_wrong_spec"),
    ("Paint-peel-off", "paint_peeling"),
    ("rust", "corrosion"),
    ("1_additional_pushpin", "extra_part"),
])
def test_defect_alias_mapping(raw, expect):
    assert TAX.map_defect(raw) == expect


def test_unknown_defect_falls_back_not_guesses():
    # 映射不上时必须退化为 other_anomaly，不能瞎猜一个具体类型
    assert TAX.map_defect("zzzz_unknown_thing") == "other_anomaly"
    assert TAX.map_defect("") == "other_anomaly"


def test_siblings_are_same_group():
    for t in TAX.siblings("fastener_missing"):
        assert TAX.group(t) == TAX.group("fastener_missing")


# ------------------------------------------------------------------ geometry
def test_connected_boxes_separates_components():
    m = np.zeros((200, 300), bool)
    m[10:40, 20:60] = True
    m[120:160, 200:260] = True
    boxes = connected_boxes(m)
    assert len(boxes) == 2
    assert all(b[2] > b[0] and b[3] > b[1] for b in boxes)


def test_connected_boxes_never_loses_tiny_defect():
    m = np.zeros((512, 512), bool)
    m[100:103, 100:103] = True          # 小到会被面积阈值过滤
    assert len(connected_boxes(m)) == 1  # 但不能丢样本


def test_region_word_nine_grid():
    assert region_word([0, 0, 30, 30], 300, 300) == "左上"
    assert region_word([270, 270, 300, 300], 300, 300) == "右下"
    assert region_word([140, 140, 160, 160], 300, 300) == "正中"


def test_qwen_box_normalisation_roundtrip():
    b = to_qwen_box([150, 100, 300, 200], 600, 400, "norm1000")
    assert b == [250, 250, 500, 500]
    assert to_qwen_box([1, 2, 3, 4], 10, 10, "abs") == [1, 2, 3, 4]


def test_qwen_box_clamped_to_range():
    b = to_qwen_box([-50, -50, 9999, 9999], 100, 100, "norm1000")
    assert min(b) >= 0 and max(b) <= 1000


def test_smart_resize_is_multiple_of_28():
    h, w = smart_resize(1023, 777)
    assert h % 28 == 0 and w % 28 == 0


# ------------------------------------------------------------------ builder
def _sample(tmp_path, label="anomalous", dtype="fastener_missing", n=2):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (400, 300), (128, 128, 128)).save(p)
    s = UnifiedSample(sample_id=f"t/{label}/{dtype}", image_path=str(p),
                      width=400, height=300, label=label, dataset="t",
                      category="panel", object_name="panel", object_zh="检查口盖",
                      aircraft_ctx="可卸口盖壁板", split="train")
    if label == "anomalous":
        for i in range(n):
            s.defects.append(Defect(type=dtype, type_raw=dtype,
                                    type_zh=TAX.zh(dtype),
                                    bbox=[10 + i * 60, 20, 60 + i * 60, 80],
                                    area_ratio=0.02, region="左上",
                                    severity="major"))
    return s


def test_builder_all_records_pass_qc(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    recs = []
    for lbl in ("anomalous", "normal"):
        for dt in ("fastener_missing", "corrosion", "crack"):
            recs += b.build(_sample(tmp_path, lbl, dt))
    assert recs
    for r in recs:
        assert check_record(r) == [], (r["task"], check_record(r), r["answer"])


def test_normal_sample_never_produces_boxes(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    for r in b.build(_sample(tmp_path, "normal")):
        if r["task"].startswith("grounding"):
            assert json.loads(r["answer"].split("\n")[0]) == []


def test_unknown_type_skips_classification(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12,
                               skip_unknown_type_tasks=True), TAX)
    tasks = {r["task"] for r in b.build(_sample(tmp_path, "anomalous",
                                                "other_anomaly"))}
    assert "classification_open" not in tasks
    assert "classification_mc" not in tasks


def test_build_is_deterministic(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=6), TAX)
    s = _sample(tmp_path)
    assert [r["qa_id"] for r in b.build(s)] == [r["qa_id"] for r in b.build(s)]


def test_counting_answer_matches_box_count(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    for r in b.build(_sample(tmp_path, n=3)):
        if r["task"] == "counting":
            assert r["count"] == len(json.loads(
                r["answer"].split("\n", 1)[1]))


def test_mc_options_unique_and_answer_consistent(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    for r in b.build(_sample(tmp_path)):
        if r["task"] == "classification_mc":
            assert len(set(r["options"])) == len(r["options"])
            assert r["answer"].startswith(r["answer_letter"])


# ------------------------------------------------------------------ qc
def test_qc_rejects_boxes_on_negative_sample():
    bad = {"task": "grounding_negative", "question": "q", "label": "normal",
           "answer": '[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]',
           "coord_mode": "norm1000"}
    assert "negative_with_boxes" in check_record(bad)


def test_qc_rejects_out_of_range_box():
    bad = {"task": "grounding_all", "question": "q", "label": "anomalous",
           "answer": '[{"bbox_2d": [0, 0, 1200, 50], "label": "x"}]',
           "coord_mode": "norm1000"}
    assert "box_out_of_range" in check_record(bad)


def test_qc_accepts_bu_hege():
    # "不合格。" 是阳性结论，不能被当成"答案说没问题"
    ok = {"task": "discrimination", "question": "q", "label": "anomalous",
          "yes_no": "yes", "answer": "不合格。该检查口盖上发现紧固件缺失。"}
    assert check_record(ok) == []


def test_run_qc_filters_bad_records():
    recs = [{"task": "description", "question": "q", "answer": "a"},
            {"task": "description", "question": "q", "answer": "缺少{obj}"}]
    ok, rep = run_qc(recs)
    assert len(ok) == 1 and rep["n_fail"] == 1


# ------------------------------------------------------------------ balance
def test_group_split_has_no_image_leakage():
    recs = [{"sample_id": f"s{i%20}", "task": "t", "image": f"i{i%20}",
             "question": f"q{i}", "answer": "a"} for i in range(200)]
    sp = group_split(recs)
    ids = {k: {r["sample_id"] for r in v} for k, v in sp.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])


def test_dedup_removes_identical_qa():
    r = {"image": "a", "question": "q", "answer": "a", "task": "t"}
    assert len(dedup([dict(r), dict(r), dict(r, question="q2")])) == 2


def test_auto_total_not_capped_by_scarcest_task():
    by_task = {"a": [0] * 1000, "b": [0] * 1000, "c": [0] * 10}
    ratio = {"a": 0.49, "b": 0.49, "c": 0.02}
    # 用最小值会得到 ~500；加权分位数应该明显更大
    assert auto_total(by_task, ratio) > 900


def test_quota_sample_respects_supply():
    recs = ([{"task": "a", "qa_id": str(i)} for i in range(100)] +
            [{"task": "b", "qa_id": "b%d" % i} for i in range(5)])
    out = quota_sample(recs, {"a": 0.5, "b": 0.5}, total=50)
    assert sum(r["task"] == "b" for r in out) <= 5
