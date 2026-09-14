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


# ------------------------------------------------------------------ 缺陷范围
def test_active_scope_downgrades_out_of_scope_type(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=12,
                      active_defect_types=["fastener_missing", "crack"])
    b = VQABuilder(cfg, TAX)
    recs = b.build(_sample(tmp_path, "anomalous", "paint_peeling"))
    # 降级成 other_anomaly：框还在（定位照做），但不再出类型题
    assert recs
    tasks = {r["task"] for r in recs}
    assert "classification_open" not in tasks and "classification_mc" not in tasks
    assert any(r["task"].startswith("grounding") for r in recs)
    for r in recs:
        assert "漆层剥落" not in r["answer"]


def test_active_scope_keeps_in_scope_type(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=12,
                      active_defect_types=["fastener_missing", "crack"])
    b = VQABuilder(cfg, TAX)
    recs = b.build(_sample(tmp_path, "anomalous", "crack"))
    assert any("裂纹" in r["answer"] for r in recs)


# ------------------------------------------------------------------ 多轮
def _multi(tmp_path, label="anomalous"):
    cfg = BuildConfig(max_qa_per_sample=12)
    cfg.task_weights = dict(cfg.task_weights, multi_turn=1.0)   # 必出，便于断言
    b = VQABuilder(cfg, TAX)
    for r in b.build(_sample(tmp_path, label)):
        if r["task"] == "multi_turn":
            return r
    return None


def test_multi_turn_structure_and_qc(tmp_path):
    for label in ("anomalous", "normal"):
        r = _multi(tmp_path, label)
        assert r is not None and r["n_turns"] >= 2
        assert r["family"] == "dialog"
        assert check_record(r) == [], check_record(r)


def test_multi_turn_normal_answers_empty_list(tmp_path):
    r = _multi(tmp_path, "normal")
    joined = " ".join(t["answer"] for t in r["turns"])
    assert "[]" in joined
    assert "bbox_2d" not in joined
    # 正常图到"复查后无异常"就该结束，不再追加签署类流程套话
    assert r["n_turns"] == 2


def test_qc_catches_boxes_in_normal_dialog():
    bad = {"task": "multi_turn", "question": "q", "answer": "a", "yes_no": "no",
           "label": "normal", "coord_mode": "norm1000",
           "turns": [{"question": "有异常吗", "answer": "未见异常。"},
                     {"question": "框出来", "answer": '[{"bbox_2d": [1,2,3,4], "label": "x"}]'}]}
    errs = check_record(bad)
    assert any("negative_with_boxes" in e for e in errs), errs


def test_exporter_emits_all_turns(tmp_path):
    from aircraft_vqa.export.qwen3vl import to_llamafactory
    r = _multi(tmp_path, "anomalous")
    out = to_llamafactory(r)
    # system + 每轮一问一答
    assert len(out["messages"]) == 1 + 2 * r["n_turns"]
    assert out["messages"][1]["content"].startswith("<image>")
    # 图片 token 只出现一次
    assert sum(m["content"].count("<image>") for m in out["messages"]
               if isinstance(m["content"], str)) == 1


# ------------------------------------------------------------------ 问法库
def test_question_bank_merges_and_is_used(tmp_path):
    import json as _json
    bank = tmp_path / "bank.json"
    bank.write_text(_json.dumps({"questions": {
        "description": ["【扩写测试】说说这个{obj}的状况。"]}}), encoding="utf-8")
    b = VQABuilder(BuildConfig(max_qa_per_sample=12,
                               question_bank=str(bank)), TAX)
    assert "【扩写测试】说说这个{obj}的状况。" in b.qpool["description"]
    assert len(b.qpool["description"]) > 1      # 种子没被顶掉


def test_question_bank_validation_rules():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gqb", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "scripts", "gen_question_bank.py"))
    gqb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gqb)

    assert gqb.validate("grounding_single", "框出图中的{defect}（JSON）。", set()) == ""
    # 缺必需占位符
    assert gqb.validate("grounding_single", "把缺陷框出来。", set())
    # 用了池外占位符
    assert gqb.validate("grounding_single", "定位{defect}在{foo}。", set())
    # 重复
    assert gqb.validate("description", "描述{obj}。", {"描述{obj}。"})


# ------------------------------------------------------------------ 合成数据
def test_synthetic_generator_covers_target_defects():
    import importlib.util
    import random as _r
    spec = importlib.util.spec_from_file_location(
        "mdd", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "scripts", "make_demo_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    names = {c["name"] for c in m.CATEGORIES}
    assert {"crack", "thread_damage"} <= names

    rng = _r.Random(0)
    seen = set()
    for i in range(40):
        _, anns = m.gen_image(i, rng, (384, 288))
        seen |= {a["category"] for a in anns}
    assert "crack" in seen                       # 裂纹只在蒙皮场景
    seen_c = set()
    for i in range(40):
        _, anns = m.gen_closeup(i, rng, (384, 288))
        seen_c |= {a["category"] for a in anns}
    assert "thread_damage" in seen_c             # 螺纹损伤只在特写场景


def test_synthetic_boxes_inside_image():
    import importlib.util
    import random as _r
    spec = importlib.util.spec_from_file_location(
        "mdd2", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "scripts", "make_demo_data.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = _r.Random(1)
    for gen in (m.gen_image, m.gen_closeup):
        for i in range(15):
            img, anns = gen(i, rng, (384, 288))
            for a in anns:
                x, y, w, h = a["bbox"]
                assert x >= 0 and y >= 0
                assert x + w <= img.width and y + h <= img.height
                assert w > 0 and h > 0


# ------------------------------------------------------------------ 问法风格
def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name.replace(".", "_"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_instruction_style_rejects_colloquial():
    gqb = _load_script("gen_question_bank.py")
    assert gqb.validate("description", "把这个{obj}瞅一下呗。", set())
    assert gqb.validate("description", "整一下{obj}的情况", set())
    # 无结尾标点 = 碎片
    assert gqb.validate("description", "描述一下这个{obj}的状况", set())
    # 行话堆砌
    assert gqb.validate("description", "按 AMM 和 SRM 的适航要求描述{obj}。", set())
    # 合格的指令式
    assert gqb.validate("description", "请描述该{obj}的外观状况。", set()) == ""


def test_loose_style_only_checks_placeholders():
    gqb = _load_script("gen_question_bank.py")
    q = "把这个{obj}瞅一下呗"
    assert gqb.validate("description", q, set(), style="loose") == ""
    assert gqb.validate("description", q, set(), style="instruction")


def test_seed_templates_are_not_colloquial():
    """种子问法自己也得守规矩，否则扩写出来的会跟着跑偏。"""
    gqb = _load_script("gen_question_bank.py")
    from aircraft_vqa.vqa import templates as TT
    pools = list(TT.QUESTIONS.values()) + [
        TT.Q_MT_T2_POS, TT.Q_MT_T2_NEG, TT.Q_MT_T3_POS]
    for pool in pools:
        for q in pool:
            if q.isascii():          # 英文问法不走中文口语规则
                continue
            for w in gqb.COLLOQUIAL:
                assert w not in q, f"种子问法口语化：{q}"


# ------------------------------------------------------------------ 通用数据混入
def test_mix_general_hits_target_ratio(tmp_path):
    bv = _load_script("build_vqa.py")
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(f'{{"i": {i}}}' for i in range(900)) + "\n",
                     encoding="utf-8")
    gen = tmp_path / "general.jsonl"
    gen.write_text("\n".join(f'{{"g": {i}}}' for i in range(500)) + "\n",
                   encoding="utf-8")
    bv._mix_general(str(train), str(gen), 0.10, seed=0)
    lines = [l for l in train.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_gen = sum('"g"' in l for l in lines)
    assert abs(n_gen / len(lines) - 0.10) < 0.01


def test_mix_general_skips_bad_input(tmp_path):
    bv = _load_script("build_vqa.py")
    train = tmp_path / "train.jsonl"
    train.write_text('{"i": 1}\n', encoding="utf-8")
    bv._mix_general(str(train), str(tmp_path / "nope.jsonl"), 0.1, 0)
    assert train.read_text(encoding="utf-8") == '{"i": 1}\n'   # 原样不动
    gen = tmp_path / "g.jsonl"
    gen.write_text('{"g": 1}\n', encoding="utf-8")
    bv._mix_general(str(train), str(gen), 0.9, 0)              # 比例不合理
    assert train.read_text(encoding="utf-8") == '{"i": 1}\n'


# ------------------------------------------------------------------ 下载器
def test_download_registry_is_consistent():
    da = _load_script(os.path.join("download", "download_all.py"))
    names = [s["name"] for s in da.SOURCES]
    assert len(names) == len(set(names))
    for s in da.SOURCES:
        assert s["mode"] in da.MODE_ZH
        for k in ("zh", "use", "dest", "probe", "license"):
            assert s.get(k), f"{s['name']} 缺字段 {k}"
        if s["mode"] == "manual":
            assert s.get("steps") and s.get("page"), f"{s['name']} 缺手动步骤"
        if s["mode"] == "auto":
            assert s.get("url") and s.get("archive")
        if s["mode"] == "keyed":
            assert s.get("env") and s.get("env_how")


def test_manual_manifest_lists_paths_and_steps(tmp_path):
    da = _load_script(os.path.join("download", "download_all.py"))
    manual = [s for s in da.SOURCES if s["mode"] == "manual"]
    out = tmp_path / "MANUAL.md"
    da.write_manual_manifest(str(tmp_path), manual, str(out))
    text = out.read_text(encoding="utf-8")
    for s in manual:
        assert s["zh"] in text
        assert s["page"] in text
        assert os.path.join(str(tmp_path), s["dest"]) in text
    assert "{full_dest}" not in text          # 占位符必须都被替换掉


# ------------------------------------------------------------------ 类别均衡
def test_cap_class_imbalance_flattens_tail():
    from aircraft_vqa.balance import cap_class_imbalance, class_distribution
    recs = ([{"task": "classification_open", "target_type": "corrosion",
              "qa_id": f"c{i}"} for i in range(100)] +
            [{"task": "classification_open", "target_type": "dent",
              "qa_id": f"d{i}"} for i in range(10)] +
            [{"task": "grounding_all", "target_type": "corrosion",
              "qa_id": f"g{i}"} for i in range(100)])
    out = cap_class_imbalance(recs, ["classification_open"], max_over_min=3.0)
    dist = class_distribution(out, ["classification_open"])
    assert dist["dent"] == 10
    assert dist["corrosion"] == 30                      # 3 × 最少类
    # 其他任务不受影响
    assert sum(r["task"] == "grounding_all" for r in out) == 100


def test_cap_class_imbalance_noop_when_already_balanced():
    from aircraft_vqa.balance import cap_class_imbalance
    recs = [{"task": "classification_mc", "target_type": t, "qa_id": f"{t}{i}"}
            for t in ("crack", "dent") for i in range(20)]
    assert len(cap_class_imbalance(recs, ["classification_mc"], 3.0)) == len(recs)


# ------------------------------------------------------------------ 配置自动启用
def test_enable_in_config_only_flips_matching_entry(tmp_path):
    da = _load_script(os.path.join("download", "download_all.py"))
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        "datasets:\n"
        "  - name: visa\n"
        "    adapter: visa\n"
        "    enabled: false   # 注释要保住\n"
        "  - name: mvtec_ad_screw\n"
        "    adapter: mvtec_ad\n"
        "    enabled: false\n", encoding="utf-8")
    changed = da.enable_in_config(str(cfg), ["visa"])
    text = cfg.read_text(encoding="utf-8")
    assert changed == ["visa"]
    assert "  - name: visa\n    adapter: visa\n    enabled: true\n" in text
    # 没点名的条目原样不动
    assert "  - name: mvtec_ad_screw\n    adapter: mvtec_ad\n    enabled: false\n" in text


def test_enable_in_config_is_idempotent(tmp_path):
    da = _load_script(os.path.join("download", "download_all.py"))
    cfg = tmp_path / "d.yaml"
    cfg.write_text("  - name: visa\n    enabled: true\n", encoding="utf-8")
    assert da.enable_in_config(str(cfg), ["visa"]) == []      # 已是 true，不重复改
    assert cfg.read_text(encoding="utf-8") == "  - name: visa\n    enabled: true\n"


def test_roboflow_downloader_parses_versions():
    rf = _load_script(os.path.join("download", "download_roboflow.py"))
    import json as _json
    meta = {"versions": [
        {"id": "ws/proj/1", "name": "v1", "images": 300},
        {"id": "ws/proj/3", "name": "v3", "images": 1115},
        {"id": "ws/proj/2", "name": "v2", "images": 700},
        {"id": "ws/proj/draft", "name": "草稿"},          # 非数字版本要被跳过
    ]}
    rf._get_json = lambda url, timeout=60: meta
    vers = rf.list_versions("ws", "proj", "k")
    assert [v["version"] for v in vers] == [1, 2, 3]        # 升序，latest 取末尾
    assert vers[-1]["images"] == 1115
    _json.dumps(vers)                                       # 可序列化


def test_roboflow_error_messages_are_actionable():
    rf = _load_script(os.path.join("download", "download_roboflow.py"))
    import urllib.error
    import io
    e401 = urllib.error.HTTPError("u", 401, "x", {}, io.BytesIO(b""))
    e404 = urllib.error.HTTPError("u", 404, "x", {}, io.BytesIO(b""))
    assert "Private API Key" in rf._explain(e401, "ws", "proj")
    assert "--list-versions" in rf._explain(e404, "ws", "proj")


# ------------------------------------------------------------------ 有序分级
def test_grade_taxonomy_is_ordered():
    order = TAX.grade_order("corrosion")
    assert order == ["good", "fair", "poor", "severe"]
    ranks = [TAX.grade_rank("corrosion", g) for g in order]
    assert ranks == sorted(ranks)
    # 等级越高严重度不降
    sev_order = ["minor", "major", "critical"]
    idx = [sev_order.index(TAX.grade_info("corrosion", g)["severity"]) for g in order]
    assert idx == sorted(idx)
    assert TAX.grade_zh("corrosion", "severe") == "截面损失"
    assert TAX.has_grades("corrosion") and not TAX.has_grades("crack")
    assert TAX.grade_info("corrosion", "不存在的等级") == {}


def _corrosion_adapter(tmp_path, grade_type="corrosion"):
    import numpy as _np
    from aircraft_vqa.adapters import build_adapter
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i, v in enumerate([1, 2, 3, 4]):
        Image.new("RGB", (128, 128), (120, 120, 120)).save(
            tmp_path / "images" / f"i{i}.png")
        m = _np.zeros((128, 128), _np.uint8)
        m[30:90, 30:90] = v
        Image.fromarray(m).save(tmp_path / "masks" / f"i{i}.png")
    spec = {"adapter": "mask_seg", "root": str(tmp_path), "name": "corr",
            "license": "x", "commercial_ok": False, "category": "metal_part",
            "images_dir": "images", "masks_dir": "masks",
            "background_values": [0],
            "class_map": {1: "good", 2: "fair", 3: "poor", 4: "severe"}}
    if grade_type:
        spec["grade_type"] = grade_type
    return build_adapter(spec, taxonomy=TAX)


def test_mask_seg_reads_grades_and_severity(tmp_path):
    got = {}
    for s in _corrosion_adapter(tmp_path).iter_samples():
        d = s.defects[0]
        got[d.grade] = (d.type, d.severity)
    assert set(got) == {"good", "fair", "poor", "severe"}
    assert all(t == "corrosion" for t, _ in got.values())
    # 严重度由等级给出，不再按面积估
    assert got["good"][1] == "minor"
    assert got["severe"][1] == "critical"


def test_grade_type_required_for_good_level(tmp_path):
    """不声明 grade_type 时 'good' 不能被当成腐蚀等级 —— 它在 MVTec 里是'正常'。"""
    for s in _corrosion_adapter(tmp_path, grade_type=None).iter_samples():
        d = s.defects[0]
        if d.type_raw == "good":
            assert d.grade == ""
            assert d.type != "corrosion"


def test_grade_assessment_answer_uses_level_wording(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=12, active_defect_types=["corrosion"])
    cfg.task_weights = dict(cfg.task_weights, grade_assessment=1.0)
    b = VQABuilder(cfg, TAX)
    seen = {}
    for s in _corrosion_adapter(tmp_path).iter_samples():
        for r in b.build(s):
            if r["task"] == "grade_assessment":
                seen[r["grade"]] = r
                assert check_record(r) == []
    assert len(seen) == 4
    assert "截面损失" in seen["severe"]["answer"]
    assert "轻微锈蚀" in seen["good"]["answer"]
    # 等级不同，处置建议必须不同，否则分级就白做了
    actions = {r["answer"].rsplit("处置建议：", 1)[-1] for r in seen.values()}
    assert len(actions) == 4


def test_ungraded_defect_produces_no_grade_task(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=12)
    cfg.task_weights = dict(cfg.task_weights, grade_assessment=1.0)
    b = VQABuilder(cfg, TAX)
    tasks = {r["task"] for r in b.build(_sample(tmp_path, "anomalous", "crack"))}
    assert "grade_assessment" not in tasks
