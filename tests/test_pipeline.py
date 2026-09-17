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
        TT.Q_MT_T2_POS, TT.Q_MT_T2_NEG, TT.Q_MT_T3_ONE, TT.Q_MT_T3_MANY]
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
    # 基准是中位数不是最少类：两个类时中位数=100，头部 cap=300（不超限，
    # corrosion 原样留下），长尾 floor=100，dent 重复采样顶到 3×=30。
    # 按最少类算的话会把 corrosion 砍到 30，白扔 70 条标注好的数据。
    assert dist["corrosion"] == 100, dist
    assert dist["dent"] == 30, dist
    assert max(dist.values()) / min(dist.values()) < 10 / 1, "失衡没改善"
    # 其他任务不受影响
    assert sum(r["task"] == "grounding_all" for r in out) == 100


def test_cap_class_imbalance_noop_when_already_balanced():
    from aircraft_vqa.balance import cap_class_imbalance
    recs = [{"task": "classification_mc", "target_type": t, "qa_id": f"{t}{i}"}
            for t in ("crack", "dent") for i in range(20)]
    assert len(cap_class_imbalance(recs, ["classification_mc"], 3.0)) == len(recs)


# ------------------------------------------------------------------ 配置自动启用
def test_enable_in_config_only_flips_matching_entry(tmp_path):
    """启用只作用于点名的源，且入库模板一个字节都不能动。"""
    from aircraft_vqa.config import load_dataset_configs
    da = _load_script(os.path.join("download", "download_all.py"))
    cfg = tmp_path / "datasets.yaml"
    body = ("datasets:\n"
            "  - name: visa\n"
            "    adapter: visa\n"
            "    enabled: false   # 注释要保住\n"
            "  - name: mvtec_ad_screw\n"
            "    adapter: mvtec_ad\n"
            "    enabled: false\n")
    cfg.write_text(body, encoding="utf-8")

    assert da.enable_in_config(str(cfg), ["visa"]) == ["visa"]
    assert cfg.read_text(encoding="utf-8") == body, "入库模板被改动了"

    # 叠上本地那份之后，启用状态才生效；没点名的依旧是关的
    specs, _ = load_dataset_configs([str(cfg)])
    assert {d["name"]: d.get("enabled") for d in specs} == {
        "visa": True, "mvtec_ad_screw": False}
    # 模板里的其他字段不能因为叠加而丢
    assert [d for d in specs if d["name"] == "visa"][0]["adapter"] == "visa"


def test_enable_in_config_is_idempotent(tmp_path):
    da = _load_script(os.path.join("download", "download_all.py"))
    cfg = tmp_path / "d.yaml"
    body = "datasets:\n  - name: visa\n    enabled: false\n"
    cfg.write_text(body, encoding="utf-8")
    assert da.enable_in_config(str(cfg), ["visa"]) == ["visa"]
    assert da.enable_in_config(str(cfg), ["visa"]) == []   # 已是 true，不重复写
    assert cfg.read_text(encoding="utf-8") == body


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


# ------------------------------------------------------------------ Real-IAD
def _realiad_root(tmp_path, payload, image_rel="switch/0001.jpg"):
    import json as _json
    (tmp_path / "realiad_jsons").mkdir()
    img_dir = tmp_path / "realiad_512" / os.path.dirname(image_rel)
    img_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (100, 100, 100)).save(
        tmp_path / "realiad_512" / image_rel)
    (tmp_path / "realiad_jsons" / "switch.json").write_text(
        _json.dumps(payload), encoding="utf-8")
    from aircraft_vqa.adapters import build_adapter
    return build_adapter(
        {"adapter": "real_iad", "root": str(tmp_path), "name": "real_iad",
         "license": "x", "commercial_ok": False,
         "json_dir": "realiad_jsons", "image_dir": "realiad_512"}, taxonomy=TAX)


def test_realiad_reads_standard_train_test_keys(tmp_path):
    ad = _realiad_root(tmp_path, {
        "train": [{"image_path": "switch/0001.jpg", "anomaly_class": "OK",
                   "view": "C1"}],
        "test": [{"image_path": "switch/0001.jpg", "anomaly_class": "scratch",
                  "view": "C2"}]})
    got = list(ad.iter_samples())
    assert len(got) == 2
    assert {s.label for s in got} == {"normal", "anomalous"}
    assert {s.meta.get("view") for s in got} == {"C1", "C2"}


def test_realiad_falls_back_to_other_top_level_keys(tmp_path):
    """顶层键不叫 train/test 时，退回找"值是一串 dict"的键。"""
    ad = _realiad_root(tmp_path, {
        "meta": {"version": 1},
        "samples": [{"image": "switch/0001.jpg", "anomaly_class": "OK"}]})
    assert len(list(ad.iter_samples())) == 1


def test_realiad_accepts_bare_list(tmp_path):
    ad = _realiad_root(tmp_path, [{"image": "switch/0001.jpg",
                                   "anomaly_class": "OK"}])
    assert len(list(ad.iter_samples())) == 1


def test_realiad_diagnoses_when_nothing_matches(tmp_path, capsys):
    ad = _realiad_root(tmp_path, {
        "samples": [{"pic": "不认识的字段名/x.jpg", "anomaly_class": "OK"}]})
    assert list(ad.iter_samples()) == []
    out = capsys.readouterr().out
    assert "读出 0 条样本" in out
    assert "条目字段" in out            # 把实际字段名亮出来
    assert "image_dir" in out           # 并指出该改哪里


# ------------------------------------------------------------------ Roboflow 批量
def test_roboflow_batch_parses_list(tmp_path, capsys):
    rb = _load_script(os.path.join("download", "roboflow_batch.py"))
    lf = tmp_path / "list.txt"
    lf.write_text(
        "# 注释行\n"
        "\n"
        "ddiisc/aircraft_skin_defects\n"
        "someone/rivet-inspection:3\n"
        "  spaced/proj  # 行尾注释\n"
        "格式不对的行\n", encoding="utf-8")
    items = rb.parse_list(str(lf))
    assert items == [
        {"ws": "ddiisc", "proj": "aircraft_skin_defects", "ver": "latest"},
        {"ws": "someone", "proj": "rivet-inspection", "ver": "3"},
        {"ws": "spaced", "proj": "proj", "ver": "latest"}]
    assert "格式不对" in capsys.readouterr().out


def test_roboflow_batch_safe_name():
    rb = _load_script(os.path.join("download", "roboflow_batch.py"))
    assert rb.safe_name("aircraft-skin-defects-631bu") == "aircraft_skin_defects_631bu"
    assert rb.safe_name("Rivet.Inspection v2") == "rivet_inspection_v2"


def test_generated_entry_is_valid_yaml_and_not_commercial():
    import yaml as _yaml
    rb = _load_script(os.path.join("download", "roboflow_batch.py"))
    entry = rb.ENTRY.format(name="rf_demo", root="/tmp/x", ws="ws", proj="proj")
    doc = _yaml.safe_load("datasets:\n" + entry)
    d = doc["datasets"][0]
    assert d["adapter"] == "coco" and d["name"] == "rf_demo"
    # 授权逐集不同，生成的条目必须默认不可商用
    assert d["commercial_ok"] is False


# ------------------------------------------------------------------ 多配置 ingest
def test_ingest_merges_configs_with_override(tmp_path):
    import yaml as _yaml
    ing = _load_script("ingest.py")
    base = tmp_path / "a.yaml"
    base.write_text(_yaml.safe_dump({
        "defaults": {"data_root": "/base"},
        "datasets": [{"name": "x", "adapter": "coco", "root": "r1"},
                     {"name": "y", "adapter": "coco", "root": "r2"}]}),
        encoding="utf-8")
    extra = tmp_path / "b.yaml"
    extra.write_text(_yaml.safe_dump({
        "datasets": [{"name": "x", "adapter": "yolo", "root": "r9"},
                     {"name": "z", "adapter": "coco", "root": "r3"}]}),
        encoding="utf-8")

    import sys as _s
    argv = _s.argv
    _s.argv = ["ingest", "--config", str(base), "--config", str(extra),
               "--out", str(tmp_path / "out"), "--data-root", str(tmp_path)]
    try:
        ing.main()          # 目录都不存在，只验配置合并不报错
    finally:
        _s.argv = argv


def test_resolve_falls_back_to_alternative_root(tmp_path):
    ing = _load_script("ingest.py")
    (tmp_path / "MVTec_AD").mkdir()
    spec = ing.resolve({"name": "m", "root": "{data_root}/mvtec_anomaly_detection",
                        "root_alternatives": ["{data_root}/MVTec_AD"]},
                       str(tmp_path))
    assert spec["root"] == str(tmp_path / "MVTec_AD")
    assert len(spec["_tried_roots"]) == 2


def test_resolve_keeps_primary_when_none_exist(tmp_path):
    ing = _load_script("ingest.py")
    spec = ing.resolve({"name": "m", "root": "{data_root}/primary",
                        "root_alternatives": ["{data_root}/other"]}, str(tmp_path))
    assert spec["root"].endswith("primary")     # 报错信息里给主路径


# ------------------------------------------------------------------ 极性与严重度
def test_discrimination_answers_have_no_bare_yes_no():
    """问法池里"是否合格"和"有没有异常"极性相反，答案不能出现裸的是/否。"""
    from aircraft_vqa.vqa import templates as TT
    for a in TT.A_DISCRIMINATION_POS + TT.A_DISCRIMINATION_NEG:
        assert not a.startswith(("是，", "是。", "否，", "否。")), a


def test_discrete_defects_keep_declared_severity(tmp_path):
    """缺一颗螺丝、一道裂纹，面积必然很小，严重度不能被面积调低。"""
    import json as _json
    from aircraft_vqa.adapters import build_adapter
    (tmp_path / "train").mkdir()
    Image.new("RGB", (1000, 1000), (120, 120, 120)).save(
        tmp_path / "train" / "a.jpg")
    ann = {"images": [{"id": 0, "file_name": "a.jpg", "width": 1000,
                       "height": 1000}],
           "categories": [{"id": 1, "name": "missing_fastener"},
                          {"id": 2, "name": "crack"},
                          {"id": 3, "name": "paint_peel_off"}],
           "annotations": [
               {"id": 1, "image_id": 0, "category_id": 1,
                "bbox": [10, 10, 12, 12], "area": 144, "iscrowd": 0},
               {"id": 2, "image_id": 0, "category_id": 2,
                "bbox": [50, 50, 14, 14], "area": 196, "iscrowd": 0},
               {"id": 3, "image_id": 0, "category_id": 3,
                "bbox": [100, 100, 15, 15], "area": 225, "iscrowd": 0}]}
    (tmp_path / "train" / "_annotations.coco.json").write_text(
        _json.dumps(ann), encoding="utf-8")
    ad = build_adapter({"adapter": "coco", "root": str(tmp_path), "name": "t",
                        "license": "x", "commercial_ok": True,
                        "category": "panel", "splits": ["train"]}, taxonomy=TAX)
    sev = {d.type: d.severity for s in ad.iter_samples() for d in s.defects}
    assert sev["fastener_missing"] == "major"      # 不因面积小被降成 minor
    assert sev["crack"] == "critical"
    assert sev["paint_peeling"] == "minor"         # 面状缺陷仍可按面积浮动


def test_area_still_scales_area_type_defects():
    from aircraft_vqa.geometry import severity_from_area
    assert TAX.area_scales_severity("corrosion")
    assert severity_from_area("major", 0.20) == "critical"   # 大面积升档
    assert not TAX.area_scales_severity("fastener_missing")


# ------------------------------------------------------------------ ShareGPT
def test_sharegpt_single_turn_shape(tmp_path):
    from aircraft_vqa.export.qwen3vl import to_sharegpt
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    r = next(x for x in b.build(_sample(tmp_path)) if not x.get("turns"))
    out = to_sharegpt(r)
    assert ["conversations", "images", "system"] == list(out)[:3]
    assert [c["from"] for c in out["conversations"]] == ["human", "gpt"]
    assert out["conversations"][0]["value"].startswith("<image>")
    assert out["conversations"][1]["value"] == r["answer"]


def test_sharegpt_multi_turn_alternates_and_has_one_image_token(tmp_path):
    from aircraft_vqa.export.qwen3vl import to_sharegpt
    cfg = BuildConfig(max_qa_per_sample=12)
    cfg.task_weights = dict(cfg.task_weights, multi_turn=1.0)
    b = VQABuilder(cfg, TAX)
    r = next(x for x in b.build(_sample(tmp_path)) if x["task"] == "multi_turn")
    out = to_sharegpt(r)
    froms = [c["from"] for c in out["conversations"]]
    assert froms == ["human", "gpt"] * r["n_turns"]
    assert sum(c["value"].count("<image>") for c in out["conversations"]) == 1


def test_sharegpt_without_system(tmp_path):
    from aircraft_vqa.export.qwen3vl import to_sharegpt
    b = VQABuilder(BuildConfig(max_qa_per_sample=12), TAX)
    r = b.build(_sample(tmp_path))[0]
    assert "system" not in to_sharegpt(r, with_system=False)


def test_sharegpt_registered_and_is_default():
    import yaml as _yaml
    from aircraft_vqa.export import EXPORTERS
    assert "sharegpt" in EXPORTERS
    cfg = _yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "build.yaml"), encoding="utf-8"))
    assert cfg["export"]["format"] == "sharegpt"


# ------------------------------------------------------------------ 问法语言
def test_lang_zh_filters_english_questions(tmp_path):
    b = VQABuilder(BuildConfig(max_qa_per_sample=12, lang="zh"), TAX)
    for pool in b.qpool.values():
        assert pool                                  # 过滤后不能把池子清空
        for q in pool:
            assert not VQABuilder._is_en(q), q
    for r in b.build(_sample(tmp_path)):
        assert not VQABuilder._is_en(r["question"])
        assert '"label": "' in r["answer"] or True   # 中文问 -> 中文标签
        for m in __import__("re").finditer(r'"label":\s*"([^"]+)"', r["answer"]):
            assert not m.group(1).isascii(), r["answer"]


def test_lang_bilingual_keeps_english_questions():
    b = VQABuilder(BuildConfig(lang="bilingual"), TAX)
    assert any(VQABuilder._is_en(q)
               for pool in b.qpool.values() for q in pool)


# ------------------------------------------------------------------ 干扰项范围
def test_mc_distractors_stay_within_active_scope(tmp_path):
    """选项里不能冒出本期不训练的缺陷类型。"""
    active = ["fastener_missing", "fastener_loose", "thread_damage", "crack",
              "corrosion", "dent", "scratch", "paint_peeling"]
    active_zh = {TAX.zh(t) for t in active}
    cfg = BuildConfig(max_qa_per_sample=12, active_defect_types=active)
    cfg.task_weights = dict(cfg.task_weights, classification_mc=1.0)
    b = VQABuilder(cfg, TAX)
    n = 0
    for dt in active:
        for r in b.build(_sample(tmp_path, "anomalous", dt)):
            if r["task"] == "classification_mc":
                n += 1
                assert set(r["options"]) <= active_zh, r["options"]
                assert check_record(r) == []
    assert n >= 4


def test_mc_shrinks_options_when_pool_too_small(tmp_path):
    """可选类型不足时少出几个选项，而不是塞兜底项凑数。"""
    cfg = BuildConfig(max_qa_per_sample=12,
                      active_defect_types=["crack", "dent"], n_options=4)
    cfg.task_weights = dict(cfg.task_weights, classification_mc=1.0)
    b = VQABuilder(cfg, TAX)
    for r in b.build(_sample(tmp_path, "anomalous", "crack")):
        if r["task"] == "classification_mc":
            assert set(r["options"]) == {"裂纹", "凹坑"}
            assert r["answer"].startswith(r["answer_letter"])
            assert check_record(r) == []


# ================================================================== 评审整改
def _build_all(tmp_path, **cfg_kw):
    """把所有任务权重拉满，一次性拿到各类样本。"""
    cfg = BuildConfig(max_qa_per_sample=30, **cfg_kw)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    out = []
    for label in ("anomalous", "normal"):
        for dt in ("fastener_missing", "crack", "corrosion", "scratch"):
            out += b.build(_sample(tmp_path, label, dt, n=2))
    return out


def test_grounding_answers_are_pure_json(tmp_path):
    """定位族 system 写了"不附加任何解释"，答案就必须是纯 JSON。"""
    for r in _build_all(tmp_path):
        if r["task"].startswith("grounding"):
            a = r["answer"].strip()
            assert a.startswith("[") and a.endswith("]"), (r["task"], a)
            json.loads(a)


def test_system_is_bound_to_output_format(tmp_path):
    """system 按输出格式绑，不按任务语义族绑 —— 语义族里混着两种输出格式。"""
    from aircraft_vqa.vqa.templates import OUTPUT_FORMAT, SYSTEM_BY_OUTPUT
    seen = set()
    for r in _build_all(tmp_path):
        ofmt = OUTPUT_FORMAT[r["task"]]
        seen.add(ofmt)
        assert r["output_format"] == ofmt, r["task"]
        assert r["system"] == SYSTEM_BY_OUTPUT[ofmt], r["task"]
        assert check_record(r) == [], (r["task"], check_record(r))
    assert {"json_only", "text", "dialog"} <= seen


def test_every_task_registers_output_format():
    """新增任务漏登记 output_format 会用错 system，这里卡死。"""
    from aircraft_vqa.vqa import templates as TT
    for task in TT.QUESTIONS:
        assert task in TT.OUTPUT_FORMAT, f"{task} 未登记输出格式"
    assert set(TT.OUTPUT_FORMAT.values()) <= set(TT.SYSTEM_BY_OUTPUT)


def test_text_tasks_never_emit_boxes(tmp_path):
    for r in _build_all(tmp_path):
        if r["output_format"] == "text":
            assert "bbox_2d" not in r["answer"], (r["task"], r["answer"])


def test_negative_answer_never_mentions_unasked_defect(tmp_path):
    """负样本模板复用正样本变量槽 -> 答案冒出问题里没提过的缺陷名。"""
    for r in _build_all(tmp_path):
        if r["task"] == "grounding_negative":
            assert r["answer"].strip() == "[]"
            assert check_record(r) == []


def test_counterfactual_asks_absent_type_and_answers_empty(tmp_path):
    n = 0
    for r in _build_all(tmp_path):
        if r["task"] == "grounding_counterfactual":
            n += 1
            assert r["answer"].strip() == "[]"
            assert r["absent_type"] not in r["image_defect_types"]
            assert r["asked_defect_types"] == [r["absent_type"]]
            assert r["answer_defect_types"] == []
            assert r["variant"] == "counterfactual"
            # 问的类型必须真的不在图里
            assert TAX.zh(r["absent_type"]) in r["question"]
            assert check_record(r) == []
    assert n >= 2, "反事实负样本没生成出来"


def test_counting_covers_zero_boundary(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    counts = set()
    for i in range(40):
        s = _sample(tmp_path, "anomalous", "crack", n=(i % 3) + 1)
        s.sample_id = f"t/count/{i}"
        for r in b.build(s):
            if r["task"] == "counting":
                counts.add(r["count"])
                assert check_record(r) == []
    assert 0 in counts and len(counts) >= 2, counts


def test_uncertainty_only_on_degraded_and_is_hedged(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    s = _sample(tmp_path, "anomalous", "crack")
    s.meta["image_quality"] = "blur"
    recs = b.build(s)
    assert recs and {r["task"] for r in recs} == {"uncertainty"}
    for r in recs:
        assert "bbox_2d" not in r["answer"]
        assert check_record(r) == []
    # 正常成像的图不该产出 uncertainty
    assert all(r["task"] != "uncertainty"
               for r in b.build(_sample(tmp_path, "anomalous", "crack")))


def test_no_fake_reasoning_phrases(tmp_path):
    """"依据是画面X处可见相应特征"是空话，等于教模型说套话装推理。"""
    for r in _build_all(tmp_path):
        text = r["answer"] + " ".join(
            t["answer"] for t in r.get("turns", []))
        assert "可见相应特征" not in text
        # 严重度是类型规则映射的，不是从图上判读的，不能声称"依据是画面…"
        assert "依据是画面" not in text


def test_no_airworthiness_verdicts(tmp_path):
    """严重度是按类型规则映射的，不是图上判读的，不能下适航结论。"""
    for r in _build_all(tmp_path):
        text = r["answer"] + " ".join(
            t["answer"] for t in r.get("turns", []))
        assert "不影响适航" not in text
        assert "可放行" not in text
    # 限定语有多个变体且只在一部分样本上带 —— 每条都带同一句会被当成
    # 机械后缀复读。这里只要求"确实有一部分带"。
    from aircraft_vqa.vqa import templates as TT
    adv = [r for r in _build_all(tmp_path)
           if r["task"] in ("severity_action", "grade_assessment")]
    if adv:
        withadv = sum(1 for r in adv
                      if any(v in r["answer"] for v in TT.ADVISORY_VARIANTS))
        assert 0 < withadv < len(adv) or len(adv) < 4


def test_discrimination_gives_count_for_multiple_defects(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    for r in b.build(_sample(tmp_path, "anomalous", "crack", n=3)):
        if r["task"] == "discrimination" and r["yes_no"] == "yes":
            assert "共 3 处" in r["answer"], r["answer"]


def test_hard_negative_region_is_near_defect():
    import random as _r
    import statistics
    from aircraft_vqa.geometry import iou
    from aircraft_vqa.vqa.negatives import sample_clean_box
    d = [[400, 300, 460, 360]]
    cx, cy = 430, 330

    def mean_dist(hard):
        rng = _r.Random(0)
        bs = [sample_clean_box(800, 600, d, rng, hard=hard) for _ in range(150)]
        bs = [b for b in bs if b]
        assert all(iou(b, d[0]) == 0.0 for b in bs)      # 仍然不重叠
        return statistics.mean(
            (((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2) ** 0.5
            for b in bs)

    assert mean_dist(True) < mean_dist(False) * 0.7


def test_mc_answer_letters_are_spread(tmp_path):
    """正确答案不能集中在某个字母上，否则模型直接背位置。"""
    from collections import Counter
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    letters = Counter()
    for i in range(120):
        s = _sample(tmp_path, "anomalous", ["crack", "corrosion", "dent",
                                            "scratch"][i % 4])
        s.sample_id = f"t/mc/{i}"
        for r in b.build(s):
            if r["task"] == "classification_mc":
                letters[r["answer_letter"]] += 1
    assert len(letters) == 4, letters
    lo, hi = min(letters.values()), max(letters.values())
    assert hi <= lo * 2.2, letters      # 分布不能太偏


def test_export_carries_id_and_meta(tmp_path):
    from aircraft_vqa.export import EXPORTERS
    r = _build_all(tmp_path)[0]
    for name, fn in EXPORTERS.items():
        out = fn(r)
        assert out.get("id") == r["qa_id"], name
        assert out["meta"]["task"] == r["task"], name
        assert "image_defect_types" in out["meta"], name
        assert "image_hw" in out["meta"], name
        assert "id" not in fn(r, with_meta=False), name


def test_term_annotation_is_off_by_default(tmp_path):
    for r in _build_all(tmp_path):
        assert "（missing fastener）" not in r["answer"]
    b = VQABuilder(BuildConfig(max_qa_per_sample=30, term_annotation=True), TAX)
    b.cfg.task_weights = {k: 1.0 for k in b.cfg.task_weights}
    texts = " ".join(r["answer"] for r in b.build(_sample(tmp_path)))
    assert "（" in texts


def test_uncertainty_fix_matches_cause(tmp_path):
    """遮挡却建议"在良好光照下补拍"是答非所问，补救措施要跟成因对上。"""
    from aircraft_vqa.vqa import templates as TT
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    assert set(TT.UNCERTAIN_FIX) == set(TT.UNCERTAIN_REASON)
    for q in TT.UNCERTAIN_REASON:
        s = _sample(tmp_path, "anomalous", "crack")
        s.sample_id = f"t/unc/{q}"
        s.meta["image_quality"] = q
        recs = b.build(s)
        assert recs
        for r in recs:
            assert TT.UNCERTAIN_FIX[q] in r["answer"], (q, r["answer"])
            assert check_record(r) == []
    # 遮挡的样本不该出现"光照"这种不相干的补救措施
    s = _sample(tmp_path, "anomalous", "crack")
    s.sample_id = "t/unc/occ2"
    s.meta["image_quality"] = "occlusion"
    assert all("照明" not in r["answer"] and "光照" not in r["answer"]
               for r in b.build(s))


# ================================================================== 二轮评审
def test_uncertainty_has_auditable_evidence(tmp_path):
    """"看不清"必须有物理依据，否则是在教模型耍赖 —— 机务场景的负向能力。"""
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    s = _sample(tmp_path, "anomalous", "crack")
    s.meta.update({"image_quality": "occlusion",
                   "degradation": {"type": "occlusion", "ratio": 0.42}})
    recs = b.build(s)
    assert recs
    for r in recs:
        assert r["degradation"]["ratio"] == 0.42
        assert check_record(r) == []
    # 没有劣化证据的 uncertainty 条目要被质检拦下
    bad = dict(recs[0]); bad.pop("degradation"); bad.pop("image_quality", None)
    assert "uncertainty_without_evidence" in check_record(bad)


def test_uncertainty_prefers_defects_present_in_image(tmp_path):
    """在劣化图上问一个不存在的类型，"没有"和"看不清"是混淆的。"""
    cfg = BuildConfig(max_qa_per_sample=30,
                      active_defect_types=["crack", "corrosion", "dent"])
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    s = _sample(tmp_path, "anomalous", "crack")
    s.meta["image_quality"] = "blur"
    asked = [r["asked_defect_types"][0] for r in b.build(s)]
    assert asked[0] == "crack"          # 图里真有的排在最前


def test_meta_defect_fields_are_semantically_separated(tmp_path):
    """image / asked / answer 三个缺陷字段混成一个会让分层评测全部失效。"""
    for r in _build_all(tmp_path):
        assert isinstance(r["image_defect_types"], list)
        assert isinstance(r["asked_defect_types"], list)
        assert isinstance(r["answer_defect_types"], list)
        # 幻觉闸门
        img, ans = set(r["image_defect_types"]), set(r["answer_defect_types"])
        assert not ans or ans <= img, (r["task"], ans, img)
        assert "label" not in r        # 已改名 image_status，避免与 bbox 的 label 混淆
        assert r["image_status"] in ("normal", "anomalous")


def test_qc_blocks_hallucinated_defect_assertion():
    bad = {"task": "grounding_all", "output_format": "json_only",
           "question": "q", "image_status": "anomalous",
           "image_defect_types": ["crack"], "answer_defect_types": ["corrosion"],
           "answer": '[{"bbox_2d": [1,2,3,4], "label": "腐蚀锈蚀"}]',
           "coord_mode": "norm1000"}
    assert "answer_asserts_absent_defect" in check_record(bad)


def test_multi_turn_disambiguates_reference(tmp_path):
    """前文有多处缺陷时，第三轮必须点名问哪一处。"""
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = dict(cfg.task_weights, multi_turn=1.0)
    b = VQABuilder(cfg, TAX)
    s = _sample(tmp_path, "anomalous", "crack", n=1)
    s.defects.append(Defect(type="corrosion", type_raw="corrosion",
                            type_zh=TAX.zh("corrosion"), bbox=[200, 20, 260, 80],
                            area_ratio=0.03, region="右下", severity="major"))
    r = next(x for x in b.build(s) if x["task"] == "multi_turn")
    t3q = r["turns"][-1]["question"]
    assert "裂纹" in t3q or "腐蚀锈蚀" in t3q, t3q     # 必须点名
    # 第一轮的方位要和缺陷一一对应
    t1a = r["turns"][0]["answer"]
    assert "裂纹位于画面" in t1a and "腐蚀锈蚀位于画面" in t1a, t1a


def test_single_defect_multi_turn_keeps_simple_reference(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = dict(cfg.task_weights, multi_turn=1.0)
    b = VQABuilder(cfg, TAX)
    r = next(x for x in b.build(_sample(tmp_path, "anomalous", "crack", n=1))
             if x["task"] == "multi_turn")
    assert "该缺陷" in r["turns"][-1]["question"] or "这个" in r["turns"][-1]["question"]


def test_mc_distractors_are_plausible_for_the_part(tmp_path):
    """拿螺纹损伤去干扰一张蒙皮图，不看图都能排掉。"""
    from aircraft_vqa.vqa.distractor import PART_DEFECTS
    active = ["fastener_missing", "fastener_loose", "thread_damage", "crack",
              "corrosion", "dent", "scratch", "paint_peeling"]
    cfg = BuildConfig(max_qa_per_sample=30, active_defect_types=active)
    cfg.task_weights = dict(cfg.task_weights, classification_mc=1.0)
    b = VQABuilder(cfg, TAX)
    allowed = {TAX.zh(t) for t in PART_DEFECTS["structure"]}
    n = 0
    for dt in ("crack", "corrosion", "dent", "scratch"):
        for r in b.build(_sample(tmp_path, "anomalous", dt)):
            if r["task"] == "classification_mc":
                n += 1
                assert set(r["options"]) <= allowed, r["options"]
    assert n >= 3


def test_pair_compare_emits_two_images(tmp_path):
    ref = tmp_path / "ref.jpg"
    Image.new("RGB", (400, 300), (128, 128, 128)).save(ref)
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = dict(cfg.task_weights, pair_compare=1.0)
    b = VQABuilder(cfg, TAX)
    b.set_reference_pool({("t", "panel"): [str(ref)] * 3})
    r = next(x for x in b.build(_sample(tmp_path, "anomalous", "crack"))
             if x["task"] == "pair_compare")
    assert len(r["images"]) == 2 and r["images"][0] == str(ref)
    from aircraft_vqa.export.qwen3vl import to_sharegpt
    out = to_sharegpt(r)
    assert len(out["images"]) == 2
    assert out["conversations"][0]["value"].count("<image>") == 2
    assert check_record(r) == []


def test_pair_compare_absent_without_reference_pool(tmp_path):
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = dict(cfg.task_weights, pair_compare=1.0)
    b = VQABuilder(cfg, TAX)          # 没注入参考图池
    assert all(r["task"] != "pair_compare"
               for r in b.build(_sample(tmp_path, "anomalous", "crack")))


def test_builder_records_template_failures(tmp_path):
    """模板异常以前被静默吞掉，整类任务凭空消失都没人发现。"""
    cfg = BuildConfig(max_qa_per_sample=30)
    b = VQABuilder(cfg, TAX)
    b._t_describe = lambda s, rng: (_ for _ in ()).throw(KeyError("boom"))
    cfg.task_weights = dict(cfg.task_weights, description=1.0)
    b.build(_sample(tmp_path))
    assert any("description" in k for k in b.failures), b.failures
    # strict 模式直接抛
    b.cfg.strict = True
    with pytest.raises(KeyError):
        b.build(_sample(tmp_path, "anomalous", "corrosion"))


def test_diversity_metrics_flag_repetition():
    from aircraft_vqa.balance import diversity
    same = [{"task": "t", "question": "问题", "answer": "完全一样的答案"}] * 50
    d = diversity(same)
    assert d["unique_answer_ratio"] < 0.05
    assert d["top20_answer_share"] == 1.0
    varied = [{"task": "t", "question": f"问题{i}", "answer": f"答案{i}不同的内容"}
              for i in range(50)]
    assert diversity(varied)["unique_answer_ratio"] == 1.0


def test_negative_breakdown_separates_two_abilities():
    from aircraft_vqa.balance import negative_breakdown
    recs = ([{"n_boxes": 0, "image_status": "normal"}] * 3 +
            [{"n_boxes": 0, "image_status": "anomalous"}] * 2 +
            [{"n_boxes": 2, "image_status": "anomalous"}] * 5)
    b = negative_breakdown(recs)
    assert b["on_normal_image"] == 3 and b["counterfactual"] == 2
    assert b["total_negative"] == 5


def test_all_answer_templates_are_fully_formatted(tmp_path):
    """答案模板加了占位符却漏 format，会静默产出带 {obj} 的脏数据。"""
    import re as _re
    ref = tmp_path / "ref.jpg"
    Image.new("RGB", (400, 300), (128, 128, 128)).save(ref)
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    b.set_reference_pool({("t", "panel"): [str(ref)] * 3})
    n = 0
    for label in ("anomalous", "normal"):
        for dt in ("crack", "corrosion", "scratch"):
            for r in b.build(_sample(tmp_path, label, dt)):
                n += 1
                text = r["answer"] + " ".join(
                    t["answer"] for t in r.get("turns", []))
                assert not _re.search(r"\{[a-z_]+\}", text), (r["task"], text)
    assert n > 20


def test_single_turn_reference_is_unambiguous(tmp_path):
    """图里有多处缺陷时，单轮问法也必须点名 —— 不然"该缺陷"没有唯一先行词。"""
    cfg = BuildConfig(max_qa_per_sample=30)
    cfg.task_weights = {k: 1.0 for k in cfg.task_weights}
    b = VQABuilder(cfg, TAX)
    s = _sample(tmp_path, "anomalous", "crack", n=1)
    s.defects.append(Defect(type="corrosion", type_raw="corrosion",
                            type_zh=TAX.zh("corrosion"), bbox=[200, 20, 260, 80],
                            area_ratio=0.03, region="右下", severity="major"))
    for r in b.build(s):
        if r["task"] == "severity_action":
            assert any(w in r["question"] for w in ("裂纹", "腐蚀锈蚀", "画面")), \
                r["question"]
        if r["task"] == "classification_open":
            assert "画面" in r["question"] or "区域" in r["question"], r["question"]
    # 单一缺陷时可以用通用指代
    r1 = [x for x in b.build(_sample(tmp_path, "anomalous", "crack", n=1))
          if x["task"] == "severity_action"]
    assert r1


def test_no_colloquial_questions_in_any_pool():
    """用户明确要求问法偏指令性；扩模板时很容易把口语化的加回来。"""
    from aircraft_vqa.vqa import templates as TT
    banned = ["帮我", "看一下这个", "看看这张", "说一下。", "大概怎么",
              "要紧吗", "有没有毛病", "放一起看", "怎么样？", "能直接装",
              "给个框", "哪个部位的件", "是好的", "有几个？", "呗", "咋"]
    pools = list(TT.QUESTIONS.values()) + [
        TT.Q_MT_T2_POS, TT.Q_MT_T2_NEG, TT.Q_MT_T3_ONE, TT.Q_MT_T3_MANY]
    for pool in pools:
        for q in pool:
            for w in banned:
                assert w not in q, f"口语化问法：{q}（命中 {w}）"


def test_protected_tasks_derived_from_output_format():
    """保护名单手维护必然漏 —— grounding_counterfactual 就漏过一次。"""
    from aircraft_vqa.vqa import templates as TT
    from aircraft_vqa.vqa.llm import PROTECTED_TASKS
    for task, fmt in TT.OUTPUT_FORMAT.items():
        if fmt in ("json_only", "text_then_json", "dialog"):
            assert task in PROTECTED_TASKS, f"{task}({fmt}) 应被保护"
    assert "classification_mc" in PROTECTED_TASKS      # 答案是"字母. 选项"
    # 纯 text 任务才参与改写
    assert "description" not in PROTECTED_TASKS
    assert "severity_action" not in PROTECTED_TASKS


def test_rewriter_rejects_fact_drift():
    from aircraft_vqa.vqa.llm import LLMRewriter
    rw = LLMRewriter("none")
    rw.provider = "openai"
    rw._call = lambda t: "画面左上有 5 处裂纹。"      # 数字被改了
    r = {"task": "description", "output_format": "text", "question": "q",
         "answer": "画面左上有 3 处裂纹。"}
    rw.rewrite(r)
    assert r["answer"] == "画面左上有 3 处裂纹。"
    assert r["rewrite_rejected"] == "fact_drift"


def test_rewriter_accepts_faithful_rewrite():
    from aircraft_vqa.vqa.llm import LLMRewriter
    rw = LLMRewriter("none")
    rw.provider = "openai"
    rw._call = lambda t: "画面左上共发现 3 处裂纹。"   # 数字一致，句式变了
    r = {"task": "description", "output_format": "text", "question": "q",
         "answer": "画面左上有 3 处裂纹。"}
    rw.rewrite(r)
    assert r.get("rewritten") is True
    assert r["answer_template"] == "画面左上有 3 处裂纹。"


def test_roboflow_versions_are_pinned_not_latest():
    """Roboflow 的 latest 往往是单类或增广版本 —— 必须钉死版本号。

    aircraft_skin_defects 的 latest(v23) 是 single class defect，
    Missing-head 这个唯一命中"螺丝缺失"的标注会直接丢掉。
    """
    da = _load_script(os.path.join("download", "download_all.py"))
    rf = [s for s in da.SOURCES if "rf" in s]
    assert rf
    for s in rf:
        v = s["rf"]["version"]
        assert v != "latest", f"{s['name']} 的版本没钉死"
        assert v.isdigit(), f"{s['name']} 的版本应为具体数字，收到 {v}"


def test_no_python_syntax_errors_in_scripts():
    """中文引号写成 ASCII 双引号会把字符串截断 —— 这个坑踩过两次。"""
    import py_compile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dirpath, _, files in os.walk(os.path.join(root, "scripts")):
        if "__pycache__" in dirpath:
            continue
        for fn in files:
            if fn.endswith(".py"):
                py_compile.compile(os.path.join(dirpath, fn), doraise=True)


def test_no_answer_template_lacks_a_placeholder():
    """答案模板必须带占位符，否则几千条样本共用一句一字不差的话。

    真实踩到的坑：5 条没有占位符的模板直接构成了 top5 答案 ——
    "该区域未发现缺陷，外观正常。" 一句出现 2066 次。模型从这种数据里
    学到的是复读，不是判读。
    """
    import re

    from aircraft_vqa.vqa import templates as T
    bad = []
    for name in dir(T):
        if not name.startswith("A_"):
            continue
        v = getattr(T, name)
        if not isinstance(v, list):
            continue
        for t in v:
            if isinstance(t, str) and not re.search(r"\{\w+\}", t):
                bad.append(f"{name}: {t}")
    assert not bad, "这些答案模板没有占位符，会整批重复：\n" + "\n".join(bad)


def test_negative_answer_pools_are_wide_enough():
    """无缺陷类回答占样本的大头，模板太少直接顶死 top20 占比。"""
    from aircraft_vqa.vqa import templates as T
    assert len(T.A_PAIR_COMPARE_NEG) >= 12, len(T.A_PAIR_COMPARE_NEG)
    assert len(T.A_REGION_NEGATIVE) >= 8, len(T.A_REGION_NEGATIVE)
    assert len(T.A_DISCRIMINATION_NEG) >= 10, len(T.A_DISCRIMINATION_NEG)


def test_every_test_file_runs_standalone():
    """顺序依赖的测试等于没测 —— 它只是碰巧被别的文件先把 sys.path 铺好。

    真实踩到两次：test_config_overlay 和 test_adapters_real_layout 里
    加载 scripts/preflight.py 的那两条，单独跑都会 ModuleNotFoundError，
    全量跑却是绿的。
    """
    import glob
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    me = os.path.basename(__file__)
    bad = []
    for f in sorted(glob.glob(os.path.join(root, "tests", "test_*.py"))):
        if os.path.basename(f) == me:      # 别递归跑自己
            continue
        r = subprocess.run([sys.executable, "-m", "pytest", f, "-q",
                            "--no-header", "-x", "--collect-only"],
                           cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{os.path.basename(f)}: {r.stdout[-300:]}")
    assert not bad, "这些文件单独收集就失败：\n" + "\n".join(bad)


def test_qc_catches_duplicated_words():
    """真实踩到的坑：size_word 返回的是完整短语"范围中等"，
    builder 的 f-string 又补了个"范围"，拼出"范围范围中等"。
    这种错误在模板扫描里看不到（它在 f-string 里），只读 stats 也看不出来，
    只有把答案原文读出来才发现。
    """
    from aircraft_vqa.qc import check_record
    bad = {"question": "这是什么缺陷？",
           "answer": "该异常是螺纹损伤，位于画面正中，范围范围中等。"}
    errs = check_record(bad)
    assert any("duplicated_word" in e for e in errs), errs
    ok = {"question": "这是什么缺陷？",
          "answer": "该异常是螺纹损伤，位于画面正中，范围中等。"}
    assert not [e for e in check_record(ok) if "duplicated" in e]


def test_no_template_double_prefixes_a_phrase_helper():
    """size_word/severity 这类返回完整短语的工具，前面不许再补前缀 ——
    模板和 builder 的 f-string 都要查。"""
    import glob
    import re as _re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        src = open(f, encoding="utf-8").read()
        for m in _re.finditer(r"范围\s*\{size[^}]*\}|范围\s*\{size_word", src):
            bad.append(f"{os.path.basename(f)}: {m.group(0)}")
        for m in _re.finditer(r"范围\{size_word\(", src):
            bad.append(f"{os.path.basename(f)}: {m.group(0)}")
    assert not bad, "这些地方会拼出叠词：" + "; ".join(bad)


def test_size_word_returns_a_complete_phrase():
    """约定：size_word 返回完整短语，调用方直接用，不要加前缀。"""
    from aircraft_vqa.geometry import size_word
    for r in (0.0005, 0.005, 0.02, 0.1, 0.3):
        assert size_word(r).startswith("范围"), size_word(r)


def test_long_running_scripts_force_line_buffering():
    """重定向到文件时 Python 默认块缓冲，要攒够几 KB 才落盘。

    真实踩到：nohup 跑七小时的改写，tail -f 看到的是空文件，
    看起来像没在跑。长任务的脚本必须自己强制行缓冲，
    不能指望调用方记得加 -u。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("build_vqa.py", "llm_smoke.py", "gen_question_bank.py"):
        src = open(os.path.join(root, "scripts", name), encoding="utf-8").read()
        assert "line_buffering=True" in src, f"{name} 没强制行缓冲"


def test_output_appears_immediately_when_redirected(tmp_path):
    """真发一次：重定向到文件后立刻就该有内容。"""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = tmp_path / "x.log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(root, "scripts",
                                                     "build_vqa.py"), "--help"],
                       stdout=f, stderr=subprocess.STDOUT, cwd=root, timeout=60)
    assert log.read_text(encoding="utf-8").strip(), "重定向后日志是空的"
