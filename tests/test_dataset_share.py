# -*- coding: utf-8 -*-
"""按数据源限流：别让某一家把域先验带偏。

真实情形：Real-IAD 两万条占全库 53%，拍的是灯箱里的端子排和电路板，
而指标要的是飞机蒙皮上的缺陷识别。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from aircraft_vqa.balance import cap_dataset_share
from aircraft_vqa.schema import UnifiedSample


def mk(ds, n, n_anom):
    return [UnifiedSample(sample_id=f"{ds}/{i}", image_path="/x.jpg",
                          width=512, height=512, dataset=ds, category="c",
                          label="anomalous" if i < n_anom else "normal")
            for i in range(n)]


def share(samples):
    c = Counter(s.dataset for s in samples)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def test_dominant_source_is_capped():
    # 25% 的上限至少要 4 个源才有解，这里给 5 个
    samples = mk("big", 2000, 1000)
    for k in "abcd":
        samples += mk(k, 300, 150)
    out, rep = cap_dataset_share(samples, 0.25)
    assert share(out)["big"] <= 0.2501, share(out)
    assert rep["big"]["before"] == 2000
    assert rep["big"]["after"] < 2000
    assert rep["a"]["after"] == 300, "没超限的源不能被动"
    assert "_note" not in rep, "这个上限是有解的，不该报无解"


def test_infeasible_cap_is_explained_not_collapsed():
    """n 个源不可能人人都低于 1/n。这种上限无解，要按均分处理并说明，
    绝不能一轮轮越砍越少最后收敛到 0 条。"""
    samples = mk("big", 2000, 1000) + mk("small", 100, 50)
    out, rep = cap_dataset_share(samples, 0.25)      # 2 个源，25% 无解
    assert out, "被砍空了"
    sh = share(out)
    assert abs(sh["big"] - 0.5) < 0.01, sh
    assert "_note" in rep and "无解" in rep["_note"], rep.get("_note")


def test_cap_preserves_within_source_anomaly_ratio():
    """分层截断 —— 整体下采样会把正负比也一起改掉。"""
    samples = mk("big", 2000, 500) + mk("a", 300, 150) + mk("b", 300, 150)
    out, _ = cap_dataset_share(samples, 0.25, seed=1)
    big = [s for s in out if s.dataset == "big"]
    got = sum(s.is_anomalous for s in big) / len(big)
    assert abs(got - 0.25) < 0.05, got          # 原比例 500/2000 = 0.25


def test_multiple_over_sources_all_land_under_cap():
    """砍了 A 会改变 B 的分母，逐个处理必然互相算错 —— 必须一起解。"""
    samples = mk("a", 1000, 500) + mk("b", 1000, 500) + mk("c", 50, 25)
    out, _ = cap_dataset_share(samples, 0.4)
    sh = share(out)
    assert all(v <= 0.4001 for v in sh.values()), sh
    # 没超限的源一条都不该少
    assert sum(1 for s in out if s.dataset == "c") == 50


def test_cap_keeps_as_much_as_possible():
    """满足上限的前提下要尽量多留 —— 砍过头等于白扔数据。"""
    samples = mk("a", 1000, 500) + mk("b", 1000, 500) + mk("c", 50, 25)
    out, _ = cap_dataset_share(samples, 0.4)
    # 解析解：c=50 固定，a=b=x，x/(2x+50)<=0.4 => x<=100
    assert sum(1 for s in out if s.dataset == "a") == 100
    assert len(out) == 250


def test_weights_oversample_and_downsample():
    samples = mk("tiny", 100, 50) + mk("mid", 400, 200)
    out, rep = cap_dataset_share(samples, 0, {"tiny": 3.0, "mid": 0.5})
    assert rep["tiny"]["after"] == 300
    assert rep["mid"]["after"] == 200


def test_weight_zero_drops_source():
    out, rep = cap_dataset_share(mk("x", 50, 25) + mk("y", 50, 25), 0, {"y": 0})
    assert rep["y"]["after"] == 0
    assert all(s.dataset == "x" for s in out)


def test_no_config_is_a_noop():
    samples = mk("a", 100, 50) + mk("b", 200, 100)
    out, rep = cap_dataset_share(samples, 0, None)
    assert len(out) == 300
    assert all(r["before"] == r["after"] for r in rep.values())


def test_deterministic():
    samples = mk("big", 800, 400) + mk("s", 100, 50)
    a, _ = cap_dataset_share(samples, 0.25, seed=7)
    b, _ = cap_dataset_share(samples, 0.25, seed=7)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]


def test_shipped_config_caps_the_dominant_source():
    """仓库里那份配置必须真的把 real_iad 压下来。"""
    import io

    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(io.open(os.path.join(root, "configs", "build.yaml"),
                                 encoding="utf-8"))
    cap = cfg.get("max_share_per_dataset")
    assert cap and 0 < cap < 0.5, cap
    # 用真实条数模拟一遍，确认航空域占比真的被顶上去了
    real = [("visa", 4416, 400), ("mvtec_ad_screw", 815, 212),
            ("mvtec_loco_screwbag", 761, 219), ("real_iad", 20100, 7505),
            ("aircraft_skin_defects", 372, 364),
            ("uts_aircraft_defect", 6803, 6803), ("corrosion_cs_vt", 440, 440),
            ("synthetic_panel", 2800, 1966), ("synthetic_closeup", 1200, 990)]
    air = {"aircraft_skin_defects", "uts_aircraft_defect",
           "synthetic_panel", "synthetic_closeup"}
    samples = [s for ds, n, a in real for s in mk(ds, n, a)]
    before = sum(1 for s in samples if s.dataset in air) / len(samples)
    out, _ = cap_dataset_share(samples, cap, cfg.get("dataset_weights"), 0)
    after = sum(1 for s in out if s.dataset in air) / len(out)
    assert before < 0.32, before
    assert after > 0.45, (before, after)


# ---------------------------------------------------------------- 类别长尾
def _cls_recs(counts):
    return [{"task": "classification_open", "target_type": k,
             "qa_id": f"{k}/{i}", "question": "q", "answer": "a"}
            for k, n in counts.items() for i in range(n)]


# 真实跑出来的分布
REAL = {"dent": 1757, "crack": 1662, "corrosion": 627, "scratch": 513,
        "fastener_missing": 357, "thread_damage": 340,
        "fastener_loose": 123, "paint_peeling": 91}


def test_rare_class_does_not_drag_everything_down():
    """真实踩到的坑：按**最少类**算封顶，paint_peeling 只有 91 条，
    cap=273，于是 dent/crack 各被砍掉 1400+，类型题整体丢 66%，
    而且扔掉的恰恰是标注最好的那些。"""
    from aircraft_vqa.balance import cap_class_imbalance
    out = cap_class_imbalance(_cls_recs(REAL), {"classification_open"}, 3.0)
    real_kept = sum(1 for r in out if not r.get("oversampled"))
    assert real_kept / sum(REAL.values()) > 0.9, real_kept


def test_imbalance_is_actually_reduced():
    from aircraft_vqa.balance import cap_class_imbalance
    out = cap_class_imbalance(_cls_recs(REAL), {"classification_open"}, 3.0)
    c = Counter(r["target_type"] for r in out)
    before = max(REAL.values()) / min(REAL.values())
    after = max(c.values()) / min(c.values())
    assert after < before / 3, (before, after)


def test_tail_is_oversampled_not_head_slashed():
    from aircraft_vqa.balance import cap_class_imbalance
    out = cap_class_imbalance(_cls_recs(REAL), {"classification_open"}, 3.0)
    c = Counter(r["target_type"] for r in out)
    assert c["paint_peeling"] > REAL["paint_peeling"], "长尾没顶上来"
    assert c["fastener_loose"] > REAL["fastener_loose"]


def test_oversample_factor_is_capped():
    """重复同一条问答会助长记忆，倍数必须封顶。"""
    from aircraft_vqa.balance import cap_class_imbalance
    counts = {"big": 3000, "mid": 1000, "tiny": 10}
    out = cap_class_imbalance(_cls_recs(counts), {"classification_open"},
                              3.0, max_oversample=2.0)
    c = Counter(r["target_type"] for r in out)
    assert c["tiny"] <= 10 * 2, c["tiny"]


def test_oversampled_rows_get_distinct_ids():
    """复制出来的行必须有各自的 qa_id，否则会被 dedup 当成同一条抹掉。"""
    from aircraft_vqa.balance import cap_class_imbalance, dedup
    out = cap_class_imbalance(_cls_recs(REAL), {"classification_open"}, 3.0)
    os_rows = [r for r in out if r.get("oversampled")]
    assert os_rows
    assert len({r["qa_id"] for r in out}) == len(out), "qa_id 撞了"


def test_other_tasks_untouched():
    from aircraft_vqa.balance import cap_class_imbalance
    recs = _cls_recs(REAL) + [{"task": "grounding_single", "qa_id": f"g/{i}"}
                              for i in range(500)]
    out = cap_class_imbalance(recs, {"classification_open"}, 3.0)
    assert sum(1 for r in out if r["task"] == "grounding_single") == 500


def test_disabled_when_ratio_not_positive():
    from aircraft_vqa.balance import cap_class_imbalance
    recs = _cls_recs(REAL)
    assert len(cap_class_imbalance(recs, set(), 3.0)) == len(recs)


# ---------------------------------------------------------------- 划分隔离
def _recs(n_samples, pair_rate, seed=0):
    """模拟真实构成：一部分 pair_compare 额外带一张别的样本的参考图。"""
    import random
    rng = random.Random(seed)
    imgs = [f"/img/{i}.jpg" for i in range(n_samples)]
    out = []
    for i in range(n_samples):
        for _ in range(4):
            r = {"sample_id": f"s/{i}", "images": [imgs[i]], "task": "x"}
            if rng.random() < pair_rate:
                r = {"sample_id": f"s/{i}", "task": "pair_compare",
                     "images": [imgs[i], rng.choice(imgs)]}
            out.append(r)
    return out


def _leak(splits):
    from itertools import combinations
    sets = {k: {p for r in v for p in r.get("images", [])}
            for k, v in splits.items()}
    return sum(len(sets[a] & sets[b]) for a, b in combinations(sets, 2))


def test_pair_compare_reference_image_does_not_cross_splits():
    """真实踩到的坑：pair_compare 从参考图池额外取一张正常件图，
    那张图属于另一个 sample。于是 A 的图在 train 当主体、
    同时在 test 给 B 当参考图 —— 模型训练时见过，测试集就是漏的。
    出厂检查在真实构建产物里抓到过（train 与 test 共用 2 张图）。"""
    from aircraft_vqa.balance import group_split
    assert _leak(group_split(_recs(3000, 0.10), (0.9, 0.05, 0.05), 0)) == 0


def test_no_leak_across_pair_rates_and_seeds():
    from aircraft_vqa.balance import group_split
    for rate in (0.0, 0.05, 0.2, 0.5):
        for seed in (0, 7):
            s = group_split(_recs(1500, rate, seed), (0.9, 0.05, 0.05), seed)
            assert _leak(s) == 0, (rate, seed)


def test_splits_are_not_degenerate_at_scale():
    """隔离不能靠把所有东西塞进 train 来实现。"""
    from aircraft_vqa.balance import group_split
    s = group_split(_recs(20000, 0.08), (0.95, 0.03, 0.02), 0)
    assert len(s["val"]) > 200, len(s["val"])
    assert len(s["test"]) > 200, len(s["test"])


def test_same_sample_still_stays_together():
    """同一个样本的多条问答仍必须在同一划分里。"""
    from aircraft_vqa.balance import group_split
    s = group_split(_recs(2000, 0.08), (0.9, 0.05, 0.05), 0)
    where = {}
    for name, rows in s.items():
        for r in rows:
            prev = where.setdefault(r["sample_id"], name)
            assert prev == name, f"{r['sample_id']} 同时出现在 {prev} 和 {name}"


def test_records_without_images_still_split():
    """没有 images 字段的记录不能被漏掉。"""
    from aircraft_vqa.balance import group_split
    recs = [{"sample_id": f"s/{i}", "task": "x"} for i in range(500)]
    s = group_split(recs, (0.9, 0.05, 0.05), 0)
    assert sum(len(v) for v in s.values()) == 500


def test_build_handles_note_entry_in_report(tmp_path):
    """cap_dataset_share 在源数少于 1/max_share 时会往报告里塞 _note 字符串，
    build_vqa 原来把报告里每一项都当 dict 取，源少时直接崩：

        TypeError: string indices must be integers

    用 --only 跑少数几个源就会踩到。
    """
    from aircraft_vqa.balance import cap_dataset_share
    samples = mk("a", 500, 250) + mk("b", 500, 250)
    _, rep = cap_dataset_share(samples, 0.25)
    assert "_note" in rep and isinstance(rep["_note"], str)
    counts = {k: v for k, v in rep.items() if isinstance(v, dict)}
    assert sum(r["after"] for r in counts.values()) > 0
    # 这一行就是崩过的那句
    assert sorted(counts.items(), key=lambda kv: -kv[1]["after"])
