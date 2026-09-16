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
