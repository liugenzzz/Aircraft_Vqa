# -*- coding: utf-8 -*-
"""扩写的问法必须真的被用上。

真实踩到的坑：classification_open 和 severity_action 从硬编码常量里取问法，
再拿合并池跟常量求交集 —— 扩写出来的问法不在常量里，全被滤掉。
库里存着 59 条，一条没用上，实测 82->81、77->79，等于白扩，而且不报错。
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest

from aircraft_vqa.schema import Defect, UnifiedSample
from aircraft_vqa.taxonomy import get_taxonomy
from aircraft_vqa.vqa import templates as T
from aircraft_vqa.vqa.builder import BuildConfig, VQABuilder

TAX = get_taxonomy()


def _sample(sid, types, region="左上"):
    return UnifiedSample(
        sample_id=sid, image_path="/x.jpg", width=800, height=600,
        label="anomalous", dataset="synthetic_panel", category="panel",
        object_name="panel", object_zh="检查口盖", aircraft_ctx="可卸口盖壁板",
        split="train", commercial_ok=True,
        defects=[Defect(type=t, type_raw=t, type_zh=TAX.zh(t),
                        bbox=[100, 100, 200, 200], area_ratio=0.02,
                        severity="major", region=region, grade="")
                 for t in types])


def _build(bank_extra, n=300):
    """用带扩写的问法库跑一批，返回 task -> 出现过的问句集合。"""
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump({"questions": bank_extra}, f, ensure_ascii=False)
        path = f.name
    try:
        b = VQABuilder(BuildConfig(question_bank=path, max_qa_per_sample=8), TAX)
        seen = {}
        rng = random.Random(0)
        for i in range(n):
            types = (["crack"] if i % 2 else ["crack", "corrosion"])
            for r in b.build(_sample(f"s/{i}", types,
                                     rng.choice(["左上", "正中", "右下"]))):
                seen.setdefault(r["task"], set()).add(r["question"])
        return seen
    finally:
        os.unlink(path)


MARKER = "【扩写标记】"


@pytest.mark.parametrize("task,extra", [
    ("classification_open", [MARKER + "请判定该{obj}上异常的缺陷类别。",
                             MARKER + "请说明画面{region}处缺陷的类别。"]),
    ("severity_action", [MARKER + "请评估{defect}的严重程度并给出处置建议。",
                         MARKER + "请说明该缺陷的等级与处理方式。"]),
    ("description", [MARKER + "请记录该{obj}的目视检查结果。"]),
    ("discrimination", [MARKER + "请判断该{obj}外观是否正常。"]),
])
def test_expanded_questions_actually_get_used(task, extra):
    """库里给了，输出里就得见得到 —— 不然扩写等于白花钱且不报错。"""
    seen = _build({task: extra})
    assert task in seen, f"{task} 一条都没生成"
    used = [q for q in seen[task] if MARKER in q]
    assert used, (f"{task} 的扩写问法一条都没被用上（库里 {len(extra)} 条，"
                  f"输出里 {len(seen[task])} 种问句）")


def test_multi_defect_images_still_get_named_questions():
    """多种缺陷同时存在时，问法必须点名问哪一处，否则指代不清。
    修复不能把这个约束一起放掉。"""
    b = VQABuilder(BuildConfig(max_qa_per_sample=8), TAX)
    for i in range(60):
        s = _sample(f"m/{i}", ["crack", "corrosion"], "右下")
        for r in b.build(s):
            if r["task"] == "severity_action":
                assert ("右下" in r["question"] or "裂纹" in r["question"]
                        or "腐蚀" in r["question"]), r["question"]


def test_single_defect_images_may_use_open_questions():
    b = VQABuilder(BuildConfig(max_qa_per_sample=8), TAX)
    got = set()
    for i in range(60):
        for r in b.build(_sample(f"o/{i}", ["crack"])):
            if r["task"] == "classification_open":
                got.add(r["question"])
    assert got, "没生成 classification_open"


def test_no_task_silently_ignores_its_bank_entry():
    """通扫一遍：库里给了扩写问法的任务，输出里都必须能看到。

    这是上面那个坑的一般形式 —— 任何任务改成从别处取问法，
    都会在这里被抓住。
    """
    bank = {t: [MARKER + q for q in qs[:2]]
            for t, qs in T.QUESTIONS.items() if qs}
    seen = _build(bank, n=400)
    missed = [t for t in seen
              if t in bank and not any(MARKER in q for q in seen[t])]
    assert not missed, f"这些任务没用上扩写问法：{missed}"
