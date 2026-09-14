# -*- coding: utf-8 -*-
"""UnifiedSample -> VQA 条目。

两大任务族：
  localization（缺陷定位）: grounding_single / grounding_all / grounding_negative
                            / referring_region / counting / region_word
  recognition （缺陷识别）: discrimination / classification_open / classification_mc
                            / description / severity_action / object_recognition

所有答案默认由模板确定性生成（可离线批量跑、可复现）；
需要更自然的表述时，再用 vqa/llm.py 做一层改写。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from ..geometry import size_word, to_qwen_box
from ..schema import UnifiedSample
from ..taxonomy import Taxonomy, get_taxonomy
from . import templates as T
from .distractor import format_options, make_options
from .negatives import sample_clean_box

LOCALIZATION_TASKS = ("grounding_single", "grounding_all", "grounding_negative",
                      "referring_region", "counting", "region_word")
RECOGNITION_TASKS = ("discrimination", "classification_open", "classification_mc",
                     "description", "severity_action", "object_recognition")


@dataclass
class BuildConfig:
    seed: int = 20260914
    coord_mode: str = "norm1000"     # norm1000 | abs
    box_label: str = "zh"            # zh | en | canonical
    max_qa_per_sample: int = 4
    # 每种任务被采纳的概率（乘以样本是否具备该任务所需标注）
    task_weights: dict = field(default_factory=lambda: {
        "grounding_single": 1.0,
        "grounding_all": 0.8,
        "grounding_negative": 1.0,
        "referring_region": 0.6,
        "counting": 0.45,
        "region_word": 0.4,
        "discrimination": 0.9,
        "classification_open": 0.7,
        "classification_mc": 0.7,
        "description": 0.8,
        "severity_action": 0.6,
        "object_recognition": 0.25,
    })
    enable_tasks: Optional[list] = None   # None = 全开
    n_options: int = 4
    # other_anomaly（源数据没有细粒度类型）时，自动跳过类型类问题
    skip_unknown_type_tasks: bool = True


class VQABuilder:
    def __init__(self, cfg: Optional[BuildConfig] = None,
                 taxonomy: Optional[Taxonomy] = None):
        self.cfg = cfg or BuildConfig()
        self.tax = taxonomy or get_taxonomy()

    # ---------------------------------------------------------- 工具
    def _rng(self, sample_id: str) -> random.Random:
        """按 sample_id 派生随机种子 —— 同一份数据重复构建结果完全一致。"""
        h = hashlib.md5(f"{self.cfg.seed}:{sample_id}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _enabled(self, task: str) -> bool:
        if self.cfg.enable_tasks is not None and task not in self.cfg.enable_tasks:
            return False
        return self.cfg.task_weights.get(task, 0) > 0

    @staticmethod
    def _is_en(text: str) -> bool:
        """判断问题是否为英文问法 —— 英文问就用英文标签作答，避免中英混搭。"""
        letters = sum(ch.isascii() and ch.isalpha() for ch in text)
        return letters > max(8, len(text) * 0.4)

    def _label_text(self, defect_type: str, en: bool = False) -> str:
        if en or self.cfg.box_label == "en":
            return self.tax.en(defect_type)
        if self.cfg.box_label == "canonical":
            return defect_type
        return self.tax.zh(defect_type)

    def _boxes_json(self, s: UnifiedSample, defects: Iterable, en: bool = False) -> str:
        items = [{"bbox_2d": to_qwen_box(d.bbox, s.width, s.height, self.cfg.coord_mode),
                  "label": self._label_text(d.type, en)}
                 for d in defects if d.bbox]
        return json.dumps(items, ensure_ascii=False)

    def _qwen_box(self, s: UnifiedSample, box: list) -> list:
        return to_qwen_box(box, s.width, s.height, self.cfg.coord_mode)

    def _rec(self, s: UnifiedSample, task: str, family: str, question: str,
             answer: str, system: str, extra: Optional[dict] = None) -> dict:
        qa_id = hashlib.md5(f"{s.sample_id}|{task}|{question}".encode()).hexdigest()[:16]
        r = {
            "qa_id": qa_id,
            "sample_id": s.sample_id,
            "image": s.image_path,
            "width": s.width, "height": s.height,
            "task": task, "family": family,
            "system": system,
            "question": question.strip(),
            "answer": answer.strip(),
            "label": s.label,
            "defect_types": s.defect_types,
            "dataset": s.dataset, "category": s.category, "split": s.split,
            "object": s.object_name,
            "license": s.license, "commercial_ok": s.commercial_ok,
            "coord_mode": self.cfg.coord_mode,
        }
        if extra:
            r.update(extra)
        return r

    def _ctx(self, s: UnifiedSample) -> dict:
        return {"obj": s.object_zh, "ctx": s.aircraft_ctx}

    # ---------------------------------------------------------- 主入口
    def build(self, s: UnifiedSample) -> list:
        rng = self._rng(s.sample_id)
        cands = []
        if s.is_anomalous:
            cands += self._anomalous_tasks(s, rng)
        else:
            cands += self._normal_tasks(s, rng)

        # 按权重抽样，保证同一张图不会生成过多同质问答
        picked, seen_task = [], set()
        rng.shuffle(cands)
        for task, fn in cands:
            if len(picked) >= self.cfg.max_qa_per_sample:
                break
            if task in seen_task:
                continue
            if rng.random() > self.cfg.task_weights.get(task, 0):
                continue
            try:
                rec = fn()
            except Exception:
                rec = None
            if rec:
                picked.append(rec)
                seen_task.add(task)
        return picked

    def build_many(self, samples: Iterable[UnifiedSample]) -> Iterator[dict]:
        for s in samples:
            for r in self.build(s):
                yield r

    # ---------------------------------------------------------- 异常样本
    def _anomalous_tasks(self, s: UnifiedSample, rng: random.Random) -> list:
        out = []
        loc = s.localizable_defects()
        types = s.defect_types
        known_type = not (self.cfg.skip_unknown_type_tasks
                          and types and types[0] == "other_anomaly")

        if loc and self._enabled("grounding_single"):
            out.append(("grounding_single", lambda: self._t_grounding_single(s, rng, loc)))
        if loc and self._enabled("grounding_all"):
            out.append(("grounding_all", lambda: self._t_grounding_all(s, rng, loc)))
        if loc and self._enabled("referring_region"):
            out.append(("referring_region", lambda: self._t_referring(s, rng, loc)))
        if loc and len(loc) >= 1 and self._enabled("counting"):
            out.append(("counting", lambda: self._t_counting(s, rng, loc)))
        if loc and self._enabled("region_word"):
            out.append(("region_word", lambda: self._t_region_word(s, rng, loc)))

        if self._enabled("discrimination"):
            out.append(("discrimination", lambda: self._t_discrimination(s, rng)))
        if known_type and self._enabled("classification_open"):
            out.append(("classification_open", lambda: self._t_classify_open(s, rng)))
        if known_type and self._enabled("classification_mc"):
            out.append(("classification_mc", lambda: self._t_classify_mc(s, rng)))
        if self._enabled("description"):
            out.append(("description", lambda: self._t_describe(s, rng)))
        if known_type and self._enabled("severity_action"):
            out.append(("severity_action", lambda: self._t_severity(s, rng)))
        if self._enabled("object_recognition"):
            out.append(("object_recognition", lambda: self._t_object(s, rng)))
        return out

    # ---------------------------------------------------------- 正常样本
    def _normal_tasks(self, s: UnifiedSample, rng: random.Random) -> list:
        out = []
        if self._enabled("grounding_negative"):
            out.append(("grounding_negative", lambda: self._t_grounding_negative(s, rng)))
        if self._enabled("discrimination"):
            out.append(("discrimination", lambda: self._t_discrimination(s, rng)))
        if self._enabled("description"):
            out.append(("description", lambda: self._t_describe(s, rng)))
        if self._enabled("referring_region"):
            out.append(("referring_region", lambda: self._t_referring(s, rng, [])))
        if self._enabled("object_recognition"):
            out.append(("object_recognition", lambda: self._t_object(s, rng)))
        return out

    # ---------------------------------------------------------- 各任务实现
    def _t_grounding_single(self, s, rng, loc) -> dict:
        t = rng.choice(s.defect_types)
        same = [d for d in loc if d.type == t]
        if not same:
            return None
        q = rng.choice(T.Q_GROUNDING_SINGLE).format(
            defect=self.tax.zh(t), defect_en=self.tax.en(t), **self._ctx(s))
        en = self._is_en(q)
        return self._rec(s, "grounding_single", "localization", q,
                         self._boxes_json(s, same, en), T.SYSTEM_PROMPT_GROUNDING,
                         {"target_type": t, "n_boxes": len(same), "lang": "en" if en else "zh"})

    def _t_grounding_all(self, s, rng, loc) -> dict:
        q = rng.choice(T.Q_GROUNDING_ALL).format(**self._ctx(s))
        en = self._is_en(q)
        return self._rec(s, "grounding_all", "localization", q,
                         self._boxes_json(s, loc, en), T.SYSTEM_PROMPT_GROUNDING,
                         {"n_boxes": len(loc), "lang": "en" if en else "zh"})

    def _t_grounding_negative(self, s, rng) -> dict:
        # 正常图也问"找缺陷"，答空列表 —— 这是抑制幻觉最有效的一类样本
        t = rng.choice([k for k in self.tax.all_types() if k != "other_anomaly"])
        q = rng.choice(T.Q_GROUNDING_NEGATIVE).format(
            defect=self.tax.zh(t), **self._ctx(s))
        a = ("[]" if self._is_en(q) else
             rng.choice(T.A_GROUNDING_EMPTY).format(defect=self.tax.zh(t), **self._ctx(s)))
        return self._rec(s, "grounding_negative", "localization", q, a,
                         T.SYSTEM_PROMPT_GROUNDING, {"n_boxes": 0})

    def _t_referring(self, s, rng, loc) -> dict:
        positive = bool(loc) and rng.random() < 0.55
        if positive:
            d = rng.choice(loc)
            box = self._qwen_box(s, d.bbox)
            q = rng.choice(T.Q_REGION_YESNO).format(box=box, **self._ctx(s))
            a = rng.choice(T.A_REGION_POSITIVE).format(
                box=box, defect=self.tax.zh(d.type), region=d.region or "中部",
                size=size_word(d.area_ratio),
                severity=self.tax.severity_zh(d.severity), **self._ctx(s))
            meta = {"region_answer": "yes"}
        else:
            clean = sample_clean_box(s.width, s.height,
                                     [d.bbox for d in loc if d.bbox], rng)
            if clean is None:
                return None
            box = self._qwen_box(s, clean)
            q = rng.choice(T.Q_REGION_YESNO).format(box=box, **self._ctx(s))
            a = rng.choice(T.A_REGION_NEGATIVE).format(box=box, **self._ctx(s))
            meta = {"region_answer": "no"}
        return self._rec(s, "referring_region", "localization", q, a,
                         T.SYSTEM_PROMPT, meta)

    def _t_counting(self, s, rng, loc) -> dict:
        t = rng.choice(s.defect_types)
        same = [d for d in loc if d.type == t]
        if not same:
            return None
        q = rng.choice(T.Q_COUNT).format(defect=self.tax.zh(t), **self._ctx(s))
        a = T.A_COUNT.format(n=len(same), defect=self.tax.zh(t),
                             json=self._boxes_json(s, same))
        return self._rec(s, "counting", "localization", q, a, T.SYSTEM_PROMPT,
                         {"count": len(same), "target_type": t})

    def _t_region_word(self, s, rng, loc) -> dict:
        d = rng.choice(loc)
        q = rng.choice(T.Q_REGION_WORD).format(defect=self.tax.zh(d.type), **self._ctx(s))
        a = rng.choice(T.A_REGION_WORD).format(
            defect=self.tax.zh(d.type), region=d.region or "中部",
            size=size_word(d.area_ratio))
        return self._rec(s, "region_word", "localization", q, a, T.SYSTEM_PROMPT,
                         {"region": d.region})

    def _t_discrimination(self, s, rng) -> dict:
        q = rng.choice(T.Q_DISCRIMINATION).format(**self._ctx(s))
        if s.is_anomalous:
            names = "、".join(self.tax.zh(t) for t in s.defect_types)
            regions = [d.region for d in s.defects if d.region]
            a = rng.choice(T.A_DISCRIMINATION_POS).format(
                defect_list=names, region="、".join(dict.fromkeys(regions)) or "画面中",
                **self._ctx(s))
        else:
            a = rng.choice(T.A_DISCRIMINATION_NEG).format(**self._ctx(s))
        return self._rec(s, "discrimination", "recognition", q, a, T.SYSTEM_PROMPT,
                         {"yes_no": "yes" if s.is_anomalous else "no"})

    def _t_classify_open(self, s, rng) -> dict:
        t = s.defect_types[0]
        d = next((x for x in s.defects if x.type == t), s.defects[0])
        evidence = ""
        if d.region:
            evidence = f"依据是画面{d.region}处可见相应特征，范围{size_word(d.area_ratio)}。"
        q = rng.choice(T.Q_CLASSIFY_OPEN).format(**self._ctx(s))
        a = rng.choice(T.A_CLASSIFY_OPEN).format(
            defect=self.tax.zh(t), defect_en=self.tax.en(t), evidence=evidence)
        return self._rec(s, "classification_open", "recognition", q, a,
                         T.SYSTEM_PROMPT, {"target_type": t})

    def _t_classify_mc(self, s, rng) -> dict:
        opts, correct = make_options(self.tax, s.defect_types, self.cfg.n_options, rng)
        q = T.Q_CLASSIFY_MC.format(options=format_options(opts), **self._ctx(s))
        a = f"{correct}. {opts['ABCDEF'.index(correct)]}"
        return self._rec(s, "classification_mc", "recognition", q, a, T.SYSTEM_PROMPT,
                         {"options": opts, "answer_letter": correct,
                          "target_type": s.defect_types[0]})

    def _t_describe(self, s, rng) -> dict:
        q = rng.choice(T.Q_DESCRIBE).format(**self._ctx(s))
        if s.is_anomalous:
            items = []
            for d in s.defects[:6]:
                seg = f"画面{d.region or '中部'}的{self.tax.zh(d.type)}"
                if d.area_ratio:
                    seg += f"（{size_word(d.area_ratio)}）"
                items.append(seg)
            worst = max(s.defects, key=lambda d: ["minor", "major", "critical"].index(
                d.severity if d.severity in ("minor", "major", "critical") else "major"))
            a = T.A_DESCRIBE_POS.format(
                n=len(s.defects), items="；".join(items),
                severity=self.tax.severity_zh(worst.severity),
                severity_desc=self.tax.severity_desc(worst.severity), **self._ctx(s))
        else:
            asp = rng.sample(T.NEGATIVE_ASPECTS, k=min(3, len(T.NEGATIVE_ASPECTS)))
            a = T.A_DESCRIBE_NEG.format(negative_aspects="、".join(asp), **self._ctx(s))
        return self._rec(s, "description", "recognition", q, a, T.SYSTEM_PROMPT)

    def _t_severity(self, s, rng) -> dict:
        d = max(s.defects, key=lambda x: x.area_ratio)
        q = rng.choice(T.Q_SEVERITY).format(**self._ctx(s))
        a = T.A_SEVERITY.format(
            severity=self.tax.severity_zh(d.severity),
            severity_desc=self.tax.severity_desc(d.severity),
            defect=self.tax.zh(d.type), region=d.region or "中部",
            size=size_word(d.area_ratio), action=self.tax.action(d.type))
        return self._rec(s, "severity_action", "recognition", q, a, T.SYSTEM_PROMPT,
                         {"severity": d.severity, "target_type": d.type})

    def _t_object(self, s, rng) -> dict:
        info = self.tax.object_info(s.object_name)
        focus = T.OBJECT_FOCUS.get(info.get("role", "unknown"), T.OBJECT_FOCUS["unknown"])
        q = rng.choice(T.Q_OBJECT).format(**self._ctx(s))
        a = rng.choice(T.A_OBJECT).format(focus=focus, **self._ctx(s))
        return self._rec(s, "object_recognition", "recognition", q, a, T.SYSTEM_PROMPT)
