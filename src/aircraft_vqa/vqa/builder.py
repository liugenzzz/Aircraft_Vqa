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
                      "grounding_counterfactual", "referring_region",
                      "counting", "region_word")
RECOGNITION_TASKS = ("discrimination", "classification_open", "classification_mc",
                     "description", "severity_action", "object_recognition",
                     "grade_assessment", "uncertainty")
DIALOG_TASKS = ("multi_turn",)
COMPARE_TASKS = ("pair_compare",)
GRADING_TASKS = ("grade_assessment",)


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
        "multi_turn": 0.5,
        "grade_assessment": 0.9,   # 只有带有序等级的源才可能触发，权重给高些
        # 反事实负样本：对有缺陷的图，问一个图里没有的缺陷类型 -> []。
        # 比"正常图答空列表"难得多，是抑制"用户说有就一定有"的关键。
        "grounding_counterfactual": 1.0,
        "uncertainty": 1.0,        # 只有标了成像质量问题的样本才触发
        # 多图对比：给一张同类正常件参考图 + 待检图。机务实际就是比着看的，
        # 只有配了参考图池的数据源才会触发。
        "pair_compare": 0.8,
    })
    enable_tasks: Optional[list] = None   # None = 全开
    # True 时模板报错直接抛出而不是跳过该任务。跑大批量前先用 strict 跑一遍小样本。
    strict: bool = False
    n_options: int = 4
    # other_anomaly（源数据没有细粒度类型）时，自动跳过类型类问题
    skip_unknown_type_tasks: bool = True
    # 本期只做这些缺陷类型；不在名单里的会被降级成 other_anomaly，
    # 仍可用于定位和有无判定，但不会出"这是什么缺陷"的题。
    # None = 本体里全部类型都做。
    active_defect_types: Optional[list] = None
    # 大模型扩写出的问法库（scripts/gen_question_bank.py 产出）
    question_bank: Optional[str] = None
    # 问法语言：zh = 只用中文问法（默认）；bilingual = 保留英文问法。
    # 英文问法占比很小（约 3%），不足以真的教会双语，却会引入中英混杂的不一致；
    # 确实需要模型听懂英文指令的话再开 bilingual，并把占比提上去。
    lang: str = "zh"
    # 答案里是否给缺陷名加英文术语括注（"漆层剥落（paint peeling）"）。
    # AMM/SRM 里术语本是英文，但只在部分任务出现会显得中英混杂不一致，
    # 所以默认关掉；要开就该所有涉及类型名的答案统一开。
    term_annotation: bool = False


class VQABuilder:
    def __init__(self, cfg: Optional[BuildConfig] = None,
                 taxonomy: Optional[Taxonomy] = None):
        self.cfg = cfg or BuildConfig()
        self.tax = taxonomy or get_taxonomy()
        self.qpool = T.merged_questions(
            T.load_question_bank(self.cfg.question_bank))
        if self.cfg.lang == "zh":
            self.qpool = {k: [q for q in v if not self._is_en(q)] or v
                          for k, v in self.qpool.items()}
        self.active = (set(self.cfg.active_defect_types)
                       if self.cfg.active_defect_types else None)
        # (dataset, category) -> 该类正常件图片路径列表，供多图对比取参考图
        self.ref_pool: dict = {}
        from collections import Counter as _C
        self.failures = _C()      # 模板异常计数，构建结束后会打印出来

    def set_reference_pool(self, pool: dict) -> None:
        """注入正常件参考图池。没注入就不会产出 pair_compare 样本。"""
        self.ref_pool = pool or {}

    def _reference_image(self, s: UnifiedSample, rng) -> Optional[str]:
        pool = self.ref_pool.get((s.dataset, s.category)) or []
        pool = [p for p in pool if p != s.image_path]
        return rng.choice(pool) if pool else None

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
             answer: str, system: Optional[str] = None,
             extra: Optional[dict] = None) -> dict:
        # system 按任务族绑定：定位族要契约、识别族要事实、多轮要指代承接。
        # 全局一把梭会出现"system 说只输出 JSON、答案却带解释"这种自相矛盾。
        # system 按**输出格式**绑定。任务必须在 templates.OUTPUT_FORMAT 里登记，
        # 漏登记会直接报错而不是悄悄用错 system。
        ofmt = T.OUTPUT_FORMAT[task]
        system = system or T.SYSTEM_BY_OUTPUT[ofmt]
        qa_id = hashlib.md5(f"{s.sample_id}|{task}|{question}".encode()).hexdigest()[:16]
        r = {
            "qa_id": qa_id,
            "sample_id": s.sample_id,
            "image": s.image_path,
            "image_hw": [s.height, s.width],
            "task": task, "family": family, "output_format": ofmt,
            "system": system,
            "question": question.strip(),
            "answer": answer.strip(),
            # 三个缺陷字段语义严格分开，混成一个 defect_types 会让所有
            # 分层评测失效：图里客观有什么、这条问的是什么、答案断言存在什么，
            # 是三件不同的事。一致性校验靠
            #   answer_defect_types ⊆ image_defect_types
            "image_status": s.label,          # 原 label；和 bbox 里的 label 同名不同义
            "image_defect_types": s.defect_types,
            "asked_defect_types": [],
            "answer_defect_types": [],
            "dataset": s.dataset, "category": s.category, "split": s.split,
            "object": s.object_name,
            "license": s.license, "commercial_ok": s.commercial_ok,
            "coord_mode": self.cfg.coord_mode,
        }
        if extra:
            r.update(extra)
        # 答案里断言存在的缺陷 = JSON 框里的 label（定位类）或显式给出的
        if "answer_defect_types" not in (extra or {}):
            r["answer_defect_types"] = self._types_in_answer(r["answer"])
        return r

    def _types_in_answer(self, answer: str) -> list:
        """从答案的 JSON 框里抽出被断言存在的缺陷类型（canonical 名）。"""
        import re as _re
        m = _re.search(r"\[.*\]", answer, _re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except Exception:
            return []
        zh2canon = {self.tax.zh(t): t for t in self.tax.all_types()}
        out = []
        for it in items:
            if isinstance(it, dict):
                c = zh2canon.get(it.get("label"), it.get("label"))
                if c and c not in out:
                    out.append(c)
        return out

    def _rec_multi(self, s: UnifiedSample, turns: list, system: str,
                   extra: Optional[dict] = None) -> dict:
        """多轮条目：turns = [(问, 答), ...]，第一轮的问题带图。"""
        r = self._rec(s, "multi_turn", "dialog", turns[0][0], turns[0][1],
                      system, extra)
        r["turns"] = [{"question": q, "answer": a} for q, a in turns]
        r["n_turns"] = len(turns)
        r["answer_defect_types"] = self._types_in_answer(
            " ".join(a for _, a in turns))
        return r

    def _ctx(self, s: UnifiedSample) -> dict:
        return {"obj": s.object_zh, "ctx": s.aircraft_ctx}

    def _where_phrase(self, s: UnifiedSample) -> str:
        """一一对应的方位描述："裂纹位于画面右下，腐蚀锈蚀位于画面左下"。

        "可见裂纹、腐蚀锈蚀，位于右下、左下"这种并列写法让人对不上号，
        模型也学不到缺陷与位置的绑定关系。
        """
        parts, seen = [], set()
        for d in s.defects:
            key = (d.type, d.region)
            if key in seen or not d.region:
                continue
            seen.add(key)
            parts.append(f"{self.tax.zh(d.type)}位于画面{d.region}")
        if not parts:
            return f"可见{'、'.join(self.tax.zh(t) for t in s.defect_types)}。"
        head = f"共 {len(s.defects)} 处：" if len(s.defects) > 1 else ""
        return head + "，".join(parts) + "。"

    def _where_suffix(self, s: UnifiedSample) -> str:
        """给 A_DISCRIMINATION_POS 用的后缀形式。"""
        parts, seen = [], set()
        for d in s.defects:
            key = (d.type, d.region)
            if key in seen or not d.region:
                continue
            seen.add(key)
            parts.append(f"{self.tax.zh(d.type)}位于画面{d.region}")
        if not parts:
            return ""
        if len(s.defects) > 1:
            return f"，共 {len(s.defects)} 处：" + "，".join(parts)
        return "，" + parts[0]

    def _advisory(self, rng: random.Random) -> str:
        """限定语：多个变体，且只在一部分样本上带。

        每条都带同一句的话，模型会当成机械后缀复读，还白占 token。
        """
        if rng.random() > T.ADVISORY_RATE:
            return ""
        return rng.choice(T.ADVISORY_VARIANTS)

    def _pick_q(self, task: str, rng: random.Random, **fields) -> str:
        """从（种子 + 扩写）问法池里抽一条并填充。"""
        pool = self.qpool.get(task) or T.QUESTIONS.get(task, [""])
        return rng.choice(pool).format(**fields)

    def _pick_disambiguated(self, task: str, rng, need_named: bool,
                            named_marks: tuple, fallback: list) -> str:
        """按"有没有消歧占位符"从合并池里选问法。

        图里有多种缺陷时，问法必须点名问的是哪一处（靠 {region}/{defect}），
        否则指代不清；只有一种时开放式问法就够。

        以前是拿合并池和硬编码常量求交集来区分这两档 —— 扩写出来的问法
        不在常量里，全被滤掉，问法库对这两个任务完全不起作用（实测
        classification_open 82 -> 81、severity_action 77 -> 79，等于白扩）。
        改成按占位符判断，扩写的问法就能正确落进对应那一档。
        """
        pool = self.qpool.get(task) or []
        has = lambda q: any(m in q for m in named_marks)
        cand = [q for q in pool if has(q) == need_named]
        return rng.choice(cand or fallback)

    def _scope(self, s: UnifiedSample) -> UnifiedSample:
        """把不在本期范围内的缺陷类型降级为 other_anomaly。

        降级而不是丢弃：这些样本的框依然是真的，定位和"有无异常"照样能用，
        只是不再声称自己知道它具体是哪一类。
        """
        if self.active is None or not s.defects:
            return s
        if all(d.type in self.active for d in s.defects):
            return s
        import copy
        s2 = copy.copy(s)
        s2.defects = []
        for d in s.defects:
            if d.type in self.active:
                s2.defects.append(d)
            else:
                d2 = copy.copy(d)
                d2.type = "other_anomaly"
                d2.type_zh = self.tax.zh("other_anomaly")
                s2.defects.append(d2)
        return s2

    # ---------------------------------------------------------- 主入口
    def build(self, s: UnifiedSample) -> list:
        s = self._scope(s)
        rng = self._rng(s.sample_id)
        cands = []
        quality = s.meta.get("image_quality")
        if quality in T.UNCERTAIN_REASON:
            # 成像质量本身就不足以判读的图，只出"无法确认"样本。
            # 在这种图上再产出信誓旦旦的定位答案，等于教模型对糊图也硬答。
            # 一张糊图可以按不同缺陷类型问多次，样本利用率高一些。
            if not self._enabled("uncertainty"):
                return []
            out = []
            for rec in self._t_uncertainty_many(s, rng, quality):
                if rec and not self._seen_qa(out, rec):
                    out.append(rec)
            return out[:self.cfg.max_qa_per_sample]
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
            except Exception as e:
                # 以前这里静默吞异常，结果模板里一个 .format 用错，
                # 整类任务凭空消失都没人发现。现在记下来，strict 模式直接抛。
                self.failures[f"{task}:{type(e).__name__}: {e}"] += 1
                if self.cfg.strict:
                    raise
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
        if self._enabled("multi_turn"):
            out.append(("multi_turn", lambda: self._t_multi_turn(s, rng, loc)))
        if any(d.grade for d in s.defects) and self._enabled("grade_assessment"):
            out.append(("grade_assessment", lambda: self._t_grade(s, rng)))
        if self._enabled("grounding_counterfactual"):
            out.append(("grounding_counterfactual",
                        lambda: self._t_grounding_counterfactual(s, rng)))
        if self.ref_pool and self._enabled("pair_compare"):
            out.append(("pair_compare", lambda: self._t_pair_compare(s, rng)))
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
        if self._enabled("multi_turn"):
            out.append(("multi_turn", lambda: self._t_multi_turn(s, rng, [])))
        if self.ref_pool and self._enabled("pair_compare"):
            out.append(("pair_compare", lambda: self._t_pair_compare(s, rng)))
        return out

    # ---------------------------------------------------------- 各任务实现
    def _t_grounding_single(self, s, rng, loc) -> dict:
        t = rng.choice(s.defect_types)
        same = [d for d in loc if d.type == t]
        if not same:
            return None
        q = self._pick_q("grounding_single", rng, defect=self.tax.zh(t),
                         defect_en=self.tax.en(t), **self._ctx(s))
        en = self._is_en(q)
        return self._rec(s, "grounding_single", "localization", q,
                         self._boxes_json(s, same, en), None,
                         {"target_type": t, "n_boxes": len(same), "lang": "en" if en else "zh"})

    def _t_grounding_all(self, s, rng, loc) -> dict:
        q = self._pick_q("grounding_all", rng, **self._ctx(s))
        en = self._is_en(q)
        return self._rec(s, "grounding_all", "localization", q,
                         self._boxes_json(s, loc, en), None,
                         {"n_boxes": len(loc), "lang": "en" if en else "zh"})

    def _t_grounding_negative(self, s, rng) -> dict:
        """正常图也问"找缺陷"，答空列表 —— 抑制幻觉最有效的一类样本。

        答案是**纯 []**：定位族的 system 写了"不附加任何解释"，
        答案再跟一句说明就是亲手教模型不遵循 system。
        问法池里也不含 {defect}，否则答案会提到问题里根本没出现的缺陷名。
        """
        q = self._pick_q("grounding_negative", rng, **self._ctx(s))
        return self._rec(s, "grounding_negative", "localization", q, "[]",
                         None, {"n_boxes": 0})

    def _t_grounding_counterfactual(self, s, rng) -> dict:
        """反事实负样本：对**有缺陷**的图，问一个图里不存在的缺陷类型 -> []。

        诱导性问法（"这张照片中存在 X，请将其框出"）如果只配正样本，
        模型会学到"用户说有就一定有"，部署时随口一问就幻觉出一个框。
        必须成对造，而且负样本要用同一批问法 —— 包括那些把前提写死的。
        在有缺陷的图上问不存在的类型，比拿一张完全正常的图问要难得多。
        """
        present = set(s.defect_types)
        pool = [t for t in (self.active or self.tax.all_types())
                if t not in present and t != "other_anomaly"]
        if not pool:
            return None
        t = rng.choice(pool)
        q = self._pick_q("grounding_counterfactual", rng, defect=self.tax.zh(t),
                         defect_en=self.tax.en(t), **self._ctx(s))
        return self._rec(s, "grounding_counterfactual", "localization", q, "[]",
                         None, {"n_boxes": 0, "absent_type": t,
                                "asked_defect_types": [t],
                                "variant": "counterfactual"})

    def _t_referring(self, s, rng, loc) -> dict:
        positive = bool(loc) and rng.random() < 0.55
        if positive:
            d = rng.choice(loc)
            box = self._qwen_box(s, d.bbox)
            q = self._pick_q("referring_region", rng, box=box, **self._ctx(s))
            a = rng.choice(T.A_REGION_POSITIVE).format(
                box=box, defect=self.tax.zh(d.type), region=d.region or "中部",
                size=size_word(d.area_ratio), **self._ctx(s))
            meta = {"region_answer": "yes"}
        else:
            # hard=True：贴着缺陷旁边采，避免模型学到"框在空白处就答没有"的捷径
            clean = sample_clean_box(s.width, s.height,
                                     [d.bbox for d in loc if d.bbox], rng,
                                     hard=True)
            if clean is None:
                return None
            box = self._qwen_box(s, clean)
            q = self._pick_q("referring_region", rng, box=box, **self._ctx(s))
            a = rng.choice(T.A_REGION_NEGATIVE).format(box=box, **self._ctx(s))
            meta = {"region_answer": "no"}
        return self._rec(s, "referring_region", "localization", q, a,
                         None, meta)

    def _t_counting(self, s, rng, loc) -> dict:
        """计数要覆盖 0 / 1 / 多三种边界，只喂"1 处"模型学不会真数数。"""
        if rng.random() < 0.25:                      # 0 处：问图里没有的类型
            present = set(s.defect_types)
            pool = [t for t in (self.active or self.tax.all_types())
                    if t not in present and t != "other_anomaly"]
            if pool:
                t = rng.choice(pool)
                q = self._pick_q("counting", rng, defect=self.tax.zh(t),
                                 **self._ctx(s))
                a = rng.choice(T.A_COUNT_ZERO).format(defect=self.tax.zh(t))
                return self._rec(s, "counting", "localization", q, a, None,
                                 {"count": 0, "target_type": t,
                                  "asked_defect_types": [t],
                                  "variant": "counterfactual"})
        t = rng.choice(s.defect_types)
        same = [d for d in loc if d.type == t]
        if not same:
            return None
        q = self._pick_q("counting", rng, defect=self.tax.zh(t), **self._ctx(s))
        a = T.A_COUNT.format(n=len(same), defect=self.tax.zh(t),
                             json=self._boxes_json(s, same))
        return self._rec(s, "counting", "localization", q, a, None,
                         {"count": len(same), "target_type": t,
                          "asked_defect_types": [t]})

    def _t_region_word(self, s, rng, loc) -> dict:
        d = rng.choice(loc)
        q = self._pick_q("region_word", rng, defect=self.tax.zh(d.type),
                         **self._ctx(s))
        a = rng.choice(T.A_REGION_WORD).format(
            defect=self.tax.zh(d.type), region=d.region or "中部",
            size=size_word(d.area_ratio))
        return self._rec(s, "region_word", "localization", q, a, None,
                         {"region": d.region, "asked_defect_types": [d.type],
                          "answer_defect_types": [d.type]})

    def _t_discrimination(self, s, rng) -> dict:
        q = self._pick_q("discrimination", rng, **self._ctx(s))
        if s.is_anomalous:
            names = "、".join(self.tax.zh(t) for t in s.defect_types)
            a = rng.choice(T.A_DISCRIMINATION_POS).format(
                defect_list=names, where=self._where_suffix(s), **self._ctx(s))
        else:
            a = rng.choice(T.A_DISCRIMINATION_NEG).format(**self._ctx(s))
        return self._rec(s, "discrimination", "recognition", q, a, None,
                         {"yes_no": "yes" if s.is_anomalous else "no"})

    def _t_classify_open(self, s, rng) -> dict:
        t = s.defect_types[0]
        d = next((x for x in s.defects if x.type == t), s.defects[0])
        info = self.tax.grade_info(t, d.grade) if d.grade else {}
        # 只陈述能核对的事实。原来那句"依据是画面X处可见相应特征"是空话 ——
        # 没有任何视觉证据，等于教模型说套话装作有推理。
        if info:
            evidence = f"，程度为{info['zh']}：{info['desc']}。"
        elif d.region:
            evidence = f"，位于画面{d.region}，范围{size_word(d.area_ratio)}。"
        else:
            evidence = "。"
        name = self.tax.zh(t)
        if self.cfg.term_annotation:
            name = f"{name}（{self.tax.en(t)}）"
        named = len(s.defect_types) > 1
        q = self._pick_disambiguated(
            "classification_open", rng, named, ("{region}",),
            T.Q_CLASSIFY_NAMED if named else T.Q_CLASSIFY_OPEN
        ).format(region=d.region or "中部", **self._ctx(s))
        a = rng.choice(T.A_CLASSIFY_OPEN).format(defect=name, evidence=evidence)
        return self._rec(s, "classification_open", "recognition", q, a,
                         None, {"target_type": t, "answer_defect_types": [t]})

    def _t_classify_mc(self, s, rng) -> dict:
        role = self.tax.object_info(s.object_name).get("role")
        opts, correct = make_options(self.tax, s.defect_types, self.cfg.n_options,
                                     rng, active=self.active, part_role=role)
        q = T.Q_CLASSIFY_MC.format(options=format_options(opts), **self._ctx(s))
        a = f"{correct}. {opts['ABCDEF'.index(correct)]}"
        return self._rec(s, "classification_mc", "recognition", q, a, None,
                         {"options": opts, "answer_letter": correct,
                          "target_type": s.defect_types[0]})

    def _t_describe(self, s, rng) -> dict:
        q = self._pick_q("description", rng, **self._ctx(s))
        if s.is_anomalous:
            items = []
            for d in s.defects[:6]:
                seg = f"画面{d.region or '中部'}的{self.tax.zh(d.type)}"
                info = self.tax.grade_info(d.type, d.grade) if d.grade else {}
                if info:
                    seg += f"（{info['zh']}）"
                elif d.area_ratio:
                    seg += f"（{size_word(d.area_ratio)}）"
                items.append(seg)
            # 描述类只讲看得见的事实，严重度结论留给 severity_action 去讲
            # （那里带限定语），不要在纯描述里下判定。
            a = T.A_DESCRIBE_POS.format(
                n=len(s.defects), items="；".join(items), **self._ctx(s))
        else:
            asp = rng.sample(T.NEGATIVE_ASPECTS, k=min(3, len(T.NEGATIVE_ASPECTS)))
            a = T.A_DESCRIBE_NEG.format(negative_aspects="、".join(asp), **self._ctx(s))
        return self._rec(s, "description", "recognition", q, a, None)

    def _t_severity(self, s, rng) -> dict:
        d = self._graded(s) or max(s.defects, key=lambda x: x.area_ratio)
        info = self.tax.grade_info(d.type, d.grade) if d.grade else {}
        name = self.tax.zh(d.type)
        if info:
            name = f"{name}（{info['zh']}）"
        # 图里有多种缺陷时，问法必须点名是哪一处 —— 单轮同样存在指代歧义，
        # 用"该缺陷"配只讲其中一个的答案，是在教模型遇到歧义就默认挑第一个。
        named = len(s.defect_types) > 1
        q = self._pick_disambiguated(
            "severity_action", rng, named, ("{defect}", "{region}"),
            T.Q_SEVERITY_NAMED if named else T.Q_SEVERITY_ONE
        ).format(defect=self.tax.zh(d.type), region=d.region or "中部",
                 **self._ctx(s))
        a = rng.choice(T.A_SEVERITY).format(
            severity=self.tax.severity_zh(d.severity),
            severity_desc=self.tax.severity_desc(d.severity),
            defect=name, region=d.region or "中部",
            size=size_word(d.area_ratio),
            action=info.get("action") or self.tax.action(d.type),
            advisory=self._advisory(rng))
        return self._rec(s, "severity_action", "recognition", q, a, None,
                         {"severity": d.severity, "target_type": d.type,
                          "asked_defect_types": [d.type],
                          "answer_defect_types": [d.type]})

    def _graded(self, s) -> Optional[object]:
        """取等级最高（最严重）的那处缺陷；没有分级标注则返回 None。"""
        graded = [d for d in s.defects if d.grade]
        if not graded:
            return None
        return max(graded, key=lambda d: self.tax.grade_rank(d.type, d.grade))

    @staticmethod
    def _seen_qa(out: list, rec: dict) -> bool:
        return any(r["question"] == rec["question"] for r in out)

    def _t_pair_compare(self, s, rng) -> dict:
        """多图对比：正常件参考图 + 待检图 -> 指出差异。

        机务实际的做法就是拿正常件比对着看。这类样本能教模型
        "差异在哪"而不是"记住某类缺陷长什么样"，泛化更好。
        参考图从同一 (数据源, 类别) 的正常图里取。
        """
        ref = self._reference_image(s, rng)
        if not ref:
            return None
        q = self._pick_q("pair_compare", rng, **self._ctx(s))
        if s.is_anomalous and s.defects:
            items = "；".join(
                f"画面{d.region or '中部'}出现{self.tax.zh(d.type)}"
                for d in s.defects[:4])
            a = rng.choice(T.A_PAIR_COMPARE_POS).format(items=items,
                                                        **self._ctx(s))
            atypes = s.defect_types
        else:
            a = rng.choice(T.A_PAIR_COMPARE_NEG).format(**self._ctx(s))
            atypes = []
        r = self._rec(s, "pair_compare", "compare", q, a, None,
                      {"answer_defect_types": atypes,
                       "reference_image": ref})
        # 两张图：参考图在前，待检图在后
        r["images"] = [ref, s.image_path]
        return r

    def _t_uncertainty_many(self, s, rng, quality: str, k: int = 3) -> list:
        """同一张糊图按不同缺陷类型问 k 次 —— 糊图本来就少，别浪费。

        **优先问图里确实存在的类型**：在劣化图上问一个根本不存在的类型，
        "本来就没有"和"被挡住看不见"是混淆的，答"无法确认"说不清是哪种。
        问一个确实存在但被遮挡/模糊掩盖的缺陷，"无法确认"才立得住。
        """
        present = [t for t in s.defect_types if t != "other_anomaly"]
        others = [t for t in (self.active or self.tax.all_types())
                  if t != "other_anomaly" and t not in present]
        rng.shuffle(others)
        pool = present + others
        return [self._t_uncertainty(s, rng, quality, t) for t in pool[:k]]

    def _t_uncertainty(self, s, rng, quality: str,
                       target: Optional[str] = None) -> dict:
        """成像质量不足以判读时，答"无法确认 + 建议补拍"。

        system 里写了"无法确认时直接说明，不要臆测"，就必须有样本把这件事
        教出来，否则那句话是空指令 —— 模型照样对糊图给出自信的框。
        """
        t = target or rng.choice([k for k in (self.active or self.tax.all_types())
                                  if k != "other_anomaly"])
        reason = T.UNCERTAIN_REASON[quality]
        q = self._pick_q("uncertainty", rng, defect=self.tax.zh(t), **self._ctx(s))
        a = rng.choice(T.A_UNCERTAIN).format(
            reason=reason, fix=T.UNCERTAIN_FIX[quality], defect=self.tax.zh(t))
        return self._rec(s, "uncertainty", "recognition", q, a, None,
                         {"image_quality": quality, "target_type": t,
                          "asked_defect_types": [t], "answer_defect_types": [],
                          "degradation": s.meta.get("degradation", {})})

    def _t_grade(self, s, rng) -> dict:
        d = self._graded(s)
        if d is None:
            return None
        info = self.tax.grade_info(d.type, d.grade)
        q = self._pick_q("grade_assessment", rng, defect=self.tax.zh(d.type),
                         **self._ctx(s))
        a = rng.choice(T.A_GRADE).format(
            grade_zh=info.get("zh", d.grade), grade_desc=info.get("desc", ""),
            region=d.region or "中部", size=size_word(d.area_ratio),
            action=info.get("action") or self.tax.action(d.type),
            advisory=self._advisory(rng))
        return self._rec(s, "grade_assessment", "recognition", q, a,
                         None,
                         {"grade": d.grade, "target_type": d.type,
                          "grade_rank": self.tax.grade_rank(d.type, d.grade)})

    def _t_multi_turn(self, s, rng, loc) -> dict:
        """有无 -> 定位 -> 处置 的三轮追问，复刻机务实际问诊流程。"""
        turns = []
        t1q = self._pick_q("multi_turn", rng, **self._ctx(s))
        if s.is_anomalous:
            names = "、".join(self.tax.zh(t) for t in s.defect_types)
            turns.append((t1q, f"有异常。{self._where_phrase(s)}"))
            if loc:
                turns.append((rng.choice(T.Q_MT_T2_POS), self._boxes_json(s, loc)))
            d = max(s.defects, key=lambda x: x.area_ratio)
            # 前文提到多处缺陷时，第三轮必须点名问哪一处 —— 否则"该缺陷"
            # 指代不唯一，而答案只讲其中一个，等于教模型遇到歧义就默认挑第一个
            # 并静默丢掉其余的。
            if len(s.defect_types) > 1:
                t3q = rng.choice(T.Q_MT_T3_MANY).format(
                    defect=self.tax.zh(d.type))
            else:
                t3q = rng.choice(T.Q_MT_T3_ONE)
            turns.append((t3q, rng.choice(T.A_SEVERITY).format(
                severity=self.tax.severity_zh(d.severity),
                severity_desc=self.tax.severity_desc(d.severity),
                defect=self.tax.zh(d.type), region=d.region or "中部",
                size=size_word(d.area_ratio), action=self.tax.action(d.type),
                advisory=self._advisory(rng))))
        else:
            turns.append((t1q, rng.choice(T.A_DISCRIMINATION_NEG).format(
                **self._ctx(s))))
            t = rng.choice([k for k in (self.active or self.tax.all_types())
                            if k != "other_anomaly"])
            turns.append((rng.choice(T.Q_MT_T2_NEG).format(defect=self.tax.zh(t)),
                          "[]"))
        if len(turns) < 2:
            return None
        return self._rec_multi(s, turns, None,
                               {"yes_no": "yes" if s.is_anomalous else "no"})

    def _t_object(self, s, rng) -> dict:
        info = self.tax.object_info(s.object_name)
        focus = T.OBJECT_FOCUS.get(info.get("role", "unknown"), T.OBJECT_FOCUS["unknown"])
        q = self._pick_q("object_recognition", rng, **self._ctx(s))
        a = rng.choice(T.A_OBJECT).format(focus=focus, **self._ctx(s))
        return self._rec(s, "object_recognition", "recognition", q, a, None)
