# -*- coding: utf-8 -*-
"""可选的大模型改写层。

模板生成的答案准确但偏机械。这一层用一个辅助大模型（DashScope / OpenAI 兼容
接口皆可）把答案改写得更像人写的机务记录，**但严禁改动事实**：
坐标、数量、缺陷类型、严重度一律原样保留 —— prompt 里对此做了硬约束，
改写后还会再跑一遍 qc.check_record，改坏了就回退到模板答案。

没有配模型池时改写层自动关闭，整条流水线照常工作。
"""
from __future__ import annotations

import json
import re
from typing import Optional

REWRITE_SYSTEM = (
    "你是民航机务维修文档的润色助手。用户会给你一段由模板生成的检查记录答案，"
    "请把它改写得更自然、更像一线机务的口吻。\n"
    "硬性要求（违反即视为失败）：\n"
    "1. 不得增加、删除或修改任何数字，包括坐标、数量、比例；\n"
    "2. 不得修改缺陷类型名称和严重程度结论；\n"
    "3. 如果原文包含 JSON，必须原样保留该 JSON，一个字符都不能改；\n"
    "4. 只输出改写后的答案本身，不要解释。"
)

def _protected_tasks() -> set:
    """不送改写的任务 —— **从 output_format 自动推导，不手维护名单**。

    手维护一张表，新增任务时必然漏（grounding_counterfactual 就漏过）。
    规则很清楚：
      json_only / text_then_json  答案是结构化输出，改写只会引入风险
      dialog                      多轮里混着 JSON 轮次，整体改写会破坏结构
      classification_mc           答案是"字母. 选项"，改了就对不上 answer_letter
    只有纯 text 任务参与改写。
    """
    from .templates import OUTPUT_FORMAT
    out = {t for t, f in OUTPUT_FORMAT.items()
           if f in ("json_only", "text_then_json", "dialog")}
    out.add("classification_mc")
    return out


PROTECTED_TASKS = _protected_tasks()


class LLMRewriter:
    """答案改写。模型走 LLMPool —— 多个端点按权重路由、失败自动转移。

    只要给一个 pool 就能用；不给就是关闭状态，整条流水线照常跑。
    """

    def __init__(self, pool=None, purpose: str = "rewrite"):
        self.pool = pool
        self.purpose = purpose
        self.last_model: Optional[str] = None

    @classmethod
    def from_config(cls, path: Optional[str], purpose: str = "rewrite"):
        from ..llm import LLMPool
        pool = LLMPool.from_file_or_none(path)
        if pool and not pool.available(purpose):
            pool = None            # 配置在但没有可用模型 = 关闭
        return cls(pool, purpose)

    @property
    def enabled(self) -> bool:
        return self.pool is not None

    def _call(self, text: str) -> Optional[str]:
        out, model = self.pool.chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": text}], purpose=self.purpose)
        self.last_model = model
        return out.strip() if out else None

    @staticmethod
    def _facts(text: str) -> tuple:
        """抽出必须保持不变的事实：所有数字 + 所有 JSON 片段。"""
        nums = tuple(re.findall(r"-?\d+(?:\.\d+)?", text))
        js = re.search(r"\[.*\]", text, re.S)
        return nums, (js.group(0) if js else "")

    def rewrite(self, record: dict) -> dict:
        """就地返回（可能被改写的）record。事实校验不过就保留原答案。"""
        if not self.enabled or record.get("task") in PROTECTED_TASKS:
            return record
        original = record["answer"]
        new = self._call(original)
        if not new:
            return record
        if self._facts(original) != self._facts(new):
            record["rewrite_rejected"] = "fact_drift"
            return record
        from ..qc import check_record
        probe = dict(record, answer=new)
        if check_record(probe):
            record["rewrite_rejected"] = "qc_fail"
            return record
        record["answer_template"] = original
        record["answer"] = new
        record["rewritten"] = True
        record["rewritten_by"] = self.last_model
        return record


def load_rewriter(cfg: dict) -> LLMRewriter:
    """cfg 取自 build.yaml 的 llm 段，只认 pool_config 一个键。"""
    return LLMRewriter.from_config((cfg or {}).get("pool_config"),
                                   (cfg or {}).get("purpose", "rewrite"))
