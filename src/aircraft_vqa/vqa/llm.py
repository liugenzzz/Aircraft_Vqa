# -*- coding: utf-8 -*-
"""可选的大模型改写层。

模板生成的答案准确但偏机械。这一层用一个辅助大模型（DashScope / OpenAI 兼容
接口皆可）把答案改写得更像人写的机务记录，**但严禁改动事实**：
坐标、数量、缺陷类型、严重度一律原样保留 —— prompt 里对此做了硬约束，
改写后还会再跑一遍 qc.check_record，改坏了就回退到模板答案。

离线环境下 provider="none"，整条流水线照常工作。
"""
from __future__ import annotations

import json
import os
import re
import time
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

# 不做改写的任务：答案本身就是结构化输出，改写只会引入风险
PROTECTED_TASKS = {"grounding_single", "grounding_all", "grounding_negative",
                   "counting", "classification_mc"}


class LLMRewriter:
    def __init__(self, provider: str = "none", model: str = "",
                 base_url: str = "", api_key_env: str = "LLM_API_KEY",
                 temperature: float = 0.6, max_retries: int = 2,
                 timeout: int = 60):
        self.provider = provider
        self.model = model
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        if provider == "openai":
            if not self.api_key:
                raise RuntimeError(f"环境变量 {api_key_env} 未设置")
            from openai import OpenAI  # 延迟导入，离线时不需要装
            self._client = OpenAI(api_key=self.api_key,
                                  base_url=self.base_url or None)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def _call(self, text: str) -> Optional[str]:
        for i in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    timeout=self.timeout,
                    messages=[{"role": "system", "content": REWRITE_SYSTEM},
                              {"role": "user", "content": text}])
                return resp.choices[0].message.content.strip()
            except Exception:
                if i == self.max_retries:
                    return None
                time.sleep(1.5 * (i + 1))
        return None

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
        return record


def load_rewriter(cfg: dict) -> LLMRewriter:
    return LLMRewriter(**{k: v for k, v in (cfg or {}).items()
                          if k in LLMRewriter.__init__.__code__.co_varnames})
