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


    # ------------------------------------------------------------ 批量
    def default_workers(self) -> int:
        """按池里各模型声明的 concurrency 之和决定并发路数。

        串行跑是不现实的：纯文本答案有五万多条，一次调用十来秒，
        排下来一百六十多小时。并发数由配置里每个副本的 concurrency 决定，
        调它就能整体提速，不用改代码。
        """
        if not self.enabled:
            return 0
        return max(1, sum(max(1, m.concurrency)
                          for m in self.pool.available(self.purpose)))

    def rewrite_many(self, records: list, workers: int = 0,
                     progress_every: int = 500, log=print) -> dict:
        """并发改写一批记录，就地修改。返回统计。

        受保护的任务（JSON / 多轮 / 选择题）先在本地筛掉，不占并发名额也
        不发请求 —— 它们本来就不参与改写。
        """
        import concurrent.futures as cf
        import threading
        import time

        if not self.enabled:
            return {"enabled": False}
        todo = [r for r in records if r.get("task") not in PROTECTED_TASKS]
        n_skip = len(records) - len(todo)
        workers = workers or self.default_workers()
        lock = threading.Lock()
        state = {"done": 0, "ok": 0, "t0": time.time()}

        def one(r):
            self.rewrite(r)
            with lock:
                state["done"] += 1
                state["ok"] += int(bool(r.get("rewritten")))
                d = state["done"]
                if progress_every and d % progress_every == 0:
                    el = time.time() - state["t0"]
                    rate = d / el if el else 0
                    left = (len(todo) - d) / rate if rate else 0
                    log(f"  ...改写 {d}/{len(todo)}，成功 {state['ok']}，"
                        f"{rate:.1f} 条/秒，预计还需 {left / 60:.0f} 分钟")

        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))

        rejected = {}
        for r in todo:
            why = r.get("rewrite_rejected")
            if why:
                rejected[why] = rejected.get(why, 0) + 1
        return {"enabled": True, "workers": workers, "n_total": len(records),
                "n_skipped_protected": n_skip, "n_sent": len(todo),
                "n_rewritten": state["ok"], "rejected": rejected,
                "seconds": round(time.time() - state["t0"], 1)}


def load_rewriter(cfg: dict) -> LLMRewriter:
    """cfg 取自 build.yaml 的 llm 段，只认 pool_config 一个键。"""
    return LLMRewriter.from_config((cfg or {}).get("pool_config"),
                                   (cfg or {}).get("purpose", "rewrite"))
