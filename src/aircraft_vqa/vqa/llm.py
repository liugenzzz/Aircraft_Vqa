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
import os
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

    @staticmethod
    def cache_key(r: dict) -> str:
        """一条记录的稳定标识，用于断点续跑。

        用 qa_id 不行：它跟随机种子走，改一次配置就全变了。改用
        (任务, 问题, 模板答案) 的哈希 —— 只要这三样没变，之前改写过的
        结果就还能用。
        """
        import hashlib
        raw = "\u0000".join((r.get("task") or "",
                              r.get("question") or "",
                              r.get("answer_template") or r.get("answer") or ""))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def rewrite_many(self, records: list, workers: int = 0,
                     progress_every: int = 500, log=print,
                     cache_path: str = "") -> dict:
        """并发改写一批记录，就地修改。返回统计。

        受保护的任务（JSON / 多轮 / 选择题）先在本地筛掉，不占并发名额也
        不发请求 —— 它们本来就不参与改写。

        给了 cache_path 就边跑边落盘：五万条要跑七到十个小时，中途崩一次
        全白跑是不能接受的。重跑时先读缓存，命中的直接套用、不再发请求。
        """
        import concurrent.futures as cf
        import json as _json
        import threading
        import time

        if not self.enabled:
            return {"enabled": False}
        todo = [r for r in records if r.get("task") not in PROTECTED_TASKS]
        n_skip = len(records) - len(todo)

        # 读缓存，命中的直接套用
        cache, n_hit = {}, 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = _json.loads(line)
                    except Exception:
                        continue      # 崩在写一半时最后一行可能是残的
                    if d.get("k"):
                        cache[d["k"]] = d
            pending = []
            for r in todo:
                hit = cache.get(self.cache_key(r))
                if hit is None:
                    pending.append(r)
                    continue
                n_hit += 1
                if hit.get("a"):
                    r["answer_template"] = r["answer"]
                    r["answer"] = hit["a"]
                    r["rewritten"] = True
                    r["rewritten_by"] = hit.get("m")
                elif hit.get("why"):
                    r["rewrite_rejected"] = hit["why"]
            log(f"  断点续跑：缓存命中 {n_hit} 条，还需改写 {len(pending)} 条")
            todo = pending

        workers = workers or self.default_workers()
        lock = threading.Lock()
        fh = open(cache_path, "a", encoding="utf-8") if cache_path else None
        state = {"done": 0, "ok": 0, "t0": time.time()}

        def one(r):
            self.rewrite(r)
            with lock:
                state["done"] += 1
                ok = bool(r.get("rewritten"))
                state["ok"] += int(ok)
                if fh:
                    fh.write(_json.dumps(
                        {"k": self.cache_key(r),
                         "a": r["answer"] if ok else None,
                         "m": r.get("rewritten_by"),
                         "why": r.get("rewrite_rejected")},
                        ensure_ascii=False) + "\n")
                    if state["done"] % 200 == 0:
                        fh.flush()          # 崩了最多丢这 200 条
                d = state["done"]
                if progress_every and d % progress_every == 0:
                    el = time.time() - state["t0"]
                    rate = d / el if el else 0
                    left = (len(todo) - d) / rate if rate else 0
                    log(f"  ...改写 {d}/{len(todo)}，成功 {state['ok']}，"
                        f"{rate:.1f} 条/秒，预计还需 {left / 60:.0f} 分钟")

        try:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(one, todo))
        finally:
            if fh:
                fh.flush()
                fh.close()

        rejected = {}
        for r in records:
            why = r.get("rewrite_rejected")
            if why:
                rejected[why] = rejected.get(why, 0) + 1
        n_rewritten = sum(1 for r in records if r.get("rewritten"))
        return {"enabled": True, "workers": workers, "n_total": len(records),
                "n_skipped_protected": n_skip, "n_sent": len(todo),
                "n_cache_hit": n_hit, "n_rewritten": n_rewritten,
                "rejected": rejected,
                "seconds": round(time.time() - state["t0"], 1)}


def load_rewriter(cfg: dict) -> LLMRewriter:
    """cfg 取自 build.yaml 的 llm 段，只认 pool_config 一个键。"""
    return LLMRewriter.from_config((cfg or {}).get("pool_config"),
                                   (cfg or {}).get("purpose", "rewrite"))
