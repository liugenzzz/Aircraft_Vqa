# -*- coding: utf-8 -*-
"""模型池：多个 OpenAI 兼容端点按权重路由，带失败转移与用途级参数覆盖。

为什么需要池而不是单个模型
1. 手上模型不止一个，想按**权重**混着用（大模型保质量、小模型冲量）；
2. 某个端点挂了要能自动转到别的，而不是整批任务失败；
3. 不同用途该用不同参数 —— 扩问法要高温度求多样，改写答案要低温度求稳；
4. 调比例不该改代码。一份 JSON，改完重跑就行。

配置见 configs/llm_pool.json，字段说明在 docs/04_llm_pool.md。
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    name: str
    model: str
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    api_key: Optional[str] = None          # 不建议直接写在配置里
    weight: float = 1.0
    enabled: bool = True
    tags: list = field(default_factory=list)
    concurrency: int = 4
    timeout: int = 60
    generation: dict = field(default_factory=dict)
    notes: str = ""

    def resolved_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")

    def resolved_base_url(self) -> str:
        """归一化 base_url。

        服务方给的地址常常是完整的 `.../v1/chat/completions`，而 OpenAI SDK
        的 base_url 只要到 `/v1` —— 直接填会拼成
        `/v1/chat/completions/chat/completions`，报 404 很难查。
        这里把末尾的 chat/completions（以及 completions / embeddings）剥掉。
        """
        u = (self.base_url or "").strip().rstrip("/")
        for suffix in ("/chat/completions", "/completions", "/embeddings"):
            if u.endswith(suffix):
                u = u[: -len(suffix)]
                break
        return u


@dataclass
class PoolStats:
    calls: int = 0
    ok: int = 0
    fail: int = 0
    total_latency: float = 0.0
    by_model: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def record(self, model: str, ok: bool, latency: float,
               err: str = "") -> None:
        self.calls += 1
        self.total_latency += latency
        m = self.by_model.setdefault(
            model, {"calls": 0, "ok": 0, "fail": 0, "latency": 0.0})
        m["calls"] += 1
        m["latency"] += latency
        if ok:
            self.ok += 1
            m["ok"] += 1
        else:
            self.fail += 1
            m["fail"] += 1
            if err:
                self.errors[err] = self.errors.get(err, 0) + 1

    def summary(self) -> dict:
        avg = self.total_latency / self.calls if self.calls else 0.0
        return {
            "calls": self.calls, "ok": self.ok, "fail": self.fail,
            "success_rate": round(self.ok / self.calls, 4) if self.calls else 0.0,
            "avg_latency_s": round(avg, 2),
            "by_model": {
                k: {**v, "avg_latency_s": round(v["latency"] / v["calls"], 2)
                    if v["calls"] else 0.0}
                for k, v in self.by_model.items()},
            "errors": dict(sorted(self.errors.items(),
                                  key=lambda x: -x[1])[:10]),
        }


DEFAULT_CONFIG = {
    "version": 1,
    "strategy": "weighted",
    "request": {"timeout": 60, "max_retries": 2, "retry_backoff": 1.5,
                "failover": True},
    "generation": {"temperature": 0.6, "top_p": 0.9, "max_tokens": 1024},
    "models": [],
    "purposes": {},
}


class LLMPool:
    """按权重从模型池里挑一个调用；失败自动换一个再试。"""

    STRATEGIES = ("weighted", "round_robin", "failover")

    def __init__(self, cfg: dict):
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.request = {**DEFAULT_CONFIG["request"],
                        **(self.cfg.get("request") or {})}
        self.generation = {**DEFAULT_CONFIG["generation"],
                           **(self.cfg.get("generation") or {})}
        self.purposes = self.cfg.get("purposes") or {}
        known = set(ModelSpec.__dataclass_fields__)
        # 配置里允许写 _说明 这类注释字段，加载时忽略掉
        self.models = [ModelSpec(**{k: v for k, v in m.items() if k in known})
                       for m in (self.cfg.get("models") or [])]
        self.stats = PoolStats()
        self._clients: dict = {}
        self._rr = 0
        self._lock = threading.Lock()
        self._rng = random.Random(self.cfg.get("seed", 0))

    # ---------------------------------------------------------- 加载
    @classmethod
    def from_file(cls, path: str) -> "LLMPool":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_file_or_none(cls, path: Optional[str]) -> Optional["LLMPool"]:
        if not path or not os.path.exists(path):
            return None
        return cls.from_file(path)

    # ---------------------------------------------------------- 选模型
    def available(self, purpose: str = "") -> list:
        """某个用途可用的模型。用途没配就用全部启用的模型。"""
        pool = [m for m in self.models if m.enabled]
        spec = self.purposes.get(purpose) or {}
        names = spec.get("models")
        if names and "*" not in names:
            want = set(names)
            pool = [m for m in pool if m.name in want]
        tags = spec.get("tags")
        if tags:
            pool = [m for m in pool if set(tags) & set(m.tags)]
        return [m for m in pool if m.resolved_key() or m.base_url]

    def _strategy(self, purpose: str) -> str:
        s = (self.purposes.get(purpose) or {}).get(
            "strategy", self.cfg.get("strategy", "weighted"))
        return s if s in self.STRATEGIES else "weighted"

    def order(self, purpose: str = "") -> list:
        """返回本次调用的**尝试顺序** —— 第一个是主选，后面是失败转移备选。"""
        pool = self.available(purpose)
        if not pool:
            return []
        st = self._strategy(purpose)
        if st == "failover":
            return list(pool)                      # 按配置顺序，主备明确
        if st == "round_robin":
            with self._lock:
                i = self._rr % len(pool)
                self._rr += 1
            return pool[i:] + pool[:i]
        # weighted：按权重抽主选，其余随机兜底
        weights = [max(0.0, m.weight) for m in pool]
        if sum(weights) <= 0:
            weights = [1.0] * len(pool)
        first = self._rng.choices(pool, weights=weights, k=1)[0]
        rest = [m for m in pool if m is not first]
        self._rng.shuffle(rest)
        return [first] + rest

    # ---------------------------------------------------------- 参数
    def gen_params(self, purpose: str, spec: ModelSpec,
                   override: Optional[dict] = None) -> dict:
        """生成参数的四级覆盖：全局 < **模型** < **用途** < 调用时。

        用途排在模型后面是故意的：模型级 generation 是"这个模型自己的默认脾气"，
        用途级是"这件事要求什么"。改写答案要求温度低，就该压过模型的默认温度，
        否则用途块写了等于白写。
        """
        clean = lambda d: {k: v for k, v in (d or {}).items()
                           if not k.startswith("_")}
        out = clean(self.generation)
        out.update(clean(spec.generation))
        out.update(clean((self.purposes.get(purpose) or {}).get("generation")))
        out.update(clean(override))
        return out

    # ---------------------------------------------------------- 调用
    def _client(self, spec: ModelSpec):
        if spec.name in self._clients:
            return self._clients[spec.name]
        from openai import OpenAI          # 延迟导入，离线环境不需要装
        c = OpenAI(api_key=spec.resolved_key() or "EMPTY",
                   base_url=spec.resolved_base_url() or None)
        self._clients[spec.name] = c
        return c

    def chat(self, messages: list, purpose: str = "",
             override: Optional[dict] = None) -> tuple:
        """返回 (文本, 用的模型名)；全部失败返回 (None, None)。

        每个模型内部重试 max_retries 次；仍失败且开了 failover 就换下一个。
        """
        seq = self.order(purpose)
        if not seq:
            return None, None
        retries = int(self.request.get("max_retries", 2))
        backoff = float(self.request.get("retry_backoff", 1.5))
        failover = bool(self.request.get("failover", True))

        for spec in seq:
            params = self.gen_params(purpose, spec, override)
            for attempt in range(retries + 1):
                t0 = time.time()
                try:
                    resp = self._client(spec).chat.completions.create(
                        model=spec.model, messages=messages,
                        timeout=spec.timeout or self.request.get("timeout", 60),
                        **params)
                    text = resp.choices[0].message.content
                    self.stats.record(spec.name, True, time.time() - t0)
                    return text, spec.name
                except Exception as e:
                    self.stats.record(spec.name, False, time.time() - t0,
                                      f"{spec.name}:{type(e).__name__}")
                    if attempt < retries:
                        time.sleep(backoff ** attempt)
            if not failover:
                break
        return None, None

    # ---------------------------------------------------------- 体检
    def health(self, purpose: str = "") -> list:
        """逐个模型发一句最短的请求，报告通不通。"""
        out = []
        for spec in self.models:
            if not spec.enabled:
                out.append({"name": spec.name, "status": "disabled"})
                continue
            if not (spec.resolved_key() or spec.base_url):
                out.append({"name": spec.name, "status": "no_credential",
                            "detail": f"环境变量 {spec.api_key_env} 未设置，"
                                      f"且没有 base_url"})
                continue
            t0 = time.time()
            try:
                resp = self._client(spec).chat.completions.create(
                    model=spec.model, timeout=min(20, spec.timeout),
                    messages=[{"role": "user", "content": "回复两个字：就绪"}],
                    max_tokens=16)
                out.append({"name": spec.name, "status": "ok",
                            "latency_s": round(time.time() - t0, 2),
                            "reply": (resp.choices[0].message.content or
                                      "")[:20]})
            except Exception as e:
                out.append({"name": spec.name, "status": "fail",
                            "detail": f"{type(e).__name__}: {e}"[:160]})
        return out

    def describe(self) -> str:
        lines = [f"策略 {self.cfg.get('strategy')}，"
                 f"重试 {self.request.get('max_retries')} 次，"
                 f"失败转移 {'开' if self.request.get('failover') else '关'}"]
        tot = sum(m.weight for m in self.models if m.enabled) or 1.0
        for m in self.models:
            flag = "" if m.enabled else "（停用）"
            share = f"{m.weight / tot:.0%}" if m.enabled else "-"
            lines.append(f"  {m.name:22s}{flag} 权重 {m.weight:<5g} 占比 {share:>5s}"
                         f"  {m.model}  {m.resolved_base_url() or '(默认端点)'}")
        for p, spec in self.purposes.items():
            who = spec.get("models") or (
                [f"tag:{t}" for t in spec.get("tags", [])] or ["*"])
            gen = {k: v for k, v in (spec.get("generation") or {}).items()
                   if not k.startswith("_")}
            n = len(self.available(p))
            lines.append(f"  用途 {p:12s} 策略 {spec.get('strategy', '继承'):12s}"
                         f" 可用 {n} 个 {who}  {gen}")
        return "\n".join(lines)
