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
import re
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------- 思维链
# Qwen3 / DeepSeek-R1 这类模型默认会输出思维链。服务端开了 reasoning parser
# 时它落在 message.reasoning_content，content 是干净的；没开 parser 时
# <think>...</think> 会原样留在 content 里，必须剥掉再用 —— 否则思维链会被
# 当成答案写进训练数据。
_R_TAG = r"think|thinking|reasoning|reason"
_R_BLOCK = re.compile(rf"<\s*({_R_TAG})\s*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_R_CLOSE = re.compile(rf"^.*<\s*/\s*(?:{_R_TAG})\s*>", re.I | re.S)
_R_OPEN = re.compile(rf"<\s*(?:{_R_TAG})\s*>", re.I)

# 默认关思考。provider 可以覆盖具体字段，或整体 disable_thinking: false 保留。
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def strip_reasoning(text: str) -> str:
    """剥掉思维链，只留最终回答。三种形态都要管：

    1. 成对 <think>...</think>  -> 整块删掉
    2. 只有闭合标签（开标签被 chat template 吃掉，思维链以前缀形式返回）
       -> 丢掉最后一个 </think> 之前的全部内容
    3. 只有开标签（输出被 max_tokens 截断）-> 其后全是思维链，没有可用回答
    """
    if not text:
        return ""
    cleaned = _R_BLOCK.sub("", text)
    if _R_CLOSE.search(cleaned):
        cleaned = _R_CLOSE.sub("", cleaned, count=1)
    m = _R_OPEN.search(cleaned)
    if m:
        cleaned = cleaned[:m.start()]
    return cleaned.strip()


def _msg_text(message) -> tuple:
    """从 SDK 的 message 对象里取 (content, reasoning_content)。

    reasoning_content 不是 OpenAI 官方字段，SDK 会把它丢进 model_extra；
    不同服务端叫法也不一样，几个常见的都试一遍。
    """
    get = (message.get if isinstance(message, dict)
           else lambda k, d=None: getattr(message, k, d))
    content = get("content", "") or ""
    reasoning = ""
    extra = get("model_extra", None) or {}
    for k in ("reasoning_content", "reasoning", "thinking"):
        v = get(k, None) or (extra.get(k) if isinstance(extra, dict) else None)
        if isinstance(v, str) and v.strip():
            reasoning = v
            break
    return str(content), str(reasoning)


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
    # 关思考。vLLM 走 extra_body.chat_template_kwargs 传给 chat template。
    disable_thinking: bool = True
    chat_template_kwargs: dict = field(default_factory=dict)
    notes: str = ""

    def template_kwargs(self) -> dict:
        if not self.disable_thinking:
            return dict(self.chat_template_kwargs)
        merged = dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
        merged.update(self.chat_template_kwargs or {})   # 显式配置优先
        return merged

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
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, ok: bool, latency: float,
               err: str = "") -> None:
        # 并发调用时 self.calls += 1 这种读-改-写会丢计数，统计就不准了
        with self._lock:
            self._record(model, ok, latency, err)

    def _record(self, model: str, ok: bool, latency: float, err: str) -> None:
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
        self._warned_key = set()
        self._clients: dict = {}
        self._rr = 0
        self._lock = threading.Lock()
        self._rng = random.Random(self.cfg.get("seed", 0))

    # ---------------------------------------------------------- 加载
    @classmethod
    @staticmethod
    def _merge_local(cfg: dict, path: str) -> dict:
        """叠加 configs/xxx.local.json —— 本机的 key 和端点差异写那儿，不入库。

        仓库是公开的，凭据一旦提交就抹不掉；和 datasets.local.yaml 一个路子。
        models 按 name 浅合并，只写要改的字段，其余从模板继承。
        """
        import json as _json
        base = path[:-5] if path.endswith(".json") else path
        lp = base + ".local.json"
        if not os.path.exists(lp):
            return cfg
        with open(lp, encoding="utf-8") as f:
            loc = _json.load(f) or {}
        out = dict(cfg)
        by_name = {m.get("name"): dict(m) for m in out.get("models", [])}
        order = [m.get("name") for m in out.get("models", [])]
        for m in loc.pop("models", []) or []:
            n = m.get("name")
            if not n:
                continue
            if n in by_name:
                by_name[n].update(m)
            else:
                by_name[n] = dict(m)
                order.append(n)
        for k, v in loc.items():
            if k == "purposes" and isinstance(v, dict):
                merged = dict(out.get("purposes") or {})
                merged.update(v)
                out[k] = merged
            else:
                out[k] = v
        out["models"] = [by_name[n] for n in order]
        print(f"[pool] 已叠加本机配置 {lp}")
        return out

    @classmethod
    def from_file(cls, path: str) -> "LLMPool":
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cls._merge_local(cfg, path))

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
        usable = []
        for m in pool:
            if m.resolved_key() or m.base_url:
                # 配了 api_key_env 却没设环境变量 —— 请求会失败然后静默转移到
                # 备选，把关模型就这么被降级掉了，必须说一声。
                if m.api_key is None and not os.environ.get(m.api_key_env):
                    if m.name not in self._warned_key:
                        self._warned_key.add(m.name)
                        print(f"[pool] {m.name} 需要环境变量 {m.api_key_env}，"
                              "当前没设 —— 调用会失败并转移到备选模型"
                              f"（purpose={purpose or '默认'} 会因此降级）")
                usable.append(m)
        return usable

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
                    kw = dict(params)
                    tkw = spec.template_kwargs()
                    if tkw:
                        # vLLM/SGLang 认的是 extra_body.chat_template_kwargs；
                        # 调用方自己传了就别覆盖他。
                        eb = dict(kw.pop("extra_body", None) or {})
                        eb.setdefault("chat_template_kwargs", tkw)
                        kw["extra_body"] = eb
                    resp = self._client(spec).chat.completions.create(
                        model=spec.model, messages=messages,
                        timeout=spec.timeout or self.request.get("timeout", 60),
                        **kw)
                    content, reasoning = _msg_text(resp.choices[0].message)
                    text = strip_reasoning(content)
                    if not text and reasoning.strip():
                        # 回答整个落在思维链里 —— 多半是没关成思考又被 max_tokens
                        # 截断。当成失败让它重试/转移，别把空串写进数据。
                        raise RuntimeError(
                            f"{spec.name} 只返回了思维链没有正文，"
                            "确认服务端吃 chat_template_kwargs.enable_thinking=false，"
                            "或调大 max_tokens")
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
                kw = {}
                tkw = spec.template_kwargs()
                if tkw:
                    kw["extra_body"] = {"chat_template_kwargs": tkw}
                # max_tokens 给宽一点：万一服务端不吃 enable_thinking，
                # 16 个 token 全用来想事情了，正文一个字都出不来，
                # 体检就会把一个其实能用的副本报成"只返回了思维链"。
                resp = self._client(spec).chat.completions.create(
                    model=spec.model, timeout=min(30, spec.timeout),
                    messages=[{"role": "user", "content": "回复两个字：就绪"}],
                    max_tokens=256, **kw)
                content, reasoning = _msg_text(resp.choices[0].message)
                text = strip_reasoning(content)
                row = {"name": spec.name,
                       "latency_s": round(time.time() - t0, 2),
                       "reply": text[:20]}
                if text:
                    row["status"] = "ok"
                    if reasoning.strip() or content != text:
                        # 能用，但服务端没关掉思考 —— 批量跑会白烧大量 token
                        row["status"] = "ok_thinking_on"
                        row["detail"] = ("返回里带思维链，已剥离。"
                                         "服务端没吃 enable_thinking=false，"
                                         "批量跑会多花不少 token")
                else:
                    row["status"] = "only_reasoning"
                    row["detail"] = "只返回了思维链，没有正文"
                out.append(row)
            except Exception as e:
                row = {"name": spec.name, "status": "fail",
                       "detail": f"{type(e).__name__}: {e}"[:160]}
                # 401/403 且压根没解析出 key —— 那不是"端点连不上"也不是
                # "key 无效"，是根本没带凭据。分开报，别让人去查网络。
                if not spec.resolved_key() and any(
                        c in str(e) for c in ("401", "403", "Unauthorized",
                                              "Forbidden")):
                    row["status"] = "no_credential"
                    row["detail"] = (
                        f"没有解析到 key（api_key_env={spec.api_key_env} 未设置，"
                        "本机也没有 .local.json），发出去的是空凭据")
                out.append(row)
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
