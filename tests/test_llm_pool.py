# -*- coding: utf-8 -*-
"""模型池：路由、参数覆盖、失败转移、统计。

这些逻辑不调真实模型也能完整验证 —— 把 client 换成假的即可。
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import pytest

from aircraft_vqa.llm import LLMPool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeResp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {
            "content": text})()})()]


class _FakeClient:
    """按模型名决定成功还是抛错，用来验证失败转移。"""

    def __init__(self, name, fail=False, record=None):
        self.name, self.fail, self.record = name, fail, record
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, model, messages, timeout=None, **params):
        if self.record is not None:
            self.record.append((self.name, params))
        if self.fail:
            raise RuntimeError(f"{self.name} 挂了")
        return _FakeResp(f"来自 {self.name}")


def _pool(models, **cfg):
    return LLMPool({"models": models, **cfg})


def _patch(pool, failing=(), record=None):
    pool._clients = {m.name: _FakeClient(m.name, m.name in failing, record)
                     for m in pool.models}


# ------------------------------------------------------------------ 路由
def test_weighted_routing_follows_weights():
    p = _pool([{"name": "big", "model": "m", "base_url": "u", "weight": 4},
               {"name": "small", "model": "m", "base_url": "u", "weight": 1}])
    c = Counter(p.order()[0].name for _ in range(4000))
    assert 0.75 < c["big"] / 4000 < 0.85, c


def test_round_robin_cycles_all_models():
    p = _pool([{"name": f"m{i}", "model": "m", "base_url": "u"}
               for i in range(3)], strategy="round_robin")
    firsts = [p.order()[0].name for _ in range(6)]
    assert firsts == ["m0", "m1", "m2", "m0", "m1", "m2"]


def test_failover_strategy_keeps_config_order():
    p = _pool([{"name": "primary", "model": "m", "base_url": "u"},
               {"name": "backup", "model": "m", "base_url": "u"}],
              strategy="failover")
    assert [m.name for m in p.order()] == ["primary", "backup"]


def test_disabled_and_credential_less_models_are_excluded():
    p = _pool([{"name": "on", "model": "m", "base_url": "u"},
               {"name": "off", "model": "m", "base_url": "u", "enabled": False},
               {"name": "nokey", "model": "m", "api_key_env": "NO_SUCH_ENV_X"}])
    assert [m.name for m in p.available()] == ["on"]


def test_purpose_filters_by_name_and_tag():
    p = _pool([{"name": "a", "model": "m", "base_url": "u", "tags": ["judge"]},
               {"name": "b", "model": "m", "base_url": "u", "tags": ["cheap"]}],
              purposes={"j": {"tags": ["judge"]}, "only_b": {"models": ["b"]}})
    assert [m.name for m in p.available("j")] == ["a"]
    assert [m.name for m in p.available("only_b")] == ["b"]
    assert len(p.available("unknown_purpose")) == 2      # 没配就用全部


# ------------------------------------------------------------------ 参数
def test_generation_override_order_global_model_purpose_call():
    """用途要压过模型 —— 模型级是"这个模型的默认脾气"，
    用途级是"这件事要求什么"，后者是任务需求，必须更高优先级。"""
    p = _pool([{"name": "a", "model": "m", "base_url": "u",
                "generation": {"temperature": 0.7, "top_p": 0.8}}],
              generation={"temperature": 0.6, "top_p": 0.9, "max_tokens": 100},
              purposes={"rw": {"generation": {"temperature": 0.2}}})
    g = p.gen_params("rw", p.models[0])
    assert g["temperature"] == 0.2      # 用途赢过模型
    assert g["top_p"] == 0.8            # 用途没设，模型赢过全局
    assert g["max_tokens"] == 100       # 都没设，用全局
    g2 = p.gen_params("rw", p.models[0], {"temperature": 0.0})
    assert g2["temperature"] == 0.0     # 调用时最高


def test_comment_fields_are_ignored():
    """配置里允许写 _说明 这类注释字段。"""
    p = LLMPool({"models": [{"name": "a", "model": "m", "base_url": "u",
                             "_说明": "注释"}],
                 "generation": {"temperature": 0.5, "_备注": "x"},
                 "purposes": {"rw": {"generation": {"_c": "y", "top_p": 0.3}}}})
    g = p.gen_params("rw", p.models[0])
    assert g["temperature"] == 0.5 and g["top_p"] == 0.3
    assert not any(k.startswith("_") for k in g)


# ------------------------------------------------------------------ 调用
def test_chat_falls_over_to_next_model():
    p = _pool([{"name": "bad", "model": "m", "base_url": "u", "weight": 100},
               {"name": "good", "model": "m", "base_url": "u", "weight": 0.001}],
              request={"max_retries": 0, "retry_backoff": 0, "failover": True})
    _patch(p, failing={"bad"})
    text, used = p.chat([{"role": "user", "content": "x"}])
    assert used == "good" and "good" in text
    assert p.stats.by_model["bad"]["fail"] >= 1
    assert p.stats.by_model["good"]["ok"] == 1


def test_chat_without_failover_gives_up():
    p = _pool([{"name": "bad", "model": "m", "base_url": "u"},
               {"name": "good", "model": "m", "base_url": "u"}],
              strategy="failover",
              request={"max_retries": 0, "retry_backoff": 0, "failover": False})
    _patch(p, failing={"bad"})
    assert p.chat([{"role": "user", "content": "x"}]) == (None, None)


def test_chat_retries_before_switching():
    p = _pool([{"name": "bad", "model": "m", "base_url": "u"}],
              request={"max_retries": 2, "retry_backoff": 0, "failover": True})
    _patch(p, failing={"bad"})
    p.chat([{"role": "user", "content": "x"}])
    assert p.stats.by_model["bad"]["calls"] == 3      # 1 次 + 2 次重试


def test_chat_passes_purpose_params_through():
    rec = []
    p = _pool([{"name": "a", "model": "m", "base_url": "u"}],
              purposes={"hot": {"generation": {"temperature": 0.95}}})
    _patch(p, record=rec)
    p.chat([{"role": "user", "content": "x"}], purpose="hot")
    assert rec[0][1]["temperature"] == 0.95


def test_stats_summary_shape():
    p = _pool([{"name": "a", "model": "m", "base_url": "u"}])
    _patch(p)
    p.chat([{"role": "user", "content": "x"}])
    s = p.stats.summary()
    assert s["calls"] == 1 and s["success_rate"] == 1.0
    assert "a" in s["by_model"]
    json.dumps(s)                                    # 必须可序列化


def test_empty_pool_returns_none():
    p = _pool([])
    assert p.chat([{"role": "user", "content": "x"}]) == (None, None)
    assert p.available() == []


# ------------------------------------------------------------------ 仓库配置
def test_shipped_config_is_valid():
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    assert p.models and p.purposes
    for purpose in ("rewrite", "paraphrase", "judge"):
        assert purpose in p.purposes
    # 模板里不能留真实密钥
    raw = open(path, encoding="utf-8").read()
    assert "sk-" not in raw
    for m in p.models:
        assert m.api_key is None, "别把 key 写进配置文件，用 api_key_env"


@pytest.mark.parametrize("given,want", [
    ("http://10.0.0.1:8001/v1/chat/completions", "http://10.0.0.1:8001/v1"),
    ("http://10.0.0.1:8001/v1/", "http://10.0.0.1:8001/v1"),
    ("http://10.0.0.1:8001/v1", "http://10.0.0.1:8001/v1"),
    ("https://x.com/compatible-mode/v1/chat/completions",
     "https://x.com/compatible-mode/v1"),
    ("", ""),
])
def test_base_url_normalisation(given, want):
    """服务方给的常是完整 .../v1/chat/completions，SDK 只要到 /v1。
    直接填会拼成 /v1/chat/completions/chat/completions，报 404 很难查。"""
    from aircraft_vqa.llm import ModelSpec
    assert ModelSpec(name="a", model="m",
                     base_url=given).resolved_base_url() == want


def test_rewriter_disabled_without_usable_pool():
    from aircraft_vqa.vqa.llm import LLMRewriter
    assert not LLMRewriter.from_config(None).enabled
    assert not LLMRewriter.from_config("/no/such/file.json").enabled


# ------------------------------------------------------------------ --set
def test_build_set_overrides_nested_keys():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bv", os.path.join(REPO, "scripts", "build_vqa.py"))
    bv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bv)
    cfg = {"normal_per_anomalous": 1.0, "task_ratio": {"a": 0.1}}
    bv.apply_overrides(cfg, ["normal_per_anomalous=0.5",
                             "task_ratio.a=0.3",
                             "task_ratio.b=0.2",
                             'active_defect_types=["crack"]',
                             "export.format=swift"])
    assert cfg["normal_per_anomalous"] == 0.5
    assert cfg["task_ratio"] == {"a": 0.3, "b": 0.2}
    assert cfg["active_defect_types"] == ["crack"]
    assert cfg["export"]["format"] == "swift"
    with pytest.raises(SystemExit):
        bv.apply_overrides({}, ["没有等号"])
