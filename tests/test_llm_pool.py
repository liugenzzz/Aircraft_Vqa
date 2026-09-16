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
    # 这个仓库是**公开**的：任何 key 都不许进入库文件，内网的也不行。
    # 内网不可达只是说别人连不上那台机器，不等于可以把凭据和内网拓扑
    # 发布到公网 —— 而且 git 历史抹不掉。key 走环境变量或
    # configs/llm_pool.local.json（已 gitignore）。
    # 注意看的是**入库文件本身**，不是 from_file 的结果 ——
    # 后者会叠上本机的 .local.json，那里面有 key 是正常的。
    import json as _json
    for m in _json.load(open(path, encoding="utf-8"))["models"]:
        assert "api_key" not in m, (
            f"{m['name']} 把 key 写进了入库配置。仓库是公开的，"
            "改用 api_key_env 或 configs/llm_pool.local.json")
    raw = open(path, encoding="utf-8").read()
    for pat in ("sk-", "sk:", "local-pool-key"):
        assert pat not in raw, f"入库配置里出现了 {pat}"


def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).hostname or "").strip()


def _is_private(host: str) -> bool:
    """RFC1918 私网地址 / 回环。主机名一律当公网。"""
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def test_local_overlay_supplies_keys(tmp_path):
    """本机 .local.json 叠在模板上补 key，模板本身保持干净。"""
    import json
    base = {"models": [{"name": "a", "model": "m",
                        "base_url": "http://10.0.0.1:8001/v1",
                        "api_key_env": "NOPE_NOT_SET"}],
            "purposes": {}}
    f = tmp_path / "pool.json"
    f.write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "pool.local.json").write_text(
        json.dumps({"models": [{"name": "a", "api_key": "k123"}]}),
        encoding="utf-8")
    m = LLMPool.from_file(str(f)).models[0]
    assert m.resolved_key() == "k123"
    assert m.model == "m" and m.base_url.endswith("8001/v1"), "模板字段被覆盖丢了"


def test_local_overlay_is_optional(tmp_path):
    import json
    f = tmp_path / "pool.json"
    f.write_text(json.dumps({"models": [{"name": "a", "model": "m",
                                         "base_url": "http://10.0.0.1:8001/v1"}],
                             "purposes": {}}), encoding="utf-8")
    assert LLMPool.from_file(str(f)).models[0].api_key is None


def test_local_overlay_can_add_a_model(tmp_path):
    import json
    f = tmp_path / "pool.json"
    f.write_text(json.dumps({"models": [{"name": "a", "model": "m",
                                         "base_url": "http://10.0.0.1:8001/v1"}],
                             "purposes": {}}), encoding="utf-8")
    (tmp_path / "pool.local.json").write_text(
        json.dumps({"models": [{"name": "b", "model": "m2",
                                "base_url": "http://10.0.0.2:8001/v1",
                                "api_key": "k"}]}), encoding="utf-8")
    names = [m.name for m in LLMPool.from_file(str(f)).models]
    assert names == ["a", "b"], names


def test_local_overlay_file_is_gitignored():
    ig = open(os.path.join(REPO, ".gitignore"), encoding="utf-8").read()
    assert "configs/*.local.json" in ig


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


# ------------------------------------------------------------------ 思维链
@pytest.mark.parametrize("raw,want", [
    ("<think>盘算一下</think>最终答案", "最终答案"),
    ("琢磨半天</think>最终答案", "最终答案"),          # 开标签被 chat template 吃掉
    ("<THINK>x</THINK> 答案", "答案"),                # 大小写
    ("< think >x</ think >答案", "答案"),              # 带空格
    ("答案在前<think>被截断了没闭合", "答案在前"),      # max_tokens 截断
    ("<reasoning>a</reasoning>b", "b"),
    ("<think>a</think>中间<think>b</think>尾", "中间尾"),
    ("干净的答案", "干净的答案"),
    ("", ""),
])
def test_strip_reasoning(raw, want):
    """思维链必须剥干净 —— 漏一个就是把模型的内心戏写进训练答案。"""
    from aircraft_vqa.llm.pool import strip_reasoning
    assert strip_reasoning(raw) == want


def test_thinking_disabled_by_default():
    from aircraft_vqa.llm.pool import ModelSpec
    assert ModelSpec(name="a", model="m").template_kwargs() == {
        "enable_thinking": False}


def test_thinking_can_be_kept_explicitly():
    from aircraft_vqa.llm.pool import ModelSpec
    m = ModelSpec(name="a", model="m", disable_thinking=False)
    assert m.template_kwargs() == {}


def test_explicit_template_kwargs_win():
    from aircraft_vqa.llm.pool import ModelSpec
    m = ModelSpec(name="a", model="m",
                  chat_template_kwargs={"enable_thinking": True, "x": 1})
    assert m.template_kwargs() == {"enable_thinking": True, "x": 1}


def test_reasoning_content_is_read_from_model_extra():
    """reasoning_content 不是 OpenAI 官方字段，SDK 会把它丢进 model_extra。"""
    from aircraft_vqa.llm.pool import _msg_text

    class Msg:
        content = ""
        model_extra = {"reasoning_content": "内心戏"}
    assert _msg_text(Msg()) == ("", "内心戏")
    assert _msg_text({"content": "答案", "reasoning_content": "戏"}) == ("答案", "戏")


def test_shipped_config_disables_thinking_everywhere():
    path = os.path.join(REPO, "configs", "llm_pool.json")
    for m in LLMPool.from_file(path).models:
        assert m.template_kwargs().get("enable_thinking") is False, m.name


def test_judge_prefers_the_big_model():
    """judge 用 failover，按配置顺序取主选 —— 把关模型必须排第一，
    否则 7 个 27B 会把它挤到永远轮不上。"""
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    seq = p.order("judge")
    assert seq, "judge 没有可用模型"
    assert "122b" in seq[0].name.lower(), [m.name for m in seq[:3]]
    assert len(seq) > 1, "把关模型挂了要有备选"


def test_big_model_stays_out_of_bulk_work():
    """122B 不该被拉去干扩写/改写这种量大的活。"""
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    for purpose in ("rewrite", "paraphrase"):
        names = [m.name.lower() for m in p.order(purpose)]
        assert not any("122b" in n for n in names), (purpose, names)
        # 别写死副本数 —— 副本会上下线，写死了每次增减都要来改测试。
        # 要盯的是"干活的副本不止一个"和"把关模型没被拉进来"。
        assert len(names) >= 2, (purpose, names)
        assert len(names) == len([m for m in p.models
                                  if "122b" not in m.name.lower()]), names


def test_shipped_config_token_budgets_are_not_stingy():
    """真实踩到的坑：max_tokens 写死 512，扩写 40 条问法必然截断，
    16 个任务全军覆没，报的却是"模型没写 JSON 数组"。

    模型是 128k 上下文，这里抠 token 省不下什么，却会把长输出腰斩，
    而截断的表现是"解析失败"，很难一眼看出是配额不够。
    """
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    floor = {"rewrite": 1024, "paraphrase": 4096, "judge": 1024}
    for purpose, least in floor.items():
        got = p.gen_params(purpose, p.order(purpose)[0]).get("max_tokens", 0)
        assert got >= least, f"{purpose} 的 max_tokens={got}，至少要 {least}"


def test_judge_budget_accounts_for_thinking():
    """122B 开着思考，思维链先吃配额。judge 给得太小，想完就没额度写结论，
    直接变成 only_reasoning —— 把关模型静默失灵，还不报错。"""
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    judge = p.order("judge")[0]
    assert "122b" in judge.name.lower()
    assert p.gen_params("judge", judge)["max_tokens"] >= 2048


def test_comment_keys_never_reach_the_api():
    """配置里用 _注 写说明很方便，但绝不能跟着请求发出去。"""
    path = os.path.join(REPO, "configs", "llm_pool.json")
    p = LLMPool.from_file(path)
    for purpose in ("rewrite", "paraphrase", "judge"):
        g = p.gen_params(purpose, p.order(purpose)[0])
        assert not [k for k in g if k.startswith("_")], (purpose, g)
