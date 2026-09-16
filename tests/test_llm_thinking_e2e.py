# -*- coding: utf-8 -*-
"""跑一个假 vLLM，验三种真实响应形态都处理对。

单测剥离函数只能证明正则没写错；真正容易出问题的是"参数有没有传出去"、
"SDK 把 reasoning_content 放哪了"、"拿不到正文时是失败还是静默返回空串"，
这些只有真发一次 HTTP 才看得出来。
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest

pytest.importorskip("openai")

from aircraft_vqa.llm.pool import LLMPool

SEEN = {}          # mode -> 服务端实际收到的 body


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        mode = self.server.mode
        SEEN[mode] = body
        if mode == "obedient":
            msg = {"role": "assistant", "content": "就绪"}
        elif mode == "tag_in_content":
            msg = {"role": "assistant",
                   "content": "<think>让我想想</think>就绪"}
        else:                                   # reasoning_field
            msg = {"role": "assistant", "content": "",
                   "reasoning_content": "想了很久"}
        raw = json.dumps({"id": "1", "object": "chat.completion",
                          "model": body.get("model", "m"),
                          "choices": [{"index": 0, "message": msg,
                                       "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture(scope="module")
def servers():
    out = {}
    for mode in ("obedient", "tag_in_content", "reasoning_field"):
        s = HTTPServer(("127.0.0.1", 0), _H)
        s.mode = mode
        threading.Thread(target=s.serve_forever, daemon=True).start()
        out[mode] = s.server_address[1]
    yield out


def _pool(port):
    return LLMPool({"models": [{"name": "m1", "model": "m",
                                "base_url": f"http://127.0.0.1:{port}/v1",
                                "api_key": "x", "tags": ["rewrite"]}],
                    "purposes": {"rewrite": {"tags": ["rewrite"]}}})


def test_enable_thinking_actually_reaches_the_server(servers):
    """关思考的参数必须真的发出去，不能只是配置里写着。"""
    _pool(servers["obedient"]).chat([{"role": "user", "content": "hi"}],
                                    purpose="rewrite")
    body = SEEN["obedient"]
    assert body.get("chat_template_kwargs") == {"enable_thinking": False}, body


def test_think_tag_in_content_is_stripped(servers):
    text, who = _pool(servers["tag_in_content"]).chat(
        [{"role": "user", "content": "hi"}], purpose="rewrite")
    assert text == "就绪", text
    assert who == "m1"


def test_reasoning_only_fails_instead_of_returning_empty(servers):
    """正文全落在思维链里时必须当失败，触发重试/转移 ——
    返回空串会把空答案写进训练数据，而且一声不吭。"""
    text, who = _pool(servers["reasoning_field"]).chat(
        [{"role": "user", "content": "hi"}], purpose="rewrite")
    assert text is None and who is None


def test_failover_skips_the_broken_replica(servers):
    """一个副本只会吐思维链时，要转移到正常副本而不是整体失败。"""
    pool = LLMPool({"models": [
        {"name": "bad", "model": "m",
         "base_url": f"http://127.0.0.1:{servers['reasoning_field']}/v1",
         "api_key": "x", "tags": ["rewrite"]},
        {"name": "good", "model": "m",
         "base_url": f"http://127.0.0.1:{servers['obedient']}/v1",
         "api_key": "x", "tags": ["rewrite"]}],
        "request": {"max_retries": 0, "failover": True},
        "purposes": {"rewrite": {"tags": ["rewrite"],
                                 "strategy": "failover"}}})
    text, who = pool.chat([{"role": "user", "content": "hi"}],
                          purpose="rewrite")
    assert text == "就绪" and who == "good"


def test_health_reports_thinking_left_on(servers):
    rows = _pool(servers["tag_in_content"]).health()
    assert rows[0]["status"] == "ok_thinking_on", rows
    assert rows[0]["reply"] == "就绪"


def test_health_flags_reasoning_only(servers):
    rows = _pool(servers["reasoning_field"]).health()
    assert rows[0]["status"] == "only_reasoning", rows


def test_health_clean_replica_is_plain_ok(servers):
    rows = _pool(servers["obedient"]).health()
    assert rows[0]["status"] == "ok", rows


# ---------------------------------------------------------------- 缺凭据
class _H401(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        key = self.headers.get("authorization", "").replace("Bearer ", "").strip()
        if key != "good-key":
            raw = json.dumps({"error": "Unauthorized"}).encode()
            self.send_response(401)
        else:
            raw = json.dumps({"id": "1", "object": "chat.completion", "model": "m",
                              "choices": [{"index": 0, "message": {
                                  "role": "assistant", "content": "就绪"},
                                  "finish_reason": "stop"}]}).encode()
            self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture(scope="module")
def auth_server():
    s = HTTPServer(("127.0.0.1", 0), _H401)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s.server_address[1]


def test_missing_key_is_reported_as_no_credential_not_fail(auth_server):
    """真实踩到的坑：本机 key 文件是 gitignore 的，git pull 不会带过去，
    于是 key 解析成空串、请求照发、服务端回 401，被报成
    "端点连不上 / 模型 id 不对"，让人跑去查网络。"""
    pool = LLMPool({"models": [{"name": "m1", "model": "m",
                                "base_url": f"http://127.0.0.1:{auth_server}/v1",
                                "api_key_env": "DEFINITELY_NOT_SET"}],
                    "purposes": {}})
    row = pool.health()[0]
    assert row["status"] == "no_credential", row
    assert "空凭据" in row["detail"], row


def test_wrong_key_still_reports_fail(auth_server):
    """key 有但不对，那就是真的 fail，不能也说成"没带 key"。"""
    pool = LLMPool({"models": [{"name": "m1", "model": "m",
                                "base_url": f"http://127.0.0.1:{auth_server}/v1",
                                "api_key": "wrong-key"}],
                    "purposes": {}})
    assert pool.health()[0]["status"] == "fail"


def test_init_local_creates_keys_for_private_endpoints_only(tmp_path, capsys):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    spec = importlib.util.spec_from_file_location(
        "llm_pool_check", os.path.join(root, "scripts", "llm_pool_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({"models": [
        {"name": "inner", "model": "m", "base_url": "http://10.1.2.3:8001/v1"},
        {"name": "outer", "model": "m", "base_url": "https://api.example.com/v1"},
    ], "purposes": {}}), encoding="utf-8")

    assert m.init_local(str(cfg), "k1") == 0
    doc = json.loads((tmp_path / "pool.local.json").read_text(encoding="utf-8"))
    names = {x["name"]: x["api_key"] for x in doc["models"]}
    assert names == {"inner": "k1"}, names          # 公网端点不自动填
    out = capsys.readouterr().out
    assert "公网" in out and "--judge-key" in out


def test_init_local_can_fill_public_endpoint_explicitly(tmp_path):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "llm_pool_check2", os.path.join(root, "scripts", "llm_pool_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg = tmp_path / "p.json"
    cfg.write_text(json.dumps({"models": [
        {"name": "outer", "model": "m", "base_url": "https://api.example.com/v1"}],
        "purposes": {}}), encoding="utf-8")
    m.init_local(str(cfg), "k1", "judge-k")
    doc = json.loads((tmp_path / "p.local.json").read_text(encoding="utf-8"))
    assert doc["models"][0]["api_key"] == "judge-k"
