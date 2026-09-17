# -*- coding: utf-8 -*-
"""改写层的并发与事实保护。

串行跑不现实：纯文本答案五万多条、一次十来秒，排下来一百六十多小时。
但并发之后事实保护不能松 —— 改写动了数字就必须回退到模板答案。
"""
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import pytest

pytest.importorskip("openai")

from aircraft_vqa.vqa.llm import PROTECTED_TASKS, LLMRewriter

CALLS = []
_L = threading.Lock()


class _T(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    MODE = "polish"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        src = body["messages"][-1]["content"]
        with _L:
            CALLS.append(src)
            i = len(CALLS)
        mode = self.server.mode
        if mode == "corrupt":                       # 改数字 -> 该被拦
            out = re.sub(r"\d+", lambda m: str(int(m.group()) + 1), src, count=1)
        elif mode == "thinking":                    # 带思维链
            out = f"<think>想想</think>{src}（已复查）"
        elif mode == "empty":
            out = ""
        else:
            out = src.replace("。", "，现场已记录。", 1) if "。" in src else src + "。"
        raw = json.dumps({"id": str(i), "object": "chat.completion", "model": "m",
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant",
                                                   "content": out}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    made = []

    def start(mode="polish"):
        CALLS.clear()
        s = _T(("127.0.0.1", 0), _H)
        s.mode = mode
        threading.Thread(target=s.serve_forever, daemon=True).start()
        made.append(s)
        return s.server_address[1]
    yield start
    for s in made:
        s.shutdown()


def _rw(port, n_models=4, conc=4):
    from aircraft_vqa.llm import LLMPool
    pool = LLMPool({"strategy": "round_robin", "models": [
        {"name": f"r{i}", "model": "m", "concurrency": conc,
         "base_url": f"http://127.0.0.1:{port}/v1", "api_key": "x",
         "tags": ["rewrite"]} for i in range(n_models)],
        "purposes": {"rewrite": {"tags": ["rewrite"]}}})
    return LLMRewriter(pool, "rewrite")


def _recs(n):
    return [{"task": "description", "question": "请描述该口盖的状况。",
             "answer": f"检查发现 {i % 4 + 1} 处异常，位于画面左上。"}
            for i in range(n)]


def test_workers_come_from_configured_concurrency(server):
    rw = _rw(server(), n_models=4, conc=4)
    assert rw.default_workers() == 16


def test_rewrite_many_processes_everything(server):
    rw = _rw(server())
    recs = _recs(120)
    st = rw.rewrite_many(recs, progress_every=0)
    assert st["n_sent"] == 120
    assert st["n_rewritten"] == 120, st
    assert len(CALLS) == 120
    assert all(r.get("rewritten") for r in recs)


def test_protected_tasks_never_hit_the_network(server):
    """JSON / 多轮 / 选择题本来就不参与改写，不该占并发也不该发请求。"""
    rw = _rw(server())
    prot = sorted(PROTECTED_TASKS)[0]
    recs = _recs(20) + [{"task": prot, "question": "q", "answer": "[]"}
                        for _ in range(30)]
    st = rw.rewrite_many(recs, progress_every=0)
    assert st["n_skipped_protected"] == 30
    assert st["n_sent"] == 20
    assert len(CALLS) == 20, "受保护的任务也发请求了"


def test_fact_drift_is_rejected_under_concurrency(server):
    """并发不能把事实保护冲掉 —— 改了数字必须整条回退。"""
    rw = _rw(server("corrupt"))
    recs = _recs(60)
    st = rw.rewrite_many(recs, progress_every=0)
    assert st["n_rewritten"] == 0, st
    assert st["rejected"].get("fact_drift") == 60
    for r in recs:
        assert "检查发现" in r["answer"], "回退后答案被改坏了"
        assert not r.get("rewritten")


def test_thinking_is_stripped_in_rewrite_path(server):
    rw = _rw(server("thinking"))
    recs = _recs(20)
    rw.rewrite_many(recs, progress_every=0)
    for r in recs:
        assert "<think>" not in r["answer"]
        assert "（已复查）" in r["answer"]


def test_empty_response_keeps_the_template_answer(server):
    rw = _rw(server("empty"))
    recs = _recs(20)
    before = [r["answer"] for r in recs]
    rw.rewrite_many(recs, progress_every=0)
    assert [r["answer"] for r in recs] == before


def test_concurrency_actually_overlaps(server):
    """16 路并发的耗时应当明显低于串行 —— 不然等于白写。"""
    import time
    rw = _rw(server(), n_models=4, conc=4)
    recs = _recs(64)
    t0 = time.time()
    rw.rewrite_many(recs, workers=1, progress_every=0)
    serial = time.time() - t0
    for r in recs:
        r.pop("rewritten", None)
        r["answer"] = r.pop("answer_template", r["answer"])
    t0 = time.time()
    rw.rewrite_many(recs, workers=16, progress_every=0)
    par = time.time() - t0
    assert par < serial * 0.7, f"串行 {serial:.2f}s vs 并发 {par:.2f}s"


def test_original_answer_is_kept_for_comparison(server):
    rw = _rw(server())
    recs = _recs(10)
    orig = [r["answer"] for r in recs]
    rw.rewrite_many(recs, progress_every=0)
    for r, o in zip(recs, orig):
        assert r["answer_template"] == o
        assert r["answer"] != o
