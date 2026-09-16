# -*- coding: utf-8 -*-
"""扩写问法：并发 + 多轮补足。

一轮定生死是不够的 —— 风格校验会拒掉相当一部分，拿到手常常不足量，
只能靠人猜着调 --per-task 重跑。
"""
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

pytest.importorskip("openai")

CALLS = []
_LOCK = threading.Lock()


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        prompt = body["messages"][0]["content"]
        with _LOCK:
            CALLS.append(prompt)
            i = len(CALLS)
        # 每轮只给 3 条合格的，外加几条必被拒的 -> 逼出多轮
        good = [f"请检查该{{obj}}并列出第{i}-{k}项异常。" for k in range(3)]
        bad = ["有啥问题呗？", "检查一下哈", "{不存在的占位符}写点什么"]
        msg = {"role": "assistant",
               "content": "<think>想想</think>" +
                          json.dumps(good + bad, ensure_ascii=False)}
        raw = json.dumps({"id": "1", "object": "chat.completion", "model": "m",
                          "choices": [{"index": 0, "message": msg,
                                       "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_pool(tmp_path):
    CALLS.clear()
    s = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    port = s.server_address[1]
    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({
        "strategy": "round_robin",
        "models": [{"name": c, "model": "m",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "x", "tags": ["paraphrase"]} for c in "abc"],
        "purposes": {"paraphrase": {"tags": ["paraphrase"]}},
    }), encoding="utf-8")
    yield cfg
    s.shutdown()


def _run(cfg, out, *extra):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "gen_question_bank.py"),
         "--llm-config", str(cfg), "--out", str(out), *extra],
        capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_multiple_rounds_reach_the_target(fake_pool, tmp_path):
    """假模型每轮只给 3 条合格的，目标 6 条 -> 必须问第二轮。"""
    out = tmp_path / "qb.json"
    _run(fake_pool, out, "--per-task", "6", "--max-rounds", "3",
         "--tasks", "description")
    bank = json.loads(out.read_text(encoding="utf-8"))
    assert len(bank["questions"]["description"]) == 6
    assert len(CALLS) >= 2, "只问了一轮，没补足"


def test_single_round_would_fall_short(fake_pool, tmp_path):
    """把轮数限成 1 就应该不足量 —— 反证多轮确实在起作用。"""
    out = tmp_path / "qb.json"
    _run(fake_pool, out, "--per-task", "6", "--max-rounds", "1",
         "--tasks", "description")
    bank = json.loads(out.read_text(encoding="utf-8"))
    assert len(bank["questions"]["description"]) == 3


def test_later_rounds_tell_the_model_what_it_already_has(fake_pool, tmp_path):
    """不把已收下的发回去，多轮之间会大量重复，白烧调用。"""
    out = tmp_path / "qb.json"
    _run(fake_pool, out, "--per-task", "6", "--max-rounds", "3",
         "--tasks", "description")
    assert any("不要重复" in p for p in CALLS[1:]), CALLS[-1][-200:]


def test_style_validation_rejects_colloquial(fake_pool, tmp_path):
    """口语化问法会把模型语言层带偏，必须拒掉。"""
    out = tmp_path / "qb.json"
    log = _run(fake_pool, out, "--per-task", "6", "--tasks", "description")
    assert "口语化" in log, log
    bank = json.loads(out.read_text(encoding="utf-8"))
    assert not any("呗" in q or "哈" in q
                   for q in bank["questions"]["description"])


def test_work_is_spread_across_replicas(fake_pool, tmp_path):
    """7 个副本闲着、一条一条串行跑是浪费。"""
    out = tmp_path / "qb.json"
    log = _run(fake_pool, out, "--per-task", "6", "--max-rounds", "2",
               "--tasks", "description", "discrimination", "counting")
    m = re.search(r"并发 (\d+) 路", log)
    assert m and int(m.group(1)) > 1, log
    used = json.loads(re.search(r"模型池调用统计：(\{.*\})", log).group(1))
    assert len(used["by_model"]) > 1, used


def test_generated_questions_are_formattable(fake_pool, tmp_path):
    """留下来的问法必须能 format —— 占位符对不上会在构建时才炸。"""
    from aircraft_vqa.vqa import templates as T
    out = tmp_path / "qb.json"
    _run(fake_pool, out, "--per-task", "6", "--tasks", "description")
    dummy = {p: "x" for p in
             set().union(*[set(v) for v in T.ALLOWED_PLACEHOLDERS.values()])}
    for q in json.loads(out.read_text(encoding="utf-8"))["questions"]["description"]:
        q.format(**dummy)
