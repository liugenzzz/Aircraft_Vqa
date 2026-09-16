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


# ---------------------------------------------------------------- 截断
class _HTrunc(BaseHTTPRequestHandler):
    """严格按 max_tokens 截断，复刻真实 vLLM 的行为。"""
    CH_PER_TOK = 1.1

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        prompt = body["messages"][0]["content"]
        mt = int(body.get("max_tokens") or 512)
        with _LOCK:
            CALLS.append(prompt)
            i = len(CALLS)
        m = re.search(r"请再写 (\d+) 条", prompt)
        want = int(m.group(1)) if m else 20
        qs = [f"请检查该{{obj}}并逐项列出第{i}轮第{k}处可见异常。"
              for k in range(want)]
        full = json.dumps(qs, ensure_ascii=False)
        budget = int(mt * self.CH_PER_TOK)
        text = full if len(full) <= budget else full[:budget]
        fr = "stop" if len(full) <= budget else "length"
        raw = json.dumps({"id": "1", "object": "chat.completion", "model": "m",
                          "choices": [{"index": 0, "finish_reason": fr,
                                       "message": {"role": "assistant",
                                                   "content": text}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def trunc_pool(tmp_path):
    CALLS.clear()
    s = HTTPServer(("127.0.0.1", 0), _HTrunc)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    port = s.server_address[1]
    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({
        "models": [{"name": "a", "model": "m",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "x", "tags": ["paraphrase"]}],
        # 用途里就是这个害人的固定 512
        "purposes": {"paraphrase": {"tags": ["paraphrase"],
                                    "generation": {"max_tokens": 512}}},
    }), encoding="utf-8")
    yield cfg
    s.shutdown()


def test_budget_scales_with_the_ask(trunc_pool, tmp_path):
    """真实踩到的坑：用途里 max_tokens 固定 512，要 40 条问法时必然截断，
    16 个任务全部 0 条，报的却是"返回里没有 JSON 数组"，看不出是被截了。"""
    out = tmp_path / "qb.json"
    _run(trunc_pool, out, "--per-task", "40", "--chunk", "25",
         "--max-rounds", "3", "--tasks", "description")
    bank = json.loads(out.read_text(encoding="utf-8"))
    assert len(bank["questions"]["description"]) >= 25, bank["stats"]


def test_prompt_count_matches_the_budget(trunc_pool, tmp_path):
    """prompt 里写"再写 N 条"和预算必须同一个 N。
    只在续轮改写的话，第一轮写着 per_task 而预算按 chunk 算，照样截断。"""
    out = tmp_path / "qb.json"
    _run(trunc_pool, out, "--per-task", "40", "--chunk", "12",
         "--max-rounds", "1", "--tasks", "description")
    assert CALLS, "一次都没调用"
    m = re.search(r"请再写 (\d+) 条", CALLS[0])
    assert m and int(m.group(1)) <= 12, CALLS[0][:200]


def test_truncated_output_is_salvaged(tmp_path):
    """被切掉尾巴时，前面写完整的那些是好的，整批丢掉太浪费。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gqb", os.path.join(REPO, "scripts", "gen_question_bank.py"))
    g = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    spec.loader.exec_module(g)

    v, salvaged = g.parse_array('["请检查该部件。", "请列出全部异常。", "请标出裂')
    assert v == ["请检查该部件。", "请列出全部异常。"] and salvaged
    v, salvaged = g.parse_array('["请检查。", "请列出。"]')
    assert v == ["请检查。", "请列出。"] and not salvaged
    assert g.parse_array("我觉得这个任务应该这样理解……") == (None, False)
    # 转义引号不能把捡取逻辑带偏
    v, _ = g.parse_array(r'["请查看\"焊缝\"处。", "请列出。"')
    assert v == ['请查看"焊缝"处。', "请列出。"], v


class _HCapped(_HTrunc):
    """服务端自己有硬上限（max_model_len 小），客户端要多少都没用。"""
    def do_POST(self):
        import io as _io
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        d = json.loads(body or b"{}")
        d["max_tokens"] = min(int(d.get("max_tokens") or 512), 200)
        raw = json.dumps(d).encode()
        self.headers.replace_header("content-length", str(len(raw))) \
            if "content-length" in self.headers else None
        self.rfile = _io.BytesIO(raw)
        self.headers["content-length"] = str(len(raw))
        _HTrunc.do_POST(self)


@pytest.fixture
def capped_pool(tmp_path):
    CALLS.clear()
    s = HTTPServer(("127.0.0.1", 0), _HCapped)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    cfg = tmp_path / "pool.json"
    cfg.write_text(json.dumps({
        "models": [{"name": "a", "model": "m",
                    "base_url": f"http://127.0.0.1:{s.server_address[1]}/v1",
                    "api_key": "x", "tags": ["paraphrase"]}],
        "purposes": {"paraphrase": {"tags": ["paraphrase"]}},
    }), encoding="utf-8")
    yield cfg
    s.shutdown()


def test_truncation_is_named_not_blamed_on_the_model(capped_pool, tmp_path):
    """报"返回里没有 JSON 数组"会让人去查模型听不听话，
    实际原因是输出被截断 —— 必须点名，并且要把末尾打出来。"""
    out = tmp_path / "qb.json"
    _run(capped_pool, out, "--per-task", "40", "--chunk", "40",
         "--max-rounds", "1", "--tasks", "description")
    bank = json.loads(out.read_text(encoding="utf-8"))
    rej = json.dumps(bank["stats"]["description"]["rejected"], ensure_ascii=False)
    # 要么抢救成功（记了"抢救"），要么明说被截断 —— 都不能只说"没有 JSON 数组"
    assert "抢救" in rej or "截断" in rej, rej


def test_salvage_keeps_what_was_written(capped_pool, tmp_path):
    """服务端硬封顶时也不该颗粒无收 —— 写完整的那些要留下。"""
    out = tmp_path / "qb.json"
    _run(capped_pool, out, "--per-task", "40", "--chunk", "40",
         "--max-rounds", "3", "--tasks", "description")
    got = json.loads(out.read_text(encoding="utf-8"))["questions"]["description"]
    assert got, "被截断就一条都不要了，太浪费"
