#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型池体检：逐个端点发一句最短请求，确认真的连得上。

跑批量之前先跑这个。端点填错、key 没设、模型 id 不对，
在这里两秒就能看出来，不用等跑到一半才发现全在失败转移。

    python scripts/llm_pool_check.py
    python scripts/llm_pool_check.py --config configs/llm_pool.prod.json
    python scripts/llm_pool_check.py --purpose paraphrase   # 只看该用途可用的
    python scripts/llm_pool_check.py --show                 # 只看配置，不发请求
"""
from __future__ import annotations

import argparse
import os
import sys

import _bootstrap  # noqa: F401

from aircraft_vqa.llm import LLMPool

ADVICE = {
    "only_reasoning": "只返回了思维链没有正文 —— 调大 max_tokens，"
                      "或确认服务端吃 enable_thinking=false",
    "no_credential": "环境变量没设，或者 base_url 也没填",
    "disabled": "配置里 enabled: false",
    "fail": "端点连不上 / 模型 id 不对 / key 无效，看 detail",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_pool.json")
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--show", action="store_true", help="只打印配置，不发请求")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"配置不存在：{args.config}")
        return 1
    pool = LLMPool.from_file(args.config)

    print(f"配置：{args.config}")
    print(pool.describe())

    if args.purpose:
        avail = pool.available(args.purpose)
        print(f"\n用途 {args.purpose} 可用模型：{[m.name for m in avail] or '无'}")
        print(f"尝试顺序：{[m.name for m in pool.order(args.purpose)]}")
        if avail:
            print(f"生成参数：{pool.gen_params(args.purpose, avail[0])}")

    if args.show:
        return 0

    try:
        import openai  # noqa: F401
    except ImportError:
        print("\n缺少 openai 包，装一下再体检：pip install openai")
        return 1

    print("\n逐个体检中……")
    rows = pool.health()
    ok = 0
    warn = 0
    for r in rows:
        st = r["status"]
        if st == "ok":
            ok += 1
            print(f"  ✓ {r['name']:22s} {r['latency_s']:>5.2f}s  "
                  f"回复「{r['reply']}」")
        elif st == "ok_thinking_on":
            ok += 1
            warn += 1
            print(f"  ⚠ {r['name']:22s} {r['latency_s']:>5.2f}s  "
                  f"回复「{r['reply']}」（带思维链，已剥离）")
        else:
            print(f"  ✗ {r['name']:22s} {st}  —— {ADVICE.get(st, '')}")
            if r.get("detail"):
                print(f"      {r['detail']}")

    print(f"\n可用 {ok}/{len(rows)}")
    if warn:
        print(f"其中 {warn} 个服务端没关掉思考。答案会被客户端剥干净，不影响正确性，"
              "但批量跑会白烧不少 token。\n"
              "  想在服务端关：vLLM 启动加 --reasoning-parser，或确认它吃 "
              "chat_template_kwargs.enable_thinking=false")
    if ok == 0:
        print("一个都连不上。检查：\n"
              "  1. base_url 是否带 /v1 后缀\n"
              "  2. api_key_env 指向的环境变量是否已 export\n"
              "  3. model 是否是服务端真实的模型 id（vLLM 用 --served-model-name）\n"
              "  4. 本地服务是否在跑：curl <base_url>/models")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
