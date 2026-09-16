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
    "no_credential": "没带 key。本机 key 文件是 gitignore 的，git pull 不会"
                     "带过来，生成一份：\n"
                     "        python scripts/llm_pool_check.py "
                     "--init-local <你的key>",
    "disabled": "配置里 enabled: false",
    "fail": "端点连不上 / 模型 id 不对 / key 无效，看 detail",
}


def init_local(config_path: str, key: str, judge_key: str = None) -> int:
    """生成本机 key 文件。

    这个文件是 gitignore 的 —— 这正是它存在的意义（仓库是公开的），
    但也意味着 git pull 不会把它带到别的机器上，每台机器得各自生成一次。
    不生成的话 key 解析成空串，请求照发，服务端回 401，
    看起来像"端点连不上"，其实是压根没带凭据。
    """
    import json as _json
    base = config_path[:-5] if config_path.endswith(".json") else config_path
    out = base + ".local.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = _json.load(f)

    from urllib.parse import urlparse
    import ipaddress

    def is_private(url: str) -> bool:
        host = (urlparse(url).hostname or "")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False          # 主机名一律当公网
        return ip.is_private or ip.is_loopback

    models, n_pub = [], 0
    for m in cfg.get("models", []):
        if is_private(m.get("base_url", "")):
            models.append({"name": m["name"], "api_key": key})
        elif judge_key:
            models.append({"name": m["name"], "api_key": judge_key})
            n_pub += 1
        else:
            n_pub += 1
    doc = {"_说明": "本机 key，已 gitignore，不会进仓库。git pull 不会带它，"
                   "换机器要重新生成。",
           "models": models}
    if os.path.exists(out):
        print(f"{out} 已存在，先备份成 {out}.bak")
        os.replace(out, out + ".bak")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print(f"已生成 {out}：{len(models)} 个端点填了 key")
    if n_pub and not judge_key:
        print(f"还有 {n_pub} 个**公网**端点没填 —— 它们的 key 是真凭据，"
              "确认要写进本机文件就加 --judge-key <KEY>，"
              "或者 export 对应的环境变量")
    print("接着跑：python scripts/llm_pool_check.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_pool.json")
    ap.add_argument("--init-local", metavar="KEY", default=None,
                    help="生成本机 key 文件 configs/llm_pool.local.json（已 "
                         "gitignore，git pull 不会带过来，每台机器要各自生成）。"
                         "给一个 KEY 就填给所有内网副本；"
                         "另配公网端点用 --judge-key")
    ap.add_argument("--judge-key", default=None,
                    help="配合 --init-local：单独给公网/把关端点的 key")
    ap.add_argument("--purpose", default=None)
    ap.add_argument("--show", action="store_true", help="只打印配置，不发请求")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"配置不存在：{args.config}")
        return 1
    if args.init_local:
        return init_local(args.config, args.init_local, args.judge_key)

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
              "  0. 本机 key 文件在不在 —— 它是 gitignore 的，git pull 不会"
              "带过来，换台机器就得重新生成：\n"
              "     python scripts/llm_pool_check.py --init-local <你的key>\n"
              "  1. base_url 是否带 /v1 后缀\n"
              "  2. api_key_env 指向的环境变量是否已 export\n"
              "  3. model 是否是服务端真实的模型 id（vLLM 用 --served-model-name）\n"
              "  4. 本地服务是否在跑：curl <base_url>/models")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
