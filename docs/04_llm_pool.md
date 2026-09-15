# 模型池配置

多个 OpenAI 兼容端点按权重/轮询路由，失败自动转移。**改配置不用动代码。**

```bash
export LOCAL_LLM_KEY=你的key            # key 走环境变量，不要写进配置文件
python scripts/llm_pool_check.py        # 先体检，确认端点通
python scripts/llm_smoke.py --n 100     # 再跑烟测
```

## 为什么要池子

1. 手上不止一个端点，想**摊平负载**（同一模型的多个副本）或**按权重混用**
   （大模型保质量、小模型冲量）；
2. 某个副本挂了要能自动转到别的，而不是整批任务失败；
3. 不同用途该用不同参数 —— 扩问法要高温度求多样，改写答案要低温度求稳；
4. 调比例不该改代码。

## 配置结构

```jsonc
{
  "strategy": "round_robin",       // weighted | round_robin | failover
  "request": {
    "timeout": 120,                 // 单次请求超时（秒）
    "max_retries": 1,               // 同一个模型内部重试次数
    "retry_backoff": 1.5,           // 重试退避基数，第 n 次等 backoff^n 秒
    "failover": true                // 重试完还失败就换池里下一个模型
  },
  "generation": {                   // 全局默认生成参数
    "temperature": 0.6, "top_p": 0.9, "max_tokens": 1024
  },
  "models": [ /* 见下 */ ],
  "purposes": { /* 见下 */ }
}
```

### 三种路由策略

| 策略 | 行为 | 适用 |
|---|---|---|
| `round_robin` | 依次轮流 | **同一模型的多个副本**，摊平负载 |
| `weighted` | 按 `weight` 抽主选 | 不同档位的模型混用，如 大:小 = 3:1 |
| `failover` | 严格按配置顺序，主备明确 | 有明确主力和兜底的场景 |

不管哪种策略，主选失败后都会依次尝试后面的（`failover: true` 时）。

### 模型条目

```jsonc
{
  "name": "qwen27b-8001",           // 池内唯一标识，统计与日志按它归类
  "enabled": true,                  // false = 不参与路由（副本临时下线就改这个）
  "weight": 1,                      // weighted 策略下的权重
  "model": "Qwen3.8-27B",           // 服务端真实的模型 id
  "base_url": "http://x:8001/v1/chat/completions",
  "api_key_env": "LOCAL_LLM_KEY",   // 从该环境变量取 key
  "tags": ["rewrite", "judge"],     // 供 purposes 按标签筛选
  "concurrency": 8,                 // 预留：并发上限
  "timeout": 120,                   // 覆盖全局超时
  "generation": {"temperature": 0.7} // 这个模型自己的默认脾气
}
```

> **base_url 会自动归一化**：服务方给的常是完整的 `.../v1/chat/completions`，
> 而 SDK 的 `base_url` 只要到 `/v1`。直接填完整路径会拼成
> `/v1/chat/completions/chat/completions` 报 404，很难查，所以池子会把末尾的
> `chat/completions` 剥掉。两种写法都能用。

> **key 一律走 `api_key_env`**，别用 `api_key` 直接写进文件 —— 配置是要进 git 的。

### 用途（purposes）

按"这件事要什么参数"分组，而不是按模型分：

```jsonc
"purposes": {
  "rewrite":    {"models": ["*"], "strategy": "round_robin",
                 "generation": {"temperature": 0.55, "max_tokens": 512}},
  "paraphrase": {"models": ["*"], "generation": {"temperature": 0.95}},
  "judge":      {"tags": ["judge"], "strategy": "failover",
                 "generation": {"temperature": 0.0}}
}
```

- `models`: 限定用哪几个（`["*"]` 或省略 = 全部）
- `tags`: 按标签筛（和 `models` 可二选一）
- `strategy`: 覆盖全局策略
- `generation`: 覆盖生成参数

### 生成参数的覆盖顺序

```
全局 generation  <  模型 generation  <  用途 generation  <  调用时传入
```

**用途排在模型后面是故意的**：模型级是"这个模型自己的默认脾气"，
用途级是"这件事要求什么"。改写答案要求温度低，就该压过模型的默认温度，
否则用途块写了等于白写。

## 扩容

同一模型的多个副本，照抄条目改 `name` 和 `base_url` 即可，轮询会自动摊平。
副本临时下线把 `enabled` 改成 `false`，不用删条目。

## 谁在用这个池

| 脚本 | 用途 | 说明 |
|---|---|---|
| `scripts/llm_pool_check.py` | — | 逐个端点体检，跑批量前先跑 |
| `scripts/llm_smoke.py` | `rewrite` | 100 条改写烟测，含事实指纹比对 |
| `scripts/gen_question_bank.py` | `paraphrase` | 扩写问法，高温度求多样 |
| `scripts/build_vqa.py --llm-rewrite` | `rewrite` | 全量改写（烟测过了再跑）|

## 排错

| 现象 | 原因 |
|---|---|
| `no_credential` | `api_key_env` 指的环境变量没 export，且没填 base_url |
| `fail` + 404 | base_url 路径不对（本仓库已自动剥 `chat/completions`），或 `model` 不是服务端真实 id（vLLM 看 `--served-model-name`）|
| `fail` + 连接超时 | 端点不通。`curl <base_url>/models` 试一下 |
| 全部转移后仍失败 | `llm_pool_check.py` 看是哪几个挂了 |

## 另一半"动态调整"：构建参数

模型池管的是大模型侧。数据构建侧的配比、平衡系数也能临时覆盖，不用改 yaml：

```bash
python scripts/build_vqa.py \
  --set normal_per_anomalous=0.5 \
  --set task_ratio.grounding_single=0.25 \
  --set 'active_defect_types=["crack","corrosion"]' \
  --set class_max_over_min=2.0
```

支持点号路径，值按 JSON 解析（解析不了当字符串）。适合批量跑对比实验 ——
每次改 yaml 容易忘了改回来。
