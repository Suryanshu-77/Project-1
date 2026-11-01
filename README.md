# tierwise

A self-hosted, OpenAI-compatible LLM gateway that picks the cheapest model expected to answer
each request well, across Anthropic, OpenAI, Gemini and local Ollama models, and measures whether
that choice paid off. Point any OpenAI SDK client at it by changing `base_url`.

**Benchmark:** on 3,943 held-out prompts, tierwise's router keeps 95% of GPT-4's quality at 44%
of its cost (random routing: 63%). Method, out-of-distribution results and limitations are in
[BENCHMARK.md](BENCHMARK.md).

Status: early development (M5: rules plus a learned router, measured by an eval harness, across
four providers with streaming, a model registry, cost tracking and a request log).
Design: [docs/superpowers/specs/2026-11-01-tierwise-design.md](docs/superpowers/specs/2026-11-01-tierwise-design.md)

## Quickstart

```bash
cp .env.example .env        # then add keys for the providers you use
uv sync
uv run python -m tierwise   # serves http://127.0.0.1:8000
```

Use it from any OpenAI SDK client by changing the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="auto",              # let tierwise choose; or any id from config/registry.yaml
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False,              # stream=True works too
)
```

`GET /v1/models` lists the models whose provider is configured. Each response carries
`X-Tierwise-Model`, `X-Tierwise-Provider`, `X-Tierwise-Reason` and `X-Tierwise-Request-Id`
headers, plus `X-Tierwise-Cost-USD` on non-streamed replies.

## Automatic routing

Send one of these as `model` and tierwise picks the model for you:

| `model` | What happens |
|---|---|
| `auto` (default) | score the request's difficulty, pick the cheapest model good enough for it |
| `auto:cheap` / `auto:quality` | same, with the quality bar lowered or raised one notch |
| `tier:basic` / `tier:moderate` / `tier:flagship` | cheapest capable model in that tier, no scoring |
| a registry id, e.g. `claude-sonnet-5-5` | exactly that model |

Put `[critical]` or `[easy]` anywhere in your last message to force the hardest or easiest
routing; the tag is removed before the request is forwarded.

How a model is chosen:

1. **Difficulty.** `src/tierwise/router/rules.py` scores the request from cheap signals: length,
   code, math, high-stakes words ("production", "security"), reasoning words ("design",
   "prove"), easy words ("translate", "hi"), tools, images and conversation length. When the
   rules aren't sure, the [learned router](#learned-router) decides instead.
2. **Quality bar.** `bar = 0.30 + 0.65 x difficulty`, shifted by `auto:cheap`/`auto:quality`,
   and raised one step when the scorer is unsure (a score near 0.5).
3. **Selection.** Models that lack a needed capability (tools, vision, JSON) or context window
   are dropped. Of those that clear the bar, the one with the lowest *estimated* cost wins,
   counting prompt-cache discounts. A 422 `no_eligible_model` means nothing could handle it.
4. **Affinity.** Conversations stick to the model whose prompt cache is still warm, unless
   switching would save more than 10%.

### Reasoning effort

Thinking tokens are billed as output, so tierwise picks a **model and an effort level
together**: each model with an `efforts` block in the registry becomes one option per level,
and the cheapest option that still clears the quality bar wins.

- **Scale:** `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Sent to Anthropic as
  `output_config.effort`, to OpenAI as `reasoning_effort`, and to Gemini as
  `thinkingConfig.thinkingLevel`. When the chosen level is the provider's default, nothing
  is sent.
- **Quality:** a level below the default lowers quality by its registry value at difficulty
  0.5, scaled with the request's difficulty: low effort costs little on easy requests and a
  lot on hard ones.
- **Caching:** `effort_in_cache_key: true` (Anthropic) means a different effort can't reuse
  the conversation's prompt cache, so tierwise prices that switch as a cache miss.
- **Your effort wins:** send `reasoning_effort` and tierwise only considers models that
  support that level (422 `no_eligible_model` if none do).
- `X-Tierwise-Effort` reports the level sent, or `default`; `scripts/costs.py` breaks cost
  down by effort.

`X-Tierwise-Reason` says why: `rules` (scored), `learned` (the classifier decided), `override`
(tag, tier or explicit model), `affinity` (kept on its warm model) or `fallback` (router error,
or no model cleared the bar).
Retune the scorer by editing `WEIGHTS` in `rules.py` and running
`uv run pytest tests/router/test_rules.py`, which checks it against 30 labeled prompts.

### Learned router

The rules router decides alone only when it is confident (confidence of 0.8 or more, about 1.5%
of real prompts); everywhere else a trained classifier decides, and the rules can only raise its
difficulty, never lower it. It embeds the last two user turns and the head of the system prompt (at most 256
tokens) with MiniLM on ONNX Runtime, and a small classifier head predicts P(a cheap model's
answer won't be good enough). Inference runs in a two-thread pool, never on the event loop.

It is trained on [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset)
(Apache-2.0): prompts whose Mixtral answer a GPT-4 judge graded below 4 of 5 are labeled
"needs a strong model". Held-out results (`models/router/metrics.json`):

| | Test ROC-AUC |
|---|---|
| Rules router | 0.589 |
| Learned router (MiniLM + MLP) | **0.735** |

Trained on 32k prompts, 13% of which needed a strong model. Validation picked the MLP over
logistic regression (0.745 vs 0.731 ROC-AUC) and a decision threshold of 0.15. On CPU the
encoder plus classifier takes 6.3 ms at p50 (`scripts/bench_router.py`).

The tokenizer and trained classifier are committed; the 90 MB ONNX encoder is not. Without it
the gateway logs a warning and routes with rules only. To build it, or to retrain:

```bash
uv sync --group train
uv run python -m training.export_onnx             # models/router/model.onnx
uv run python -m training.prepare_data --limit 40000
uv run python -m training.train                   # classifier.npz, router.json, metrics.json
uv run python scripts/bench_router.py             # p50 latency, budget 10 ms
```

Set `TIERWISE_LEARNED_ROUTER=false` to turn it off; `TIERWISE_ROUTER_DIR` points elsewhere.

## Providers

| Provider | Enable with | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | prompt-cache reads and writes are priced separately |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` points it at any OpenAI-compatible host |
| Gemini | `GEMINI_API_KEY` | native API; only `data:` image URLs are accepted |
| Ollama | `OLLAMA_BASE_URL` | free local models, e.g. `http://localhost:11434/v1` |

## Model registry

Every servable model lives in [config/registry.yaml](config/registry.yaml) with its provider,
prices (USD per 1M tokens), context window and capabilities. To add a model, add an entry:

```yaml
  - id: my-model            # the name clients send
    provider: openai
    upstream_id: real-name  # optional, if the provider calls it something else
    price_in: 0.50
    price_out: 1.50
    price_cache_read: 0.05  # optional; defaults to price_in
    context: 128000
    caps: [tools, json]
    quality: 0.6
    tier: basic
```

The gateway refuses to start if the file is invalid. Verify prices against each provider's
official pricing page; the shipped values are a starting point.

## Costs and the request log

Every request is logged (SQLite by default, `TIERWISE_DB_URL`) with tokens, cost and the cost
the same tokens would have had on the baseline model. Summarize it with:

```bash
uv run python scripts/costs.py
```

Check every configured provider end to end, plain and streamed:

```bash
uv run python scripts/smoke.py
```

```bash
uv run python scripts/smoke.py --stream
```

See which model `auto` picks for an easy, a medium and a critical prompt:

```bash
uv run python scripts/smoke.py --auto
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```
