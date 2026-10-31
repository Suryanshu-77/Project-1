# tierwise

A self-hosted, OpenAI-compatible LLM gateway that picks the cheapest model expected to answer
each request well, across Anthropic, OpenAI, Gemini and local Ollama models, and measures whether
that choice paid off. Point any OpenAI SDK client at it by changing `base_url`.

Status: early development (M2: four providers, streaming, model registry, cost tracking and a
request log; automatic routing arrives in M3).
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
    model="claude-haiku-5-5",  # any id from config/registry.yaml
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False,              # stream=True works too
)
```

`GET /v1/models` lists the models whose provider is configured. Each response carries
`X-Tierwise-Model`, `X-Tierwise-Provider`, `X-Tierwise-Reason` and `X-Tierwise-Request-Id`
headers, plus `X-Tierwise-Cost-USD` on non-streamed replies.

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

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```
