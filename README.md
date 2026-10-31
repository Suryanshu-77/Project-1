# tierwise

A self-hosted, OpenAI-compatible LLM gateway that picks the cheapest model expected to answer
each request well, across Anthropic, OpenAI, Gemini and local Ollama models, and measures whether
that choice paid off. Point any OpenAI SDK client at it by changing `base_url`.

Status: early development (M1: explicit-model passthrough to Anthropic and OpenAI).
Design: [docs/superpowers/specs/2026-11-01-tierwise-design.md](docs/superpowers/specs/2026-11-01-tierwise-design.md)

## Quickstart

```bash
cp .env.example .env        # then add your provider keys
uv sync
uv run python -m tierwise   # serves http://127.0.0.1:8000
```

Use it from any OpenAI SDK client by changing the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="claude-haiku-5-5",  # or an OpenAI model such as gpt-4.1-mini
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Each response carries `X-Tierwise-Model`, `X-Tierwise-Provider`, `X-Tierwise-Reason` and
`X-Tierwise-Request-Id` headers. Check every configured provider end to end with:

```bash
uv run python scripts/smoke.py
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```
