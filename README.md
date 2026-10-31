# tierwise

A self-hosted, OpenAI-compatible LLM gateway that picks the cheapest model expected to answer
each request well, across Anthropic, OpenAI, Gemini and local Ollama models, and measures whether
that choice paid off. Point any OpenAI SDK client at it by changing `base_url`.

Status: early development (M1: explicit-model passthrough to Anthropic and OpenAI).
Design: [docs/superpowers/specs/2026-11-01-tierwise-design.md](docs/superpowers/specs/2026-11-01-tierwise-design.md)

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```
