"""Live smoke test: send one tiny prompt per model through a running tierwise gateway.

Usage: uv run python scripts/smoke.py [--model MODEL ...] [--stream] [--auto] [--base-url URL]
By default it tries the basic-tier model of each provider in config/registry.yaml.
Models whose provider has no key configured (404 model_not_found) are skipped.
"""

import argparse
import sys
import time

import openai

from tierwise.registry import load_registry

PROMPT = "Reply with the single word: pong"
AUTO_PROMPTS = [
    "hi",
    "Explain how a hash map handles collisions, with a short example.",
    "[critical] Design a zero-downtime migration plan for a 4 TB production database.",
]


def default_models() -> list[str]:
    registry = load_registry("config/registry.yaml")
    seen: dict[str, str] = {}
    for model in registry.models:
        if model.tier == "basic":
            seen.setdefault(model.provider, model.id)
    return list(seen.values())


def run_plain(client: openai.OpenAI, model: str) -> str:
    started = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "user", "content": PROMPT}], max_tokens=16
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    completion = raw.parse()
    provider = raw.headers.get("X-Tierwise-Provider")
    cost = raw.headers.get("X-Tierwise-Cost-USD")
    content = completion.choices[0].message.content
    return f"via {provider} in {elapsed_ms:.0f} ms, ${cost}: {content!r}"


def run_stream(client: openai.OpenAI, model: str) -> str:
    started = time.perf_counter()
    first_token_ms: float | None = None
    parts: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=16,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            parts.append(chunk.choices[0].delta.content)
    total_ms = (time.perf_counter() - started) * 1000
    ttft = f"{first_token_ms:.0f}" if first_token_ms is not None else "-"
    return f"streamed, first token {ttft} ms, total {total_ms:.0f} ms: {''.join(parts)!r}"


def run_auto(client: openai.OpenAI) -> int:
    failures = 0
    for prompt in AUTO_PROMPTS:
        try:
            raw = client.chat.completions.with_raw_response.create(
                model="auto", messages=[{"role": "user", "content": prompt}], max_tokens=16
            )
        except openai.APIError as exc:
            print(f"FAIL {prompt[:40]!r}: {exc}")
            failures += 1
            continue
        model = raw.headers.get("X-Tierwise-Model")
        reason = raw.headers.get("X-Tierwise-Reason")
        effort = raw.headers.get("X-Tierwise-Effort")
        cost = raw.headers.get("X-Tierwise-Cost-USD")
        print(f"OK   {prompt[:40]!r} -> {model} @ {effort} effort ({reason}), ${cost}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--auto", action="store_true", help="show what auto routing picks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    if args.auto:
        return run_auto(client)
    failures = 0
    for model in args.models or default_models():
        try:
            summary = run_stream(client, model) if args.stream else run_plain(client, model)
        except openai.NotFoundError:
            print(f"SKIP {model}: provider not configured")
            continue
        except openai.APIError as exc:
            print(f"FAIL {model}: {exc}")
            failures += 1
            continue
        print(f"OK   {model} {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
