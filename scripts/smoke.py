"""Live smoke test: send one tiny prompt per model through a running tierwise gateway.

Usage: uv run python scripts/smoke.py [--model MODEL ...] [--base-url URL]
Models whose provider has no key configured (404 model_not_found) are skipped.
"""

import argparse
import sys
import time

import openai

DEFAULT_MODELS = ["claude-haiku-5-5", "gpt-4.1-mini"]
PROMPT = "Reply with the single word: pong"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    failures = 0
    for model in args.models or DEFAULT_MODELS:
        started = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": PROMPT}], max_tokens=16
            )
        except openai.NotFoundError:
            print(f"SKIP {model}: provider not configured")
            continue
        except openai.APIError as exc:
            print(f"FAIL {model}: {exc}")
            failures += 1
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        completion = raw.parse()
        provider = raw.headers.get("X-Tierwise-Provider")
        content = completion.choices[0].message.content
        print(f"OK   {model} via {provider} in {elapsed_ms:.0f} ms: {content!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
