"""A deterministic OpenAI-compatible model server for the compose stack and its smoke check.

It answers ``POST /v1/chat/completions`` (JSON or SSE) with a short canned reply whose usage
scales with the prompt, so the dashboard shows real-looking traffic without any API keys.
``GET /stats`` reports how many requests and distinct client connections it has seen, which
the smoke check uses to confirm the gateway reuses pooled connections.

Run: uvicorn --app-dir scripts fake_upstream:app --host 0.0.0.0 --port 8080
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

STATS: dict[str, Any] = {"requests": 0, "connections": set()}
REPLY = "This is a canned answer from the fake upstream."
JUDGE_MARKER = "tierwise-judge-v1"  # must match tierwise.flywheel.JUDGE_MARKER
HARD = ("design", "prove", "debug", "production", "architecture")


def _judge(messages: list[dict[str, Any]]) -> str | None:
    """A deterministic fake judge: cheap answers to hard-looking prompts score 2, others 5,
    so flywheel labels carry a signal a retrained router can learn."""
    first = messages[0] if messages else {}
    if first.get("role") != "system" or JUDGE_MARKER not in str(first.get("content")):
        return None
    conversation = str(messages[-1].get("content")).split("Answer A:")[0].lower()
    return json.dumps({"score": 2 if any(word in conversation for word in HARD) else 5})


def _usage(messages: list[dict[str, Any]]) -> dict[str, int]:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return {"prompt_tokens": max(1, chars // 4), "completion_tokens": len(REPLY) // 4}


async def chat(request: Request) -> Response:
    STATS["requests"] += 1
    if request.client is not None:
        STATS["connections"].add((request.client.host, request.client.port))
    body = await request.json()
    model, usage = body.get("model", "fake"), _usage(body.get("messages", []))
    base = {"id": "fake-1", "model": model, "created": int(time.time())}
    if not body.get("stream"):
        message = {"role": "assistant", "content": _judge(body.get("messages", [])) or REPLY}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return JSONResponse({**base, "object": "chat.completion", "choices": [choice],
                             "usage": usage})  # fmt: skip

    async def events() -> AsyncIterator[str]:
        for i, word in enumerate(REPLY.split(" ")):
            delta = {"content": word if i == 0 else f" {word}"}
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta}]})}\n\n"
        done = {"index": 0, "delta": {}, "finish_reason": "stop"}
        yield f"data: {json.dumps({**base, 'choices': [done]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def stats(_: Request) -> Response:
    return JSONResponse({"requests": STATS["requests"], "connections": len(STATS["connections"])})


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/stats", stats, methods=["GET"]),
    ]
)
