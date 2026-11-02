"""A deterministic OpenAI-compatible model server for the compose stack and its smoke check.

It answers ``POST /v1/chat/completions`` (JSON or SSE) with a short canned reply whose usage
scales with the prompt, so the dashboard shows real-looking traffic without any API keys.
``GET /stats`` reports how many requests and distinct client connections it has seen, which
the smoke check uses to confirm the gateway reuses pooled connections.

Two knobs make the dashboard demo look like real traffic; both are off by default:
``FAKE_DELAY_S`` adds latency that grows with model size (small 1x, medium 2.5x, large 5x,
+-30%), and ``FAKE_FAIL_RATE`` answers that share of requests with a 503, which the gateway
retries and fails over. Judge calls never fail.

Run: uvicorn --app-dir scripts fake_upstream:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import os
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

STATS: dict[str, Any] = {"requests": 0, "connections": set()}
REPLY = "This is a canned answer from the fake upstream ({model})."
JUDGE_MARKER = "tierwise-judge-v2"  # must match tierwise.flywheel.JUDGE_MARKER
HARD = ("design", "prove", "debug", "production", "architecture")
DELAY_S = float(os.environ.get("FAKE_DELAY_S", "0"))
FAIL_RATE = float(os.environ.get("FAKE_FAIL_RATE", "0"))
SIZE = {"small": 1.0, "medium": 2.5, "large": 5.0}


def _judge(messages: list[dict[str, Any]]) -> str | None:
    """A deterministic fake judge: on hard-looking prompts the small model's answer scores 2,
    every other answer 5, so flywheel labels carry a signal a retrained router can learn."""
    first = messages[0] if messages else {}
    if first.get("role") != "system" or JUDGE_MARKER not in str(first.get("content")):
        return None
    # from the end: the conversation itself may quote "Answer A:" (judge_messages' layout)
    rest, b = str(messages[-1].get("content")).rsplit("\n\nAnswer B:\n", 1)
    conversation, a = rest.rsplit("\n\nAnswer A:\n", 1)
    hard = any(word in conversation.lower() for word in HARD)
    return json.dumps({key: 2 if hard and "(small)" in text else 5
                       for key, text in (("a", a), ("b", b))})  # fmt: skip


def _usage(messages: list[dict[str, Any]]) -> dict[str, int]:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return {"prompt_tokens": max(1, chars // 4), "completion_tokens": len(REPLY) // 4}


async def chat(request: Request) -> Response:
    STATS["requests"] += 1
    if request.client is not None:
        STATS["connections"].add((request.client.host, request.client.port))
    body = await request.json()
    model, usage = body.get("model", "fake"), _usage(body.get("messages", []))
    judged = _judge(body.get("messages", []))
    if judged is None and random.random() < FAIL_RATE:
        return JSONResponse({"error": {"message": "fake overload", "type": "server_error"}},
                            status_code=503)  # fmt: skip
    if DELAY_S:
        await asyncio.sleep(DELAY_S * SIZE.get(model, 1.0) * random.uniform(0.7, 1.3))
    base = {"id": "fake-1", "model": model, "created": int(time.time())}
    if not body.get("stream"):
        message = {"role": "assistant", "content": judged or REPLY.format(model=model)}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return JSONResponse({**base, "object": "chat.completion", "choices": [choice],
                             "usage": usage})  # fmt: skip

    async def events() -> AsyncIterator[str]:
        for i, word in enumerate(REPLY.format(model=model).split(" ")):
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
