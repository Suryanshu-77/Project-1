"""Serve a provider's normalized chunks as an OpenAI-compatible server-sent event stream."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from fastapi.responses import StreamingResponse

from tierwise.api.errors import gateway_error_from
from tierwise.providers.base import ProviderError
from tierwise.schema import StreamChunk, Usage, chunk_to_openai

OnDone = Callable[[Usage | None, int, str | None], None]
STATUS_CLIENT_CLOSED = 499
STATUS_UPSTREAM_FAILED_MID_STREAM = 502


def _event(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def sse_response(
    chunks: AsyncGenerator[StreamChunk, None],
    first: StreamChunk | None,
    *,
    request_id: str,
    model: str,
    headers: dict[str, str],
    on_done: OnDone,
) -> StreamingResponse:
    """Stream ``first`` and then the rest of ``chunks``.

    ``on_done(usage, status, error_code)`` runs exactly once, however the stream ends: normally
    (200), after an upstream failure that could only be reported in-band (502), or because
    the client went away (499, which also closes the upstream request).
    """
    created = int(time.time())

    async def body() -> AsyncIterator[str]:
        usage: Usage | None = None
        status, code = 200, None

        def render(chunk: StreamChunk) -> str:
            nonlocal usage
            if chunk.usage is not None:
                usage = chunk.usage
            return _event(chunk_to_openai(chunk, id=request_id, model=model, created=created))

        try:
            if first is not None:
                yield render(first)
            async for chunk in chunks:
                yield render(chunk)
            yield _event("[DONE]")
        except ProviderError as exc:
            # Headers are already sent, so the error can only travel inside the stream.
            err = gateway_error_from(exc)
            status, code = STATUS_UPSTREAM_FAILED_MID_STREAM, err.code
            yield _event({"error": {"message": err.message, "type": err.type, "code": err.code}})
            yield _event("[DONE]")
        except (asyncio.CancelledError, GeneratorExit):
            status, code = STATUS_CLIENT_CLOSED, "client_closed"
            raise
        finally:
            await chunks.aclose()
            on_done(usage, status, code)

    stream_headers = {**headers, "Cache-Control": "no-cache"}
    return StreamingResponse(body(), media_type="text/event-stream", headers=stream_headers)
