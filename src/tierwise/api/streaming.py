"""Serve a provider's normalized chunks as an OpenAI-compatible server-sent event stream."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from tierwise.api.errors import gateway_error_from
from tierwise.providers.base import ProviderError
from tierwise.schema import StreamChunk, Usage, chunk_to_openai

logger = logging.getLogger(__name__)

OnDone = Callable[[Usage | None, int, str | None], None]
STATUS_CLIENT_CLOSED = 499
STATUS_UPSTREAM_FAILED_MID_STREAM = 502


def _event(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n"


def _error_event(message: str, type_: str, code: str) -> str:
    return _event({"error": {"message": message, "type": type_, "code": code}})


class _Outcome:
    """Reports a stream's outcome exactly once, however and wherever the stream ends."""

    def __init__(self, chunks: AsyncGenerator[StreamChunk, None], on_done: OnDone) -> None:
        self.chunks = chunks
        self.on_done = on_done
        self.reported = False

    async def finish(self, usage: Usage | None, status: int, code: str | None) -> None:
        if not self.reported:
            self.reported = True
            self.on_done(usage, status, code)  # record first: closing upstream may fail
        await self.chunks.aclose()


class _ReportingStreamingResponse(StreamingResponse):
    """A StreamingResponse that still reports when it is cancelled before its body runs.

    On ASGI spec < 2.4 Starlette races the body against a disconnect listener; if the client
    is already gone, the body generator is cancelled before its first step and its own
    ``finally`` never executes.
    """

    def __init__(self, content: AsyncIterator[str], outcome: _Outcome, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._outcome = outcome

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._outcome.finish(None, STATUS_CLIENT_CLOSED, "client_closed")


def sse_response(
    chunks: AsyncGenerator[StreamChunk, None],
    first: StreamChunk | None,
    *,
    request_id: str,
    model: str,
    headers: dict[str, str],
    include_usage: bool,
    on_done: OnDone,
) -> StreamingResponse:
    """Stream ``first`` and then the rest of ``chunks``.

    ``on_done(usage, status, error_code)`` runs exactly once: 200 when the stream completes,
    502 after a failure that could only be reported in-band, 499 when the client went away
    (which also closes the upstream request). The final usage chunk is only sent to the client
    when it asked for it (``stream_options.include_usage``), as OpenAI does, but usage is
    always captured for the log.
    """
    created = int(time.time())
    outcome = _Outcome(chunks, on_done)

    async def body() -> AsyncIterator[str]:
        usage: Usage | None = None
        status, code = 200, None

        def render(chunk: StreamChunk) -> str | None:
            nonlocal usage
            if chunk.usage is not None:
                usage = chunk.usage
                if not include_usage:
                    return None
            return _event(chunk_to_openai(chunk, id=request_id, model=model, created=created))

        try:
            pending = [first] if first is not None else []
            for chunk in pending:
                if (event := render(chunk)) is not None:
                    yield event
            async for chunk in chunks:
                if (event := render(chunk)) is not None:
                    yield event
            yield _event("[DONE]")
        except ProviderError as exc:
            # Headers are already sent, so the error can only travel inside the stream.
            err = gateway_error_from(exc)
            status, code = STATUS_UPSTREAM_FAILED_MID_STREAM, err.code
            yield _error_event(err.message, err.type, err.code)
            yield _event("[DONE]")
        except (asyncio.CancelledError, GeneratorExit):
            status, code = STATUS_CLIENT_CLOSED, "client_closed"
            raise
        except Exception:
            logger.exception("unexpected error while streaming %s", request_id)
            status, code = STATUS_UPSTREAM_FAILED_MID_STREAM, "upstream_error"
            yield _error_event("The upstream stream failed unexpectedly.", "api_error", code)
            yield _event("[DONE]")
        finally:
            await outcome.finish(usage, status, code)

    stream_headers = {**headers, "Cache-Control": "no-cache"}
    return _ReportingStreamingResponse(
        body(), outcome, media_type="text/event-stream", headers=stream_headers
    )
