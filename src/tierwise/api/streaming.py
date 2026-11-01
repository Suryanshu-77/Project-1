"""Serve a provider's normalized chunks as an OpenAI-compatible server-sent event stream."""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass
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


@dataclass
class Primed:
    """An upstream stream, held until its first real content (see ``prime_stream``)."""

    chunks: AsyncGenerator[StreamChunk, None]
    buffered: list[StreamChunk]
    pending: "asyncio.Future[StreamChunk] | None" = None  # a read still in flight at commit

    async def close(self) -> None:
        """Close upstream; a read still in flight must be cancelled first (an async generator
        can't be closed while it is running)."""
        if self.pending is not None and not self.pending.done():
            self.pending.cancel()
            with contextlib.suppress(BaseException):
                await self.pending
        await self.chunks.aclose()


def _has_content(chunk: StreamChunk) -> bool:
    return bool(chunk.content) or bool(chunk.tool_calls)


async def prime_stream(
    chunks: AsyncGenerator[StreamChunk, None], hold_s: float = 3.0, hold_bytes: int = 4096
) -> Primed:
    """Read upstream until the first content or tool-call delta, the end of the stream, or
    the hold cap (time or buffered bytes), without sending anything to the client.

    An error raised inside this window propagates (after closing upstream), so the caller
    can still fail over invisibly. Reads use a separate task, never ``wait_for`` on the
    generator: cancelling a pending ``anext`` would close a healthy but slow stream.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + hold_s
    primed = Primed(chunks, [])
    size = 0
    try:
        while True:
            read = asyncio.ensure_future(anext(chunks))
            primed.pending = read
            done, _ = await asyncio.wait({read}, timeout=max(deadline - loop.time(), 0.0))
            if not done:
                return primed  # the hold timed out: commit with the read still in flight
            primed.pending = None
            try:
                chunk = read.result()
            except StopAsyncIteration:
                return primed
            primed.buffered.append(chunk)
            size += len(chunk.model_dump_json())
            if _has_content(chunk) or size >= hold_bytes:
                return primed
    except BaseException:
        await primed.close()
        raise


class _Outcome:
    """Reports a stream's outcome exactly once, however and wherever the stream ends."""

    def __init__(self, primed: Primed, on_done: OnDone) -> None:
        self.primed = primed
        self.on_done = on_done
        self.reported = False

    async def finish(self, usage: Usage | None, status: int, code: str | None) -> None:
        if not self.reported:
            self.reported = True
            self.on_done(usage, status, code)  # record first: closing upstream may fail
        await self.primed.close()


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
    primed: Primed,
    *,
    request_id: str,
    model: str,
    headers: dict[str, str],
    include_usage: bool,
    on_done: OnDone,
    collect: list[StreamChunk] | None = None,
) -> StreamingResponse:
    """Stream the held chunks, then the read in flight (if any), then the rest.

    ``on_done(usage, status, error_code)`` runs exactly once: 200 when the stream completes,
    502 after a failure that could only be reported in-band, 499 when the client went away
    (which also closes the upstream request). The final usage chunk is only sent to the client
    when it asked for it (``stream_options.include_usage``), as OpenAI does, but usage is
    always captured for the log.
    """
    created = int(time.time())
    outcome = _Outcome(primed, on_done)

    async def body() -> AsyncIterator[str]:
        usage: Usage | None = None
        status, code = 200, None

        def render(chunk: StreamChunk) -> str | None:
            nonlocal usage
            if collect is not None:
                collect.append(chunk)
            if chunk.usage is not None:
                usage = chunk.usage
                if not include_usage:
                    return None
            return _event(chunk_to_openai(chunk, id=request_id, model=model, created=created))

        try:
            for chunk in primed.buffered:
                if (event := render(chunk)) is not None:
                    yield event
            ended = False
            if primed.pending is not None:
                try:
                    chunk = await primed.pending
                except StopAsyncIteration:
                    ended = True
                else:
                    if (event := render(chunk)) is not None:
                        yield event
                primed.pending = None
            if not ended:
                async for chunk in primed.chunks:
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
