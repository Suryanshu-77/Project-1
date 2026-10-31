"""Shared provider plumbing: the adapter protocol, typed errors and the HTTP send helper."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import httpx

from tierwise.schema import ChatRequest, ChatResponse, StreamChunk

MAX_ERROR_CHARS = 500


class ProviderError(Exception):
    def __init__(self, provider: str, status: int | None, message: str) -> None:
        super().__init__(f"{provider} error (status={status}): {message}")
        self.provider = provider
        self.status = status
        self.message = message


class RetryableError(ProviderError):
    """Transient: worth retrying or failing over (rate limits, 5xx, timeouts)."""


class FatalError(ProviderError):
    """The request itself is bad; retrying or failing over will not help."""


class AuthError(ProviderError):
    """Our provider credentials were rejected: a gateway configuration problem."""


class Provider(Protocol):
    name: str

    async def chat(self, req: ChatRequest, model_id: str) -> ChatResponse: ...

    def stream(self, req: ChatRequest, model_id: str) -> AsyncIterator[StreamChunk]: ...


def classify_http_error(provider: str, status: int, body: str) -> ProviderError:
    """Map a non-2xx upstream response to the error type that decides what happens next."""
    message = body[:MAX_ERROR_CHARS]
    if status in (401, 403):
        return AuthError(provider, status, message)
    if status == 429 or status >= 500:
        return RetryableError(provider, status, message)
    return FatalError(provider, status, message)


async def send(
    client: httpx.AsyncClient, provider: str, method: str, url: str, **kw: Any
) -> dict[str, Any]:
    try:
        response = await client.request(method, url, **kw)
    except httpx.TransportError as exc:  # timeouts, refused or dropped connections
        raise RetryableError(provider, None, type(exc).__name__) from exc
    if not response.is_success:
        raise classify_http_error(provider, response.status_code, response.text)
    try:
        data: dict[str, Any] = response.json()
    except ValueError as exc:  # e.g. a captive-proxy HTML page served with 200
        raise RetryableError(
            provider, response.status_code, response.text[:MAX_ERROR_CHARS]
        ) from exc
    return data


def parse_or_retry[T](
    provider: str, data: dict[str, Any], parse: Callable[[dict[str, Any]], T]
) -> T:
    """Run a response parser; a 2xx body we cannot understand is treated as a transient failure."""
    try:
        return parse(data)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        message = f"malformed response: {exc!r}"[:MAX_ERROR_CHARS]
        raise RetryableError(provider, 200, message) from exc


async def stream_sse(
    client: httpx.AsyncClient, provider: str, method: str, url: str, **kw: Any
) -> AsyncIterator[dict[str, Any]]:
    """Yield the JSON payload of each ``data:`` line of a server-sent event stream.

    The upstream response is always closed, including when the consumer stops early.
    """
    try:
        async with client.stream(method, url, **kw) as response:
            if not response.is_success:
                body = (await response.aread()).decode(errors="replace")
                raise classify_http_error(provider, response.status_code, body)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank separators, event: names, : comments
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except ValueError as exc:
                    message = f"malformed stream event: {data}"[:MAX_ERROR_CHARS]
                    raise RetryableError(provider, response.status_code, message) from exc
    except httpx.TransportError as exc:
        raise RetryableError(provider, None, type(exc).__name__) from exc
