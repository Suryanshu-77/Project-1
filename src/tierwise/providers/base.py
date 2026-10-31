"""Shared provider plumbing: the adapter protocol, typed errors and the HTTP send helper."""

from collections.abc import Callable
from typing import Any, Protocol

import httpx

from tierwise.schema import ChatRequest, ChatResponse

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
