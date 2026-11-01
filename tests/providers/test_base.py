import httpx
import pytest
import respx

from tierwise.providers.base import (
    AuthError,
    FatalError,
    RetryableError,
    parse_or_retry,
    send,
    stream_sse,
)

URL = "https://up.test/x"


async def _call() -> dict[str, object]:
    async with httpx.AsyncClient() as client:
        return await send(client, "up", "POST", URL, json={})


@respx.mock
async def test_429_is_retryable() -> None:
    respx.post(URL).respond(429, json={"error": "slow down"})
    with pytest.raises(RetryableError):
        await _call()


@respx.mock
async def test_503_is_retryable() -> None:
    respx.post(URL).respond(503, text="unavailable")
    with pytest.raises(RetryableError):
        await _call()


@pytest.mark.parametrize("status", [401, 403])
@respx.mock
async def test_401_and_403_are_auth(status: int) -> None:
    respx.post(URL).respond(status, json={"error": "bad key"})
    with pytest.raises(AuthError):
        await _call()


@pytest.mark.parametrize("status", [400, 404])
@respx.mock
async def test_400_and_404_are_fatal(status: int) -> None:
    respx.post(URL).respond(status, json={"error": "bad request"})
    with pytest.raises(FatalError) as info:
        await _call()
    assert info.value.status == status


@respx.mock
async def test_timeout_is_retryable() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("timed out"))
    with pytest.raises(RetryableError):
        await _call()


@respx.mock
async def test_non_json_error_body_truncated() -> None:
    respx.post(URL).respond(502, text="<html>" + "x" * 2000)
    with pytest.raises(RetryableError) as info:
        await _call()
    assert len(info.value.message) <= 500
    assert info.value.provider == "up"


@respx.mock
async def test_2xx_non_json_body_is_retryable() -> None:
    respx.post(URL).respond(200, text="<html>captive portal" + "x" * 2000)
    with pytest.raises(RetryableError) as info:
        await _call()
    assert len(info.value.message) <= 500


@respx.mock
async def test_dropped_connection_is_retryable() -> None:
    respx.post(URL).mock(side_effect=httpx.RemoteProtocolError("peer closed connection"))
    with pytest.raises(RetryableError):
        await _call()


@respx.mock
async def test_2xx_returns_json() -> None:
    respx.post(URL).respond(200, json={"ok": True})
    assert await _call() == {"ok": True}


SSE_HEADERS = {"content-type": "text/event-stream"}


async def _collect(limit: int | None = None) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    async with httpx.AsyncClient() as client:
        stream = stream_sse(client, "up", "POST", URL, json={})
        async for item in stream:
            items.append(item)
            if limit is not None and len(items) >= limit:
                await stream.aclose()
                break
    return items


@respx.mock
async def test_sse_yields_data_json_skips_done_and_events() -> None:
    body = (
        b": keep-alive comment\n\n"
        b'event: message_start\ndata: {"a": 1}\n\n'
        b'data: {"b": 2}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(URL).respond(200, content=body, headers=SSE_HEADERS)
    assert await _collect() == [{"a": 1}, {"b": 2}]


@respx.mock
async def test_sse_non_2xx_raises_classified() -> None:
    respx.post(URL).respond(429, json={"error": "slow"})
    with pytest.raises(RetryableError):
        await _collect()


@respx.mock
async def test_sse_bad_json_line_is_retryable() -> None:
    respx.post(URL).respond(200, content=b"data: {not json\n\n", headers=SSE_HEADERS)
    with pytest.raises(RetryableError):
        await _collect()


@respx.mock
async def test_sse_early_exit_closes_response() -> None:
    body = b'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
    route = respx.post(URL).respond(200, content=body, headers=SSE_HEADERS)
    assert await _collect(limit=1) == [{"a": 1}]
    assert route.calls.last.response.is_closed


def test_parse_or_retry_attribute_error_is_retryable() -> None:
    def parse(data: dict[str, object]) -> object:
        return data["choices"][0].get("delta")  # type: ignore[index]

    with pytest.raises(RetryableError):
        parse_or_retry("up", {"choices": ["x"]}, parse)


@respx.mock
async def test_decoding_error_is_retryable() -> None:
    respx.post(URL).mock(side_effect=httpx.DecodingError("bad gzip"))
    with pytest.raises(RetryableError):
        await _call()


@respx.mock
async def test_connect_error_is_connection_failure() -> None:
    from tierwise.providers.base import ConnectionFailure

    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ConnectionFailure):
        await _call()


@respx.mock
async def test_read_timeout_is_retryable_but_not_connection_failure() -> None:
    from tierwise.providers.base import ConnectionFailure

    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(RetryableError) as info:
        await _call()
    assert not isinstance(info.value, ConnectionFailure)


@respx.mock
async def test_stream_connect_error_is_connection_failure() -> None:
    from tierwise.providers.base import ConnectionFailure

    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ConnectionFailure):
        async with httpx.AsyncClient() as client:
            async for _ in stream_sse(client, "up", "POST", URL, json={}):
                pass
