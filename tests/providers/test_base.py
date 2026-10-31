import httpx
import pytest
import respx

from tierwise.providers.base import AuthError, FatalError, RetryableError, send

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
async def test_2xx_returns_json() -> None:
    respx.post(URL).respond(200, json={"ok": True})
    assert await _call() == {"ok": True}
