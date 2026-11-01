import httpx

from tests.api.helpers import body, gateway, make_key, ok_upstream


async def test_rate_limited_is_429_with_retry_after() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return ok_upstream(request)

    async with gateway(handler, require_api_key=True) as (app, client):
        key, _ = await make_key(app, rpm=1)
        auth = {"Authorization": f"Bearer {key}"}
        first = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"), headers=auth)
        second = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"), headers=auth)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"
    assert int(second.headers["Retry-After"]) >= 1
    assert len(calls) == 1
