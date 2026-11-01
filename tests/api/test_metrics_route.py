import httpx

from tests.api.helpers import body, gateway, openai_chunk, sse
from tests.api.test_stream_hold import TWO


async def test_metrics_endpoint_after_requests() -> None:
    async with gateway() as (_, client):
        for _ in range(2):
            await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert 'tierwise_requests_total{model="gpt-4.1-mini",source="override",status="200"} 2.0' in (
        resp.text
    )


async def test_streamed_request_counted_once_with_status() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        return sse(openai_chunk({"content": "po"}), done=False, raw_tail=b"data: {x\n\n")

    async with gateway(broken, registry=TWO) as (app, client):
        await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        text = (await client.get("/metrics")).text
    lines = [line for line in text.splitlines() if line.startswith("tierwise_requests_total{")]
    assert len(lines) == 1 and 'status="502"' in lines[0] and lines[0].endswith(" 1.0")


async def test_unknown_model_does_not_add_label() -> None:
    async with gateway() as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("typo-xyz"))
        assert resp.status_code == 404
        text = (await client.get("/metrics")).text
    assert "typo-xyz" not in text
