import httpx
import pytest

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


import asyncio  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from tierwise.api.streaming import Primed, sse_response  # noqa: E402
from tierwise.schema import StreamChunk  # noqa: E402


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def _chat(client: httpx.AsyncClient, key: str) -> httpx.Response:
    return await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"), headers=_auth(key))


async def test_over_budget_is_429_before_upstream() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return ok_upstream(request)

    async with gateway(handler, require_api_key=True) as (app, client):
        key, _ = await make_key(app, budget=0.0)
        resp = await _chat(client, key)
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "budget_exceeded"
    assert calls == []


async def test_success_debits_actual_cost(tmp_path: Path) -> None:
    async with gateway(db_dir=tmp_path, require_api_key=True) as (app, client):
        key, row = await make_key(app, budget=10.0)
        resp = await _chat(client, key)
        assert resp.status_code == 200
        spent = app.state.budgets.spent(row.id)
    [logged] = await app.state.log.fetch_all()
    assert spent == pytest.approx(logged.cost_usd) and spent > 0
    [(key_id, _, usage_spent, requests)] = await app.state.log.fetch_usage()
    assert (key_id, requests) == (row.id, 1) and usage_spent == pytest.approx(spent)


async def test_failed_request_is_not_debited() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    async with gateway(down, require_api_key=True) as (app, client):
        key, row = await make_key(app, budget=10.0)
        resp = await _chat(client, key)
        assert resp.status_code == 400
        assert app.state.budgets.spent(row.id) == 0.0
        assert app.state.budgets.reserved(row.id) == 0.0


async def test_concurrent_requests_cannot_overshoot() -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return ok_upstream(request)

    async with gateway(slow, require_api_key=True) as (app, client):
        from tierwise.budgets import estimate_for
        from tierwise.schema import ChatRequest

        request = ChatRequest.model_validate(body("gpt-4.1-mini"))
        estimate = estimate_for(app.state.registry.get("gpt-4.1-mini"), request)
        key, _ = await make_key(app, budget=estimate * 3.5)
        responses = await asyncio.gather(*(
            client.post("/v1/chat/completions", json=body("gpt-4.1-mini"), headers=_auth(key))
            for _ in range(20)
        ))  # fmt: skip
    codes = [r.status_code for r in responses]
    assert codes.count(200) == 3 and codes.count(429) == 17


async def test_abandoned_stream_releases_reservation() -> None:
    async with gateway(require_api_key=True) as (app, client):
        key, row = await make_key(app, budget=10.0)
        budgets = app.state.budgets
        keyrow = app.state.keys.find(key)
        reservation = budgets.reserve(keyrow, 0.5)

        async def upstream():  # type: ignore[no-untyped-def]
            yield StreamChunk(content="a")
            yield StreamChunk(content="b")

        def on_done(usage, status, code):  # type: ignore[no-untyped-def]
            budgets.settle(reservation, keyrow, None if usage is None else 1.0)

        response = sse_response(
            Primed(upstream(), []), request_id="r", model="m", headers={},
            include_usage=False, on_done=on_done,
        )  # fmt: skip
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()  # type: ignore[attr-defined]
        assert budgets.reserved(row.id) == 0.0 and budgets.spent(row.id) == 0.0


async def test_stream_failing_after_content_still_debits_an_estimate(tmp_path: Path) -> None:
    from tests.api.helpers import openai_chunk, sse
    from tests.api.test_stream_hold import TWO

    def broken(request: httpx.Request) -> httpx.Response:  # content, then no usage chunk
        return sse(openai_chunk({"content": "x" * 400}), done=False, raw_tail=b"data: {x\n\n")

    async with gateway(broken, registry=TWO, db_dir=tmp_path, require_api_key=True) as (
        app, client,
    ):  # fmt: skip
        key, row = await make_key(app, budget=10.0)
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth(key),
        )
        assert '"error"' in resp.text  # failed in-band after content
        spent = app.state.budgets.spent(row.id)
        assert app.state.budgets.reserved(row.id) == 0.0
    assert spent > 0  # the tokens were generated and billed upstream


async def _cancel_once(app: Any, client: httpx.AsyncClient, key: str, ready: Any) -> None:
    """Start a request, wait until `ready()` holds, then cancel it like a client disconnect."""
    task = asyncio.create_task(_chat(client, key))
    for _ in range(200):
        if ready():
            break
        await asyncio.sleep(0.01)
    assert ready(), "the request never reached the point under test"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelled_request_releases_reservation(tmp_path: Path) -> None:
    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()  # never answers: the client gives up first
        raise AssertionError("unreachable")

    async with gateway(hang, db_dir=tmp_path, require_api_key=True) as (app, client):
        key, row = await make_key(app, budget=10.0)
        budgets = app.state.budgets
        await _cancel_once(app, client, key, lambda: budgets.reserved(row.id) > 0)
        assert budgets.reserved(row.id) == 0.0 and budgets.spent(row.id) == 0.0
    [logged] = await app.state.log.fetch_all()
    assert logged.status == 499


async def test_cancel_before_reservation_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    from tierwise.api import routes

    entered: list[bool] = []

    async def stuck_route(*args: Any, **kwargs: Any) -> Any:
        entered.append(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(routes, "route", stuck_route)
    async with gateway(require_api_key=True) as (app, client):
        key, row = await make_key(app, budget=10.0)
        await _cancel_once(app, client, key, lambda: bool(entered))
        assert app.state.budgets.reserved(row.id) == 0.0
