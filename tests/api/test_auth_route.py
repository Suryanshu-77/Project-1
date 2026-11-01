from pathlib import Path

import httpx

from tests.api.helpers import body, gateway, make_key, ok_upstream


def _counting() -> tuple[list[str], object]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return ok_upstream(request)

    return calls, handler


async def test_missing_key_is_401_when_required() -> None:
    calls, handler = _counting()
    async with gateway(handler, require_api_key=True) as (_, client):
        resp = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert calls == []


async def test_valid_key_logs_key_id(tmp_path: Path) -> None:
    async with gateway(db_dir=tmp_path, require_api_key=True) as (app, client):
        key, row = await make_key(app)
        resp = await client.post(
            "/v1/chat/completions",
            json=body("gpt-4.1-mini"),
            headers={"Authorization": f"Bearer {key}"},
        )
    assert resp.status_code == 200
    [logged] = await app.state.log.fetch_all()
    assert logged.key_id == row.id


async def test_revoked_key_rejected_after_refresh() -> None:
    async with gateway(require_api_key=True) as (app, client):
        key, row = await make_key(app)
        await app.state.log.revoke_key(row.id)
        app.state.keys.replace(await app.state.log.list_keys())
        resp = await client.post(
            "/v1/chat/completions",
            json=body("gpt-4.1-mini"),
            headers={"Authorization": f"Bearer {key}"},
        )
    assert resp.status_code == 401


async def test_models_requires_key_healthz_does_not() -> None:
    async with gateway(require_api_key=True) as (_, client):
        assert (await client.get("/v1/models")).status_code == 401
        assert (await client.get("/healthz")).status_code == 200


async def test_auth_off_by_default_still_tracks_valid_key(tmp_path: Path) -> None:
    async with gateway(db_dir=tmp_path) as (app, client):
        key, row = await make_key(app)
        anonymous = await client.post("/v1/chat/completions", json=body("gpt-4.1-mini"))
        keyed = await client.post(
            "/v1/chat/completions",
            json=body("gpt-4.1-mini"),
            headers={"Authorization": f"Bearer {key}"},
        )
    assert anonymous.status_code == keyed.status_code == 200
    ids = sorted((r.key_id or "") for r in await app.state.log.fetch_all())
    assert ids == ["", row.id]
