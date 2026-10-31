from tests.api.helpers import gateway


async def test_lists_only_configured_providers() -> None:
    async with gateway(gemini_api_key=None, ollama_base_url=None) as (_, client):
        resp = await client.get("/v1/models")
    payload = resp.json()
    assert payload["object"] == "list"
    ids = {m["id"]: m["owned_by"] for m in payload["data"]}
    assert ids == {
        "claude-haiku-5-5": "anthropic",
        "claude-opus-5-5": "anthropic",
        "gpt-4.1-mini": "openai",
    }
    assert all(m["object"] == "model" for m in payload["data"])
