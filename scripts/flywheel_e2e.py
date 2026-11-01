"""The M9 exit check: sample → label → retrain → gate, end to end, at $0.

Runs in-process against the fake upstream (its fake judge scores hard-looking prompts 2 and easy
ones 5). Needs the encoder: `tierwise fetch-model` (CI) or a local models/router/model.onnx.
Usage: uv run --group retrain python scripts/flywheel_e2e.py
"""

import asyncio
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))  # training/ and evaluation/ live at the repo root

import fake_upstream  # noqa: E402

from tierwise.api.app import build_providers, create_app  # noqa: E402
from tierwise.config import Settings  # noqa: E402
from tierwise.flywheel import label_pending  # noqa: E402
from tierwise.keys import generate_key, hash_key  # noqa: E402
from tierwise.registry import load_registry  # noqa: E402
from tierwise.store import RequestLog  # noqa: E402
from training import retrain  # noqa: E402

REGISTRY = load_registry(ROOT / "deploy" / "registry.compose.yaml")
ARTIFACTS = ("classifier.npz", "router.json", "metrics.json", "tokenizer.json", "NOTICE")
THINGS = ["billing", "chat", "search", "payments", "logging", "auth", "video", "inventory",
          "booking", "analytics", "email", "storage", "maps", "music", "news"]  # fmt: skip
HARD = ["Design a multi-region {} architecture and compare the trade-offs step by step.",
        "Debug this production outage in our {} service: latency spiked after a migration.",
        "Prove that our {} scheduler can never deadlock under concurrent writers.",
        "Design the database schema and caching architecture for a {} platform."]  # fmt: skip
EASY = ["Translate 'good morning' into Spanish for the {} page.",
        "What is a one-word name for a {} app?",
        "Say hi to the {} team.",
        "Give me a synonym for 'fast' to use in the {} docs."]  # fmt: skip


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_fake_upstream() -> int:
    port = _free_port()
    config = uvicorn.Config(fake_upstream.app, host="127.0.0.1", port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True).start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/stats", timeout=1).raise_for_status()
            return port
        except httpx.HTTPError:
            time.sleep(0.1)
    fail("fake upstream did not start")
    return port


def router_copy(target: Path) -> Path:
    target.mkdir()
    source = ROOT / "models" / "router"
    for name in ARTIFACTS:
        shutil.copy(source / name, target / name)
    if (source / "model.onnx").exists():  # local checkout; in CI fetch-model's cache is used
        shutil.copy(source / "model.onnx", target / "model.onnx")
    return target


async def send_traffic(settings: Settings) -> int:
    app = create_app(settings, registry=REGISTRY)
    sent = 0
    async with app.router.lifespan_context(app):
        if app.state.learned is None:
            fail("learned router not loaded: run `tierwise fetch-model` first")
        key = generate_key()
        await app.state.log.create_key("e2e", hash_key(key), key[:10], None, None, True)
        app.state.keys.replace(await app.state.log.list_keys())
        transport = httpx.ASGITransport(app=app)
        auth = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            for templates in (HARD, EASY):
                for template in templates:
                    for thing in THINGS:
                        payload = {"model": "auto", "messages": [
                            {"role": "user", "content": template.format(thing)}]}  # fmt: skip
                        resp = await client.post("/v1/chat/completions", json=payload,
                                                 headers=auth)  # fmt: skip
                        if resp.status_code != 200:
                            fail(f"request failed: {resp.status_code} {resp.text}")
                        sent += 1
    return sent


async def label(settings: Settings) -> None:
    clients: list[httpx.AsyncClient] = []

    def client_for(_: str) -> httpx.AsyncClient:
        clients.append(httpx.AsyncClient(timeout=30))
        return clients[-1]

    log = RequestLog(settings.db_url)
    await log.start()
    try:
        judge = REGISTRY.get(REGISTRY.baseline_model)
        assert judge is not None
        run = await label_pending(log, REGISTRY, build_providers(settings, client_for), judge,
                                  budget_usd=100.0, limit=1000)  # fmt: skip
    finally:
        await log.stop()
        for client in clients:
            await client.aclose()
    print(f"ok   labeled {run.labeled}, failed {run.failed}")
    if run.labeled < 100 or run.failed:
        fail("expected every sample to be labeled")


async def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="tierwise-e2e-"))
    port = start_fake_upstream()
    router = router_copy(work / "router")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_url=f"sqlite+aiosqlite:///{(work / 'e2e.db').as_posix()}",
        router_dir=str(router), registry_path=str(ROOT / "deploy" / "registry.compose.yaml"),
        ollama_base_url=f"http://127.0.0.1:{port}/v1", require_api_key=True,
        store_payloads=True, shadow_rate=1.0, retries=0,
    )  # fmt: skip
    print(f"ok   sent {await send_traffic(settings)} requests")
    await label(settings)

    for attempt in (1, 2):  # the second run has no new labels: it must still decide cleanly
        before = (router / "router.json").read_bytes()
        report = await retrain.run(settings.db_url, router, REGISTRY, promote=True,
                                   data_dir=work / "no-base-data")  # fmt: skip
        print(f"retrain #{attempt}:", json.dumps(asdict(report), indent=2))
        changed = (router / "router.json").read_bytes() != before
        if not report.reason:
            fail("the gate gave no reason")
        if changed != report.promoted:
            fail(f"router.json changed={changed} but promoted={report.promoted}")
        if not (router / "versions" / report.version / "classifier.npz").exists():
            fail("the challenger was not kept in versions/")
        time.sleep(1.1)  # versions are per second
    print("M9 exit check passed: sampled, labeled, retrained and gated end to end.")


if __name__ == "__main__":
    asyncio.run(main())
