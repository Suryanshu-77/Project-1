"""Application factory: registry, provider clients, learned router and the request log."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from tierwise import __version__
from tierwise.affinity import AffinityTable
from tierwise.api.errors import install_error_handlers
from tierwise.api.routes import router
from tierwise.cache import SemanticCache
from tierwise.config import Settings
from tierwise.providers.anthropic import AnthropicProvider
from tierwise.providers.base import Provider
from tierwise.providers.gemini import GeminiProvider
from tierwise.providers.ollama import OllamaProvider
from tierwise.providers.openai import OpenAIProvider
from tierwise.registry import Registry, load_registry
from tierwise.reliability.breaker import Breakers
from tierwise.reliability.retry import RetryPolicy
from tierwise.router.learned import load_learned
from tierwise.store import RequestLog

LIMITS = httpx.Limits(max_keepalive_connections=100, max_connections=250, keepalive_expiry=30)
TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=2.0)


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    registry: Registry | None = None,
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Fail fast: an invalid registry stops startup instead of breaking requests later.
        app.state.registry = registry or load_registry(config.registry_path)

        # One pooled client per provider for the app's whole life: no per-request handshakes.
        clients: dict[str, httpx.AsyncClient] = {}
        providers: dict[str, Provider] = {}

        def client_for(name: str) -> httpx.AsyncClient:
            clients[name] = httpx.AsyncClient(limits=LIMITS, timeout=TIMEOUT, transport=transport)
            return clients[name]

        if config.anthropic_api_key:
            providers["anthropic"] = AnthropicProvider(
                client_for("anthropic"), config.anthropic_api_key
            )
        if config.openai_api_key:
            providers["openai"] = OpenAIProvider(
                client_for("openai"), config.openai_api_key, config.openai_base_url
            )
        if config.gemini_api_key:
            providers["gemini"] = GeminiProvider(client_for("gemini"), config.gemini_api_key)
        if config.ollama_base_url:
            providers["ollama"] = OllamaProvider(client_for("ollama"), config.ollama_base_url)
        app.state.clients = clients
        app.state.providers = providers
        app.state.affinity = AffinityTable()
        app.state.breakers = Breakers(
            config.breaker_failures, config.breaker_window_s, config.breaker_cooldown_s
        )
        app.state.retry = RetryPolicy(config.retries, config.retry_base_s, config.retry_max_s)
        app.state.stream_hold = (config.stream_hold_s, config.stream_hold_bytes)

        # Loading and warming the ONNX session blocks; keep it off the event loop.
        learned = await asyncio.to_thread(load_learned, config)
        app.state.learned = learned
        app.state.cache = (
            SemanticCache(
                learned.embed_text,
                config.cache_similarity,
                config.cache_ttl_s,
                config.cache_max_entries,
            )
            if learned is not None and config.cache_enabled
            else None
        )
        app.state.cache_max_temperature = config.cache_max_temperature

        log = RequestLog(config.db_url)
        await log.start()
        app.state.log = log
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()
            await log.stop()
            if learned is not None:
                learned.close()

    app = FastAPI(title="tierwise", version=__version__, lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(router)
    return app
