"""Application factory and lifespan-managed provider clients."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from tierwise import __version__
from tierwise.api.errors import install_error_handlers
from tierwise.api.routes import router
from tierwise.config import Settings
from tierwise.providers.anthropic import AnthropicProvider
from tierwise.providers.base import Provider
from tierwise.providers.openai import OpenAIProvider

LIMITS = httpx.Limits(max_keepalive_connections=100, max_connections=250, keepalive_expiry=30)
TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=2.0)


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
        app.state.clients = clients
        app.state.providers = providers
        try:
            yield
        finally:
            for client in clients.values():
                await client.aclose()

    app = FastAPI(title="tierwise", version=__version__, lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(router)
    return app
