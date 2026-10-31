"""Local models via Ollama's OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)."""

import httpx

from tierwise.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    name = "ollama"
    include_usage = False  # older Ollama versions reject stream_options

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        super().__init__(client, api_key="", base_url=base_url)

    def _headers(self) -> dict[str, str]:
        return {}  # no authentication for a local server
