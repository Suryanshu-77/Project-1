"""Gateway settings, read from the environment (and an optional .env file)."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str | None = None
    ollama_base_url: str | None = None  # e.g. http://localhost:11434/v1; unset disables Ollama
    registry_path: str = Field(
        default="config/registry.yaml",
        validation_alias=AliasChoices("TIERWISE_REGISTRY", "registry_path"),
    )
    db_url: str = Field(
        default="sqlite+aiosqlite:///tierwise.db",
        validation_alias=AliasChoices("TIERWISE_DB_URL", "db_url"),
    )
    router_dir: str = Field(
        default="models/router",
        validation_alias=AliasChoices("TIERWISE_ROUTER_DIR", "router_dir"),
    )
    router_threads: int = 2
    router_provider: str = "CPUExecutionProvider"
    learned_router: bool = Field(
        default=True, validation_alias=AliasChoices("TIERWISE_LEARNED_ROUTER", "learned_router")
    )
    retries: int = 2
    retry_base_s: float = 0.25
    retry_max_s: float = 2.0
    breaker_failures: int = 5
    breaker_window_s: float = 60.0
    breaker_cooldown_s: float = 30.0
    stream_hold_s: float = 3.0
    stream_hold_bytes: int = 4096
    host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("TIERWISE_HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("TIERWISE_PORT", "port"))
