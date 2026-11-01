"""Gateway settings, read from the environment (and an optional .env file)."""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str | None = None
    ollama_base_url: str | None = None  # e.g. http://localhost:11434/v1; unset disables Ollama
    registry_path: str | None = Field(  # None: the repo's config/, else the packaged copy
        default=None,
        validation_alias=AliasChoices("TIERWISE_REGISTRY", "registry_path"),
    )
    db_url: str = Field(
        default="sqlite+aiosqlite:///tierwise.db",
        validation_alias=AliasChoices("TIERWISE_DB_URL", "db_url"),
    )
    router_dir: str | None = Field(  # None: the repo's models/router, else the packaged copy
        default=None,
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
    cache_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("TIERWISE_CACHE", "cache_enabled")
    )
    cache_similarity: float = 0.95
    cache_ttl_s: float = 3600
    cache_max_entries: int = 10_000
    cache_max_temperature: float = 0.3
    require_api_key: bool = Field(
        default=False,
        validation_alias=AliasChoices("TIERWISE_REQUIRE_KEY", "require_api_key"),
    )
    keys_refresh_s: float = 30.0
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "otel_exporter_otlp_endpoint"),
    )
    store_payloads: bool = Field(  # the flywheel's first opt-in: keep prompts and answers
        default=False,
        validation_alias=AliasChoices("TIERWISE_STORE_PAYLOADS", "store_payloads"),
    )
    shadow_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("TIERWISE_SHADOW_RATE", "shadow_rate"),
    )
    shadow_budget_usd: float = Field(
        default=5.0,
        ge=0.0,
        validation_alias=AliasChoices("TIERWISE_SHADOW_BUDGET_USD", "shadow_budget_usd"),
    )
    judge_model: str | None = Field(  # None: the registry's baseline_model
        default=None, validation_alias=AliasChoices("TIERWISE_JUDGE_MODEL", "judge_model")
    )
    host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("TIERWISE_HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("TIERWISE_PORT", "port"))


CACHE_ROUTER_DIR = Path.home() / ".cache" / "tierwise" / "router"  # where fetch-model writes


def packaged_data() -> Traversable:
    """The registry and router artifacts shipped inside the wheel."""
    return files("tierwise") / "data"


def _resolve(explicit: str | None, repo_default: str, packaged: str) -> Path:
    if explicit is not None:
        return Path(explicit)  # used verbatim: a missing explicit path is an error, not a fallback
    if Path(repo_default).exists():
        return Path(repo_default)
    return Path(str(packaged_data() / packaged))


def registry_file(settings: Settings) -> Path:
    return _resolve(settings.registry_path, "config/registry.yaml", "registry.yaml")


def router_directory(settings: Settings) -> Path:
    return _resolve(settings.router_dir, "models/router", "router")
