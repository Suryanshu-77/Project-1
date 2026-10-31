"""Gateway settings, read from the environment (and an optional .env file)."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("TIERWISE_HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("TIERWISE_PORT", "port"))
