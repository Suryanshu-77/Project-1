"""The model registry: every routable model, its provider, prices and capabilities.

Loaded from YAML and validated once at startup, so adding a model is a config change.
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ProviderName = Literal["anthropic", "openai", "gemini", "ollama"]
Capability = Literal["tools", "vision", "json"]


class RegistryError(ValueError):
    """The registry file is unreadable or invalid."""


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    provider: ProviderName
    upstream_id: str | None = None
    price_in: float = Field(ge=0)  # USD per 1M tokens
    price_out: float = Field(ge=0)
    price_cache_read: float | None = Field(default=None, ge=0)  # None: priced as price_in
    price_cache_write: float | None = Field(default=None, ge=0)
    cache_ttl_s: int | None = None
    context: int = Field(gt=0)
    caps: set[Capability]
    quality: float = Field(ge=0, le=1)
    tier: Literal["basic", "moderate", "flagship"]

    @property
    def wire_id(self) -> str:
        """The model name sent to the provider."""
        return self.upstream_id or self.id


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_model: str
    models: list[ModelSpec]

    @model_validator(mode="after")
    def _check(self) -> "Registry":
        duplicates = [mid for mid, n in Counter(m.id for m in self.models).items() if n > 1]
        if duplicates:
            raise ValueError(f"duplicate model ids: {duplicates}")
        if self.get(self.baseline_model) is None:
            raise ValueError(f"baseline_model '{self.baseline_model}' is not a listed model")
        return self

    def get(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.id == model_id), None)

    def baseline(self) -> ModelSpec:
        spec = self.get(self.baseline_model)
        assert spec is not None  # guaranteed by _check
        return spec


def load_registry(path: str | Path) -> Registry:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return Registry.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise RegistryError(f"invalid registry {path}: {exc}") from exc
