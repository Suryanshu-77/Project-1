"""The model registry: every routable model, its provider, prices and capabilities.

Loaded from YAML and validated once at startup, so adding a model is a config change.
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tierwise.schema import EFFORT_ORDER as EFFORT_ORDER
from tierwise.schema import EffortLevel as EffortLevel

ProviderName = Literal["anthropic", "openai", "gemini", "ollama"]
Capability = Literal["tools", "vision", "json"]


class RegistryError(ValueError):
    """The registry file is unreadable or invalid."""


class EffortLevelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality: float  # shift from the model's quality at its default effort
    thinking_tokens: int = Field(ge=0)  # expected reasoning tokens per request


class Efforts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: EffortLevel  # what the provider does when no effort is sent
    levels: dict[EffortLevel, EffortLevelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_listed(self) -> "Efforts":
        if self.default not in self.levels:
            raise ValueError(f"default effort '{self.default}' is not one of the levels")
        return self


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
    efforts: Efforts | None = None
    effort_in_cache_key: bool = False  # True where changing effort invalidates the prompt cache

    @model_validator(mode="after")
    def _effort_quality_in_range(self) -> "ModelSpec":
        for level, spec in self.efforts.levels.items() if self.efforts else ():
            if not 0.0 <= self.quality + spec.quality <= 1.0:
                raise ValueError(f"effort '{level}' takes quality outside 0-1")
        return self

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
