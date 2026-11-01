from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tierwise.registry import ModelSpec, RegistryError, load_registry

ENTRY = """
  - id: {id}
    provider: anthropic
    price_in: 1.0
    price_out: 2.0
    context: 1000
    caps: [tools]
    quality: {quality}
    tier: basic
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _registry(baseline: str = "a", entries: str | None = None) -> str:
    models = entries if entries is not None else ENTRY.format(id="a", quality=0.5)
    return f"baseline_model: {baseline}\nmodels:{models}"


def test_loads_shipped_registry() -> None:
    registry = load_registry("config/registry.yaml")
    providers = {model.provider for model in registry.models}
    assert providers == {"anthropic", "openai", "gemini", "ollama"}
    assert registry.baseline().tier == "flagship"


def test_duplicate_id_rejected(tmp_path: Path) -> None:
    entries = ENTRY.format(id="a", quality=0.5) + ENTRY.format(id="a", quality=0.6)
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, _registry(entries=entries)))


def test_unknown_baseline_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, _registry(baseline="missing")))


def test_unknown_field_rejected(tmp_path: Path) -> None:
    entries = ENTRY.format(id="a", quality=0.5) + "    colour: blue\n"
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, _registry(entries=entries)))


def test_quality_out_of_range_rejected(tmp_path: Path) -> None:
    entries = ENTRY.format(id="a", quality=1.5)
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, _registry(entries=entries)))


def test_unreadable_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        load_registry(_write(tmp_path, "models: [unclosed"))


def test_wire_id(tmp_path: Path) -> None:
    entries = ENTRY.format(id="a", quality=0.5) + ENTRY.format(id="b", quality=0.5)
    entries += "    upstream_id: llama3.2\n"
    registry = load_registry(_write(tmp_path, _registry(entries=entries)))
    a, b = registry.get("a"), registry.get("b")
    assert a is not None and a.wire_id == "a"
    assert b is not None and b.wire_id == "llama3.2"
    assert registry.get("nope") is None


def test_shipped_registry_cache_ttls_match_real_caching() -> None:
    # OpenAI and Gemini cache prompt prefixes automatically. Anthropic only caches when the
    # request sets cache_control, which the gateway doesn't do yet, so it must not look warm.
    for model in load_registry("config/registry.yaml").models:
        if model.provider in ("openai", "gemini"):
            assert model.cache_ttl_s, f"{model.id} caches automatically but has no TTL"
        else:
            assert model.cache_ttl_s is None, f"{model.id} would be treated as warm"


def _spec(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "id": "m", "provider": "anthropic", "price_in": 1.0, "price_out": 2.0,
        "context": 1000, "caps": [], "quality": 0.5, "tier": "basic",
    }  # fmt: skip
    fields.update(overrides)
    return fields


def _levels(**levels: float) -> dict[str, Any]:
    return {name: {"quality": q, "thinking_tokens": 100} for name, q in levels.items()}


def test_efforts_default_must_be_listed() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_spec(efforts={"default": "high", "levels": _levels(low=-0.05)}))


def test_efforts_unknown_level_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(_spec(efforts={"default": "ultra", "levels": _levels(ultra=0.0)}))


def test_effective_quality_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            _spec(quality=0.95, efforts={"default": "low", "levels": _levels(low=0.0, high=0.10)})
        )


def test_model_without_efforts_is_valid() -> None:
    spec = ModelSpec.model_validate(_spec())
    assert spec.efforts is None
    assert spec.effort_in_cache_key is False


def test_shipped_registry_efforts() -> None:
    registry = load_registry("config/registry.yaml")
    for model in registry.models:
        if model.provider == "anthropic":
            assert model.effort_in_cache_key
            assert model.efforts is not None
            assert model.efforts.default in model.efforts.levels
        if model.provider == "ollama" or model.id == "gpt-4.1-nano":
            assert model.efforts is None
