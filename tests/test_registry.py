from pathlib import Path

import pytest

from tierwise.registry import RegistryError, load_registry

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
