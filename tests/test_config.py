from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tierwise import config
from tierwise.api.app import create_app
from tierwise.config import Settings, registry_file, router_directory
from tierwise.registry import RegistryError


def _settings(**kw: str) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def test_explicit_registry_is_used_verbatim() -> None:
    assert registry_file(_settings(registry_path="x.yaml")) == Path("x.yaml")
    assert router_directory(_settings(router_dir="r")) == Path("r")


def test_repo_default_wins_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "registry.yaml").write_text("models: []")
    (tmp_path / "models" / "router").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert registry_file(_settings()) == Path("config/registry.yaml")
    assert router_directory(_settings()) == Path("models/router")


def test_packaged_fallback_when_repo_default_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "packaged_data", lambda: tmp_path / "pkg")
    assert registry_file(_settings()) == tmp_path / "pkg" / "registry.yaml"
    assert router_directory(_settings()) == tmp_path / "pkg" / "router"


def test_missing_explicit_registry_fails_startup(tmp_path: Path) -> None:
    app = create_app(_settings(registry_path=str(tmp_path / "nope.yaml")))
    with pytest.raises(RegistryError, match="nope.yaml"), TestClient(app):
        pass
