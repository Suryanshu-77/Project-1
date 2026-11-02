from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import AliasChoices

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


def test_import_survives_missing_home(tmp_path: Path) -> None:
    """Containers under an arbitrary uid have no passwd entry and often no HOME."""
    import subprocess
    import sys
    import tempfile

    code = """
from pathlib import Path

def no_home(cls=None):
    raise RuntimeError("Could not determine home directory.")

Path.home = classmethod(no_home)
import tierwise.config as config
print(config.CACHE_ROUTER_DIR)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().startswith(tempfile.gettempdir())


def test_env_example_documents_every_setting() -> None:
    documented = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        env = alias.choices[0] if isinstance(alias, AliasChoices) else name.upper()
        assert f"{env}=" in documented, f"{env} is missing from .env.example"
