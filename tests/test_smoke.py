from importlib.metadata import version

import pytest

import tierwise
from tierwise.__main__ import main


def test_version_matches_the_installed_package() -> None:
    assert tierwise.__version__ == version("tierwise")


def test_cli_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.strip() == f"tierwise {version('tierwise')}"
