import hashlib
import io
from pathlib import Path
from typing import BinaryIO

import pytest

from tierwise import fetch
from tierwise.__main__ import main
from tierwise.fetch import FetchError, fetch_model

GOOD = hashlib.sha256(b"abc").hexdigest()


def _opener(data: bytes = b"abc"):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def open_(url: str) -> BinaryIO:
        calls.append(url)
        return io.BytesIO(data)

    return open_, calls


def test_writes_file_when_hash_matches(tmp_path: Path) -> None:
    opener, _ = _opener()
    path = fetch_model(tmp_path, sha256=GOOD, opener=opener)
    assert path == tmp_path / "model.onnx"
    assert path.read_bytes() == b"abc"


def test_hash_mismatch_leaves_nothing(tmp_path: Path) -> None:
    opener, _ = _opener(b"tampered")
    with pytest.raises(FetchError, match="sha256"):
        fetch_model(tmp_path, sha256=GOOD, opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_existing_good_file_is_not_redownloaded(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"abc")
    opener, calls = _opener()
    assert fetch_model(tmp_path, sha256=GOOD, opener=opener) == tmp_path / "model.onnx"
    assert calls == []


def test_stream_error_leaves_nothing(tmp_path: Path) -> None:
    class Broken(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise OSError("connection reset")

    with pytest.raises(FetchError, match="connection reset"):
        fetch_model(tmp_path, sha256=GOOD, opener=lambda url: Broken())
    assert list(tmp_path.iterdir()) == []


def test_cli_fetch_model_prints_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(fetch, "fetch_model", lambda out: seen.append(out) or out / "model.onnx")
    assert main(["fetch-model", "--out", str(tmp_path)]) == 0
    assert seen == [tmp_path]
    assert str(tmp_path / "model.onnx") in capsys.readouterr().out


def test_oversized_download_stops_and_leaves_nothing(tmp_path: Path) -> None:
    read: list[int] = []

    class Endless(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            read.append(size)
            return b"x" * 1024  # a broken or hostile mirror that never stops

    with pytest.raises(FetchError, match="larger than"):
        fetch_model(tmp_path, sha256=GOOD, opener=lambda url: Endless(), max_bytes=10_000)
    assert list(tmp_path.iterdir()) == []
    assert len(read) < 20  # stopped soon after the cap, not at the end of the stream
