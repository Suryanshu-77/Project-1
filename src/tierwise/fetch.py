"""Download the learned router's encoder (`tierwise fetch-model`).

The 90 MB MiniLM ONNX file is too big to ship in the wheel, so it is fetched once from Hugging
Face at a pinned commit and checked against its sha256 before it replaces anything.
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

MODEL_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx"
)
MODEL_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
MODEL_SIZE = 90_405_214  # bytes; a mirror sending more is stopped before it fills the disk
CHUNK = 1 << 20


class FetchError(Exception):
    pass


def _open(url: str) -> BinaryIO:
    return cast(BinaryIO, urllib.request.urlopen(url, timeout=60))  # noqa: S310 (pinned https URL)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch_model(
    out_dir: Path,
    url: str = MODEL_URL,
    sha256: str = MODEL_SHA256,
    opener: Callable[[str], BinaryIO] = _open,
    max_bytes: int = MODEL_SIZE,
) -> Path:
    """Write `out_dir/model.onnx` atomically; a bad hash or a broken download leaves nothing."""
    return download_file(url, out_dir / "model.onnx", sha256, max_bytes, opener)


def download_file(
    url: str,
    target: Path,
    sha256: str,
    max_bytes: int,
    opener: Callable[[str], BinaryIO] = _open,
) -> Path:
    """Download a pinned file to `target` atomically, unless it is already there with this
    sha256; a bad hash, an oversized or a broken download leaves nothing."""
    if target.exists() and _sha256(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}-", suffix=".part",
                                      delete=False)  # fmt: skip
    try:
        digest, written, reported = hashlib.sha256(), 0, 0
        with tmp, opener(url) as source:
            while block := source.read(CHUNK):
                tmp.write(block)
                digest.update(block)
                written += len(block)
                if written > max_bytes:
                    raise FetchError(f"download of {url} is larger than {max_bytes} bytes")
                if written - reported >= 10 * CHUNK:
                    print(f"  {written / CHUNK:.0f} MB", file=sys.stderr)
                    reported = written
        if digest.hexdigest() != sha256:
            raise FetchError(f"sha256 mismatch for {url}: got {digest.hexdigest()}")
        os.replace(tmp.name, target)
        return target
    except OSError as exc:
        raise FetchError(f"download of {url} failed: {exc}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)
