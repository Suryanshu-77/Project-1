import asyncio
import re
import sqlite3
from pathlib import Path

import pytest

from tierwise.__main__ import main
from tierwise.budgets import month_of
from tierwise.keys import hash_key
from tierwise.store import RequestLog

KEY = re.compile(r"tw-[A-Za-z0-9_-]{40,}")


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli.db"
    monkeypatch.setenv("TIERWISE_DB_URL", f"sqlite+aiosqlite:///{path.as_posix()}")
    return path


def _create(capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[str, str]:
    assert main(["keys", "create", "--name", "team-a", *extra]) == 0
    out = capsys.readouterr().out
    key = KEY.search(out)
    key_id = re.search(r"id:\s*(\w+)", out)
    assert key is not None and key_id is not None, out
    return key.group(0), key_id.group(1)


def test_create_prints_key_once_and_stores_only_hash(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key, _ = _create(capsys, "--budget", "50", "--rpm", "60")
    with sqlite3.connect(db) as conn:
        stored = conn.execute("select key_hash, monthly_budget_usd, rpm from api_keys").fetchall()
        everything = " ".join(str(v) for row in conn.execute("select * from api_keys") for v in row)
    assert stored == [(hash_key(key), 50.0, 60)]
    assert key not in everything


def test_list_shows_no_secrets(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    key, _ = _create(capsys)
    assert main(["keys", "list"]) == 0
    out = capsys.readouterr().out
    assert "team-a" in out and key[:10] in out
    assert key not in out and hash_key(key) not in out


def test_revoke_then_revoke_again_exits_1(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, key_id = _create(capsys)
    assert main(["keys", "revoke", key_id]) == 0
    assert main(["keys", "revoke", key_id]) == 1
    assert main(["keys", "revoke", "nope"]) == 1
    assert main(["keys", "list"]) == 0
    assert "revoked" in capsys.readouterr().out


def test_list_shows_month_spend(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import time

    _, key_id = _create(capsys)

    async def debit() -> None:
        log = RequestLog(f"sqlite+aiosqlite:///{db.as_posix()}")
        await log.start()
        log.submit_debit(key_id, month_of(time.time()), 1.25)
        await log.stop()

    asyncio.run(debit())
    assert main(["keys", "list"]) == 0
    assert "1.25" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [["--rpm", "0"], ["--rpm", "-5"], ["--budget", "-1"]])
def test_create_rejects_invalid_limits(db: Path, bad: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["keys", "create", "--name", "x", *bad])
    assert info.value.code == 2
