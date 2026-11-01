import pytest

from tierwise.api.auth import authenticate
from tierwise.api.errors import InvalidApiKey
from tierwise.keys import KEY_PREFIX, KeyStore, generate_key, hash_key
from tierwise.store import KeyRow


def _row(key: str, revoked: float | None = None) -> KeyRow:
    return KeyRow("id1", "team", hash_key(key), key[:10], None, None, 0.0, revoked)


def test_generated_key_shape() -> None:
    a, b = generate_key(), generate_key()
    assert a.startswith(KEY_PREFIX) and len(a) >= 40 and a != b


def test_find_only_active_keys() -> None:
    active, revoked = generate_key(), generate_key()
    store = KeyStore()
    store.replace([_row(active), _row(revoked, revoked=1.0)])
    assert store.find(active) is not None
    assert store.find(revoked) is None
    assert store.find("tw-unknown") is None


KEY = generate_key()


@pytest.mark.parametrize("header", [f"Bearer {KEY}", f"bearer {KEY}", f"  Bearer   {KEY}  "])
def test_authenticate_accepts_bearer_variants(header: str) -> None:
    store = KeyStore()
    store.replace([_row(KEY)])
    assert authenticate(header, store, required=True) is not None


@pytest.mark.parametrize("header", [None, f"Basic {KEY}", "Bearer", "Bearer wrong", ""])
def test_authenticate_rejects(header: str | None) -> None:
    store = KeyStore()
    store.replace([_row(KEY)])
    with pytest.raises(InvalidApiKey):
        authenticate(header, store, required=True)
    assert authenticate(header, store, required=False) is None
