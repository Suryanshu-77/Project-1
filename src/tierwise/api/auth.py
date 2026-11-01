"""Bearer-token authentication against the gateway's API keys."""

from tierwise.api.errors import InvalidApiKey
from tierwise.keys import KeyStore
from tierwise.store import KeyRow


def authenticate(header: str | None, store: KeyStore, required: bool) -> KeyRow | None:
    """The key named by ``Authorization: Bearer <key>``.

    When keys are required, a missing, malformed, unknown or revoked key is a 401. When they
    aren't, a valid key is still resolved so its spend is tracked, and anything else is
    anonymous (None).
    """
    parts = (header or "").split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        row = store.find(parts[1])
        if row is not None:
            return row
    if required:
        raise InvalidApiKey()
    return None
