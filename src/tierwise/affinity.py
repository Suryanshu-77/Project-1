"""Conversation memory: which model served which prompt prefix, and when.

Providers cache the start of a prompt, so the model that served a conversation's earlier
turns can be cheaper for the next turn than a "cheaper" model whose cache is cold.
"""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from tierwise.schema import ChatRequest
from tierwise.tokens import estimate_text, text_of


def split_prefix(req: ChatRequest) -> tuple[str | None, int]:
    """Hash and estimated size of everything before the last user message.

    The prefix is the system/developer messages, the tool definitions and all earlier turns:
    the part a provider can serve from its prompt cache. An empty prefix has no hash.
    """
    last_user = max((i for i, m in enumerate(req.messages) if m.role == "user"), default=None)
    earlier = req.messages if last_user is None else req.messages[:last_user]
    system_after = [m for m in req.messages[len(earlier) :] if m.role in ("system", "developer")]
    prefix_messages = [*earlier, *system_after]
    tools = [t.model_dump(exclude_none=True) for t in req.tools or []]
    if not prefix_messages and not tools:
        return None, 0
    payload: dict[str, Any] = {
        "messages": [m.model_dump(exclude_none=True) for m in prefix_messages],
        "tools": tools,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    text = "".join(text_of(m.content) for m in prefix_messages) + json.dumps(tools)
    return hashlib.sha256(canonical.encode()).hexdigest(), estimate_text(text)


class AffinityTable:
    """In-memory LRU of prefix hash -> {model id: when it last served that prefix}."""

    def __init__(
        self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._entries: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._max_entries = max_entries
        self._clock = clock

    def record(self, prefix_hash: str, model_id: str) -> None:
        models = self._entries.pop(prefix_hash, {})
        models[model_id] = self._clock()
        self._entries[prefix_hash] = models
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def warm_models(self, prefix_hash: str | None) -> dict[str, float]:
        if prefix_hash is None:
            return {}
        return dict(self._entries.get(prefix_hash, {}))

    def is_warm(self, prefix_hash: str | None, model_id: str, ttl: int | None) -> bool:
        served_at = self.warm_models(prefix_hash).get(model_id)
        return served_at is not None and ttl is not None and self._clock() - served_at <= ttl

    def incumbent(
        self, prefix_hash: str | None, ttl_for: Callable[[str], int | None]
    ) -> str | None:
        """The most recent model whose cache for this prefix should still be warm."""
        now = self._clock()
        warm = [
            (served_at, model_id)
            for model_id, served_at in self.warm_models(prefix_hash).items()
            if (ttl := ttl_for(model_id)) is not None and now - served_at <= ttl
        ]
        return max(warm)[1] if warm else None
