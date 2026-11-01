"""Conversation memory: which model served which prompt prefix, and when.

Providers cache the start of a prompt, so the model that served a conversation's earlier
turns can be cheaper for the next turn than a "cheaper" model whose cache is cold.
"""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from tierwise.schema import ChatRequest
from tierwise.tokens import estimate_text, text_of


def _items(req: ChatRequest) -> list[tuple[str, int]]:
    """The prompt as cache-able items, in the order providers see them: tools, then messages."""
    items: list[tuple[str, int]] = []
    if req.tools:
        tools = [t.model_dump(exclude_none=True) for t in req.tools]
        items.append((_canonical({"tools": tools}), estimate_text(json.dumps(tools))))
    for m in req.messages:
        items.append(
            (_canonical(m.model_dump(exclude_none=True)), estimate_text(text_of(m.content)))
        )
    return items


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prefix_boundaries(req: ChatRequest) -> list[tuple[str, int]]:
    """(rolling hash, cumulative estimated tokens) after each item of the prompt.

    Providers cache the *beginning* of a prompt, so turn N+1 of a conversation can reuse the
    cache written for turn N: the whole turn-N prompt is one of turn N+1's boundaries.
    The last entry covers the full prompt.
    """
    boundaries: list[tuple[str, int]] = []
    digest, tokens = b"", 0
    for canonical, item_tokens in _items(req):
        digest = hashlib.sha256(digest + canonical.encode()).digest()
        tokens += item_tokens
        boundaries.append((digest.hex(), tokens))
    return boundaries


def split_prefix(req: ChatRequest) -> tuple[str | None, int]:
    """Hash and size of everything before the last user message (the conversation so far)."""
    last_user = max((i for i, m in enumerate(req.messages) if m.role == "user"), default=None)
    count = (1 if req.tools else 0) + (len(req.messages) if last_user is None else last_user)
    if count == 0:
        return None, 0
    return prefix_boundaries(req)[count - 1]


class AffinityTable:
    """In-memory LRU of prefix hash -> {(model id, effort sent): when it last served it}."""

    def __init__(
        self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._entries: OrderedDict[str, dict[tuple[str, str | None], float]] = OrderedDict()
        self._max_entries = max_entries
        self._clock = clock

    def record(self, prefix_hash: str, model_id: str, effort: str | None = None) -> None:
        served = self._entries.pop(prefix_hash, {})
        served[(model_id, effort)] = self._clock()
        self._entries[prefix_hash] = served
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def warm_models(self, prefix_hash: str | None) -> dict[str, float]:
        """Model id -> most recent time it served this prefix, at any effort."""
        latest: dict[str, float] = {}
        for (model_id, _), served_at in self._entries.get(prefix_hash or "", {}).items():
            latest[model_id] = max(served_at, latest.get(model_id, served_at))
        return latest

    def is_warm(
        self,
        prefix_hash: str | None,
        model_id: str,
        ttl: int | None,
        effort: str | None = None,
        effort_matters: bool = False,
    ) -> bool:
        """Whether ``model_id`` still has this prefix cached.

        Where effort is part of the provider's cache key, only the same effort counts.
        """
        if ttl is None:
            return False
        now = self._clock()
        return any(
            mid == model_id and (not effort_matters or used == effort) and now - at <= ttl
            for (mid, used), at in self._entries.get(prefix_hash or "", {}).items()
        )

    def longest_warm(
        self,
        boundaries: Sequence[tuple[str, int]],
        model_id: str,
        ttl: int | None,
        effort: str | None = None,
        effort_matters: bool = False,
    ) -> int:
        """Estimated tokens of the longest boundary ``model_id`` has warm in its cache."""
        for prefix_hash, tokens in reversed(boundaries):
            if self.is_warm(prefix_hash, model_id, ttl, effort, effort_matters):
                return tokens
        return 0

    def chain_incumbent(
        self, boundaries: Sequence[tuple[str, int]], ttl_for: Callable[[str], int | None]
    ) -> str | None:
        """The warm model for the longest boundary that has one."""
        for prefix_hash, _ in reversed(boundaries):
            if (model_id := self.incumbent(prefix_hash, ttl_for)) is not None:
                return model_id
        return None

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
