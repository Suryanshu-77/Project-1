"""Semantic cache: answer a repeated question without calling a model.

The last user turn is matched by meaning (cosine similarity of MiniLM embeddings).
Everything else that shapes the answer is matched exactly, through the scope: the routing
target and tag, the required capabilities, and every earlier message.
"""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

import numpy as np

from tierwise.router.router import RoutingResult
from tierwise.schema import Usage
from tierwise.tokens import text_of

# Embeds a last user turn to a unit vector, or None when it is too long to embed faithfully
# (the encoder truncates), in which case only an exact text match counts.
Embed = Callable[[str], Awaitable[np.ndarray | None]]


@dataclass(frozen=True)
class CachedAnswer:
    content: str
    model_id: str
    provider: str
    usage: Usage
    finish_reason: str


@dataclass(frozen=True)
class CacheKey:
    scope: str
    text_hash: str
    vector: np.ndarray | None


@dataclass
class _Entry:
    key: CacheKey
    answer: CachedAnswer
    created: float


def last_user_text(result: RoutingResult) -> str:
    return text_of(result.request.messages[-1].content)


def cacheable(result: RoutingResult, headers: Mapping[str, str], max_temperature: float) -> bool:
    req = result.request
    temperature = 1.0 if req.temperature is None else req.temperature  # unset: provider default
    return (
        result.target.kind != "model"  # a named model is a passthrough
        and not req.tools
        and "vision" not in result.required_caps  # images aren't part of the embedded text
        and req.messages[-1].role == "user"
        and temperature <= max_temperature
        and "no-cache" not in headers.get("cache-control", "").lower()
    )


def scope_key(result: RoutingResult) -> str:
    """Everything but the last user turn's text, hashed: it must match exactly."""
    decision = result.decision
    tag = decision.difficulty if decision is not None and decision.source == "override" else None
    earlier = [m.model_dump(mode="json") for m in result.request.messages[:-1]]
    parts = {
        "model": result.request.model,
        "tag": tag,
        "caps": sorted(result.required_caps),
        "earlier": earlier,
        "response_format": result.request.response_format,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


class SemanticCache:
    """An in-memory index (numpy), oldest entries evicted first."""

    def __init__(
        self,
        embed: Embed,
        similarity: float = 0.95,
        ttl_s: float = 3600,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._embed = embed
        self.similarity = similarity
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)

    async def key(self, scope: str, text: str) -> CacheKey:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return CacheKey(scope, digest, await self._embed(text))

    def lookup(self, key: CacheKey) -> CachedAnswer | None:
        now = self._clock()
        best: tuple[float, CachedAnswer] | None = None
        for entry_id, entry in list(self._entries.items()):
            if now - entry.created > self.ttl_s:
                del self._entries[entry_id]
                continue
            if entry.key.scope != key.scope:
                continue
            if entry.key.text_hash == key.text_hash:
                return entry.answer
            if key.vector is None or entry.key.vector is None:
                continue  # long text: exact matches only
            score = float(np.dot(key.vector, entry.key.vector))
            if score >= self.similarity and (best is None or score > best[0]):
                best = (score, entry.answer)
        return None if best is None else best[1]

    def store(self, key: CacheKey, answer: CachedAnswer) -> None:
        self._entries[self._next_id] = _Entry(key, answer, self._clock())
        self._next_id += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
