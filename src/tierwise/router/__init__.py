"""Difficulty routing: decide how capable a model a request needs."""

from dataclasses import dataclass, field
from typing import Any, Literal

Source = Literal["override", "rules", "learned", "fallback"]


@dataclass(frozen=True)
class RoutingDecision:
    """How hard a request looks (0 = trivial, 1 = needs the best model) and how sure we are."""

    difficulty: float
    confidence: float
    source: Source
    signals: dict[str, Any] = field(default_factory=dict)
