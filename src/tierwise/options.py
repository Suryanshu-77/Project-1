"""(model, effort) options: what the selector actually chooses between.

A model with an ``efforts`` block becomes one option per level; a model without one is a
single option that leaves effort to the provider.
"""

from dataclasses import dataclass

from tierwise.registry import ModelSpec
from tierwise.schema import EFFORT_ORDER, EffortLevel


@dataclass(frozen=True)
class Option:
    spec: ModelSpec
    effort: EffortLevel | None  # None: the model has no effort control
    quality: float  # effective quality at this effort
    thinking_tokens: int  # expected reasoning tokens at this effort
    send: bool  # whether an effort field goes on the wire

    @property
    def wire_effort(self) -> EffortLevel | None:
        """The effort actually sent to the provider (None: provider default)."""
        return self.effort if self.send else None

    @property
    def header_value(self) -> str:
        return self.effort if self.send and self.effort is not None else "default"

    @property
    def rank(self) -> int:
        """Position on the effort scale; -1 for models without effort control."""
        return EFFORT_ORDER.index(self.effort) if self.effort is not None else -1


def expand(spec: ModelSpec) -> list[Option]:
    """All options for ``spec``, lowest effort first.

    Effective quality is the model's quality plus the level's shift; the provider's
    default level sends nothing, every other level is sent.
    """
    if spec.efforts is None:
        return [Option(spec=spec, effort=None, quality=spec.quality, thinking_tokens=0, send=False)]

    return [
        Option(
            spec=spec,
            effort=level,
            quality=min(1.0, max(0.0, spec.quality + spec.efforts.levels[level].quality)),
            thinking_tokens=spec.efforts.levels[level].thinking_tokens,
            send=(level != spec.efforts.default),
        )
        for level in EFFORT_ORDER
        if level in spec.efforts.levels
    ]


def forced(spec: ModelSpec, level: str) -> Option | None:
    """The option at ``level`` when the client asked for it: always sent, or None."""
    if spec.efforts is None or level not in spec.efforts.levels:
        return None
    shift = spec.efforts.levels[level]
    return Option(
        spec=spec,
        effort=level,
        quality=min(1.0, max(0.0, spec.quality + shift.quality)),
        thinking_tokens=shift.thinking_tokens,
        send=True,
    )
