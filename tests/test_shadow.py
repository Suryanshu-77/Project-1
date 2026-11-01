import pytest

from tierwise.router import RoutingDecision
from tierwise.shadow import sample_rate, should_sample
from tierwise.store import KeyRow


def _key(allow_shadow: bool = True) -> KeyRow:
    return KeyRow("k1", "team", "h", "tw-x", None, None, 0.0, None, allow_shadow)


DECISION = RoutingDecision(difficulty=0.3, confidence=0.9, source="rules")


@pytest.mark.parametrize("base", [0.0, 0.02, 0.5, 1.0])
def test_sample_rate_contract(base: float) -> None:
    rates = {c: sample_rate(base, c) for c in (None, 0.1, 0.5, 0.95)}
    assert all(0.0 <= r <= 1.0 for r in rates.values()), rates
    assert rates[None] == base and rates[0.95] == base  # sure (or unknown): the base rate
    if 0.0 < base < 1.0:
        assert rates[0.1] > base  # unsure routes are where labels teach the most
    assert rates[0.1] >= rates[0.5] >= rates[0.95]  # never less sampling for less confidence


@pytest.mark.parametrize(
    ("key", "decision", "store_payloads", "expected"),
    [
        (_key(), DECISION, True, True),
        (_key(allow_shadow=False), DECISION, True, False),
        (_key(), DECISION, False, False),
        (_key(), None, True, False),  # a concrete model or tier: no routing decision to learn
        (None, DECISION, True, False),  # no key, no consent
    ],
)
def test_should_sample_requires_all_opt_ins(
    key: KeyRow | None, decision: RoutingDecision | None, store_payloads: bool, expected: bool
) -> None:
    assert should_sample(key, decision, store_payloads, 1.0, rng=lambda: 0.0) is expected


def test_should_sample_respects_rate() -> None:
    assert not should_sample(_key(), DECISION, True, 0.02, rng=lambda: 0.5)
    assert should_sample(_key(), DECISION, True, 0.02, rng=lambda: 0.01)
