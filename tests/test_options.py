from typing import Any

import pytest

from tierwise.options import expand, forced
from tierwise.registry import ModelSpec


def _spec(**overrides: Any) -> ModelSpec:
    fields: dict[str, Any] = {
        "id": "m", "provider": "anthropic", "price_in": 1.0, "price_out": 2.0,
        "context": 1000, "caps": [], "quality": 0.80, "tier": "moderate",
    }  # fmt: skip
    fields.update(overrides)
    return ModelSpec.model_validate(fields)


EFFORTS = {
    "default": "medium",
    "levels": {
        "high": {"quality": 0.03, "thinking_tokens": 2000},
        "low": {"quality": -0.04, "thinking_tokens": 150},
        "medium": {"quality": 0.0, "thinking_tokens": 600},
    },
}


def test_model_without_efforts_has_one_default_option() -> None:
    [option] = expand(_spec())
    assert option.effort is None
    assert option.send is False
    assert option.quality == 0.80
    assert option.thinking_tokens == 0
    assert option.header_value == "default"
    assert option.wire_effort is None


def test_one_option_per_level_in_effort_order() -> None:
    options = expand(_spec(efforts=EFFORTS))
    assert [o.effort for o in options] == ["low", "medium", "high"]


def test_effective_quality_and_thinking_tokens_per_level() -> None:
    by_level = {o.effort: o for o in expand(_spec(efforts=EFFORTS))}
    assert by_level["low"].quality == pytest.approx(0.76)
    assert by_level["high"].quality == pytest.approx(0.83)
    assert by_level["low"].thinking_tokens == 150


def test_default_level_sends_nothing() -> None:
    by_level = {o.effort: o for o in expand(_spec(efforts=EFFORTS))}
    assert by_level["medium"].send is False
    assert by_level["medium"].wire_effort is None
    assert by_level["medium"].header_value == "default"
    assert by_level["low"].send is True
    assert by_level["low"].wire_effort == "low"
    assert by_level["low"].header_value == "low"


def test_forced_level_is_always_sent() -> None:
    option = forced(_spec(efforts=EFFORTS), "medium")
    assert option is not None
    assert option.send is True
    assert option.wire_effort == "medium"


def test_forced_unknown_level_is_none() -> None:
    assert forced(_spec(efforts=EFFORTS), "max") is None
    assert forced(_spec(), "low") is None


def test_rank_follows_effort_scale() -> None:
    options = expand(_spec(efforts=EFFORTS))
    assert [o.rank for o in options] == [1, 2, 3]
    assert expand(_spec())[0].rank == -1
