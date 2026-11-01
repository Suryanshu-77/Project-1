import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.datasets import load_gpt4_test, row_from_gpt4, row_from_sprout
from training.prepare_data import split_of


def _gpt4(prompt: str = "q", score: int = 3, **kw: Any) -> dict[str, Any]:
    return {"prompt": prompt, "label": int(score < 4), "mixtral_score": score,
            "prompt_chars": 400, "gpt4_chars": 800, "mixtral_chars": 800, **kw}  # fmt: skip


def _prompt_in(split: str) -> str:
    return next(p for i in range(10_000) if split_of(p := f"prompt {i}") == split)


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_row_from_gpt4_quality_and_cost() -> None:
    row = row_from_gpt4(_gpt4(score=3))
    assert (row.q_cheap, row.q_strong) == (0.5, 1.0)
    assert row.cost_strong == pytest.approx((100 * 10 + 200 * 30) / 1e6)
    assert row.cost_cheap == pytest.approx((100 * 0.6 + 200 * 0.6) / 1e6)


def test_load_gpt4_test_reads_test_rows(tmp_path: Path) -> None:
    rows = load_gpt4_test(_write(tmp_path / "t.jsonl", [_gpt4(_prompt_in("test"), score=5)]))
    assert len(rows) == 1 and rows[0].q_cheap == 1.0


def test_load_gpt4_test_rejects_train_prompts(tmp_path: Path) -> None:
    path = _write(tmp_path / "t.jsonl", [_gpt4(_prompt_in("test")), _gpt4(_prompt_in("train"))])
    with pytest.raises(ValueError, match="row 1"):
        load_gpt4_test(path)


def test_load_gpt4_test_requires_new_fields(tmp_path: Path) -> None:
    path = _write(tmp_path / "t.jsonl", [{"prompt": _prompt_in("test"), "label": 0}])
    with pytest.raises(ValueError, match="prepare_data"):
        load_gpt4_test(path)


def _sprout(cheap: float | None = 0.0, strong: float | None = 1.0) -> dict[str, Any]:
    def model(score: float | None) -> dict[str, Any] | None:
        if score is None:
            return None
        return {"score": score, "num_input_tokens": 1000, "num_output_tokens": 500,
                "response": "r", "judge_response": "{}"}  # fmt: skip

    return {"prompt": "p", "openai-gpt-4o-mini": model(cheap), "openai-gpt-4o": model(strong)}


def test_row_from_sprout() -> None:
    row = row_from_sprout(_sprout(0.25, 1.0), "openai-gpt-4o-mini", "openai-gpt-4o")
    assert row is not None
    assert (row.q_cheap, row.q_strong) == (0.25, 1.0)
    assert row.cost_cheap == pytest.approx((1000 * 0.15 + 500 * 0.6) / 1e6)
    assert row.cost_strong == pytest.approx((1000 * 2.5 + 500 * 10) / 1e6)


def test_row_from_sprout_missing_model_is_none() -> None:
    assert row_from_sprout(_sprout(cheap=None), "openai-gpt-4o-mini", "openai-gpt-4o") is None


def test_row_from_sprout_bad_score_raises() -> None:
    with pytest.raises(ValueError):
        row_from_sprout(_sprout(cheap=4.0), "openai-gpt-4o-mini", "openai-gpt-4o")


def test_load_gpt4_test_keeps_unicode_line_separators(tmp_path: Path) -> None:
    prompt = next(p for i in range(10_000) if split_of(p := f"line break {i}") == "test")
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(_gpt4(prompt), ensure_ascii=False) + "\n", encoding="utf-8")
    [row] = load_gpt4_test(path)
    assert row.prompt == prompt
