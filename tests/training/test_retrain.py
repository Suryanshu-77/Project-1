import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from tierwise.registry import load_registry  # noqa: E402
from tierwise.schema import ChatRequest  # noqa: E402
from tierwise.store import RequestLog, ShadowSample  # noqa: E402
from training import retrain  # noqa: E402
from training.prepare_data import split_of  # noqa: E402
from training.retrain import NotEnoughLabels, prompt_of, run, shadow_rows  # noqa: E402

ROOT = Path(__file__).parents[2]
REGISTRY = load_registry(ROOT / "deploy" / "registry.compose.yaml")
HARD = ("design", "prove", "debug", "architecture")
NOW = 1793577600.0  # 2026-11-02T00:00:00Z
FILES = ("classifier.npz", "router.json", "metrics.json")


def fake_embed(requests: list[ChatRequest]) -> np.ndarray:
    """384-dim like MiniLM: axis 0 carries 'looks hard', the rest is per-prompt noise."""
    out = np.zeros((len(requests), 384), dtype=np.float32)
    for i, req in enumerate(requests):
        text = prompt_of(req).lower()
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        out[i] = np.random.default_rng(seed).normal(scale=0.1, size=384)
        out[i, 0] = 3.0 if any(word in text for word in HARD) else -3.0
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _prompts(n: int) -> list[tuple[str, bool]]:
    hard = [(f"Design a {i}-tier architecture for system {i}", True) for i in range(n // 2)]
    easy = [(f"Translate word number {i} into Spanish", False) for i in range(n - n // 2)]
    return hard + easy


def _labeled(i: int, prompt: str, hard: bool) -> ShadowSample:
    request = {"model": "auto", "messages": [{"role": "user", "content": prompt}]}
    routed, contrast = ("large", "small") if hard else ("small", "large")
    return ShadowSample(
        id=f"s{i}", ts=1000.0 + i, key_id="k", request_json=json.dumps(request),
        routed_model=routed, routed_answer="a routed answer", difficulty=0.5, confidence=0.4,
        source="rules", status="labeled", contrast_model=contrast,
        contrast_answer="a contrast answer", judge_score=2 if hard else 5, needs_strong=hard,
        label_cost_usd=0.01, labeled_ts=NOW,
    )  # fmt: skip


async def _db(tmp_path: Path, samples: list[ShadowSample]) -> str:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'r.db').as_posix()}"
    log = RequestLog(url)
    await log.start()
    for sample in samples:
        log.submit_sample(sample)
    await log.stop()
    return url


def _router(tmp_path: Path) -> Path:
    router = tmp_path / "router"
    router.mkdir()
    for name in (*FILES, "tokenizer.json", "NOTICE"):
        shutil.copy(ROOT / "models" / "router" / name, router / name)
    return router


async def _run(tmp_path: Path, samples: list[ShadowSample], promote: bool = True):  # type: ignore[no-untyped-def]
    url = await _db(tmp_path, samples)
    return await run(url, _router(tmp_path), REGISTRY, promote=promote,
                     data_dir=tmp_path / "no-base-data", embed=fake_embed,
                     now=lambda: NOW)  # fmt: skip


def _samples(n: int = 400) -> list[ShadowSample]:
    return [_labeled(i, p, hard) for i, (p, hard) in enumerate(_prompts(n))]


def test_shadow_rows_split_by_prompt_hash() -> None:
    splits = shadow_rows(_samples(), REGISTRY, critical=set())
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for r in rows:
            assert split == split_of(r.prompt)
            assert seen.setdefault(r.prompt, split) == split  # never in two splits
    assert sum(len(rows) for rows in splits.values()) == 400


def test_shadow_eval_rows_price_cheap_and_strong() -> None:
    splits = shadow_rows(_samples(20), REGISTRY, critical=set())
    for rows in splits.values():
        for r in rows:
            assert r.row.q_strong == 1.0 and r.row.q_cheap == (0.0 if r.needs_strong else 1.0)
            assert 0 < r.row.cost_cheap < r.row.cost_strong  # small vs large, either way round


def test_critical_prompts_excluded_from_training() -> None:
    hard_prompt = "Design a 0-tier architecture for system 0"
    splits = shadow_rows(_samples(20), REGISTRY, critical={hard_prompt.lower()})
    prompts = [r.prompt for rows in splits.values() for r in rows]
    assert hard_prompt not in prompts and len(prompts) == 19


async def test_refuses_below_20_labels(tmp_path: Path) -> None:
    with pytest.raises(NotEnoughLabels):
        await _run(tmp_path, _samples(12))


async def test_dry_run_writes_version_but_not_router_dir(tmp_path: Path) -> None:
    report = await _run(tmp_path, _samples(), promote=False)
    router = tmp_path / "router"
    assert report.version == "20261102T000000Z" and not report.promoted
    assert {p.name for p in (router / "versions" / report.version).iterdir()} == set(FILES)
    for name in FILES:  # the champion is untouched
        assert (router / name).read_bytes() == (ROOT / "models" / "router" / name).read_bytes()
    assert report.reason and report.n_labels == 400


async def test_promote_commits_by_pointing_router_json_at_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrain, "decide", lambda champ, chall: (True, "forced"))
    report = await _run(tmp_path, _samples())
    router = tmp_path / "router"
    assert report.decision and report.promoted and report.reason == "forced"
    config = json.loads((router / "router.json").read_text())
    assert config["version"] == report.version
    assert config["weights"] == f"versions/{report.version}/classifier.npz"
    # the shipped weights are never overwritten; they're archived under their content hash
    shipped = (ROOT / "models" / "router" / "classifier.npz").read_bytes()
    assert (router / "classifier.npz").read_bytes() == shipped
    [archive] = [d for d in (router / "versions").iterdir() if d.name.startswith("initial-")]
    archived = json.loads((archive / "router.json").read_text())
    assert (router / archived["weights"]).read_bytes() == shipped
    assert not list(router.glob("*.tmp"))


async def test_failed_commit_leaves_the_champion_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrain, "decide", lambda champ, chall: (True, "forced"))

    def crash(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(retrain.os, "replace", crash)
    with pytest.raises(OSError):
        await _run(tmp_path, _samples())
    for name in ("classifier.npz", "router.json"):
        champion = (tmp_path / "router" / name).read_bytes()
        assert champion == (ROOT / "models" / "router" / name).read_bytes()


async def test_two_unversioned_champions_are_both_archived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrain, "decide", lambda champ, chall: (True, "forced"))
    url = await _db(tmp_path, _samples())
    router = _router(tmp_path)

    async def once(clock: float) -> None:
        await run(url, router, REGISTRY, promote=True, data_dir=tmp_path / "none",
                  embed=fake_embed, now=lambda: clock)  # fmt: skip

    await once(NOW)
    # someone re-runs training/train.py: new weights, router.json without version or weights
    rng = np.random.default_rng(7)
    with np.load(router / "versions" / "20261102T000000Z" / "classifier.npz") as data:
        np.savez(router / "classifier.npz",
                 **{k: data[k] + rng.normal(scale=0.01, size=data[k].shape) for k in data.files})
    config = json.loads((router / "router.json").read_text())
    for field in ("version", "weights"):
        config.pop(field)
    (router / "router.json").write_text(json.dumps(config))
    await once(NOW + 60)
    archives = [d for d in (router / "versions").iterdir() if d.name.startswith("initial-")]
    assert len(archives) == 2  # neither hand-trained champion was lost


async def test_too_few_held_out_rows_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrain, "decide", lambda champ, chall: (True, "forced"))
    report = await _run(tmp_path, _samples(40))
    assert not report.decision and not report.promoted
    assert "held-out" in report.reason


async def test_reject_leaves_router_dir_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrain, "decide", lambda champ, chall: (False, "nope"))
    report = await _run(tmp_path, _samples())
    assert not report.decision and not report.promoted
    for name in FILES:
        champion = (tmp_path / "router" / name).read_bytes()
        assert champion == (ROOT / "models" / "router" / name).read_bytes()


async def test_challenger_learns_the_shadow_signal(tmp_path: Path) -> None:
    report = await _run(tmp_path, _samples(), promote=False)
    assert report.challenger.auc is not None and report.challenger.auc > 0.95


def test_cli_exit_2_on_too_few_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*args: object, **kwargs: object) -> None:
        raise NotEnoughLabels("only 3 labeled samples; need 20")

    monkeypatch.setattr(retrain, "run", refuse)
    assert retrain.main(["--db-url", "sqlite+aiosqlite://", "--router-dir", str(tmp_path)]) == 2


def test_default_embedder_finds_encoder_in_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = ROOT / "models" / "router" / "model.onnx"
    if not real.exists():
        pytest.skip("needs the real encoder (tierwise fetch-model)")
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(real, cache / "model.onnx")
    monkeypatch.setattr("tierwise.router.learned.CACHE_ROUTER_DIR", cache)
    embed = retrain.default_embedder(_router(tmp_path))  # no model.onnx in the router dir
    vectors = embed([ChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})])
    assert vectors.shape == (1, 384)


async def test_critical_check_uses_the_deployed_hybrid() -> None:
    hard = retrain._critical(retrain.CRITICAL)
    assert len(hard) == 10
    # a classifier that calls every hard prompt easy still routes none of them basic: in the
    # gateway, rules can only raise difficulty. So this check is a floor guaranteed by the
    # rules, and the gate adds a classifier-only non-regression check on top (gate.decide).
    assert await retrain.critical_misroutes(hard, np.zeros(10), 0.5, REGISTRY) == 0
    assert await retrain.critical_misroutes(hard, np.ones(10), 0.5, REGISTRY) == 0


def test_retrain_imports_without_torch() -> None:
    """The `retrain` dependency group has no torch: nothing retrain imports may need it."""
    import subprocess
    import sys

    # torch "not installed": its import fails the way it does in CI
    code = """
import sys

class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            raise ModuleNotFoundError(name)

sys.meta_path.insert(0, NoTorch())
import training.retrain
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


async def test_crash_leaves_no_partial_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("crashed mid-gate")

    monkeypatch.setattr(retrain, "decide", boom)  # after the weights are written
    with pytest.raises(RuntimeError):
        await _run(tmp_path, _samples())
    versions = tmp_path / "router" / "versions"
    left = [p.name for p in versions.iterdir()] if versions.exists() else []
    assert left == []  # no half-written folder that rollback could point at


def test_encoder_dim_mismatch_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = _router(tmp_path)
    config = json.loads((router / "router.json").read_text())
    (router / "router.json").write_text(json.dumps({**config, "dim": 768}))

    class Encoder:
        dim = 384

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(retrain, "OnnxEncoder", Encoder)
    with pytest.raises(ValueError, match="384.*768|768.*384"):
        retrain.default_embedder(router)
