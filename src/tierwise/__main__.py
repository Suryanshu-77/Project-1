"""Run the gateway (`tierwise`, `python -m tierwise`), manage API keys (`tierwise keys`) or
download the learned router's encoder (`tierwise fetch-model`)."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import uvicorn

from tierwise import __version__, calibration, fetch, flywheel
from tierwise.api.app import LIMITS, TIMEOUT, build_providers, create_app
from tierwise.budgets import month_of
from tierwise.config import CACHE_ROUTER_DIR, JUDGE_CUTS_FILE, Settings, registry_file
from tierwise.keys import generate_key, hash_key
from tierwise.registry import load_registry
from tierwise.store import RequestLog


def _rpm(value: str) -> int:
    rpm = int(value)
    if rpm < 1:
        raise argparse.ArgumentTypeError("must be at least 1 (omit --rpm for no limit)")
    return rpm


def _budget(value: str) -> float:
    budget = float(value)
    if budget < 0:
        raise argparse.ArgumentTypeError("must be zero or more (omit --budget for no limit)")
    return budget


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tierwise")
    parser.add_argument("--version", action="version", version=f"tierwise {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the gateway (the default)")
    keys = commands.add_parser("keys", help="manage gateway API keys").add_subparsers(
        dest="action", required=True
    )
    create = keys.add_parser("create", help="create a key (printed once)")
    create.add_argument("--name", required=True)
    create.add_argument("--budget", type=_budget, help="monthly budget in USD")
    create.add_argument("--rpm", type=_rpm, help="requests per minute")
    create.add_argument(
        "--allow-shadow",
        action="store_true",
        help="let the data flywheel sample this key's traffic",
    )
    keys.add_parser("list", help="list keys (no secrets)")
    revoke = keys.add_parser("revoke", help="revoke a key by id")
    revoke.add_argument("id")
    fetch_model = commands.add_parser(
        "fetch-model", help="download the learned router's encoder (90 MB, sha256-checked)"
    )
    fetch_model.add_argument("--out", type=Path, default=CACHE_ROUTER_DIR)
    fw = commands.add_parser("flywheel", help="data flywheel: label shadow samples")
    fw_actions = fw.add_subparsers(dest="action", required=True)
    label = fw_actions.add_parser(
        "label", help="contrast + judge pending samples (paid calls, monthly budget)"
    )
    label.add_argument("--limit", type=_rpm, default=500, help="most samples to label this run")
    calibrate = fw_actions.add_parser(
        "calibrate", help="choose the judge's needs-strong cut from SPROUT reference pairs"
    )
    calibrate.add_argument(
        "--dry-run", action="store_true", help="fetch and check the reference, print the cost"
    )
    calibrate.add_argument(
        "--max-usd", type=_budget, default=0.10, help="stop (and save nothing) past this spend"
    )
    calibrate.add_argument("--cuts-file", type=Path, default=JUDGE_CUTS_FILE)
    return parser


async def _keys(args: argparse.Namespace, db_url: str) -> int:
    log = RequestLog(db_url)
    await log.start()
    try:
        if args.action == "create":
            key = generate_key()
            row = await log.create_key(
                args.name, hash_key(key), key[:10], args.budget, args.rpm, args.allow_shadow
            )
            print(f"id: {row.id}\nkey: {key}\nStore the key now: it is not shown again.")
            return 0
        if args.action == "revoke":
            if await log.revoke_key(args.id):
                print(f"revoked {args.id}")
                return 0
            print(f"no active key with id {args.id}", file=sys.stderr)
            return 1
        spend = await log.load_month_spend(month_of(time.time()))
        print(f"{'id':<14}{'name':<16}{'prefix':<12}{'budget':>10}{'spent':>10}{'rpm':>6}  status")
        for row in await log.list_keys():
            budget = "-" if row.monthly_budget_usd is None else f"{row.monthly_budget_usd:.2f}"
            rpm = "-" if row.rpm is None else str(row.rpm)
            status = "active" if row.revoked_ts is None else "revoked"
            print(f"{row.id:<14}{row.name:<16}{row.prefix:<12}{budget:>10}"
                  f"{spend.get(row.id, 0.0):>10.2f}{rpm:>6}  {status}")  # fmt: skip
        return 0
    finally:
        await log.stop()


async def _flywheel_label(args: argparse.Namespace, settings: Settings) -> int:
    registry = load_registry(registry_file(settings))
    judge = registry.get(settings.judge_model or registry.baseline_model)
    if judge is None:
        print(f"judge model {settings.judge_model!r} is not in the registry", file=sys.stderr)
        return 1
    clients: list[httpx.AsyncClient] = []

    def client_for(_: str) -> httpx.AsyncClient:
        clients.append(httpx.AsyncClient(limits=LIMITS, timeout=TIMEOUT))
        return clients[-1]

    log = RequestLog(settings.db_url)
    await log.start()
    try:
        providers = build_providers(settings, client_for)
        run = await flywheel.label_pending(
            log, registry, providers, judge, settings.shadow_budget_usd, args.limit
        )
    except ValueError as exc:
        print(f"flywheel label failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await log.stop()
        for client in clients:
            await client.aclose()
    print(f"labeled {run.labeled}, failed {run.failed}, left {run.left}; "
          f"spent ${run.spent_usd:.2f} this run")  # fmt: skip
    if run.stopped_by_budget:
        print(f"stopped: monthly label budget ${settings.shadow_budget_usd:.2f} reached "
              "(TIERWISE_SHADOW_BUDGET_USD)")  # fmt: skip
    return 0


async def _flywheel_calibrate(args: argparse.Namespace, settings: Settings) -> int:
    registry = load_registry(registry_file(settings))
    judge = registry.get(settings.judge_model or registry.baseline_model)
    if judge is None:
        print(f"judge model {settings.judge_model!r} is not in the registry", file=sys.stderr)
        return 1
    try:
        pairs = calibration.fetch_reference(calibration.reference_file())
    except calibration.CalibrationError as exc:
        print(f"flywheel calibrate failed: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        cost = calibration.estimate_usd(pairs, judge)
        print(f"dry run: {len(pairs)} calls to judge {judge.id}, ~${cost:.2f}; no calls made")
        return 0
    clients: list[httpx.AsyncClient] = []

    def client_for(_: str) -> httpx.AsyncClient:
        clients.append(httpx.AsyncClient(limits=LIMITS, timeout=TIMEOUT))
        return clients[-1]

    try:
        provider = build_providers(settings, client_for).get(judge.provider)
        if provider is None:
            print(f"judge {judge.id}: provider {judge.provider!r} is not configured",
                  file=sys.stderr)  # fmt: skip
            return 1
        judged = await calibration.judge_reference(pairs, provider, judge, args.max_usd)
    finally:
        for client in clients:
            await client.aclose()
    if judged.stopped_by_cap:
        print(f"incomplete: stopped at --max-usd ${args.max_usd:.2f} after "
              f"{len(judged.scores)} of {len(pairs)} pairs; nothing saved",
              file=sys.stderr)  # fmt: skip
        return 1
    if judged.unscored > calibration.MAX_UNSCORED * len(pairs):
        print(f"incomplete: {judged.unscored} of {len(pairs)} pairs unscored (more than "
              f"{calibration.MAX_UNSCORED:.0%}); nothing saved", file=sys.stderr)  # fmt: skip
        return 1
    scored = [p for p in pairs if p.pid in judged.scores]
    gold = [p.gold for p in scored]
    reference_rate = sum(gold) / len(gold)
    stats = [calibration.cut_stats([judged.scores[p.pid] for p in scored], gold, cut)
             for cut in calibration.CANDIDATE_CUTS]  # fmt: skip
    best = calibration.choose_cut(stats, reference_rate)
    if best is None:
        print(f"judge {judge.id}: no cut labels at half to twice G's rate; nothing saved",
              file=sys.stderr)  # fmt: skip
        for s in stats:
            print(f"  cut {s.cut}: rate {s.rate:.1%} (G {reference_rate:.1%}), "
                  f"precision {s.precision:.2f}, recall {s.recall:.2f}",
                  file=sys.stderr)  # fmt: skip
        return 1
    calibration.save_cut(args.cuts_file, judge.id, {
        "cut": best.cut, "rate": best.rate, "reference_rate": reference_rate,
        "precision": best.precision, "recall": best.recall, "f1": best.f1, "n": len(scored),
        "spent_usd": judged.spent_usd, "ts": time.time(),
    })  # fmt: skip
    print(f"judge {judge.id}: cut {best.cut} (rate {best.rate:.1%}, G {reference_rate:.1%}, "
          f"precision {best.precision:.2f}, recall {best.recall:.2f}, n {len(scored)}, "
          f"${judged.spent_usd:.2f}); saved to {args.cuts_file}")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch-model":
        try:
            path = fetch.fetch_model(args.out)
        except fetch.FetchError as exc:
            print(f"fetch-model failed: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {path}")
        return 0
    settings = Settings()
    if args.command == "flywheel":
        if args.action == "calibrate":
            return asyncio.run(_flywheel_calibrate(args, settings))
        return asyncio.run(_flywheel_label(args, settings))
    if args.command == "keys":
        return asyncio.run(_keys(args, settings.db_url))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
