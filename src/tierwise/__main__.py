"""Run the gateway (`tierwise`, `python -m tierwise`), manage API keys (`tierwise keys`) or
download the learned router's encoder (`tierwise fetch-model`)."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import uvicorn

from tierwise import fetch
from tierwise.api.app import create_app
from tierwise.budgets import month_of
from tierwise.config import CACHE_ROUTER_DIR, Settings
from tierwise.keys import generate_key, hash_key
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
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the gateway (the default)")
    keys = commands.add_parser("keys", help="manage gateway API keys").add_subparsers(
        dest="action", required=True
    )
    create = keys.add_parser("create", help="create a key (printed once)")
    create.add_argument("--name", required=True)
    create.add_argument("--budget", type=_budget, help="monthly budget in USD")
    create.add_argument("--rpm", type=_rpm, help="requests per minute")
    keys.add_parser("list", help="list keys (no secrets)")
    revoke = keys.add_parser("revoke", help="revoke a key by id")
    revoke.add_argument("id")
    fetch_model = commands.add_parser(
        "fetch-model", help="download the learned router's encoder (90 MB, sha256-checked)"
    )
    fetch_model.add_argument("--out", type=Path, default=CACHE_ROUTER_DIR)
    return parser


async def _keys(args: argparse.Namespace, db_url: str) -> int:
    log = RequestLog(db_url)
    await log.start()
    try:
        if args.action == "create":
            key = generate_key()
            row = await log.create_key(args.name, hash_key(key), key[:10], args.budget, args.rpm)
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
    if args.command == "keys":
        return asyncio.run(_keys(args, settings.db_url))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
