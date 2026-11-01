"""Show routing at work: prompts of rising difficulty through the compose stack, one table.

Usage (stack up first: docker compose -f deploy/docker-compose.yml up -d --build):
    uv run python scripts/demo.py
"""

from pathlib import Path

import httpx
import yaml
from compose_smoke import GATEWAY, PROMPTS, create_key, wait_for

REGISTRY = Path(__file__).parents[1] / "deploy" / "registry.compose.yaml"


def main() -> None:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    models = {m["id"]: m for m in registry["models"]}
    flagship = models[registry["baseline_model"]]
    client = httpx.Client(timeout=30)
    wait_for("gateway /healthz", lambda: client.get(f"{GATEWAY}/healthz").is_success, 120)
    auth = {"Authorization": f"Bearer {create_key('demo')}"}
    live = lambda: client.get(f"{GATEWAY}/v1/models", headers=auth).is_success  # noqa: E731
    wait_for("demo key live", live, 10)

    print(f"\n{'prompt':<50}{'model':<8}{'effort':<9}{'reason':<16}{'cost $':>10}")
    total = baseline = 0.0
    for prompt in PROMPTS:
        payload = {"model": "auto", "messages": [{"role": "user", "content": prompt}]}
        resp = client.post(f"{GATEWAY}/v1/chat/completions", json=payload, headers=auth)
        resp.raise_for_status()
        h, usage = resp.headers, resp.json()["usage"]
        cost = float(h["X-Tierwise-Cost-USD"])
        total += cost
        baseline += (usage["prompt_tokens"] * flagship["price_in"]
                     + usage["completion_tokens"] * flagship["price_out"]) / 1e6  # fmt: skip
        shown = prompt if len(prompt) <= 48 else prompt[:45] + "..."
        print(f"{shown:<50}{h['X-Tierwise-Model']:<8}{h.get('X-Tierwise-Effort', '-'):<9}"
              f"{h['X-Tierwise-Reason']:<16}{cost:>10.6f}")  # fmt: skip
    saved = 1 - total / baseline if baseline else 0.0
    print(f"\ntotal ${total:.6f} vs ${baseline:.6f} always-{flagship['id']}: {saved:.0%} saved")


if __name__ == "__main__":
    main()
