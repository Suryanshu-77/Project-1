"""M7 exit check: drive the running compose stack and confirm every part works.

Usage (stack already up):
    docker compose -f deploy/docker-compose.yml up -d --build
    uv run python scripts/compose_smoke.py
Exits non-zero on the first failed check, after printing the compose logs.
"""

import re
import subprocess
import sys
import time
from collections.abc import Callable

import httpx

COMPOSE = ["docker", "compose", "-f", "deploy/docker-compose.yml"]
GATEWAY, PROMETHEUS = "http://localhost:8000", "http://localhost:9090"
GRAFANA, JAEGER, UPSTREAM = "http://localhost:3000", "http://localhost:16686", "http://localhost:8080"
PROMPTS = [
    "hi, thanks",
    "What is the capital of France?",
    "Translate 'good morning' into Spanish.",
    "Design a multi-tenant SaaS billing architecture and compare the trade-offs step by step.",
    "Debug this production outage: checkout latency spiked after a migration.",
    "Prove that the square root of 2 is irrational.",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    subprocess.run([*COMPOSE, "logs", "--tail", "60"], check=False)
    sys.exit(1)


def wait_for(what: str, check: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                print(f"ok   {what}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    fail(f"timed out waiting for {what}")


def create_key(name: str, *flags: str) -> str:
    out = subprocess.run(
        [*COMPOSE, "exec", "-T", "gateway", "tierwise", "keys", "create", "--name", name, *flags],
        capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    match = re.search(r"tw-[A-Za-z0-9_-]{40,}", out)
    if match is None:
        fail(f"could not create key {name}: {out!r}")
    assert match is not None
    return match.group(0)


def main() -> None:
    client = httpx.Client(timeout=30)
    wait_for("gateway /healthz", lambda: client.get(f"{GATEWAY}/healthz").is_success, 120)
    key = create_key("smoke", "--budget", "5", "--rpm", "600")
    limited = create_key("smoke-limited", "--rpm", "1")
    time.sleep(3)  # KEYS_REFRESH_S=2 in compose
    auth = {"Authorization": f"Bearer {key}"}

    sent = 0
    for i in range(60):
        prompt = PROMPTS[i % len(PROMPTS)]
        stream = i % 6 == 5
        payload = {"model": "auto", "stream": stream, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
        resp = client.post(f"{GATEWAY}/v1/chat/completions", json=payload, headers=auth)
        if resp.status_code != 200:
            fail(f"request {i} returned {resp.status_code}: {resp.text[:200]}")
        sent += 1
    print(f"ok   {sent} requests through the gateway")

    if client.post(f"{GATEWAY}/v1/chat/completions", json={"messages": [{"role": "user",
                   "content": "hi"}]}).status_code != 401:  # fmt: skip
        fail("a request without a key was not rejected")
    print("ok   missing key -> 401")
    limited_auth = {"Authorization": f"Bearer {limited}"}
    body = {"messages": [{"role": "user", "content": "hi"}]}
    codes = [client.post(f"{GATEWAY}/v1/chat/completions", json=body, headers=limited_auth)
             .status_code for _ in range(2)]  # fmt: skip
    if codes != [200, 429]:
        fail(f"rpm=1 key returned {codes}, expected [200, 429]")
    print("ok   rate limit -> 429")

    def prometheus_has_traffic() -> bool:
        data = client.get(f"{PROMETHEUS}/api/v1/query",
                          params={"query": "sum(tierwise_requests_total)"}).json()  # fmt: skip
        result = data["data"]["result"]
        return bool(result) and float(result[0]["value"][1]) >= sent

    wait_for("Prometheus scraped the traffic", prometheus_has_traffic, 60)
    wait_for(
        "Grafana serves the tierwise dashboard",
        lambda: client.get(f"{GRAFANA}/api/dashboards/uid/tierwise").is_success,
        60,
    )
    wait_for(
        "Jaeger received tierwise traces",
        lambda: "tierwise" in client.get(f"{JAEGER}/api/services").json().get("data") or [],
        60,
    )

    stats = client.get(f"{UPSTREAM}/stats").json()
    if stats["requests"] < sent or stats["connections"] > 10:
        fail(f"connection pool not reused: {stats}")
    print(f"ok   pool reuse: {stats['requests']} upstream requests over "
          f"{stats['connections']} connections")  # fmt: skip
    print("\nM7 exit check passed: the stack is up and the dashboard has data.")


if __name__ == "__main__":
    main()
