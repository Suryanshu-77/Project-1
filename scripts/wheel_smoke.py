"""Check an installed wheel works with no repo checkout (the M8 exit check).

Run with the wheel's own Python from an empty directory, so nothing resolves against the repo:
    cd "$(mktemp -d)" && /path/to/venv/bin/python /path/to/scripts/wheel_smoke.py
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

import tierwise
from tierwise.api.app import create_app
from tierwise.config import Settings, router_directory


def main() -> None:
    package = Path(tierwise.__file__).parent
    if Path("config").exists() or Path("models").exists():
        sys.exit("run this from an empty directory, not the repo")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = router_directory(settings)
    assert router.is_relative_to(package), f"router dir {router} is not the packaged copy"
    assert (router / "router.json").exists(), "router.json missing from the wheel"
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert len(app.state.registry.models) > 0, "packaged registry is empty"
    print(f"ok   wheel serves with {len(app.state.registry.models)} packaged models")


if __name__ == "__main__":
    main()
