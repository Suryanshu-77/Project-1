"""Protocol 4: run `tierwise flywheel calibrate` for real, with Claude Haiku via AICredits.

    uv run --group train python experiments/flywheel_local/calibrate_haiku.py [--dry-run]

.env's JUDGE_API_KEY / JUDGE_BASE_URL become tierwise's OpenAI-compatible provider (never
printed); the cut goes to judge_cuts_haiku.json here, not to ~/.cache. See PROTOCOL-4.md.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pairs import env  # noqa: E402

from tierwise.__main__ import main  # noqa: E402

if __name__ == "__main__":
    cfg = env(HERE.parents[1] / ".env")
    os.environ |= {
        "OPENAI_API_KEY": cfg["JUDGE_API_KEY"],
        "OPENAI_BASE_URL": cfg["JUDGE_BASE_URL"],
        "TIERWISE_REGISTRY": str(HERE / "registry.judge.yaml"),
        "TIERWISE_JUDGE_MODEL": "claude-haiku-5-5",
    }
    sys.exit(main(["flywheel", "calibrate", "--cuts-file", str(HERE / "judge_cuts_haiku.json"),
                   "--max-usd", "0.20", *sys.argv[1:]]))  # fmt: skip
