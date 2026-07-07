#!/usr/bin/env python3
"""Grade a single user's submission for a single challenge; used by CI to gate PRs.

Usage:
    python scripts/grade_submission.py <challenge_id> <username>

Exits 0 (and prints the pytest output) if all tests pass or if the user has no
submission for this challenge. Exits 1 if tests fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPO_ROOT  # noqa: E402
from app.grader import grade  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: grade_submission.py <challenge_id> <username>", file=sys.stderr)
        return 2

    challenge_id, username = sys.argv[1], sys.argv[2]
    solution = REPO_ROOT / challenge_id / "submissions" / username / "solution.py"
    if not solution.is_file():
        print(f"No submission found at {solution}, nothing to grade.")
        return 0

    result = grade(challenge_id, solution.read_text(encoding="utf-8"))
    print(result["output"])
    print(f"\n{result['passed']}/{result['total']} tests passed")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
