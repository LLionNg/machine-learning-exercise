#!/usr/bin/env python3
"""Grade every submission and (re)write each challenge's SCOREBOARD.md.

Usage:
    python scripts/update_scoreboards.py                 # all challenges
    python scripts/update_scoreboards.py challenge-1-... # specific challenges

This is the heavy, batch counterpart to the read-only aggregation the web API
does at request time. Run it locally or in CI after submissions change.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.challenges import list_challenges  # noqa: E402
from app.config import REPO_ROOT  # noqa: E402
from app.grader import grade  # noqa: E402


def update_one(challenge_id: str) -> None:
    challenge_dir = REPO_ROOT / challenge_id
    submissions_root = challenge_dir / "submissions"

    rows: list[tuple[str, int, int]] = []
    if submissions_root.is_dir():
        for user_dir in sorted(submissions_root.iterdir()):
            solution = user_dir / "solution.py"
            if not solution.is_file():
                continue
            result = grade(challenge_id, solution.read_text(encoding="utf-8"))
            rows.append((user_dir.name, result["passed"], result["total"]))
            status = "PASS" if result["ok"] else "fail"
            print(f"  [{status}] {user_dir.name}: {result['passed']}/{result['total']}")

    rows.sort(key=lambda r: (-r[1], r[0].lower()))

    lines = [
        f"# Scoreboard for {challenge_id}",
        "| Username | Passed Tests | Total Tests |",
        "|----------|--------------|-------------|",
    ]
    lines += [f"| {name} | {passed} | {total} |" for name, passed, total in rows]
    (challenge_dir / "SCOREBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = sys.argv[1:] or [c["id"] for c in list_challenges()]
    for challenge_id in targets:
        print(f"Grading {challenge_id} ...")
        update_one(challenge_id)
    print("Done.")


if __name__ == "__main__":
    main()
