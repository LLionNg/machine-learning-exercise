#!/usr/bin/env python3
"""Grade every submission and (re)write each challenge's SCOREBOARD.md.

Usage:
    python scripts/update_scoreboards.py                 # all challenges
    python scripts/update_scoreboards.py challenge-1-... # specific challenges

The web app calls the same grading path synchronously after a save (see
`app.scoreboard.regenerate_scoreboard`), so this script is mainly for batch/CI
regeneration (e.g. after several submissions land via PR merges).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.challenges import list_challenges  # noqa: E402
from app.scoreboard import regenerate_scoreboard  # noqa: E402


def main() -> None:
    targets = sys.argv[1:] or [c["id"] for c in list_challenges()]
    for challenge_id in targets:
        print(f"Grading {challenge_id} ...")
        for r in regenerate_scoreboard(challenge_id):
            status = "PASS" if r["total"] > 0 and r["passed"] == r["total"] else "fail"
            print(f"  [{status}] {r['username']}: {r['passed']}/{r['total']}")
    print("Done.")


if __name__ == "__main__":
    main()
