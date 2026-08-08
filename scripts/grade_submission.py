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


def find_submission(challenge_id: str, username: str) -> Path | None:
    """Locate <challenge_id>/submissions/<username>/solution.py, ignoring case.

    GitHub logins are case-insensitive, and pr-tests.yml compares the submission
    directory to the PR author case-insensitively too. Resolving the path with
    exact case here would make a directory like `submissions/octocat` invisible
    to an author whose login renders as `Octocat`: the grader would report
    "nothing to grade", exit 0, and the PR would go green without ever being
    tested. Match the same way CI validates.
    """
    submissions = REPO_ROOT / challenge_id / "submissions"
    if not submissions.is_dir():
        return None

    exact = submissions / username / "solution.py"
    if exact.is_file():
        return exact

    target = username.lower()
    for user_dir in sorted(submissions.iterdir()):
        if not user_dir.is_dir() or user_dir.name.lower() != target:
            continue
        solution = user_dir / "solution.py"
        if solution.is_file():
            return solution
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: grade_submission.py <challenge_id> <username>", file=sys.stderr)
        return 2

    challenge_id, username = sys.argv[1], sys.argv[2]
    solution = find_submission(challenge_id, username)
    if solution is None:
        print(
            f"No solution.py for {username!r} under "
            f"{REPO_ROOT / challenge_id / 'submissions'}, nothing to grade."
        )
        return 0

    print(f"Grading {solution.relative_to(REPO_ROOT).as_posix()}")
    result = grade(challenge_id, solution.read_text(encoding="utf-8"))
    print(result["output"])
    print(f"\n{result['passed']}/{result['total']} tests passed")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
