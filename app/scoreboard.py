"""Read per-challenge SCOREBOARD.md files and aggregate a main leaderboard.

Writing the scoreboards (which runs the grader on every submission) lives in
`scripts/update_scoreboards.py`; this module only *reads* them, so the API stays
fast.
"""

from __future__ import annotations

import re
from pathlib import Path

from .challenges import list_challenges
from .config import REPO_ROOT

_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")

# Pseudo-users that hold the maintainer's answer-key solutions -- kept for
# grading/verification but hidden from the public leaderboard and scoreboards.
_HIDDEN_USERS = {"reference"}


def parse_scoreboard(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "Username" in stripped or "---" in stripped or stripped.startswith("#"):
            continue
        m = _ROW_RE.match(stripped)
        if m:
            rows.append(
                {"username": m.group(1).strip(), "passed": int(m.group(2)), "total": int(m.group(3))}
            )
    return rows


def challenge_scoreboard(challenge_id: str) -> list[dict]:
    rows = [
        r
        for r in parse_scoreboard(REPO_ROOT / challenge_id / "SCOREBOARD.md")
        if r["username"].lower() not in _HIDDEN_USERS
    ]
    rows.sort(key=lambda r: (-r["passed"], r["username"].lower()))
    return rows


def leaderboard() -> dict:
    challenges = list_challenges()
    users: dict[str, dict] = {}
    for c in challenges:
        for r in challenge_scoreboard(c["id"]):
            u = users.setdefault(
                r["username"], {"username": r["username"], "solved": 0, "challenges": {}}
            )
            completed = r["total"] > 0 and r["passed"] == r["total"]
            u["challenges"][c["id"]] = completed
            if completed:
                u["solved"] += 1
    board = sorted(users.values(), key=lambda u: (-u["solved"], u["username"].lower()))
    return {"totalChallenges": len(challenges), "users": board}
