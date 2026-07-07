"""Read per-challenge SCOREBOARD.md files and aggregate a main leaderboard.

The web app never writes SCOREBOARD.md itself (same as go-interview-practice):
that file is git-tracked and only ever rewritten by `scripts/update_scoreboards.py`,
run locally or by CI after a merge. That keeps a contributor's PR from ever
touching a shared generated file -- it only ever adds their own submission.

Instead, `app.main.api_save` records a result in `_LIVE` below, an in-memory
overlay that `challenge_scoreboard`/`leaderboard` merge in on read. This makes
a save show up immediately for anyone browsing the same running instance,
without persisting to disk; it resets when the server restarts.
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

# In-memory overlay: challenge_id -> username -> {"username", "passed", "total"}.
# Not persisted -- see the module docstring.
_LIVE: dict[str, dict[str, dict]] = {}


def record_live_result(challenge_id: str, username: str, passed: int, total: int) -> None:
    _LIVE.setdefault(challenge_id, {})[username] = {
        "username": username,
        "passed": passed,
        "total": total,
    }


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
    rows_by_user = {
        r["username"]: r
        for r in parse_scoreboard(REPO_ROOT / challenge_id / "SCOREBOARD.md")
        if r["username"].lower() not in _HIDDEN_USERS
    }
    for username, r in _LIVE.get(challenge_id, {}).items():
        if username.lower() not in _HIDDEN_USERS:
            rows_by_user[username] = r
    rows = list(rows_by_user.values())
    rows.sort(key=lambda r: (-r["passed"], r["username"].lower()))
    return rows


def regenerate_scoreboard(challenge_id: str) -> list[dict]:
    """Grade every submission for `challenge_id` and rewrite its SCOREBOARD.md.

    Used by `scripts/update_scoreboards.py` (local batch runs and CI) -- never
    called from the web app's request handlers.
    """
    from .grader import grade  # local import: keeps the read path grader-free

    challenge_dir = REPO_ROOT / challenge_id
    submissions_root = challenge_dir / "submissions"

    rows: list[dict] = []
    if submissions_root.is_dir():
        for user_dir in sorted(submissions_root.iterdir()):
            solution = user_dir / "solution.py"
            if not solution.is_file():
                continue
            result = grade(challenge_id, solution.read_text(encoding="utf-8"))
            rows.append(
                {"username": user_dir.name, "passed": result["passed"], "total": result["total"]}
            )

    rows.sort(key=lambda r: (-r["passed"], r["username"].lower()))

    lines = [
        f"# Scoreboard for {challenge_id}",
        "| Username | Passed Tests | Total Tests |",
        "|----------|--------------|-------------|",
    ]
    lines += [f"| {r['username']} | {r['passed']} | {r['total']} |" for r in rows]
    (challenge_dir / "SCOREBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
