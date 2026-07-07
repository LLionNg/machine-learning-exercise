"""FastAPI application: serves the challenge API and the built frontend."""

from __future__ import annotations

import re
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .challenges import get_challenge, list_challenges
from .config import REPO_ROOT
from .grader import grade, safe_challenge_dir
from .scoreboard import challenge_scoreboard, leaderboard, record_live_result

app = FastAPI(title="ML Practice")

# In dev the Vite server proxies /api, so this is only a convenience.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    challengeId: str
    code: str


class SaveRequest(BaseModel):
    challengeId: str
    username: str
    code: str


@app.get("/api/challenges")
def api_challenges() -> list[dict]:
    return list_challenges()


@app.get("/api/challenges/{challenge_id}")
def api_challenge(challenge_id: str) -> dict:
    challenge = get_challenge(challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    return challenge


@app.post("/api/run")
def api_run(req: RunRequest) -> dict:
    return grade(req.challengeId, req.code)


@app.get("/api/scoreboard/{challenge_id}")
def api_scoreboard(challenge_id: str) -> list[dict]:
    return challenge_scoreboard(challenge_id)


@app.get("/api/leaderboard")
def api_leaderboard() -> dict:
    return leaderboard()


def _git(*args: str) -> str:
    """Run a git command in the repo root; return stdout (empty on failure)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _github_user_from_url(url: str) -> str:
    """Extract the GitHub handle from a remote URL, or '' if it isn't GitHub."""
    for pattern in (r"git@github\.com:([^/]+)/", r"https?://github\.com/([^/]+)/"):
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return ""


@app.get("/api/git-username")
def api_git_username() -> dict:
    """Auto-detect the user's GitHub username (like go-interview-practice).

    A fork's `origin` points at github.com/<handle>/... so that is the most
    reliable source. A user may have several remotes (someone with both GitLab
    and GitHub set up may have a GitLab `origin`), so we scan every remote for a
    github.com URL. Only as a last resort do we use the git-config identity,
    which may not be a GitHub account -- in that case the UI's "Change Username"
    lets the user correct it."""
    # 1. Prefer origin if it is a github.com remote.
    user = _github_user_from_url(_git("remote", "get-url", "origin"))
    if user:
        return {"username": user, "source": "remote-origin", "success": True}
    # 2. Otherwise scan all remotes for any github.com URL.
    for name in _git("remote").splitlines():
        name = name.strip()
        if not name:
            continue
        user = _github_user_from_url(_git("remote", "get-url", name))
        if user:
            return {"username": user, "source": f"remote-{name}", "success": True}
    # 3. Fall back to git config (may not be a GitHub handle).
    name = _git("config", "--get", "user.name")
    if name:
        return {"username": name, "source": "git-config", "success": True}
    email = _git("config", "--get", "user.email")
    if email and "@" in email:
        return {"username": email.split("@", 1)[0], "source": "git-config", "success": True}
    return {"username": "", "source": "not-found", "success": False}


@app.get("/api/user-stats/{username}")
def api_user_stats(username: str) -> dict:
    """Aggregate a single user's progress across all challenges, like
    go-interview's profile dropdown: attempted / completed / average score /
    main-scoreboard rank."""
    challenges = list_challenges()
    passed_total: list[tuple[int, int]] = []
    for c in challenges:
        for r in challenge_scoreboard(c["id"]):
            if r["username"] == username:
                passed_total.append((r["passed"], r["total"]))
                break
    attempted = len(passed_total)
    completed = sum(1 for p, t in passed_total if t > 0 and p == t)
    scores = [round(p / t * 100) for p, t in passed_total if t > 0]
    avg_score = round(sum(scores) / len(scores)) if scores else 0
    board = leaderboard()["users"]
    rank = next((i + 1 for i, u in enumerate(board) if u["username"] == username), 0)
    return {
        "attempted": attempted,
        "completed": completed,
        "avgScore": avg_score,
        "rank": rank,
        "totalChallenges": len(challenges),
        "success": True,
    }


@app.get("/api/submission/{challenge_id}/{username}")
def api_get_submission(challenge_id: str, username: str) -> dict:
    """Return a previously-saved submission's code so the editor can restore it
    (like go-interview loading an existing solution). Empty if none exists."""
    d = safe_challenge_dir(challenge_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    uname = "".join(ch for ch in username if ch.isalnum() or ch in "-_")
    solution = d / "submissions" / uname / "solution.py"
    if uname and solution.is_file():
        return {"code": solution.read_text(encoding="utf-8"), "exists": True}
    return {"code": "", "exists": False}


@app.post("/api/save")
def api_save(req: SaveRequest) -> dict:
    d = safe_challenge_dir(req.challengeId)
    if d is None:
        raise HTTPException(status_code=404, detail="Challenge not found")
    username = "".join(ch for ch in req.username if ch.isalnum() or ch in "-_")
    if not username:
        raise HTTPException(status_code=400, detail="Invalid username")
    submission_dir = d / "submissions" / username
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "solution.py").write_text(req.code, encoding="utf-8")
    # Grade and record the result in the in-memory live scoreboard, so it's
    # visible immediately to anyone browsing this running instance. The
    # git-tracked SCOREBOARD.md is untouched -- it's only rewritten by
    # scripts/update_scoreboards.py, so a PR never needs to include it.
    result = grade(req.challengeId, req.code)
    record_live_result(req.challengeId, username, result["passed"], result["total"])
    rel = f"{req.challengeId}/submissions/{username}/solution.py"
    return {
        "success": True,
        "filePath": rel,
        "gitCommands": [
            f"git add {rel}",
            f'git commit -m "Add {req.challengeId} solution by {username}"',
            "git push origin main",
        ],
    }


# --------------------------------------------------------------------------- #
# Serve the built frontend (frontend/dist) if it exists. In development the
# frontend runs on the Vite dev server instead.
# --------------------------------------------------------------------------- #
_DIST = REPO_ROOT / "frontend" / "dist"
if _DIST.is_dir():
    if (_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}")
    def _spa(path: str) -> FileResponse:
        candidate = _DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")  # SPA fallback
