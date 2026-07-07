"""Shared configuration and paths for the ML Practice platform."""

from pathlib import Path

# Repo root is the parent of the `app/` package.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Challenge directories are `challenge-*` folders in the repo root.
CHALLENGE_GLOB = "challenge-*"

# Hard cap on how long a single grading run may take (seconds). Guards against
# infinite loops in submitted code.
GRADER_TIMEOUT = 60
