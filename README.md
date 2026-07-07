# Machine Learning Practice

This repository is inspired by [RezaSi's Golang interview practice](https://github.com/RezaSi/go-interview-practice)

Hands-on coding challenges for **Python ML/DL**. Each challenge ships with a
starter template, a canonical test suite, learning material, and hints. You
implement the solution, run the tests locally, and (later) submit it for the
leaderboard - a self-hosted, fork-and-PR practice platform in the spirit of
[go-interview-practice](https://github.com/RezaSi/go-interview-practice), for
machine learning.

## Challenges

| # | Challenge | Topic | Difficulty |
|---|---|---|---|
| 1 | [Pydantic Models for an ML Training Config](challenge-1-pydantic-ml-config) | Pydantic v2 validation | Beginner |
| 2 | [PyTorch Tensor Operations](challenge-2-tensor-operations) | Tensors: shape, indexing, broadcasting, matmul | Beginner |
| 3 | [A Small Neural Network in PyTorch](challenge-3-pytorch-neural-network) | PyTorch `nn.Module`, training loop | Beginner-Intermediate |

*More challenges coming.*

## How a challenge is structured

```
challenge-N-<slug>/
|-- README.md              # problem statement
|-- metadata.json          # title, difficulty, tags, package requirements
|-- solution_template.py   # skeleton you edit (has TODOs)
|-- test_solution.py       # canonical tests - the source of truth (don't edit)
|-- learning.md            # concept primer with references to official docs
|-- hints.md               # progressive hints
|-- run_tests.sh           # local grader (uv run pytest)
`-- submissions/           # one folder per user
    `-- <username>/solution.py
```

## Solve a challenge

```bash
# from the repo root, once: create the environment with uv
uv sync

cd challenge-1-pydantic-ml-config

# set up your submission
mkdir -p submissions/<your-username>
cp solution_template.py submissions/<your-username>/solution.py

# implement submissions/<your-username>/solution.py, then run the tests
./run_tests.sh            # enter your username when prompted
```

The grader copies your `solution.py` next to `test_solution.py` in a temporary
directory and runs `pytest` - your solution passes when all tests pass.

## Web UI

A local web app (FastAPI + React/Vite) lets you browse challenges, edit and run
code in the browser, and watch the leaderboard.

```bash
# 0. From the repo root, install Python deps with uv (creates .venv)
uv sync

# 1. Backend - serves the API (and the built UI) on :20700
uv run uvicorn app.main:app --port 20700 --reload

# 2a. Frontend in dev mode - hot reload on :5173, proxies /api to :20700
cd frontend && npm install && npm run dev      # open http://localhost:5173

# 2b. ...or build once and let the backend serve everything from :20700
cd frontend && npm install && npm run build    # open http://localhost:20700
```

Regenerate the scoreboards (grades every submission and rewrites each
`SCOREBOARD.md`; the API aggregates those into the leaderboard live):

```bash
uv run python scripts/update_scoreboards.py                             # all challenges
uv run python scripts/update_scoreboards.py challenge-3-pytorch-neural-network  # one
```

### Platform layout

```
app/            FastAPI backend
  challenges.py   discover challenge-* dirs, render markdown
  grader.py       run a submission against tests in a temp dir (pytest)
  scoreboard.py   read SCOREBOARD.md files, aggregate the leaderboard
  main.py         API routes + serves frontend/dist
frontend/       React + Vite + Tailwind + shadcn/ui (neutral theme, light/dark)
scripts/        update_scoreboards.py (batch grader), grade_submission.py (one-user CI grader)
```

## Contribution flow (CI)

Like [go-interview-practice](https://github.com/RezaSi/go-interview-practice), submissions
land via fork + PR and the repo automates the rest with GitHub Actions:

- **`.github/workflows/pr-tests.yml`** - on every PR to `main`: checks the PR only touches
  `challenge-*/submissions/<pr-author>/` (files elsewhere need a maintainer to add the
  `manual-approval-granted` label), then runs `scripts/grade_submission.py` for each changed
  challenge. The final `pr-status` job is the single check to require in branch protection.
- **`.github/workflows/auto-merge.yml`** - every 2 hours, squash-merges any open PR whose
  `pr-status` check has been green for 2+ days (tweak `WAIT_MS` in the workflow to change the
  wait), then regenerates scoreboards for the challenges it touched.
- **`.github/workflows/update-scoreboards.yml`** - on any direct push to `main` under
  `challenge-*/submissions/**` (e.g. a maintainer merging manually), regenerates the affected
  `SCOREBOARD.md` files via `scripts/update_scoreboards.py` and commits them back.

All three commit as `github-actions[bot]`; no extra secrets are needed beyond the default
`GITHUB_TOKEN`.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node 18+ (only for the web UI frontend)

`uv sync` installs everything (fastapi, a CPU build of torch, pydantic, numpy,
pytest, ...) into `.venv`. All challenges are **CPU-only** and run in seconds; no
GPU required.
