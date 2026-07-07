"""Discover challenges by scanning `challenge-*` directories in the repo root."""

from __future__ import annotations

import json
import re
from pathlib import Path

import markdown as md

from .config import CHALLENGE_GLOB, REPO_ROOT
from .grader import safe_challenge_dir

_HINT_HEADER = re.compile(r"^##\s+Hint\s+\d+:\s*(.*)$", re.IGNORECASE)


def _to_html(markdown_text: str) -> str:
    return md.markdown(
        markdown_text,
        extensions=["fenced_code", "codehilite", "tables"],
        extension_configs={"codehilite": {"guess_lang": False}},
    )


def _parse_hints(markdown_text: str) -> list[dict]:
    """Split a hints.md file into progressive hints by `## Hint N: Title`
    headers (same convention as go-interview). Content before the first header
    is ignored. Each hint's body is rendered to HTML."""
    hints: list[dict] = []
    current: dict | None = None
    for line in markdown_text.splitlines():
        m = _HINT_HEADER.match(line.strip())
        if m:
            if current:
                hints.append(current)
            current = {"title": m.group(1).strip(), "content": ""}
        elif current is not None:
            current["content"] += line + "\n"
    if current:
        hints.append(current)
    return [{"title": h["title"], "html": _to_html(h["content"].strip())} for h in hints]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _load_meta(challenge_dir: Path) -> dict:
    meta_path = challenge_dir / "metadata.json"
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _summary(challenge_dir: Path) -> dict:
    meta = _load_meta(challenge_dir)
    return {
        "id": challenge_dir.name,
        "title": meta.get("title", challenge_dir.name),
        "shortDescription": meta.get("short_description", ""),
        "difficulty": meta.get("difficulty", "Unknown"),
        "category": meta.get("category", ""),
        "tags": meta.get("tags", []),
        "estimatedTime": meta.get("estimated_time", ""),
        "testCount": meta.get("test_count"),
        "order": meta.get("order", 999),
    }


def list_challenges() -> list[dict]:
    challenges = [
        _summary(d)
        for d in REPO_ROOT.glob(CHALLENGE_GLOB)
        if d.is_dir() and (d / "test_solution.py").is_file()
    ]
    challenges.sort(key=lambda c: (c["order"], c["id"]))
    return challenges


def get_challenge(challenge_id: str) -> dict | None:
    d = safe_challenge_dir(challenge_id)
    if d is None:
        return None

    detail = _summary(d)
    detail.update(
        {
            "template": _read(d / "solution_template.py"),
            "testFile": _read(d / "test_solution.py"),
            "readmeHtml": _to_html(_read(d / "README.md")),
            "learningHtml": _to_html(_read(d / "learning.md")),
            "hintsHtml": _to_html(_read(d / "hints.md")),
            "hints": _parse_hints(_read(d / "hints.md")),
        }
    )
    return detail
