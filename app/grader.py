"""The grader: run a submission against a challenge's canonical tests.

Mirrors the local `run_tests.sh` flow -- copy the participant's `solution.py`
next to the challenge's `test_solution.py` in a throwaway directory and run
`pytest` -- but in-process so the web UI and scoreboard scripts can reuse it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import GRADER_TIMEOUT, REPO_ROOT


def safe_challenge_dir(challenge_id: str) -> Path | None:
    """Resolve a challenge id to its directory, rejecting path traversal."""
    if "/" in challenge_id or "\\" in challenge_id or ".." in challenge_id:
        return None
    if not challenge_id.startswith("challenge-"):
        return None
    d = REPO_ROOT / challenge_id
    if d.parent != REPO_ROOT or not (d / "test_solution.py").is_file():
        return None
    return d


def _parse_junit(xml_path: Path) -> tuple[int, int, int]:
    """Return (passed, total, failed) from a pytest junit-xml report."""
    if not xml_path.is_file():
        return 0, 0, 0
    root = ET.parse(xml_path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return 0, 0, 0
    total = int(suite.get("tests", 0))
    failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failed - skipped
    return passed, total, failed


def grade(challenge_id: str, code: str) -> dict:
    """Run `code` against the challenge's tests. Returns a result dict."""
    d = safe_challenge_dir(challenge_id)
    if d is None:
        return {
            "ok": False, "passed": 0, "total": 0, "failed": 0,
            "durationMs": 0, "output": f"Unknown challenge: {challenge_id!r}",
        }

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="mlx-grade-") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        shutil.copy(d / "test_solution.py", tmp / "test_solution.py")
        xml = tmp / "report.xml"
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-q",
                    "-p", "no:cacheprovider",
                    "--color=no",
                    "--override-ini=addopts=",
                    "--junitxml", str(xml),
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=GRADER_TIMEOUT,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            return {
                "ok": False, "passed": 0, "total": 0, "failed": 0,
                "durationMs": int((time.time() - start) * 1000),
                "output": f"Timed out after {GRADER_TIMEOUT}s (possible infinite loop).",
            }
        passed, total, failed = _parse_junit(xml)

    return {
        "ok": total > 0 and passed == total,
        "passed": passed,
        "total": total,
        "failed": failed,
        "durationMs": int((time.time() - start) * 1000),
        "output": output.strip(),
    }
