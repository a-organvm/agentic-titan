"""Tests for the advisory AI quality gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / ".ci" / "ai_quality_gate.py"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    _run(["git", "init"], root)
    _run(["git", "config", "user.email", "test@example.invalid"], root)
    _run(["git", "config", "user.name", "Test User"], root)


def _track(root: Path, relative_path: str, text: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _run(["git", "add", relative_path], root)


def test_advisory_gate_reports_duplication_security_and_logic(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    repeated = """
def first() -> int:
    value = 1
    value += 1
    value += 2
    value += 3
    value += 4
    value += 5
    return value
"""
    _track(
        tmp_path,
        "titan/example_a.py",
        repeated
        + """
def bad(flag: bool) -> int:
    if flag and not flag:
        return 1
    return 0
""",
    )
    _track(
        tmp_path,
        "titan/example_b.py",
        repeated.replace("first", "second") + f'\n{"API_" + "KEY"} = "12345678901234567890"\n',
    )

    output = tmp_path / "report.md"
    proc = _run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--window-size",
            "5",
        ],
        tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    report = output.read_text(encoding="utf-8")
    assert "AI Quality Gate Report" in report
    assert "### Duplication" in report
    assert "hardcoded-secret-assignment" in report
    assert "direct self-contradiction" in report
    assert "Advisory mode does not block merge" in report


def test_strict_gate_fails_on_error_findings(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(tmp_path, "titan/secret.py", f'{"pass" + "word"} = "12345678901234567890"\n')

    proc = _run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.md"),
            "--mode",
            "strict",
        ],
        tmp_path,
    )

    assert proc.returncode == 1


def test_clean_repo_has_empty_report(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _track(
        tmp_path,
        "titan/clean.py",
        """
def add(left: int, right: int) -> int:
    return left + right
""",
    )

    output = tmp_path / "report.md"
    proc = _run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        tmp_path,
    )

    assert proc.returncode == 0
    assert "No duplication, security, or logic-risk findings detected." in output.read_text(
        encoding="utf-8"
    )
