#!/usr/bin/env python3
"""Advisory quality gate for AI-assisted code changes.

The gate intentionally uses only the Python standard library so it can run in
pull-request CI without adding another toolchain. It reports three classes of
findings:

- duplicated normalized code windows
- security-sensitive patterns common in generated code
- simple logic-risk AST patterns
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / ".ci" / "ai_quality_gate_report.md"

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx"}
PYTHON_SUFFIXES = {".py"}
EXCLUDED_PREFIXES = (
    ".git/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".ci/allow_secret_baseline.txt",
    ".ci/ai_quality_gate_report.md",
    ".ci/baseline_",
    ".ci/current_",
    ".venv/",
    "__pycache__/",
    "htmlcov/",
)
EXCLUDED_PATH_PARTS = {"__pycache__", "node_modules", "htmlcov"}
SECURITY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "hardcoded-secret-assignment",
        "ERROR",
        r"(?i)\b(api[_-]?key|secret|password|token|credential)\b\s*[:=]\s*['\"][^'\"\n]{12,}['\"]",
    ),
    (
        "shell-true",
        "WARNING",
        r"\bsubprocess\.(run|call|Popen|check_call|check_output)\([^)\n]*shell\s*=\s*True",
    ),
    ("eval-call", "WARNING", r"\beval\s*\("),
    ("pickle-load", "WARNING", r"\bpickle\.loads?\s*\("),
)


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    path: str
    line: int
    message: str


def _run_git(root: Path, args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def list_tracked_files(root: Path) -> list[Path]:
    files = [Path(line) for line in _run_git(root, ["ls-files"]).splitlines() if line.strip()]
    return [path for path in files if should_scan(path)]


def should_scan(path: Path) -> bool:
    posix_path = path.as_posix()
    if any(posix_path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return False
    return path.suffix in TEXT_SUFFIXES


def read_text(root: Path, path: Path) -> str | None:
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def normalize_code_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("#", "//", "/*", "*", "*/")):
        return None
    return re.sub(r"\s+", " ", stripped)


def collect_duplication_findings(
    root: Path,
    files: Iterable[Path],
    *,
    window_size: int,
    max_findings: int,
) -> list[Finding]:
    seen: dict[str, tuple[Path, int, str]] = {}
    findings: list[Finding] = []

    for path in files:
        if path.suffix not in CODE_SUFFIXES or path.parts[0] == "tests":
            continue
        text = read_text(root, path)
        if text is None:
            continue

        normalized: list[tuple[int, str]] = []
        for index, line in enumerate(text.splitlines(), start=1):
            normalized_line = normalize_code_line(line)
            if normalized_line is not None:
                normalized.append((index, normalized_line))

        if len(normalized) < window_size:
            continue

        for start in range(0, len(normalized) - window_size + 1):
            line_number = normalized[start][0]
            window = [line for _, line in normalized[start : start + window_size]]
            digest = hashlib.sha256("\n".join(window).encode("utf-8")).hexdigest()
            first = seen.get(digest)
            if first is None:
                seen[digest] = (path, line_number, window[0])
                continue
            first_path, first_line, first_preview = first
            if first_path == path and abs(first_line - line_number) < window_size:
                continue
            findings.append(
                Finding(
                    category="duplication",
                    severity="WARNING",
                    path=path.as_posix(),
                    line=line_number,
                    message=(
                        f"{window_size}-line normalized block duplicates "
                        f"{first_path.as_posix()}:{first_line} ({first_preview[:80]})"
                    ),
                )
            )
            if len(findings) >= max_findings:
                return findings

    return findings


def collect_security_findings(
    root: Path,
    files: Iterable[Path],
    *,
    max_findings: int,
) -> list[Finding]:
    findings: list[Finding] = []

    for path in files:
        if path.parts[0] in {"docs", "tests"}:
            continue
        text = read_text(root, path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_name, severity, pattern in SECURITY_PATTERNS:
                if re.search(pattern, line):
                    findings.append(
                        Finding(
                            category="security",
                            severity=severity,
                            path=path.as_posix(),
                            line=line_number,
                            message=f"{pattern_name}: review generated code for unsafe default",
                        )
                    )
                    if len(findings) >= max_findings:
                        return findings

    return findings


class LogicRiskVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            if not self._contains_exit(node.body):
                self.findings.append(
                    Finding(
                        category="logic",
                        severity="WARNING",
                        path=self.path.as_posix(),
                        line=node.lineno,
                        message="while True block has no obvious break, return, or raise",
                    )
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test = ast.dump(node.test, include_attributes=False)
        if self._has_contradictory_self_check(node.test):
            self.findings.append(
                Finding(
                    category="logic",
                    severity="WARNING",
                    path=self.path.as_posix(),
                    line=node.lineno,
                    message="condition contains a direct self-contradiction",
                )
            )
        if node.body and node.orelse:
            body_dump = [ast.dump(stmt, include_attributes=False) for stmt in node.body]
            else_dump = [ast.dump(stmt, include_attributes=False) for stmt in node.orelse]
            if body_dump == else_dump:
                self.findings.append(
                    Finding(
                        category="logic",
                        severity="WARNING",
                        path=self.path.as_posix(),
                        line=node.lineno,
                        message=f"if/else branches are identical for {test[:80]}",
                    )
                )
        self.generic_visit(node)

    @staticmethod
    def _contains_exit(nodes: Sequence[ast.stmt]) -> bool:
        for node in ast.walk(ast.Module(body=list(nodes), type_ignores=[])):
            if isinstance(node, ast.Break | ast.Return | ast.Raise):
                return True
        return False

    @staticmethod
    def _has_contradictory_self_check(node: ast.AST) -> bool:
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
            return False

        positive: set[str] = set()
        negative: set[str] = set()
        for value in node.values:
            if isinstance(value, ast.Name):
                positive.add(value.id)
            elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
                if isinstance(value.operand, ast.Name):
                    negative.add(value.operand.id)
        return bool(positive & negative)


def collect_logic_findings(
    root: Path,
    files: Iterable[Path],
    *,
    max_findings: int,
) -> list[Finding]:
    findings: list[Finding] = []

    for path in files:
        if path.suffix not in PYTHON_SUFFIXES or path.parts[0] == "tests":
            continue
        text = read_text(root, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text, filename=path.as_posix())
        except SyntaxError as exc:
            findings.append(
                Finding(
                    category="logic",
                    severity="ERROR",
                    path=path.as_posix(),
                    line=exc.lineno or 1,
                    message=f"syntax error prevents logic scan: {exc.msg}",
                )
            )
            continue
        visitor = LogicRiskVisitor(path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        if len(findings) >= max_findings:
            return findings[:max_findings]

    return findings


def render_report(findings: Sequence[Finding], mode: str) -> str:
    categories = ("duplication", "security", "logic")
    severities = ("ERROR", "WARNING")
    counts = {
        category: {
            severity: sum(
                1
                for finding in findings
                if finding.category == category and finding.severity == severity
            )
            for severity in severities
        }
        for category in categories
    }

    lines = [
        "<!-- ai-quality-gate-report -->",
        "## AI Quality Gate Report",
        "",
        f"Mode: `{mode}`",
        "",
        "| Category | Errors | Warnings |",
        "| --- | ---: | ---: |",
    ]
    for category in categories:
        error_count = counts[category]["ERROR"]
        warning_count = counts[category]["WARNING"]
        lines.append(f"| {category.title()} | {error_count} | {warning_count} |")

    if not findings:
        lines.extend(["", "No duplication, security, or logic-risk findings detected."])
        return "\n".join(lines) + "\n"

    for category in categories:
        category_findings = [finding for finding in findings if finding.category == category]
        if not category_findings:
            continue
        lines.extend(["", f"### {category.title()}"])
        for finding in category_findings:
            lines.append(
                f"- `{finding.severity}` `{finding.path}:{finding.line}` - {finding.message}"
            )

    if mode == "advisory":
        lines.extend(
            [
                "",
                (
                    "Advisory mode does not block merge. "
                    "Review findings before strict mode is enabled."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def run_gate(
    *,
    root: Path,
    mode: str,
    output: Path,
    window_size: int,
    max_findings: int,
) -> int:
    files = list_tracked_files(root)
    findings = [
        *collect_duplication_findings(
            root,
            files,
            window_size=window_size,
            max_findings=max_findings,
        ),
        *collect_security_findings(root, files, max_findings=max_findings),
        *collect_logic_findings(root, files, max_findings=max_findings),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(findings, mode)
    output.write_text(report, encoding="utf-8")
    print(report)

    if mode == "strict" and any(finding.severity == "ERROR" for finding in findings):
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("advisory", "strict"), default="advisory")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-findings", type=int, default=50)
    args = parser.parse_args()

    if args.window_size < 3:
        parser.error("--window-size must be at least 3")
    if args.max_findings < 1:
        parser.error("--max-findings must be at least 1")

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    return run_gate(
        root=root,
        mode=args.mode,
        output=output,
        window_size=args.window_size,
        max_findings=args.max_findings,
    )


if __name__ == "__main__":
    sys.exit(main())
