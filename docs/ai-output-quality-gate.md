# F-66: AI Output Quality Gate in CI

> CI step targeting AI-generated code with duplication, security, and logic error detection.

## Motivation

AI-generated code passes syntax checks and often passes tests, but can introduce subtle issues: duplicated logic that already exists elsewhere in the codebase, security anti-patterns, unreachable code, and logic errors that tests don't cover. A dedicated quality gate catches these before merge.

## Gate Architecture

```
PR opened/updated
  │
  ├── 1. Standard CI (tests, lint, typecheck)
  │
  └── 2. AI Quality Gate (runs after tests pass)
       ├── Duplication detection
       ├── Security scanning
       ├── Logic error detection
       └── Report (advisory or blocking)
```

The quality gate runs as a separate GitHub Actions job that depends on the test job passing first. This ensures we only spend compute on code that is otherwise valid.

## Detection Categories

### 1. Duplication Detection

**Problem**: AI models frequently regenerate logic that already exists in the codebase, creating maintenance burden and inconsistency.

**Tools**:
- `jscpd` — copy/paste detector for any language
- `ruff` (Python) — detects some duplicate patterns
- Custom: compare new functions against existing function signatures

**Configuration**:

```yaml
# .jscpd.json
{
  "threshold": 20,
  "reporters": ["json", "github"],
  "ignore": ["**/test_*", "**/*.test.*", "**/node_modules/**"],
  "minLines": 5,
  "minTokens": 50
}
```

**CI Step**:

```yaml
- name: Duplication check
  run: |
    npx jscpd --config .jscpd.json --output reports/duplication.json
    python3 scripts/check-duplication.py reports/duplication.json --threshold 20
```

### 2. Security Scanning

**Problem**: AI models can generate code with security vulnerabilities — SQL injection, command injection, hardcoded credentials, insecure defaults.

**Tools**:
- `semgrep` — pattern-based security scanner (supports Python, TypeScript, Go)
- `ruff` — catches some security issues (S rules)
- `eslint-plugin-security` — TypeScript/JavaScript security patterns
- Custom: credential pattern detection (`sk-`, `ghp_`, `AKIA`, API keys)

**Semgrep Rules**:

```yaml
# .semgrep/ai-security.yaml
rules:
  - id: hardcoded-secret
    patterns:
      - pattern: |
          $KEY = "..."
      - metavariable-regex:
          metavariable: $KEY
          regex: (api_key|secret|password|token|credential)
    message: "Possible hardcoded secret in AI-generated code"
    severity: ERROR

  - id: shell-injection
    pattern: |
      subprocess.run($CMD, shell=True, ...)
    message: "Shell injection risk — use list form instead"
    severity: WARNING

  - id: sql-injection
    pattern: |
      cursor.execute(f"...")
    message: "Possible SQL injection — use parameterized queries"
    severity: ERROR
```

**CI Step**:

```yaml
- name: Security scan
  run: |
    semgrep --config .semgrep/ --json --output reports/security.json
    semgrep --config p/python --json --output reports/security-community.json
```

### 3. Logic Error Detection

**Problem**: AI-generated code can contain subtle logic errors that pass tests but fail in edge cases.

**Checks**:

| Check | Tool | Language |
|-------|------|----------|
| Unreachable code | `ruff` (F811, F841), `eslint` (no-unreachable) | Python, TypeScript |
| Infinite loops | Custom AST analysis | Python |
| Off-by-one | Custom: range/slice boundary checks | Python |
| Unused imports/variables | `ruff` (F401, F841), `tsc --noUnusedLocals` | Python, TypeScript |
| Type narrowing gaps | `mypy --strict` | Python |
| Null/undefined access | `tsc --strictNullChecks` | TypeScript |
| Dead branches | `semgrep` pattern matching | Both |

**Custom Logic Checker**:

```python
# scripts/check-logic-errors.py
import ast
import sys

class LogicErrorVisitor(ast.NodeVisitor):
    def visit_For(self, node):
        """Detect potentially infinite loops."""
        # Check for missing break/return in while True patterns
        ...

    def visit_If(self, node):
        """Detect contradictory conditions."""
        # Check for `if x and not x` patterns
        ...

    def visit_Subscript(self, node):
        """Flag hardcoded indices that may be off-by-one."""
        ...
```

## Operating Modes

### Advisory Mode (Default)

The quality gate reports findings as PR comments but does not block merge:

```yaml
- name: AI Quality Gate (Advisory)
  if: always()
  run: |
    python3 scripts/ai-quality-report.py \
      --duplication reports/duplication.json \
      --security reports/security.json \
      --logic reports/logic.json \
      --mode advisory \
      --output reports/quality-gate.md

- name: Post PR Comment
  uses: actions/github-script@v7
  with:
    script: |
      const report = fs.readFileSync('reports/quality-gate.md', 'utf8');
      github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body: report
      });
```

### Strict Mode (Flagships)

For flagship repos (`orchestration-start-here`, `agentic-titan`), the gate blocks merge on ERROR-level findings:

```yaml
- name: AI Quality Gate (Strict)
  if: always()
  run: |
    python3 scripts/ai-quality-report.py \
      --duplication reports/duplication.json \
      --security reports/security.json \
      --logic reports/logic.json \
      --mode strict \
      --output reports/quality-gate.md
    # Exit code 1 if any ERROR-level findings
```

### Mode Selection

```yaml
# In the workflow
env:
  QUALITY_GATE_MODE: ${{ contains(fromJSON('["orchestration-start-here", "agentic-titan"]'), github.event.repository.name) && 'strict' || 'advisory' }}
```

## Report Format

```markdown
## AI Quality Gate Report

### Summary
| Category | Findings | Severity |
|----------|----------|----------|
| Duplication | 2 | WARNING |
| Security | 0 | — |
| Logic | 1 | WARNING |

### Duplication (2 findings)
- `src/utils.py:45-62` duplicates `src/helpers.py:12-29` (78% similarity)
- `src/api/routes.py:100-115` duplicates `src/api/handlers.py:30-45` (85% similarity)

### Logic (1 finding)
- `src/processor.py:88` — Loop variable `i` shadows outer scope variable (ruff F811)

**Mode**: Advisory — these findings do not block merge.
```

## GitHub Action Workflow

```yaml
# .github/workflows/ai-quality-gate.yml
name: AI Quality Gate

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/ -v

  quality-gate:
    needs: tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for duplication comparison

      - name: Install tools
        run: |
          pip install semgrep ruff
          npm install -g jscpd

      - name: Duplication check
        run: npx jscpd --config .jscpd.json --output reports/duplication.json
        continue-on-error: true

      - name: Security scan
        run: semgrep --config .semgrep/ --json --output reports/security.json
        continue-on-error: true

      - name: Logic check
        run: |
          ruff check . --output-format json > reports/ruff.json || true
          python3 scripts/check-logic-errors.py --output reports/logic.json

      - name: Generate report
        run: |
          python3 scripts/ai-quality-report.py \
            --duplication reports/duplication.json \
            --security reports/security.json \
            --logic reports/logic.json \
            --mode ${{ env.QUALITY_GATE_MODE }} \
            --output reports/quality-gate.md

      - name: Post PR comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('reports/quality-gate.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: report
            });
```

## Tuning and False Positives

- **Duplication threshold**: Start at 20% similarity, tune based on false positive rate
- **Semgrep rules**: Start with community rulesets, add custom rules as patterns emerge
- **Ignore patterns**: Test files, generated code, vendored dependencies
- **Baseline**: On first run, establish baseline findings count; only flag new findings in PRs

## References

- `ci-quality-gates.md` — existing CI quality gates documentation
- `ci-governance-ownership.md` — CI governance model
- `removable-orchestration-layers.md` (F-18) — quality gate is a removable layer
- `cost-latency-monitoring.md` (F-31) — quality gate findings correlate with AI output quality proxy
