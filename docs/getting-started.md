# Getting Started

This guide is the short path from a fresh checkout to a useful local
development loop for Agentic Titan.

## 1. Prepare Python

Use Python 3.11 or newer. The CI matrix covers Python 3.11 and 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,dashboard,auth,ratelimit]"
```

If you use `uv`, the equivalent local command is:

```bash
uv run --python 3.11 --with '.[dev,dashboard,auth,ratelimit]' pytest
```

## 2. Run The Quality Loop

Run these before opening a pull request:

```bash
ruff check .
mypy .
pytest
```

For the same coverage mode used in CI:

```bash
pytest --cov=. --cov-report=xml --cov-report=term --cov-report=html
```

## 3. Work In The Right Layer

The repo is organized as root-level Python packages:

| Path | Purpose |
| --- | --- |
| `titan/` | CLI, API, orchestration, safety, learning, and workflow code. |
| `hive/` | Collective behavior primitives: memory, topology, stigmergy, and experiments. |
| `agents/` | Agent archetypes and reusable agent behaviors. |
| `adapters/` | LLM provider adapters and routing. |
| `runtime/` | Local and isolated execution backends. |
| `dashboard/` | FastAPI dashboard and templates. |
| `.ci/` | Local governance and quality scripts used by CI. |

## 4. Choose The Smallest Verification Set First

For a narrow fix, start with the relevant test file:

```bash
pytest tests/test_hive/test_perceptual_gating.py
pytest tests/ci/test_ai_quality_gate.py
```

Then run the broader gates before pushing:

```bash
ruff check .
mypy .
pytest --cov=. --cov-report=xml --cov-report=term --cov-report=html
```

## 5. Respect The Governance Boundary

This repo participates in ORGAN-IV orchestration. Keep dependencies flowing in
the approved direction, do not commit secrets, and do not bypass the
branch-protection dependency validation status. If a pull request is blocked on
`validate-dependencies`, run the canonical dependency validator and publish a
status only after it passes.
