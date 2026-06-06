# Contributing to Agentic Titan

Polymorphic multi-agent orchestration framework. Python 3.11+, MIT licensed.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/agentic-titan.git
cd agentic-titan
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For optional extras (dashboard, metrics, vector-db, etc.), see `pyproject.toml`.

For a fuller first-run workflow, see [docs/getting-started.md](docs/getting-started.md).

## Tests

```bash
pytest                          # all tests (asyncio_mode = "auto" — no decorator needed)
pytest tests/integration/       # integration only
pytest -k "test_topology"       # pattern match
pytest --cov=. --cov-report=term  # with coverage
```

Async tests run automatically — `asyncio_mode = "auto"` is configured in `pyproject.toml`, so you do not need `@pytest.mark.asyncio`.

## Linting

```bash
ruff check .                    # lint (line-length=100, target=py311)
ruff check . --fix              # auto-fix
mypy .                          # strict type checking
```

Ruff rules: `["E", "F", "I", "N", "W", "UP"]` — errors, pyflakes, isort, naming, warnings, pyupgrade.

## Package Layout

Top-level packages (not under `src/`):

```
titan/          # CLI entry point (titan.cli:main)
agents/         # Agent archetypes and definitions
hive/           # Hive mind — fission/fusion, learning, memory, pheromone fields
adapters/       # LLM adapters (Ollama, Anthropic, OpenAI, Groq)
runtime/        # Runtime fabric (local, Docker, OpenFaaS, Firecracker)
orchestrator/   # Topology engine — 9 coordination patterns
dashboard/      # FastAPI dashboard
```

Imports resolve from the repo root (`pythonpath = ["."]` in pytest config).

## Code Style

- PEP 8, enforced by ruff
- Type hints on all function signatures
- Pydantic models for data structures
- Docstrings on public APIs
- Line length: 100 characters

## Submitting a PR

1. Fork and create a branch: `git checkout -b feat/your-feature`
2. Make changes, add tests
3. Run `ruff check . && mypy . && pytest` — all three must pass
4. Commit with imperative mood: `feat: add cooldown to fission-fusion transitions`
5. Push and open a PR against `main`
6. Reference the issue number in the PR description

One feature or fix per PR. Large changes should be discussed in an issue first.

## Good First Issues

Issues labeled [`good first issue`](https://github.com/organvm-iv-taxis/agentic-titan/labels/good%20first%20issue) are self-contained tasks with clear scope. Each issue body includes the relevant file paths and a concrete proposal.

## Questions

Open an issue or check [orchestration-start-here](https://github.com/organvm-iv-taxis/orchestration-start-here) for system-wide context.
