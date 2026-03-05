# CLAUDE.md — agentic-titan

**ORGAN IV** (Orchestration) · `organvm-iv-taxis/agentic-titan`
**Status:** ACTIVE · **Branch:** `main`

## What This Repo Is

Polymorphic Agent Swarm Architecture: model-agnostic, self-organizing multi-agent system with 6 topologies, 1,095+ tests (adversarial, chaos, e2e, integration, performance, MCP, Ray), 18 completed phases.

## Stack

**Languages:** Python, HTML, JavaScript
**Build:** Python (pip/setuptools)
**Testing:** pytest (likely)

## Directory Structure

```
📁 .ci/
📁 .github/
📁 .meta/
📁 .serena/
📁 .stress-results/
📁 absorb-alchemize/
📁 adapters/
📁 agents/
📁 alembic/
📁 dashboard/
📁 data/
📁 demos/
📁 deploy/
📁 docs/
    adr
    api.md
    ci-governance-ownership.md
    ci-quality-gates.md
    claude-code-setup.md
    deploy-smoke-evidence.md
    deploy-smoke-runbook.md
    release-approver-signoff.md
    release-closure-checklist.md
    release-evidence-template.md
    ... (12 items)
📁 examples/
📁 hive/
📁 mcp/
📁 plans/
📁 runtime/
📁 specs/
📁 tests/
    adversarial
    analysis
    api
    archetypes
    auth
    batch
    chaos
    conftest.py
    e2e
    evaluation
    ... (20 items)
📁 titan/
📁 tools/
  .dbxignore
  .gitignore
  CHANGELOG.md
  GEMINI.md
  LICENSE
  README.md
  alembic.ini
  mypy.ini
  pyproject.toml
  renovate.json
  seed.yaml
```

## Key Files

- `README.md` — Project documentation
- `pyproject.toml` — Python project config
- `seed.yaml` — ORGANVM orchestration metadata
- `tests/` — Test suite

## Development

```bash
pip install -e .    # Install in development mode
pytest              # Run tests
```

## ORGANVM Context

This repository is part of the **ORGANVM** eight-organ creative-institutional system.
It belongs to **ORGAN IV (Orchestration)** under the `organvm-iv-taxis` GitHub organization.

**Dependencies:**
- organvm-i-theoria/recursive-engine--generative-entity

**Registry:** [`registry-v2.json`](https://github.com/meta-organvm/organvm-corpvs-testamentvm/blob/main/registry-v2.json)
**Corpus:** [`organvm-corpvs-testamentvm`](https://github.com/meta-organvm/organvm-corpvs-testamentvm)

<!-- ORGANVM:AUTO:START -->
## System Context (auto-generated — do not edit)

**Organ:** ORGAN-IV (Orchestration) | **Tier:** flagship | **Status:** PUBLIC_PROCESS
**Org:** `organvm-iv-taxis` | **Repo:** `agentic-titan`

### Edges
- **Produces** → `organvm-iv-taxis/agent--claude-smith`: dependency
- **Produces** → `organvm-v-logos/public-process`: dependency
- **Consumes** ← `organvm-i-theoria/recursive-engine--generative-entity`: dependency

### Siblings in Orchestration
`orchestration-start-here`, `petasum-super-petasum`, `universal-node-network`, `.github`, `agent--claude-smith`, `a-i--skills`

### Governance
- *Standard ORGANVM governance applies*

*Last synced: 2026-02-24T12:41:28Z*
<!-- ORGANVM:AUTO:END -->


## ⚡ Conductor OS Integration
This repository is a managed component of the ORGANVM meta-workspace.
- **Orchestration:** Use `conductor patch` for system status and work queue.
- **Lifecycle:** Follow the `FRAME -> SHAPE -> BUILD -> PROVE` workflow.
- **Governance:** Promotions are managed via `conductor wip promote`.
- **Intelligence:** Conductor MCP tools are available for routing and mission synthesis.
