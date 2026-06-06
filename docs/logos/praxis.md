# Praxis

Praxis records the active remediation path for keeping Agentic Titan coherent.

## Maintenance Loop

1. Keep `ruff check .`, `mypy .`, and the pytest suite green.
2. Treat branch-protection dependency validation as a real governance gate.
3. Keep generated reports and local coverage artifacts out of commits.
4. Add tests beside behavior changes, especially in topology, safety, routing,
   and dashboard surfaces.
5. Prefer advisory reporting before turning new quality gates into blocking
   gates.

## Near-Term Work

- Calibrate advisory AI quality findings into a reviewed baseline.
- Continue reducing duplicate adapter/router code surfaced by the advisory
  gate.
- Keep topology-gating behavior covered as fission-fusion and stigmergic
  coordination evolve.
- Split large research ideas into implementation-ready issues only after their
  contracts, tests, and success criteria are clear.
