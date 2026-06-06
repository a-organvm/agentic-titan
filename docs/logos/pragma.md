# Pragma

Pragma records the concrete state of the repository.

## Current Surface

- Python package layout is root-based: `titan/`, `hive/`, `agents/`,
  `adapters/`, `runtime/`, and `dashboard/`.
- CI covers linting, strict type checking, tests with coverage, CodeQL, secret
  pattern detection, release drafting, and an advisory AI quality gate.
- The topology layer includes fission-fusion behavior, stigmergic traces,
  perceptual gating, multi-scale neighborhoods, and convergence experiments.
- The adapter layer supports provider routing across local and cloud model
  backends.
- The dashboard and gateway expose operational surfaces for inspection and
  orchestration.

## Known Boundaries

- Long-horizon research issues remain open for topology concepts that do not
  yet have an implementation path.
- Local-inference sovereignty remains an R&D track, not a completed guarantee.
- Cross-model replay/diff has a design document, but its CLI and persistence
  implementation remain future work.
