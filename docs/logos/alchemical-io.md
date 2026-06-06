# Alchemical I/O

Alchemical I/O records the repository's transformation pattern: what enters the
system, how it is processed, and what returns to the broader ORGANVM graph.

## Inputs

- User tasks and agent specifications.
- Provider responses from local and cloud LLM adapters.
- Shared-memory traces, topology metrics, and dashboard events.
- Governance constraints from the dependency graph and branch protection.

## Transmutation

Inputs pass through routing, topology selection, safety checks, and workflow
execution. The system should preserve enough evidence to audit each transition:
which agent acted, which model or runtime was selected, which gate approved or
blocked action, and which tests protect the behavior.

## Outputs

- Agent results and workflow artifacts.
- Metrics, audit logs, and dashboard-visible state.
- Dependency-compatible code and documentation changes.
- Public process material when changes need a narrative record outside this
  repository.
