# F-75: Agent Config Fragmentation Audit

> Audit of agent configuration across 7+ tools and proposal for a unified config hierarchy.

## Problem

The ORGANVM workspace uses multiple AI coding agents, each with its own configuration format, file location, and settings vocabulary. The same conceptual settings (model selection, temperature, allowed directories, API keys) are configured independently in 7+ places, leading to drift, contradiction, and maintenance burden.

## Current Configuration Landscape

### Tool-by-Tool Inventory

| Tool | Config Files | Location | Format |
|------|-------------|----------|--------|
| **Claude Code** | `CLAUDE.md`, `.claude/settings.json`, `.claude/commands/` | Repo root, `~/.claude/` | Markdown, JSON |
| **Goose** | `profiles.yaml`, `permission.yaml` | `~/.config/goose/`, `.goose/` | YAML |
| **aider** | `.aider.conf.yml`, `.aiderignore` | Repo root, `~/.aider.conf.yml` | YAML |
| **Codex** | `codex.yaml` | Repo root | YAML |
| **Cursor** | `.cursorrules`, `.cursor/settings.json` | Repo root | Text, JSON |
| **Copilot** | `.github/copilot-instructions.md`, `.copilot/` | Repo root | Markdown |
| **agentic-titan** | `titan-config.yaml`, agent specs | Repo root, `specs/` | YAML |

### Settings That Diverge

| Setting | Claude Code | Goose | aider | Cursor | Copilot |
|---------|-------------|-------|-------|--------|---------|
| Model | Implicit (Claude) | `profiles.yaml` | `--model` / config | `.cursor/settings.json` | Implicit (GPT-4) |
| Temperature | Not configurable | `profiles.yaml` | `--temperature` | Settings | Not configurable |
| Context window | Auto-detected | `context_window` | `--map-tokens` | Auto | Auto |
| Allowed dirs | `.claude/settings.json` | `permission.yaml` | `.aiderignore` | Workspace trust | Not configurable |
| API keys | Env vars | Env vars / config | Env vars | Settings | GitHub auth |
| Ignore patterns | Not applicable | `permission.yaml` | `.aiderignore` | `.cursorignore` | Not applicable |
| Commit style | Manual | Manual | Auto (configurable) | Manual | Suggestions only |

### Observed Contradictions

1. **Model selection**: Claude Code always uses Claude; aider might be configured for GPT-4; Goose might use Ollama — no single source of truth for "which model should this repo use"
2. **Ignore patterns**: `.aiderignore` excludes `registry.json`; `.cursorignore` might not; Goose `permission.yaml` has its own deny list — identical intent, three files
3. **API keys**: `ANTHROPIC_API_KEY` in shell env, `OPENAI_API_KEY` in a different env, Goose profile references a third — key rotation requires updating multiple places
4. **Allowed directories**: Claude Code trusts workspace root; Goose permission.yaml restricts to `src/` and `tests/`; aider has no restriction — inconsistent sandboxing

## Proposed Canonical Config Hierarchy

### 4-Level Override Chain

```
Level 1: System Defaults (ORGANVM-wide)
  └── Level 2: Organ Defaults (per organ)
       └── Level 3: Repo Overrides (per repo)
            └── Level 4: Session Overrides (per invocation)
```

Each level overrides the one above. Unset values inherit from the parent level.

### Level 1: System Defaults

```yaml
# ~/Workspace/.agent-config.yaml (or in meta-organvm governance)
agent_config:
  version: "1.0"
  defaults:
    model:
      primary: "claude-sonnet-4-20250514"
      local: "qwen2.5-coder:14b"
      fallback: "gpt-4o"
    temperature: 0.1
    context_window: 200000
    commit_style: "conventional"    # feat:, fix:, docs:, etc.
    auto_commit: false
    protected_files:
      - "registry.json"
      - "registry-v2.json"
      - "governance-rules.json"
      - "seed.yaml"
      - "system-metrics.json"
    ignored_paths:
      - "node_modules/"
      - ".venv/"
      - "__pycache__/"
      - "*.lock"
      - ".build/"
      - "dist/"
    secrets:
      provider: "1password"
      vault: "ORGANVM"
```

### Level 2: Organ Defaults

```yaml
# organvm-iv-taxis/.agent-config.yaml
agent_config:
  inherits: "system"
  overrides:
    model:
      primary: "claude-sonnet-4-20250514"  # ORGAN-IV prefers Claude for orchestration
    protected_files:
      - "governance-rules.json"
      - "registry.json"
    additional_context:
      - "Read CLAUDE.md before any work"
      - "Check seed.yaml for repo contracts"
```

### Level 3: Repo Overrides

```yaml
# agentic-titan/.agent-config.yaml
agent_config:
  inherits: "organ"
  overrides:
    model:
      local: "deepseek-coder-v2:16b"  # Better for Python than default
    temperature: 0.0                    # Deterministic for orchestration code
    additional_ignored_paths:
      - "dashboard/static/"
    test_command: "pytest"
    lint_command: "ruff check ."
    typecheck_command: "mypy ."
```

### Level 4: Session Overrides

```bash
# CLI flags or environment variables
AGENT_MODEL=ollama/qwen2.5-coder:14b
AGENT_TEMPERATURE=0.3
AGENT_CONTEXT_WINDOW=32768
```

### Integration with seed.yaml

The canonical config can be embedded in seed.yaml's `agent_config` section:

```yaml
# seed.yaml
name: agentic-titan
organ: IV
tier: flagship
promotion_status: PUBLIC_PROCESS
agent_config:
  model: "claude-sonnet-4-20250514"
  temperature: 0.0
  test_command: "pytest"
  lint_command: "ruff check ."
```

## Unified Config Format

### `.agent-config.yaml` Schema

```yaml
# Full schema
agent_config:
  version: "1.0"
  inherits: "system" | "organ" | null

  # Model selection
  model:
    primary: string          # Default model for this context
    local: string            # Preferred local/Ollama model
    fallback: string         # Fallback if primary unavailable

  # Generation parameters
  temperature: float         # 0.0 - 2.0
  context_window: int        # Token limit
  max_output_tokens: int     # Output limit

  # Safety
  protected_files: list[str]       # Files agents must not overwrite
  ignored_paths: list[str]         # Paths excluded from context/editing
  allowed_directories: list[str]   # Sandboxed write directories

  # Git
  commit_style: string       # "conventional", "freeform"
  auto_commit: bool          # Whether agents should auto-commit

  # Commands
  test_command: string
  lint_command: string
  typecheck_command: string
  build_command: string

  # Secrets
  secrets:
    provider: string         # "1password", "keychain", "env"
    vault: string            # For 1Password
    items: dict              # Key-value mappings

  # Additional context
  additional_context: list[string]  # Instructions for all agents
```

## Conflict Detection Script

A script that reads all agent config files in a repo and flags contradictions:

```python
#!/usr/bin/env python3
"""Detect contradictions across agent configuration files."""

import yaml
import json
from pathlib import Path

CONFIG_FILES = {
    "claude": [".claude/settings.json", "CLAUDE.md"],
    "goose": [".goose/permission.yaml", "~/.config/goose/profiles.yaml"],
    "aider": [".aider.conf.yml", ".aiderignore"],
    "cursor": [".cursorrules", ".cursor/settings.json"],
    "copilot": [".github/copilot-instructions.md"],
    "titan": [".agent-config.yaml", "seed.yaml"],
}

def detect_conflicts(repo_path: Path) -> list[dict]:
    conflicts = []
    configs = {}

    for tool, files in CONFIG_FILES.items():
        for f in files:
            path = repo_path / f
            if path.exists():
                configs[tool] = parse_config(path)

    # Check model conflicts
    models = {tool: c.get("model") for tool, c in configs.items() if c.get("model")}
    if len(set(str(m) for m in models.values())) > 1:
        conflicts.append({
            "type": "model_divergence",
            "details": models,
            "severity": "warning",
        })

    # Check ignore pattern conflicts
    ignores = {tool: set(c.get("ignore", [])) for tool, c in configs.items()}
    all_ignores = set().union(*ignores.values())
    for pattern in all_ignores:
        tools_with = [t for t, i in ignores.items() if pattern in i]
        tools_without = [t for t, i in ignores.items() if pattern not in i and i]
        if tools_with and tools_without:
            conflicts.append({
                "type": "ignore_pattern_inconsistency",
                "pattern": pattern,
                "included_by": tools_with,
                "missing_from": tools_without,
                "severity": "info",
            })

    # Check protected file coverage
    canonical_protected = configs.get("titan", {}).get("protected_files", [])
    for tool, config in configs.items():
        tool_denied = config.get("denied_files", []) or config.get("protected_files", [])
        missing = set(canonical_protected) - set(tool_denied)
        if missing and tool_denied:  # Only flag if tool has a deny list at all
            conflicts.append({
                "type": "protection_gap",
                "tool": tool,
                "unprotected": list(missing),
                "severity": "error",
            })

    return conflicts

if __name__ == "__main__":
    import sys
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    conflicts = detect_conflicts(repo)
    for c in conflicts:
        severity = c["severity"].upper()
        print(f"[{severity}] {c['type']}: {json.dumps(c, indent=2)}")
    sys.exit(1 if any(c["severity"] == "error" for c in conflicts) else 0)
```

### CI Integration

```yaml
# .github/workflows/config-audit.yml
- name: Agent config consistency check
  run: python3 scripts/check-agent-configs.py . --strict
```

## Migration Path

1. **Phase 1**: Create `.agent-config.yaml` at system and organ levels as the canonical source
2. **Phase 2**: Write generators that produce tool-specific configs from `.agent-config.yaml`:
   - `.agent-config.yaml` -> `.aider.conf.yml`
   - `.agent-config.yaml` -> `.goose/permission.yaml`
   - `.agent-config.yaml` -> `.cursorrules`
3. **Phase 3**: Run conflict detection in CI for all repos
4. **Phase 4**: Auto-generate tool configs from canonical source in CI

## References

- seed.yaml schema — existing per-repo contract
- CLAUDE.md conventions — Claude Code configuration
- `goose-evaluation.md` (F-25) — Goose configuration details
- `aider-evaluation.md` (F-26) — aider configuration details
- `mcp-protocol-support.md` (F-27) — MCP server configuration overlap
