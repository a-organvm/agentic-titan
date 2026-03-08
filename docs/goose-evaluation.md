# F-25: Goose Evaluation and Integration

> Evaluation of Goose (by Block) as primary local agent for the ORGANVM ecosystem.

## Overview

Goose is an open-source AI agent developed by Block (formerly Square) designed for local-first developer workflows. It runs entirely on the developer's machine, supports multiple LLM providers including local models via Ollama, and integrates with the MCP (Model Context Protocol) ecosystem.

**Repository**: https://github.com/block/goose
**License**: Apache 2.0

## Installation

```bash
# Homebrew (macOS)
brew install goose

# pip
pip install goose-ai

# From source
git clone https://github.com/block/goose.git
cd goose && cargo build --release
```

## Key Features

### Ollama Provider Support

Goose supports local LLM inference via Ollama as a first-class provider:

```yaml
# ~/.config/goose/profiles.yaml
default:
  provider: ollama
  model: deepseek-coder-v2:16b
  context_window: 32768
```

Any OpenAI-compatible endpoint works via `OPENAI_API_BASE`:

```bash
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=unused
goose session start
```

### Permission System

`permission.yaml` provides fine-grained sandboxing:

```yaml
# .goose/permission.yaml
permissions:
  file_read:
    allowed:
      - "src/**"
      - "tests/**"
      - "docs/**"
    denied:
      - "**/.env"
      - "**/credentials*"
  file_write:
    allowed:
      - "src/**"
      - "tests/**"
    denied:
      - "seed.yaml"          # Protected governance file
      - "registry.json"      # Protected registry
  shell:
    allowed:
      - "pytest*"
      - "ruff*"
      - "git status"
      - "git diff"
    denied:
      - "rm -rf*"
      - "git push --force*"
```

This aligns with ORGANVM's data integrity rules — protected files can be explicitly denied at the agent level.

### MCP Server Integration

Goose natively consumes MCP servers, enabling tool connectivity:

```yaml
# ~/.config/goose/profiles.yaml
default:
  provider: anthropic
  model: claude-sonnet-4-20250514
  mcpServers:
    filesystem:
      command: "node"
      args: ["/path/to/mcp-servers/dist/filesystem.js"]
    memory:
      command: "node"
      args: ["/path/to/mcp-servers/dist/memory.js"]
```

### Persistent Tool Context

Goose maintains a `.goose/` directory per project containing:
- Session history (conversation logs)
- Tool output caches
- File edit history
- Context snapshots for session resumption

This is the basis for the F-74 PTC bridge — see `goose-titan-ptc-bridge.md`.

## Configuration for ORGANVM

### Recommended Profile

```yaml
# ~/.config/goose/profiles.yaml
organvm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  context_window: 200000
  mcpServers:
    filesystem:
      command: "node"
      args: ["~/Workspace/mcp-servers/dist/filesystem.js"]
      env:
        ALLOWED_DIRS: "~/Workspace"
    memory:
      command: "node"
      args: ["~/Workspace/mcp-servers/dist/memory.js"]
  toolkits:
    - developer
    - repo_context
```

### Local Model Configuration

```yaml
local:
  provider: ollama
  model: qwen2.5-coder:14b
  context_window: 32768
  temperature: 0.1
```

```bash
# Start with local profile
goose session start --profile local
```

## Integration with Agentic-Titan

### Agent Type Wrapper

Goose sessions can be wrapped as a Titan agent type:

```python
# agents/goose_agent.py
class GooseAgent(BaseAgent):
    """Wraps a Goose session as a Titan-managed agent."""

    async def execute(self, task: Task) -> AgentResult:
        # Start goose session with task prompt
        # Read .goose context for results
        # Convert to Titan result format
        ...
```

### Context Bridge (F-74)

Goose `.goose/` context directories can be imported into Titan's agent memory system:
- `.goose/sessions/` → Titan session history
- `.goose/context/` → Titan agent memory
- Bidirectional sync enables switching between Goose interactive sessions and Titan orchestrated workflows

### Unified Secrets Interface

Both Goose and agentic-titan need access to secrets. A unified approach:

| Secret Store | Goose Access | Titan Access |
|-------------|-------------|-------------|
| 1Password | `op://vault/item/field` via CLI | `SecretResolver` in agent--claude-smith |
| macOS Keychain | `security find-generic-password` | Python `keyring` library |
| Environment | `GOOSE_API_KEY` etc. | `LLMConfig.api_key` |

Proposed: a shared secrets bridge that both tools read from, backed by 1Password with Keychain fallback.

## Strengths

- **Local-first**: runs entirely on developer machine, no cloud dependency for orchestration
- **MCP native**: first-class MCP server integration aligns with ORGANVM's MCP infrastructure
- **Permission model**: sandboxing prevents destructive operations — critical for governance
- **Session persistence**: `.goose/` context survives across sessions
- **Open source**: Apache 2.0, active development by Block engineering
- **Provider agnostic**: supports Anthropic, OpenAI, Ollama, and any OpenAI-compatible endpoint

## Weaknesses

- **Limited model support**: best with Claude and GPT-4 class models; smaller local models may underperform
- **Community size**: smaller community than aider or Cursor; fewer third-party extensions
- **Rust codebase**: harder to contribute to or customize than Python-based alternatives
- **No native git integration**: unlike aider, doesn't auto-commit or understand repo structure deeply
- **Early maturity**: API surface and config format may change between releases

## Verdict

**Recommended for local agent workflows.**

Goose fills a specific niche in the ORGANVM stack: interactive, local-first, MCP-native agent sessions. It complements (rather than replaces) aider for code editing and Claude Code for complex multi-file tasks. The permission system and MCP integration make it the strongest candidate for a governed local agent.

### Recommended Usage Patterns

| Workflow | Tool |
|----------|------|
| Interactive code exploration | Goose |
| Single-file editing with git | aider |
| Multi-file orchestrated tasks | agentic-titan |
| Complex codebase reasoning | Claude Code |

### Next Steps

1. Install and configure Goose with ORGANVM profiles
2. Implement F-74 PTC bridge for context sharing with Titan
3. Write `GooseAgent` wrapper for Titan integration
4. Test with local Ollama models for fully offline workflows
