# Titan MCP Server

Expose Titan agents via the Model Context Protocol (MCP).

## Quick Start

### 1. Add to Claude Code Settings

Add this to your `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "titan": {
      "command": "/path/to/agentic-titan/.venv/bin/python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/agentic-titan"
    }
  }
}
```

### 2. Verify Installation

```bash
# Test the server
cd /path/to/agentic-titan
.venv/bin/python -m titan.cli mcp test
```

## Available Tools

| Tool | Description |
|------|-------------|
| `spawn_agent` | Create a new agent (researcher, coder, reviewer, orchestrator, simple) |
| `agent_status` | Check agent progress by session ID |
| `agent_result` | Get completed agent results |
| `list_agents` | List all active agent sessions |
| `cancel_agent` | Cancel a running agent |

## Available Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Agent Types | `titan://agents/types` | List of available agent archetypes |
| Agent Tools | `titan://agents/tools` | List of tools available to agents |

## Example Usage (from Claude Code)

```
User: Research the latest developments in quantum computing

Claude: I'll spawn a researcher agent to investigate this topic.
[Calls spawn_agent with type="researcher", task="Research quantum computing developments"]

Result: Session sess_abc12345 created, agent running...

[Later, calls agent_result with session_id="sess_abc12345"]

Result: {comprehensive research report}
```

## Agent Types

### Researcher
- Deep research and analysis
- Multi-question investigation
- Source synthesis

### Coder
- Code generation
- Implementation tasks
- Technical solutions

### Reviewer
- Code review
- Quality assessment
- Constructive feedback

### Orchestrator
- Multi-agent coordination
- Complex task decomposition
- Result aggregation

### Simple
- Basic tool-using agent
- Direct task execution
- Configurable system prompt
