# F-26: Aider Evaluation and Integration

> Evaluation of aider as terminal pair programmer for the ORGANVM ecosystem.

## Overview

Aider is an open-source AI pair programming tool that runs in the terminal. It edits code directly in your local git repo, auto-commits changes, and maintains a repository map for context-aware edits. It is the most mature terminal-based AI coding tool as of early 2026.

**Repository**: https://github.com/Aider-AI/aider
**License**: Apache 2.0

## Installation

```bash
# Homebrew (macOS)
brew install aider

# pip (recommended for version pinning)
pip install aider-chat

# pipx (isolated install)
pipx install aider-chat
```

## Key Features

### Ollama / Local Model Support

Aider supports local inference via any OpenAI-compatible endpoint:

```bash
# Ollama
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=unused
aider --model ollama/deepseek-coder-v2:16b

# Direct Ollama integration
aider --model ollama/qwen2.5-coder:14b
```

Aider maintains a leaderboard of model performance on its coding benchmark, which helps select the best local model for the task.

### Auto-Commits

Every edit aider makes is automatically committed to git with a descriptive message:

```
aider: Added error handling to process_request function
```

Commit style is configurable:
```bash
aider --auto-commits           # default: on
aider --no-auto-commits        # disable for manual staging
aider --commit-prompt "feat:"  # custom commit prefix
```

### Repository Mapping

Aider builds a map of the entire repository structure and uses it for context:

```bash
aider --map-tokens 2048    # allocate 2048 tokens for repo map
aider --map-tokens 0       # disable repo map (saves context)
```

The repo map includes:
- File tree structure
- Function/class signatures
- Import relationships
- Recently edited files (weighted higher)

### Context Window Awareness

Aider tracks token usage and manages context budget:

```bash
aider --model claude-sonnet-4-20250514  # auto-detects 200k context
aider --map-tokens 4096                  # repo map budget
# Remaining tokens allocated to: chat history + file contents + output
```

When context fills up, aider automatically summarizes older conversation turns.

## Configuration

### `.aider.conf.yml`

```yaml
# .aider.conf.yml (per-repo)
model: claude-sonnet-4-20250514
auto-commits: true
map-tokens: 2048
dark-mode: true
stream: true

# Ignore patterns
ignore:
  - "*.lock"
  - "node_modules/"
  - ".venv/"
  - "__pycache__/"

# Git settings
attribute-author: false
attribute-committer: false
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export OPENAI_API_BASE=http://localhost:11434/v1  # for local models
```

### ORGANVM-Specific Configuration

```yaml
# .aider.conf.yml for ORGANVM repos
model: claude-sonnet-4-20250514
auto-commits: false           # ORGANVM uses manual commits with Conventional Commits
map-tokens: 2048
dark-mode: true

ignore:
  - "*.lock"
  - "node_modules/"
  - ".venv/"
  - "__pycache__/"
  - ".build/"                  # a-i--skills generated artifacts
  - "dist/"
  - "registry.json"           # protected governance file
  - "governance-rules.json"   # protected governance file
```

**Note**: `auto-commits: false` is recommended for ORGANVM repos because the project enforces Conventional Commits format and atomic, focused commits. Aider's auto-commit messages don't follow this convention.

## Integration with Agentic-Titan

### Agent Type Wrapper

Aider can be wrapped as a Titan agent type for orchestrated editing tasks:

```python
# agents/aider_agent.py
class AiderAgent(BaseAgent):
    """Wraps aider as a Titan-managed code editing agent."""

    async def execute(self, task: Task) -> AgentResult:
        # Invoke aider in non-interactive mode
        result = await run_subprocess(
            ["aider", "--no-auto-commits", "--yes", "--message", task.prompt],
            cwd=task.repo_path,
        )
        # Parse aider output for edit results
        return AgentResult(output=result.stdout, edits=parse_edits(result))
```

### Non-Interactive Mode

Aider supports scripted/non-interactive usage:

```bash
# Single message, apply and exit
aider --yes --message "Add type hints to all functions in src/utils.py"

# Read prompt from file
aider --yes --message-file prompt.txt

# Specify files to edit
aider --yes --file src/utils.py --message "Add docstrings"
```

This makes it suitable for Titan-orchestrated workflows where the human is not in the loop for individual edits.

## Strengths

- **Excellent git integration**: auto-commits, understands git history, respects `.gitignore`
- **Repository-aware**: repo map provides structural context without consuming full file contents
- **Mature**: active since 2023, large community, extensive documentation
- **Context management**: automatic summarization, configurable token budgets
- **Multi-model support**: works with Claude, GPT-4, local models via Ollama, Gemini
- **Benchmarked**: published coding benchmarks help select optimal model/settings
- **Non-interactive mode**: scriptable for orchestration integration

## Weaknesses

- **Opinionated commit style**: auto-commit messages don't follow Conventional Commits (mitigated by disabling auto-commits)
- **Limited multi-file coordination**: best for focused, single-file or small-scope edits; struggles with large refactors across many files
- **No MCP support**: cannot consume MCP servers for extended tool access (unlike Goose)
- **No permission model**: no built-in sandboxing for file write restrictions
- **Python dependency**: requires Python environment, which can conflict with project venvs (mitigated by pipx)
- **Token cost**: repo map and context management consume tokens even for simple edits

## Verdict

**Recommended for single-file editing workflows.**

Aider is the best terminal-based tool for focused code editing tasks: adding functions, fixing bugs, writing tests, adding type hints. Its git awareness and repo mapping make it context-efficient. However, it lacks the MCP integration and permission model that make Goose better for governed workflows, and it cannot match Claude Code's multi-file reasoning.

### Tool Selection Matrix

| Task | Best Tool | Why |
|------|-----------|-----|
| Add docstrings to a module | aider | Focused, single-file, git-aware |
| Fix a specific bug | aider | Quick edit with auto-commit |
| Write tests for a function | aider | Understands function signatures via repo map |
| Multi-file refactor | Claude Code | Better multi-file reasoning |
| Interactive exploration | Goose | MCP tools, persistent context |
| Orchestrated batch edits | agentic-titan | Manages multiple agents |

### Next Steps

1. Install aider and configure `.aider.conf.yml` for ORGANVM repos
2. Benchmark local model performance (deepseek-coder, qwen2.5-coder) with aider
3. Implement `AiderAgent` wrapper for Titan integration
4. Test non-interactive mode for orchestrated workflows
