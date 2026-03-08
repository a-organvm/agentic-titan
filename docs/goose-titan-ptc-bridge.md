# F-74: Goose-to-Titan PTC Bridge

> Persistent Tool Context (PTC) bridge for sharing session context between Goose and Titan.

## Overview

Goose and agentic-titan both maintain session context — conversation history, file edits, tool outputs — but in incompatible formats. The PTC bridge enables bidirectional context sharing so that work started in an interactive Goose session can be continued by Titan orchestration, and vice versa.

## Context Formats

### Goose Context (`.goose/`)

```
.goose/
├── sessions/
│   └── 2026-03-08-abc123.json    # Session transcript
├── context/
│   ├── file_edits.json           # Files modified in session
│   ├── tool_outputs.json         # Tool call results
│   └── memory.json               # Persistent memory across sessions
└── permission.yaml               # Sandboxing rules
```

**Session format** (simplified):

```json
{
  "id": "2026-03-08-abc123",
  "started_at": "2026-03-08T10:00:00Z",
  "messages": [
    {"role": "user", "content": "Add error handling to process_request"},
    {"role": "assistant", "content": "I'll add try/except blocks...", "tool_calls": [...]},
    {"role": "tool", "name": "file_edit", "content": {"path": "src/api.py", "diff": "..."}}
  ],
  "files_touched": ["src/api.py", "tests/test_api.py"],
  "tools_used": ["file_edit", "shell", "file_read"]
}
```

### Titan Agent Memory

```
$AGENTS_ROOT/<agent-id>/
├── sessions/
│   └── session-xyz.json          # Session state
├── memory/
│   ├── short_term.json           # Current task context
│   ├── long_term.json            # Persistent knowledge
│   └── tool_history.json         # Tool invocation log
└── artifacts/
    └── <output files>
```

**Session format** (simplified):

```json
{
  "session_id": "session-xyz",
  "agent_id": "code-editor-01",
  "topology": "pipeline",
  "status": "completed",
  "tasks": [
    {
      "prompt": "Add error handling to process_request",
      "result": "Added try/except blocks...",
      "tool_calls": [...],
      "tokens_used": 4500,
      "cost_usd": 0.045
    }
  ],
  "context": {
    "files_touched": ["src/api.py", "tests/test_api.py"],
    "git_branch": "feature/error-handling"
  }
}
```

## Bridge Architecture

```
Goose Session                    PTC Bridge                    Titan Session
─────────────                    ──────────                    ─────────────
.goose/sessions/   ──export──►   ptc-bridge/    ──import──►   agents/<id>/sessions/
.goose/context/    ──export──►   staging/       ──import──►   agents/<id>/memory/
                   ◄──import──                  ◄──export──
```

### Storage Location

```
$AGENTS_ROOT/ptc-bridge/
├── staging/                      # Intermediate format
│   ├── goose-to-titan/
│   │   └── <session-id>.json
│   └── titan-to-goose/
│       └── <session-id>.json
├── mappings/                     # Format conversion rules
│   ├── goose-schema.json
│   └── titan-schema.json
└── bridge.py                     # Bridge script
```

## Context Types

### File Edits

| Goose Format | Titan Format |
|-------------|-------------|
| `tool_outputs.json` with `file_edit` entries | `tool_history.json` with edit diffs |
| Stores original + modified content | Stores unified diff |

**Conversion**:

```python
def goose_edit_to_titan(goose_edit: dict) -> dict:
    return {
        "tool": "file_edit",
        "path": goose_edit["path"],
        "diff": compute_unified_diff(goose_edit["original"], goose_edit["modified"]),
        "timestamp": goose_edit["timestamp"],
    }
```

### Tool Outputs

Both systems log tool invocations. The bridge normalizes tool names:

```python
TOOL_NAME_MAP = {
    # Goose → Titan
    "file_read": "filesystem_read",
    "file_edit": "filesystem_write",
    "shell": "shell_execute",
    "web_search": "web_search",
}
```

### Conversation History

Goose stores full conversation transcripts. Titan stores task-level summaries. The bridge:
1. **Goose → Titan**: Summarize Goose conversation into task descriptions + results
2. **Titan → Goose**: Expand Titan task logs into conversational format

```python
async def summarize_goose_session(session: GooseSession, adapter: BaseAdapter) -> TitanTask:
    """Convert a Goose conversation into a Titan task summary."""
    transcript = format_messages(session.messages)
    summary = await adapter.generate(
        f"Summarize this coding session into a task description and result:\n\n{transcript}"
    )
    return TitanTask(
        prompt=extract_task(summary),
        result=extract_result(summary),
        files_touched=session.files_touched,
        tools_used=[TOOL_NAME_MAP.get(t, t) for t in session.tools_used],
    )
```

## Bridge Implementation

```python
#!/usr/bin/env python3
"""PTC Bridge: bidirectional context sharing between Goose and Titan."""

import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class BridgeConfig:
    goose_dir: Path          # .goose/ directory
    titan_agent_dir: Path    # $AGENTS_ROOT/<agent-id>/
    staging_dir: Path        # ptc-bridge/staging/

class PTCBridge:
    def __init__(self, config: BridgeConfig):
        self.config = config

    def export_goose_to_titan(self, session_id: str) -> Path:
        """Export a Goose session to Titan-compatible format."""
        # Read Goose session
        goose_session = self._read_goose_session(session_id)

        # Convert context types
        titan_session = {
            "session_id": f"goose-{session_id}",
            "source": "goose",
            "imported_at": datetime.utcnow().isoformat(),
            "tasks": [self._convert_task(msg) for msg in goose_session["messages"]],
            "context": {
                "files_touched": goose_session.get("files_touched", []),
                "tool_history": [
                    goose_edit_to_titan(edit)
                    for edit in self._extract_edits(goose_session)
                ],
            },
        }

        # Write to staging
        output = self.config.staging_dir / "goose-to-titan" / f"{session_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(titan_session, indent=2))
        return output

    def import_titan_to_goose(self, session_id: str) -> Path:
        """Export a Titan session to Goose-compatible format."""
        titan_session = self._read_titan_session(session_id)

        goose_session = {
            "id": f"titan-{session_id}",
            "started_at": titan_session.get("started_at"),
            "messages": self._expand_to_messages(titan_session["tasks"]),
            "files_touched": titan_session["context"].get("files_touched", []),
            "tools_used": list({
                self._reverse_tool_name(t["tool"])
                for t in titan_session["context"].get("tool_history", [])
            }),
        }

        output = self.config.staging_dir / "titan-to-goose" / f"{session_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(goose_session, indent=2))
        return output

    def sync(self, direction: str = "both"):
        """Sync all pending sessions."""
        if direction in ("goose-to-titan", "both"):
            for session_file in (self.config.goose_dir / "sessions").glob("*.json"):
                session_id = session_file.stem
                self.export_goose_to_titan(session_id)

        if direction in ("titan-to-goose", "both"):
            for session_file in (self.config.titan_agent_dir / "sessions").glob("*.json"):
                session_id = session_file.stem
                self.import_titan_to_goose(session_id)
```

## CLI Interface

```bash
# Export specific Goose session to Titan
python3 -m ptc_bridge export --from goose --session 2026-03-08-abc123

# Import Titan session into Goose format
python3 -m ptc_bridge export --from titan --session session-xyz

# Sync all sessions bidirectionally
python3 -m ptc_bridge sync --direction both

# List available sessions
python3 -m ptc_bridge list --source goose
python3 -m ptc_bridge list --source titan
```

## Titan Plugin Alternative

Instead of a standalone script, the bridge can be a Titan plugin:

```python
# plugins/ptc_bridge.py
from titan.plugins import Plugin

class PTCBridgePlugin(Plugin):
    name = "ptc-bridge"

    async def on_session_start(self, session):
        """Check for Goose context to import."""
        goose_dir = session.repo_path / ".goose"
        if goose_dir.exists():
            latest = self.get_latest_goose_session(goose_dir)
            if latest:
                session.import_context(self.convert(latest))

    async def on_session_end(self, session):
        """Export session context for Goose consumption."""
        self.export_to_goose(session)
```

## Workflow Example

1. Developer starts interactive Goose session to explore a bug
2. Goose session identifies the issue, makes partial fix
3. Developer runs `ptc-bridge export --from goose --session latest`
4. Titan picks up the context and runs a full test suite with the fix
5. Titan identifies additional test failures and fixes them
6. `ptc-bridge sync` makes Titan's results available in next Goose session

## Security Considerations

- **Credential scrubbing**: Strip API keys, tokens, and secrets from exported context
- **Path sanitization**: Normalize file paths to prevent directory traversal
- **Permission alignment**: Goose `permission.yaml` restrictions should be respected by Titan when importing context

## References

- `goose-evaluation.md` (F-25) — Goose feature assessment
- `aider-evaluation.md` (F-26) — aider comparison
- `mcp-protocol-support.md` (F-27) — MCP as alternative context sharing mechanism
