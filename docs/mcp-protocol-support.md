# F-27: MCP Protocol Support

> Design doc for Model Context Protocol (MCP) integration in agentic-titan.

## Overview

The Model Context Protocol (MCP) is an open standard for connecting AI models to external tools and data sources. It uses JSON-RPC 2.0 over stdio or Server-Sent Events (SSE) transports, providing a universal interface for tool discovery, invocation, and context sharing.

Agentic-titan should support MCP in two directions:
1. **As MCP server**: expose Titan's capabilities (file ops, git, registry, orchestration) to MCP clients
2. **As MCP client**: consume external MCP servers (filesystem, memory, databases, APIs)

## MCP Fundamentals

### Protocol Structure

```
Client ←→ Server (JSON-RPC 2.0)
  │
  ├── initialize        → capabilities exchange
  ├── tools/list        → discover available tools
  ├── tools/call        → invoke a tool
  ├── resources/list    → discover data resources
  ├── resources/read    → read a resource
  ├── prompts/list      → discover prompt templates
  └── prompts/get       → retrieve a prompt
```

### Transport Options

| Transport | Use Case | Latency | Security |
|-----------|----------|---------|----------|
| **stdio** | Local processes, CLI tools | Lowest | Process isolation |
| **SSE** | Remote servers, web services | Higher | TLS, auth tokens |
| **Streamable HTTP** | Scalable remote, bidirectional | Medium | TLS, auth headers |

## Server Support: Titan as MCP Server

### Exposed Tool Categories

#### Registry & Governance Tools

```json
{
  "name": "titan_registry_query",
  "description": "Query the ORGANVM registry for repo metadata",
  "inputSchema": {
    "type": "object",
    "properties": {
      "organ": { "type": "string", "description": "Organ number (I-VII or META)" },
      "status": { "type": "string", "enum": ["LOCAL", "CANDIDATE", "PUBLIC_PROCESS", "GRADUATED", "ARCHIVED"] },
      "tier": { "type": "string", "enum": ["flagship", "standard", "infrastructure"] }
    }
  }
}
```

```json
{
  "name": "titan_validate_deps",
  "description": "Validate dependency graph for back-edge violations",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": { "type": "string", "description": "Repository name to validate" }
    }
  }
}
```

#### Orchestration Tools

```json
{
  "name": "titan_run_topology",
  "description": "Execute a Titan topology (pipeline, scatter-gather, etc.)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "topology": { "type": "string", "enum": ["pipeline", "scatter_gather", "debate", "hierarchical"] },
      "spec_path": { "type": "string" },
      "prompt": { "type": "string" }
    },
    "required": ["topology", "prompt"]
  }
}
```

#### Git & File Tools

```json
{
  "name": "titan_git_status",
  "description": "Get git status for a repo in the ORGANVM workspace",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": { "type": "string", "description": "Repo path relative to workspace root" }
    },
    "required": ["repo"]
  }
}
```

### Server Implementation

```python
# mcp_server/server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("agentic-titan")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="titan_registry_query", ...),
        Tool(name="titan_validate_deps", ...),
        Tool(name="titan_run_topology", ...),
        Tool(name="titan_git_status", ...),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    match name:
        case "titan_registry_query":
            return await registry_query(arguments)
        case "titan_validate_deps":
            return await validate_deps(arguments)
        ...

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
```

### Server Registration

For Claude Code and other MCP clients:

```json
// .mcp.json or claude_desktop_config.json
{
  "mcpServers": {
    "agentic-titan": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/Users/4jp/Workspace/organvm-iv-taxis/agentic-titan"
    }
  }
}
```

## Client Support: Titan Consumes MCP Servers

### MCP Client in Adapter Layer

The existing adapter layer (`adapters/`) can be extended to consume MCP servers as tool providers:

```python
# adapters/mcp_client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientAdapter:
    """Connects to an MCP server and exposes its tools to Titan agents."""

    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self._session: ClientSession | None = None
        self._tools: list[Tool] = []

    async def connect(self):
        """Establish connection and discover tools."""
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        self._transport = await stdio_client(params)
        self._session = ClientSession(*self._transport)
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = result.tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Invoke a tool on the connected MCP server."""
        result = await self._session.call_tool(name, arguments)
        return result.content
```

### Dynamic Tool Registration

When Titan connects to an MCP server, its tools are dynamically registered in the agent's tool set:

```python
async def register_mcp_tools(agent: BaseAgent, server_config: MCPServerConfig):
    client = MCPClientAdapter(server_config)
    await client.connect()

    for tool in client.tools:
        agent.register_tool(
            name=f"mcp_{server_config.name}_{tool.name}",
            description=tool.description,
            schema=tool.inputSchema,
            handler=lambda args, t=tool.name: client.call_tool(t, args),
        )
```

### Consuming Workspace MCP Servers

```yaml
# titan-config.yaml
mcp_servers:
  filesystem:
    command: "node"
    args: ["~/Workspace/mcp-servers/dist/filesystem.js"]
    env:
      ALLOWED_DIRS: "~/Workspace"

  memory:
    command: "node"
    args: ["~/Workspace/mcp-servers/dist/memory.js"]

  sequential-thinking:
    command: "node"
    args: ["~/Workspace/mcp-servers/dist/sequential-thinking.js"]
```

## Security

### Tool Allowlists

Not all MCP tools should be available to all agents. Titan enforces per-agent allowlists:

```yaml
# Agent spec
agents:
  code-editor:
    mcp_tools:
      allow:
        - "filesystem_read_file"
        - "filesystem_write_file"
        - "filesystem_list_directory"
      deny:
        - "filesystem_delete_*"
        - "memory_*"

  researcher:
    mcp_tools:
      allow:
        - "memory_*"
        - "filesystem_read_file"
      deny:
        - "filesystem_write_file"
        - "filesystem_delete_*"
```

### Rate Limiting

Per-server and per-tool rate limits prevent runaway tool invocations:

```yaml
mcp_servers:
  filesystem:
    rate_limit:
      calls_per_minute: 60
      burst: 10
```

### Audit Logging

All MCP tool calls are logged to the audit system:

```json
{
  "timestamp": "2026-03-08T12:00:00Z",
  "agent": "code-editor",
  "mcp_server": "filesystem",
  "tool": "write_file",
  "arguments": {"path": "src/utils.py"},
  "result": "success",
  "duration_ms": 45
}
```

## Transport Selection

| Scenario | Transport | Rationale |
|----------|-----------|-----------|
| Local MCP servers (filesystem, memory) | stdio | Lowest latency, no network |
| Remote databases (Neon, Supabase) | SSE | Requires network, stateful |
| Cloud APIs (GitHub, Notion) | Streamable HTTP | Scalable, bidirectional |
| Agent-to-agent within Titan | stdio | Same machine, process isolation |

## Implementation Plan

### Phase 1: MCP Client

1. Implement `MCPClientAdapter` in `adapters/mcp_client.py`
2. Add MCP server configuration to `titan-config.yaml`
3. Dynamic tool registration for agents
4. Connect to existing workspace MCP servers (filesystem, memory)

### Phase 2: MCP Server

1. Implement Titan MCP server exposing registry and governance tools
2. Register in workspace `.mcp.json`
3. Test with Claude Code as client

### Phase 3: Security & Governance

1. Tool allowlists per agent
2. Rate limiting per server/tool
3. Audit logging integration
4. Budget tracking for MCP tool calls

### Phase 4: Advanced

1. Tool composition (chain MCP tools into workflows)
2. Resource subscriptions (watch for changes in MCP resources)
3. Prompt template sharing across agents via MCP prompts

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- `~/Workspace/mcp-servers/` — local MCP server infrastructure
- `adapters/` — existing adapter layer
- `adapters/router.py` — routing layer to extend with MCP
