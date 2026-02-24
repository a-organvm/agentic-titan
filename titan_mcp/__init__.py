"""
Titan MCP Server - Expose agents via Model Context Protocol.

Allows external tools (like Claude Code) to:
- Spawn and manage agents
- Execute agent tasks
- Monitor agent status
- Get agent results
"""

from titan_mcp.server import TitanMCPServer, create_server, run_server

__all__ = [
    "TitanMCPServer",
    "create_server",
    "run_server",
]
