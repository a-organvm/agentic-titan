"""
MCP Client Tool - Consumes tools from an external MCP server.

Allows Titan agents to use any tool exposed by an MCP server (like organvm-mcp-server).
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tools.base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("titan.tools.mcp")


class MCPToolAdapter(Tool):
    """
    Adapts a single tool from an MCP server into a Titan Tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        session: ClientSession,
    ) -> None:
        self._name = name
        self._description = description
        self._session = session
        self._parameters = self._parse_schema(input_schema)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    def _parse_schema(self, schema: dict[str, Any]) -> list[ToolParameter]:
        """Convert JSON schema to Titan ToolParameters."""
        params = []
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        for key, prop in props.items():
            params.append(
                ToolParameter(
                    name=key,
                    type=prop.get("type", "string"),
                    description=prop.get("description", ""),
                    required=key in required,
                    default=prop.get("default"),
                    enum=prop.get("enum"),
                )
            )
        return params

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Call the tool on the remote MCP server."""
        try:
            result = await self._session.call_tool(self.name, arguments=kwargs)
            
            # MCP returns a list of content (TextContent or ImageContent)
            # We concatenate text content for now
            output_text = ""
            if result.content:
                for content in result.content:
                    if content.type == "text":
                        output_text += content.text
                    elif content.type == "image":
                        output_text += f"[Image: {content.mimeType}]"
                        
            return ToolResult(success=True, output=output_text)
        except Exception as e:
            logger.error(f"MCP tool {self.name} failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))


class MCPClient:
    """
    Manages connection to an MCP server and discovers tools.
    """

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        self.params = StdioServerParameters(command=command, args=args or [], env=env)
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        """Connect to the MCP server."""
        read, write = await self.exit_stack.enter_async_context(stdio_client(self.params))
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

    async def get_tools(self) -> list[MCPToolAdapter]:
        """Discover tools from the server."""
        if not self.session:
            raise RuntimeError("Client not connected")

        response = await self.session.list_tools()
        tools = []
        for tool_info in response.tools:
            adapter = MCPToolAdapter(
                name=tool_info.name,
                description=tool_info.description or "",
                input_schema=tool_info.inputSchema,
                session=self.session,
            )
            tools.append(adapter)
        return tools

    async def close(self) -> None:
        """Close the connection."""
        await self.exit_stack.aclose()
