"""
Tool Base - Protocol and registry for tools.

Tools follow a simple protocol:
- name: Unique identifier
- description: What the tool does
- parameters: JSON schema for input
- execute(): Async execution method
"""

from __future__ import annotations

import builtins
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("titan.tools")


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        """Convert to a string message for LLM."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2, default=str)
        else:
            return f"Error: {self.error}"


@dataclass
class ToolParameter:
    """A tool parameter definition."""

    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


class Tool(ABC):
    """
    Abstract base class for tools.

    Subclasses must implement:
    - name: Tool identifier
    - description: What the tool does
    - parameters: List of parameters
    - execute(): Async execution
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with output or error
        """
        pass

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for LLM."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        schema = self.to_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return self.to_schema()


class FunctionTool(Tool):
    """
    Tool wrapper for simple functions.

    Allows creating tools from functions without subclassing.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        func: Callable[..., Awaitable[ToolResult]],
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> list[ToolParameter]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return await self._func(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(success=False, output=None, error=str(e))


class ToolRegistry:
    """
    Registry of available tools.

    Manages tool registration and lookup.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> builtins.list[str]:
        """List all tool names."""
        return builtins.list(self._tools.keys())

    def get_schemas(self, format: str = "anthropic") -> builtins.list[dict[str, Any]]:
        """Get schemas for all tools."""
        if format == "openai":
            return [t.to_openai_schema() for t in self._tools.values()]
        else:
            return [t.to_anthropic_schema() for t in self._tools.values()]

    def filter_by_capabilities(self, capabilities: builtins.list[str]) -> builtins.list[Tool]:
        """Get tools that match capabilities."""
        # Simple name-based matching for now
        return [
            t for t in self._tools.values() if any(cap in t.name.lower() for cap in capabilities)
        ]


# Singleton registry
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the default tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def register_tool(tool: Tool) -> None:
    """Register a tool with the default registry."""
    get_registry().register(tool)


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Callable[[Callable[..., Awaitable[ToolResult]]], Tool]:
    """
    Decorator to create a tool from a function.

    Example:
        @tool("greet", "Greet someone", [ToolParameter("name", "string", "Name")])
        async def greet(name: str) -> ToolResult:
            return ToolResult(success=True, output=f"Hello, {name}!")
    """

    def decorator(func: Callable[..., Awaitable[ToolResult]]) -> Tool:
        t = FunctionTool(name, description, parameters or [], func)
        register_tool(t)
        return t

    return decorator
