"""
Built-in Tools - Core tools available to all agents.

Provides:
- File operations (read, write, list)
- Web search (Tavily API integration)
- Shell commands (sandboxed)
- Math/calculation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, ToolResult, register_tool

logger = logging.getLogger("titan.tools.builtin")


# ============================================================================
# Web Search Configuration
# ============================================================================


class WebSearchConfig:
    """Configuration for web search providers."""

    # Supported providers
    PROVIDER_TAVILY = "tavily"
    PROVIDER_SERPER = "serper"
    PROVIDER_BRAVE = "brave"
    PROVIDER_SIMULATED = "simulated"

    def __init__(self) -> None:
        self.provider = os.getenv("WEB_SEARCH_PROVIDER", self.PROVIDER_TAVILY)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        self.brave_api_key = os.getenv("BRAVE_API_KEY", "")

        # Rate limiting
        self.rate_limit_requests = int(os.getenv("WEB_SEARCH_RATE_LIMIT", "10"))
        self.rate_limit_window_seconds = int(os.getenv("WEB_SEARCH_RATE_WINDOW", "60"))

        # Caching
        self.cache_enabled = os.getenv("WEB_SEARCH_CACHE", "true").lower() == "true"
        self.cache_ttl_seconds = int(os.getenv("WEB_SEARCH_CACHE_TTL", "3600"))

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is configured for the selected provider."""
        if self.provider == self.PROVIDER_TAVILY:
            return bool(self.tavily_api_key)
        elif self.provider == self.PROVIDER_SERPER:
            return bool(self.serper_api_key)
        elif self.provider == self.PROVIDER_BRAVE:
            return bool(self.brave_api_key)
        return True  # Simulated doesn't need a key


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: list[float] = []

    def is_allowed(self) -> bool:
        """Check if a request is allowed under the rate limit."""
        now = time.time()
        # Remove old requests outside the window
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        return len(self._requests) < self.max_requests

    def record_request(self) -> None:
        """Record a new request."""
        self._requests.append(time.time())

    def get_wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.is_allowed():
            return 0.0
        now = time.time()
        oldest = min(self._requests)
        return max(0.0, self.window_seconds - (now - oldest))


class SearchCache:
    """Simple in-memory cache for search results."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}

    def _make_key(self, query: str, num_results: int) -> str:
        """Create a cache key from query parameters."""
        raw = f"{query.lower().strip()}:{num_results}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, num_results: int) -> Any | None:
        """Get cached result if available and not expired."""
        key = self._make_key(query, num_results)
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return result
            # Expired, remove it
            del self._cache[key]
        return None

    def set(self, query: str, num_results: int, result: Any) -> None:
        """Cache a search result."""
        key = self._make_key(query, num_results)
        self._cache[key] = (time.time(), result)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts >= self.ttl_seconds]
        for key in expired:
            del self._cache[key]
        return len(expired)


# ============================================================================
# File Tools
# ============================================================================


class ReadFileTool(Tool):
    """Read contents of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file from the filesystem."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="max_lines",
                type="integer",
                description="Maximum number of lines to read (default: 500)",
                required=False,
                default=500,
            ),
        ]

    async def execute(self, path: str, max_lines: int = 500) -> ToolResult:  # type: ignore[override]
        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}",
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a file: {path}",
                )

            content = file_path.read_text()
            lines = content.split("\n")

            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n\n... (truncated, {len(lines) - max_lines} more lines)"

            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(file_path), "lines": len(lines)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WriteFileTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to write",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
            ToolParameter(
                name="append",
                type="boolean",
                description="Append to file instead of overwriting",
                required=False,
                default=False,
            ),
        ]

    async def execute(  # type: ignore[override]
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode) as f:
                f.write(content)

            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                metadata={"path": str(file_path), "bytes": len(content)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ListDirectoryTool(Tool):
    """List contents of a directory."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories in a given path."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the directory to list",
                required=True,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Whether to list recursively",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="max_depth",
                type="integer",
                description="Maximum depth for recursive listing",
                required=False,
                default=3,
            ),
        ]

    async def execute(  # type: ignore[override]
        self,
        path: str,
        recursive: bool = False,
        max_depth: int = 3,
    ) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Directory not found: {path}",
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a directory: {path}",
                )

            entries = []

            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    if len(rel_path.parts) <= max_depth:
                        entry_type = "dir" if item.is_dir() else "file"
                        entries.append({"path": str(rel_path), "type": entry_type})
            else:
                for item in dir_path.iterdir():
                    entry_type = "dir" if item.is_dir() else "file"
                    entries.append({"path": item.name, "type": entry_type})

            # Sort: directories first, then files
            entries.sort(key=lambda x: (x["type"] == "file", x["path"]))

            return ToolResult(
                success=True,
                output=entries,
                metadata={"path": str(dir_path), "count": len(entries)},
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Web Tools
# ============================================================================


class WebSearchTool(Tool):
    """Search the web for information using Tavily, Serper, or Brave APIs."""

    def __init__(self) -> None:
        self._config = WebSearchConfig()
        self._rate_limiter = RateLimiter(
            max_requests=self._config.rate_limit_requests,
            window_seconds=self._config.rate_limit_window_seconds,
        )
        self._cache = SearchCache(ttl_seconds=self._config.cache_ttl_seconds)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        provider = self._config.provider
        return (
            f"Search the web for information on a topic using {provider}. Returns relevant results."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                type="integer",
                description="Number of results to return (default: 5, max: 20)",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="search_depth",
                type="string",
                description="Search depth: 'basic' or 'advanced' (Tavily only, default: basic)",
                required=False,
                default="basic",
            ),
            ToolParameter(
                name="include_answer",
                type="boolean",
                description="Include AI-generated answer summary (Tavily only, default: false)",
                required=False,
                default=False,
            ),
        ]

    async def execute(  # type: ignore[override]
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = False,
    ) -> ToolResult:
        # Validate and clamp num_results
        num_results = max(1, min(num_results, 20))

        # Check cache first
        if self._config.cache_enabled:
            cached = self._cache.get(query, num_results)
            if cached:
                logger.debug(f"WebSearchTool: Cache hit for '{query}'")
                return ToolResult(
                    success=True,
                    output=cached,
                    metadata={"query": query, "cached": True, "provider": self._config.provider},
                )

        # Check rate limit
        if not self._rate_limiter.is_allowed():
            wait_time = self._rate_limiter.get_wait_time()
            return ToolResult(
                success=False,
                output=None,
                error=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds.",
                metadata={"rate_limited": True, "wait_seconds": wait_time},
            )

        # Record the request
        self._rate_limiter.record_request()

        # Route to appropriate provider
        provider = self._config.provider
        try:
            if provider == WebSearchConfig.PROVIDER_TAVILY:
                results = await self._search_tavily(
                    query, num_results, search_depth, include_answer
                )
            elif provider == WebSearchConfig.PROVIDER_SERPER:
                results = await self._search_serper(query, num_results)
            elif provider == WebSearchConfig.PROVIDER_BRAVE:
                results = await self._search_brave(query, num_results)
            else:
                # Fallback to simulated
                results = self._search_simulated(query, num_results)

            # Cache successful results
            if self._config.cache_enabled and results:
                self._cache.set(query, num_results, results)

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "provider": provider,
                    "num_results": len(results) if isinstance(results, list) else 1,
                    "cached": False,
                },
            )

        except Exception as e:
            logger.error(f"WebSearchTool: Search failed - {e}")
            # Fallback to simulated on error if configured
            if os.getenv("WEB_SEARCH_FALLBACK_SIMULATED", "true").lower() == "true":
                logger.warning("WebSearchTool: Falling back to simulated results")
                results = self._search_simulated(query, num_results)
                return ToolResult(
                    success=True,
                    output=results,
                    metadata={
                        "query": query,
                        "provider": "simulated",
                        "fallback": True,
                        "original_error": str(e),
                    },
                )
            return ToolResult(success=False, output=None, error=str(e))

    async def _search_tavily(
        self,
        query: str,
        num_results: int,
        search_depth: str,
        include_answer: bool,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Search using Tavily API."""
        if not self._config.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not configured")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._config.tavily_api_key,
                    "query": query,
                    "max_results": num_results,
                    "search_depth": search_depth,
                    "include_answer": include_answer,
                    "include_raw_content": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract results
        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
            )

        # Include answer if requested
        if include_answer and data.get("answer"):
            return {
                "answer": data["answer"],
                "results": results,
            }

        return results

    async def _search_serper(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Search using Serper API (Google Search)."""
        if not self._config.serper_api_key:
            raise ValueError("SERPER_API_KEY not configured")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self._config.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": num_results,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract organic results
        results = []
        for item in data.get("organic", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                }
            )

        return results

    async def _search_brave(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Search using Brave Search API."""
        if not self._config.brave_api_key:
            raise ValueError("BRAVE_API_KEY not configured")

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": self._config.brave_api_key,
                    "Accept": "application/json",
                },
                params={
                    "q": query,
                    "count": num_results,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract web results
        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "age": item.get("age", ""),
                }
            )

        return results

    def _search_simulated(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Return simulated search results for testing."""
        logger.warning("WebSearchTool: Using simulated response")
        return [
            {
                "title": f"Result {i + 1} for: {query}",
                "url": f"https://example.com/result{i + 1}",
                "snippet": f"This is a simulated search result for '{query}'. "
                f"In production, this would be real web content.",
                "simulated": True,
            }
            for i in range(min(num_results, 5))
        ]

    def clear_cache(self) -> None:
        """Clear the search result cache."""
        self._cache.clear()
        logger.info("WebSearchTool: Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get tool statistics."""
        return {
            "provider": self._config.provider,
            "has_api_key": self._config.has_api_key,
            "cache_enabled": self._config.cache_enabled,
            "rate_limit": (
                f"{self._config.rate_limit_requests}/{self._config.rate_limit_window_seconds}s"
            ),
        }


class WebFetchTool(Tool):
    """Fetch content from a URL."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch the content of a web page by URL."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="URL to fetch",
                required=True,
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum content length to return",
                required=False,
                default=10000,
            ),
        ]

    async def execute(self, url: str, max_length: int = 10000) -> ToolResult:  # type: ignore[override]
        try:
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

            content = response.text
            if len(content) > max_length:
                content = content[:max_length] + "\n\n... (truncated)"

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", "unknown"),
                },
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Shell Tools
# ============================================================================


class ShellCommandTool(Tool):
    """Execute a shell command."""

    # Commands that are allowed by default
    ALLOWED_COMMANDS = {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "echo",
        "pwd",
        "date",
        "whoami",
        "uname",
        "env",
        "which",
        "python",
        "pip",
        "npm",
        "node",
        "git",
        "curl",
    }

    # Commands that are never allowed
    BLOCKED_COMMANDS = {
        "rm",
        "rmdir",
        "mv",
        "cp",
        "chmod",
        "chown",
        "sudo",
        "su",
        "kill",
        "pkill",
        "shutdown",
        "reboot",
        "dd",
        "mkfs",
    }

    @property
    def name(self) -> str:
        return "shell_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. Some commands are blocked for safety. "
            f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS)[:10])}..."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Shell command to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Command timeout in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:  # type: ignore[override]
        # Extract the base command
        base_command = command.split()[0] if command.split() else ""

        # Check if command is blocked
        if base_command in self.BLOCKED_COMMANDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command '{base_command}' is blocked for safety",
            )

        # Check if command is allowed
        if base_command not in self.ALLOWED_COMMANDS:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command '{base_command}' is not in allowed list. "
                f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS)[:10])}...",
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout.decode() if stdout else ""
            error_output = stderr.decode() if stderr else ""

            if process.returncode == 0:
                return ToolResult(
                    success=True,
                    output=output or "(no output)",
                    metadata={
                        "command": command,
                        "return_code": process.returncode,
                        "stderr": error_output if error_output else None,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    output=output,
                    error=error_output or f"Command failed with code {process.returncode}",
                )

        except TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# ============================================================================
# Utility Tools
# ============================================================================


class CalculatorTool(Tool):
    """Perform mathematical calculations using ast.literal_eval for safety."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Supports basic arithmetic."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                required=True,
            ),
        ]

    async def execute(self, expression: str) -> ToolResult:  # type: ignore[override]
        import ast
        import math
        import operator

        # Supported operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Supported functions
        funcs = {
            "abs": abs,
            "round": round,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
        }

        # Supported constants
        consts = {
            "pi": math.pi,
            "e": math.e,
        }

        def safe_eval(node: ast.AST) -> float:
            """Safely evaluate an AST node."""
            if isinstance(node, ast.Constant):  # Numbers
                return float(node.value)  # type: ignore[arg-type]
            elif isinstance(node, ast.Name):  # Named constants
                if node.id in consts:
                    return consts[node.id]
                raise ValueError(f"Unknown constant: {node.id}")
            elif isinstance(node, ast.BinOp):  # Binary operations
                op_func = ops.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return float(op_func(safe_eval(node.left), safe_eval(node.right)))  # type: ignore[operator]
            elif isinstance(node, ast.UnaryOp):  # Unary operations
                op_func = ops.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return float(op_func(safe_eval(node.operand)))  # type: ignore[operator]
            elif isinstance(node, ast.Call):  # Function calls
                if isinstance(node.func, ast.Name) and node.func.id in funcs:
                    args = [safe_eval(arg) for arg in node.args]
                    return float(funcs[node.func.id](*args))  # type: ignore[operator]
                raise ValueError(f"Unknown function: {getattr(node.func, 'id', 'unknown')}")
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculation error: {e}",
            )


class JsonTool(Tool):
    """Parse or format JSON data."""

    @property
    def name(self) -> str:
        return "json_tool"

    @property
    def description(self) -> str:
        return "Parse JSON string to object, or format object as pretty JSON."

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action: 'parse' or 'format'",
                required=True,
                enum=["parse", "format"],
            ),
            ToolParameter(
                name="data",
                type="string",
                description="JSON string to parse, or object to format",
                required=True,
            ),
        ]

    async def execute(self, action: str, data: str) -> ToolResult:  # type: ignore[override]
        try:
            if action == "parse":
                result = json.loads(data)
                return ToolResult(success=True, output=result)
            elif action == "format":
                obj = json.loads(data) if isinstance(data, str) else data
                formatted = json.dumps(obj, indent=2, default=str)
                return ToolResult(success=True, output=formatted)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"JSON error: {e}",
            )


# ============================================================================
# Registration
# ============================================================================


def register_builtin_tools() -> None:
    """Register all built-in tools."""
    tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        WebSearchTool(),
        WebFetchTool(),
        ShellCommandTool(),
        CalculatorTool(),
        JsonTool(),
    ]

    for tool in tools:
        register_tool(tool)

    logger.info(f"Registered {len(tools)} built-in tools")


# Auto-register on import
register_builtin_tools()
