"""
Semantic Code Tool - LSP-based code analysis and navigation.

Provides semantic understanding of code through Language Server Protocol:
- Symbol discovery (classes, functions, variables)
- Reference finding
- Definition/declaration navigation
- Hierarchical code structure analysis

Reference: vendor/agents/serena/ LSP integration patterns

Note: Full LSP integration requires external language servers.
This implementation provides a simplified interface with fallback
to tree-sitter for basic symbol extraction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolParameter, ToolResult, register_tool

logger = logging.getLogger("titan.tools.semantic_code")


# ============================================================================
# Symbol Types (LSP-compatible)
# ============================================================================


class SymbolKind(IntEnum):
    """Symbol kinds matching LSP specification."""

    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14
    STRING = 15
    NUMBER = 16
    BOOLEAN = 17
    ARRAY = 18
    OBJECT = 19
    KEY = 20
    NULL = 21
    ENUM_MEMBER = 22
    STRUCT = 23
    EVENT = 24
    OPERATOR = 25
    TYPE_PARAMETER = 26


@dataclass
class Position:
    """A position in a file (0-based line and column)."""

    line: int
    character: int

    def to_dict(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}


@dataclass
class Range:
    """A range in a file."""

    start: Position
    end: Position

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}


@dataclass
class Location:
    """A location (file URI + range)."""

    uri: str
    range: Range

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "range": self.range.to_dict()}


@dataclass
class Symbol:
    """
    A code symbol (class, function, variable, etc.).

    Matches LSP DocumentSymbol structure with extensions.
    """

    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range | None = None
    detail: str = ""
    children: list[Symbol] = field(default_factory=list)

    # Extended fields
    file_path: str = ""
    body: str = ""
    parent_name: str = ""

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.name,
            "kind_value": self.kind.value,
            "range": self.range.to_dict(),
            "detail": self.detail,
            "file_path": self.file_path,
        }
        if self.parent_name:
            result["parent"] = self.parent_name
        if include_children and self.children:
            result["children"] = [c.to_dict(include_children) for c in self.children]
        return result

    def get_name_path(self) -> str:
        """Get hierarchical name path (e.g., 'MyClass/method')."""
        if self.parent_name:
            return f"{self.parent_name}/{self.name}"
        return self.name


# ============================================================================
# Python Symbol Extraction (Fallback when no LSP available)
# ============================================================================


def extract_python_symbols(content: str, file_path: str = "") -> list[Symbol]:
    """
    Extract symbols from Python code using regex patterns.

    This is a simplified fallback when LSP is not available.
    For production, use pyright or pylsp.
    """
    symbols: list[Symbol] = []

    # Class pattern
    class_pattern = re.compile(
        r"^(\s*)class\s+(\w+)(?:\([^)]*\))?\s*:",
        re.MULTILINE,
    )

    # Function pattern
    func_pattern = re.compile(
        r"^(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?\s*:",
        re.MULTILINE,
    )

    # Variable/constant pattern (module level assignments)
    var_pattern = re.compile(
        r"^([A-Z_][A-Z0-9_]*)\s*[=:]",
        re.MULTILINE,
    )

    lines = content.split("\n")

    # Find classes
    for match in class_pattern.finditer(content):
        indent = len(match.group(1))
        name = match.group(2)
        line_num = content[: match.start()].count("\n")

        # Determine class end by indentation
        end_line = line_num
        for i in range(line_num + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(" " * (indent + 1)):
                if not lines[i].startswith(" " * indent + " "):
                    break
            end_line = i

        symbol = Symbol(
            name=name,
            kind=SymbolKind.CLASS,
            range=Range(
                start=Position(line_num, indent),
                end=Position(end_line, len(lines[end_line]) if end_line < len(lines) else 0),
            ),
            file_path=file_path,
        )

        # Find methods within class
        class_content = "\n".join(lines[line_num : end_line + 1])
        for method_match in func_pattern.finditer(class_content):
            method_indent = len(method_match.group(1))
            if method_indent > indent:  # Inside class
                method_name = method_match.group(3)
                method_line = line_num + class_content[: method_match.start()].count("\n")

                is_async = bool(method_match.group(2))
                kind = SymbolKind.METHOD

                method_symbol = Symbol(
                    name=method_name,
                    kind=kind,
                    range=Range(
                        start=Position(method_line, method_indent),
                        end=Position(method_line, method_indent),  # Simplified
                    ),
                    detail="async" if is_async else "",
                    parent_name=name,
                    file_path=file_path,
                )
                symbol.children.append(method_symbol)

        symbols.append(symbol)

    # Find module-level functions
    for match in func_pattern.finditer(content):
        indent = len(match.group(1))
        if indent == 0:  # Module level
            name = match.group(3)
            line_num = content[: match.start()].count("\n")
            is_async = bool(match.group(2))

            symbol = Symbol(
                name=name,
                kind=SymbolKind.FUNCTION,
                range=Range(
                    start=Position(line_num, 0),
                    end=Position(line_num, 0),  # Simplified
                ),
                detail="async" if is_async else "",
                file_path=file_path,
            )
            symbols.append(symbol)

    # Find constants
    for match in var_pattern.finditer(content):
        name = match.group(1)
        line_num = content[: match.start()].count("\n")

        symbol = Symbol(
            name=name,
            kind=SymbolKind.CONSTANT,
            range=Range(
                start=Position(line_num, 0),
                end=Position(line_num, len(match.group(0))),
            ),
            file_path=file_path,
        )
        symbols.append(symbol)

    return symbols


def extract_symbols_from_file(file_path: str | Path) -> list[Symbol]:
    """
    Extract symbols from a file based on extension.

    Args:
        file_path: Path to the source file

    Returns:
        List of extracted symbols
    """
    path = Path(file_path)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    if suffix == ".py":
        return extract_python_symbols(content, str(path))
    else:
        # For other languages, return file-level symbol only
        return [
            Symbol(
                name=path.name,
                kind=SymbolKind.FILE,
                range=Range(
                    start=Position(0, 0),
                    end=Position(content.count("\n"), 0),
                ),
                file_path=str(path),
            )
        ]


# ============================================================================
# Reference Finding
# ============================================================================


def find_references_in_file(
    file_path: str | Path,
    symbol_name: str,
) -> list[dict[str, Any]]:
    """
    Find references to a symbol in a file.

    Simplified text-based search. For production, use LSP.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    references = []
    pattern = re.compile(rf"\b{re.escape(symbol_name)}\b")

    for line_num, line in enumerate(lines):
        for match in pattern.finditer(line):
            # Get context
            start = max(0, match.start() - 20)
            end = min(len(line), match.end() + 20)
            context = line[start:end]
            if start > 0:
                context = "..." + context
            if end < len(line):
                context = context + "..."

            references.append(
                {
                    "file": str(path),
                    "line": line_num + 1,
                    "column": match.start() + 1,
                    "context": context.strip(),
                }
            )

    return references


# ============================================================================
# Semantic Code Tool Implementation
# ============================================================================


class SemanticCodeTool(Tool):
    """
    Semantic code analysis tool.

    Provides symbol-level code understanding:
    - List symbols in a file
    - Find symbol definitions
    - Find references to symbols
    - Navigate code structure
    """

    @property
    def name(self) -> str:
        return "semantic_code"

    @property
    def description(self) -> str:
        return (
            "Semantic code analysis tool for understanding code structure. "
            "Actions: 'symbols' (list symbols in file), 'find' (find symbol by name), "
            "'references' (find all references), 'outline' (get code outline)."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action: 'symbols', 'find', 'references', 'outline'",
                required=True,
                enum=["symbols", "find", "references", "outline"],
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Path to source file or directory",
                required=True,
            ),
            ToolParameter(
                name="symbol_name",
                type="string",
                description="Symbol name to find or get references for",
                required=False,
            ),
            ToolParameter(
                name="kind",
                type="string",
                description="Filter by symbol kind: 'class', 'function', 'method', 'variable'",
                required=False,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Search recursively in directories",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        path = kwargs.get("path", "")

        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="'path' parameter is required",
            )

        try:
            path_obj = Path(path)

            if action == "symbols":
                if not path_obj.exists():
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Path not found: {path}",
                    )

                if path_obj.is_file():
                    symbols = extract_symbols_from_file(path_obj)
                else:
                    # Directory: get symbols from all files
                    symbols = []
                    recursive = kwargs.get("recursive", False)
                    pattern = "**/*.py" if recursive else "*.py"
                    for py_file in path_obj.glob(pattern):
                        symbols.extend(extract_symbols_from_file(py_file))

                # Filter by kind if specified
                kind_filter = kwargs.get("kind", "").lower()
                if kind_filter:
                    kind_map = {
                        "class": SymbolKind.CLASS,
                        "function": SymbolKind.FUNCTION,
                        "method": SymbolKind.METHOD,
                        "variable": SymbolKind.VARIABLE,
                        "constant": SymbolKind.CONSTANT,
                    }
                    if kind_filter in kind_map:
                        target_kind = kind_map[kind_filter]
                        symbols = [s for s in symbols if s.kind == target_kind]

                return ToolResult(
                    success=True,
                    output={
                        "path": str(path),
                        "symbol_count": len(symbols),
                        "symbols": [s.to_dict(include_children=True) for s in symbols],
                    },
                )

            elif action == "find":
                symbol_name = kwargs.get("symbol_name", "")
                if not symbol_name:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'symbol_name' parameter required for find action",
                    )

                # Search for symbol
                found = []
                if path_obj.is_file():
                    files = [path_obj]
                else:
                    recursive = kwargs.get("recursive", True)
                    pattern = "**/*.py" if recursive else "*.py"
                    files = list(path_obj.glob(pattern))

                for py_file in files:
                    symbols = extract_symbols_from_file(py_file)
                    for symbol in symbols:
                        if symbol.name == symbol_name or symbol_name in symbol.get_name_path():
                            found.append(symbol.to_dict())
                        # Check children
                        for child in symbol.children:
                            if child.name == symbol_name or symbol_name in child.get_name_path():
                                found.append(child.to_dict())

                return ToolResult(
                    success=True,
                    output={
                        "symbol_name": symbol_name,
                        "found_count": len(found),
                        "results": found[:20],  # Limit results
                    },
                )

            elif action == "references":
                symbol_name = kwargs.get("symbol_name", "")
                if not symbol_name:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'symbol_name' parameter required for references action",
                    )

                # Find references
                references = []
                if path_obj.is_file():
                    files = [path_obj]
                else:
                    recursive = kwargs.get("recursive", True)
                    pattern = "**/*.py" if recursive else "*.py"
                    files = list(path_obj.glob(pattern))

                for py_file in files:
                    refs = find_references_in_file(py_file, symbol_name)
                    references.extend(refs)

                return ToolResult(
                    success=True,
                    output={
                        "symbol_name": symbol_name,
                        "reference_count": len(references),
                        "references": references[:50],  # Limit results
                    },
                )

            elif action == "outline":
                if not path_obj.is_file():
                    return ToolResult(
                        success=False,
                        output=None,
                        error="'outline' action requires a file path",
                    )

                symbols = extract_symbols_from_file(path_obj)

                # Build outline string
                outline_lines = [f"# {path_obj.name}"]
                for symbol in symbols:
                    prefix = "class" if symbol.kind == SymbolKind.CLASS else "def"
                    outline_lines.append(f"\n{prefix} {symbol.name}")
                    for child in symbol.children:
                        child_prefix = "async def" if child.detail == "async" else "def"
                        outline_lines.append(f"    {child_prefix} {child.name}")

                return ToolResult(
                    success=True,
                    output={
                        "path": str(path_obj),
                        "outline": "\n".join(outline_lines),
                        "symbol_count": len(symbols),
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action: {action}",
                )

        except Exception as e:
            logger.exception(f"Semantic code tool error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Analysis failed: {e}",
            )


# Register the tool
semantic_code_tool = SemanticCodeTool()
register_tool(semantic_code_tool)
