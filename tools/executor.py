"""
Tool Executor - Executes tools based on LLM tool calls.

Handles:
- Parsing tool calls from LLM responses
- Executing tools with proper arguments
- Human-in-the-Loop approval for high-risk actions
- Returning results for conversation continuation
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tools.base import ToolRegistry, ToolResult, get_registry

if TYPE_CHECKING:
    from titan.persistence.audit import AuditLogger
    from titan.safety.hitl import HITLHandler

logger = logging.getLogger("titan.tools.executor")


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolExecution:
    """Result of a tool execution."""

    call: ToolCall
    result: ToolResult
    execution_time_ms: int = 0


class ToolExecutor:
    """
    Executes tools based on LLM tool calls.

    Features:
    - Parse tool calls from various formats
    - Execute tools with argument validation
    - Handle parallel tool execution
    - Timeout and error handling
    - Human-in-the-Loop approval for high-risk actions
    - Audit logging for all tool executions
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_concurrent: int = 5,
        timeout_seconds: float = 30.0,
        hitl_handler: HITLHandler | None = None,
        audit_logger: AuditLogger | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._hitl = hitl_handler
        self._audit_logger = audit_logger
        self._agent_id = agent_id
        self._session_id = session_id

    def set_context(self, agent_id: str, session_id: str) -> None:
        """Set agent and session context for logging."""
        self._agent_id = agent_id
        self._session_id = session_id

    def set_hitl_handler(self, handler: HITLHandler) -> None:
        """Set the HITL handler for approval checks."""
        self._hitl = handler

    def set_audit_logger(self, logger: AuditLogger) -> None:
        """Set the audit logger."""
        self._audit_logger = logger

    def parse_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """
        Parse tool calls from LLM response.

        Handles both Anthropic and OpenAI formats.
        """
        parsed: list[ToolCall] = []

        for tc in tool_calls:
            # Anthropic format: {id, name, arguments}
            # OpenAI format: {id, function: {name, arguments}}
            if "function" in tc:
                # OpenAI format
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                if isinstance(args_raw, str):
                    args = json.loads(args_raw)
                else:
                    args = args_raw
            else:
                # Anthropic format
                name = tc["name"]
                args = tc.get("arguments") or tc.get("input", {})

            parsed.append(
                ToolCall(
                    id=tc.get("id", f"call_{len(parsed)}"),
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        return parsed

    async def execute_one(self, call: ToolCall) -> ToolExecution:
        """Execute a single tool call with HITL check."""
        import time

        start = time.perf_counter()

        tool = self.registry.get(call.name)
        if not tool:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {call.name}",
            )
            return ToolExecution(call=call, result=result, execution_time_ms=0)

        # Log tool call to audit
        await self._audit_tool_called(call)

        # Check HITL approval if handler is configured
        if self._hitl:
            approved, approval_result = await self._hitl.check_action(
                action=f"Execute tool: {call.name}",
                agent_id=self._agent_id or "unknown",
                session_id=self._session_id or "unknown",
                tool_name=call.name,
                arguments=call.arguments,
            )

            if not approved:
                result = ToolResult(
                    success=False,
                    output=None,
                    error=(
                        "Tool execution denied: "
                        f"{approval_result.reason if approval_result else 'Approval required'}"
                    ),
                    metadata={"approval_status": "denied"},
                )
                execution_time_ms = int((time.perf_counter() - start) * 1000)
                await self._audit_tool_completed(call, result, execution_time_ms)
                return ToolExecution(
                    call=call,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )

        # Execute the tool
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    tool.execute(**call.arguments),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                result = ToolResult(
                    success=False,
                    output=None,
                    error=f"Tool timed out after {self.timeout_seconds}s",
                )
            except Exception as e:
                logger.error(f"Tool {call.name} failed: {e}", exc_info=True)
                result = ToolResult(
                    success=False,
                    output=None,
                    error=str(e),
                )

        execution_time_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"Tool {call.name} executed in {execution_time_ms}ms (success={result.success})"
        )

        # Log completion to audit
        await self._audit_tool_completed(call, result, execution_time_ms)

        return ToolExecution(
            call=call,
            result=result,
            execution_time_ms=execution_time_ms,
        )

    async def _audit_tool_called(self, call: ToolCall) -> None:
        """Log tool call to audit."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_tool_called(
                agent_id=self._agent_id or "unknown",
                session_id=self._session_id or "unknown",
                tool_name=call.name,
                arguments=call.arguments,
            )
        except Exception as e:
            logger.warning(f"Failed to audit tool call: {e}")

    async def _audit_tool_completed(
        self,
        call: ToolCall,
        result: ToolResult,
        execution_time_ms: int,
    ) -> None:
        """Log tool completion to audit."""
        if not self._audit_logger:
            return
        try:
            await self._audit_logger.log_tool_completed(
                agent_id=self._agent_id or "unknown",
                session_id=self._session_id or "unknown",
                tool_name=call.name,
                success=result.success,
                output=result.output,
                execution_time_ms=execution_time_ms,
                error=result.error,
            )
        except Exception as e:
            logger.warning(f"Failed to audit tool completion: {e}")

    async def execute_all(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = True,
    ) -> list[ToolExecution]:
        """
        Execute all tool calls.

        Args:
            tool_calls: List of tool calls from LLM
            parallel: Whether to execute in parallel

        Returns:
            List of execution results
        """
        calls = self.parse_tool_calls(tool_calls)

        if not calls:
            return []

        if parallel and len(calls) > 1:
            # Execute in parallel
            tasks = [self.execute_one(call) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            executions: list[ToolExecution] = []
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    executions.append(
                        ToolExecution(
                            call=calls[i],
                            result=ToolResult(success=False, output=None, error=str(result)),
                        )
                    )
                else:
                    executions.append(result)
            return executions
        else:
            # Execute sequentially
            return [await self.execute_one(call) for call in calls]

    def format_results_for_llm(
        self,
        executions: list[ToolExecution],
        format: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """
        Format execution results for LLM continuation.

        Args:
            executions: List of tool executions
            format: "anthropic" or "openai"

        Returns:
            List of tool result messages
        """
        if format == "openai":
            return [
                {
                    "role": "tool",
                    "tool_call_id": ex.call.id,
                    "content": ex.result.to_message(),
                }
                for ex in executions
            ]
        else:
            # Anthropic format
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": ex.call.id,
                    "content": ex.result.to_message(),
                }
                for ex in executions
            ]


# Singleton executor
_default_executor: ToolExecutor | None = None


def get_executor() -> ToolExecutor:
    """Get the default tool executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = ToolExecutor()
    return _default_executor


async def execute_tools(tool_calls: list[dict[str, Any]]) -> list[ToolExecution]:
    """Execute tool calls using the default executor."""
    return await get_executor().execute_all(tool_calls)
