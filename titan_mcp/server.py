"""
Titan MCP Server - JSON-RPC server implementing MCP protocol.

Provides:
- Tool: spawn_agent - Create and run agents
- Tool: agent_status - Check agent state
- Tool: list_agents - List active agents
- Tool: agent_result - Get agent results
- Tool: route_cognitive_task - Route tasks to optimal models
- Tool: compare_models - Compare model cognitive signatures
- Tool: start_inquiry - Start multi-perspective inquiry
- Tool: inquiry_status - Check inquiry session status
- Resource: agent_types - Available agent archetypes
- Resource: learning/stats - Learning system statistics
- Resource: models/signatures - Model cognitive signatures
- Resource: topology/current - Current topology state
- Resource: hive/events/recent - Recent hive events
- Prompts: inquiry prompts for common workflows
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from titan_mcp.notifications import (
    NotificationType,
    get_notification_manager,
    start_notifications,
    stop_notifications,
)
from titan_mcp.prompts import get_all_prompts, get_prompt, get_prompt_messages
from titan_mcp.resources import format_resource_contents, get_all_resources, read_resource

logger = logging.getLogger("titan.mcp")


# ============================================================================
# MCP Protocol Types
# ============================================================================


class MCPMethod(StrEnum):
    """MCP JSON-RPC methods."""

    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    SHUTDOWN = "shutdown"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"


@dataclass
class MCPRequest:
    """Incoming MCP request."""

    jsonrpc: str
    id: int | str | None
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Outgoing MCP response."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    inputSchema: dict[str, Any]  # noqa: N815

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPResource:
    """MCP resource definition."""

    uri: str
    name: str
    description: str
    mimeType: str = "application/json"  # noqa: N815

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType,
        }


# ============================================================================
# Agent Session Management
# ============================================================================


@dataclass
class AgentSession:
    """Active agent session."""

    id: str
    agent_type: str
    task: str
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    result: Any = None
    error: str | None = None


class AgentManager:
    """Manages agent sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def spawn_agent(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> str:
        """Spawn a new agent and return session ID."""
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        session = AgentSession(
            id=session_id,
            agent_type=agent_type,
            task=task,
            status="running",
        )
        self._sessions[session_id] = session

        # Start agent in background
        agent_task = asyncio.create_task(self._run_agent(session, **kwargs))
        self._tasks[session_id] = agent_task

        logger.info(f"Spawned agent {agent_type} with session {session_id}")
        return session_id

    async def _run_agent(self, session: AgentSession, **kwargs: Any) -> None:
        """Run an agent to completion."""
        try:
            # Import agent types
            from agents.archetypes.coder import CoderAgent
            from agents.archetypes.orchestrator import OrchestratorAgent
            from agents.archetypes.researcher import ResearcherAgent
            from agents.archetypes.reviewer import ReviewerAgent
            from agents.framework.tool_agent import SimpleToolAgent

            agent_classes = {
                "researcher": ResearcherAgent,
                "coder": CoderAgent,
                "reviewer": ReviewerAgent,
                "orchestrator": OrchestratorAgent,
                "simple": SimpleToolAgent,
            }

            agent_class = agent_classes.get(session.agent_type.lower())
            if not agent_class:
                session.status = "failed"
                session.error = f"Unknown agent type: {session.agent_type}"
                return

            # Create agent based on type
            if session.agent_type == "researcher":
                agent = agent_class(topic=session.task, **kwargs)
            elif session.agent_type == "coder":
                agent = agent_class(task_description=session.task, **kwargs)
            elif session.agent_type == "reviewer":
                agent = agent_class(content=session.task, **kwargs)
            elif session.agent_type == "orchestrator":
                agent = agent_class(task=session.task, **kwargs)
            else:
                agent = agent_class(task=session.task, **kwargs)

            # Run agent
            result = await agent.run()

            session.status = "completed" if result.success else "failed"
            session.result = result.result
            session.error = result.error

            logger.info(f"Agent {session.id} completed with status {session.status}")

        except Exception as e:
            logger.exception(f"Agent {session.id} failed: {e}")
            session.status = "failed"
            session.error = str(e)

    def get_session(self, session_id: str) -> AgentSession | None:
        """Get agent session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[AgentSession]:
        """List all sessions."""
        return list(self._sessions.values())

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        task = self._tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            session = self._sessions.get(session_id)
            if session:
                session.status = "cancelled"
            return True
        return False


# ============================================================================
# MCP Server
# ============================================================================


class TitanMCPServer:
    """
    MCP Server for Titan multi-agent system.

    Exposes agents via JSON-RPC over stdin/stdout.
    """

    def __init__(self) -> None:
        self._agent_manager = AgentManager()
        self._notification_manager = get_notification_manager()
        self._initialized = False
        self._handlers: dict[str, Callable[..., Awaitable[Any]]] = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.SHUTDOWN.value: self._handle_shutdown,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_READ.value: self._handle_resources_read,
            MCPMethod.PROMPTS_LIST.value: self._handle_prompts_list,
            MCPMethod.PROMPTS_GET.value: self._handle_prompts_get,
        }

    def get_tools(self) -> list[MCPTool]:
        """Get available MCP tools."""
        return [
            MCPTool(
                name="spawn_agent",
                description=(
                    "Spawn a new Titan agent to perform a task. "
                    "Returns a session ID to track progress."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": (
                                "Type of agent: researcher, coder, reviewer, "
                                "orchestrator, or simple"
                            ),
                            "enum": ["researcher", "coder", "reviewer", "orchestrator", "simple"],
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description for the agent",
                        },
                    },
                    "required": ["agent_type", "task"],
                },
            ),
            MCPTool(
                name="agent_status",
                description="Check the status of an agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID returned by spawn_agent",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            MCPTool(
                name="agent_result",
                description="Get the result of a completed agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID returned by spawn_agent",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            MCPTool(
                name="list_agents",
                description="List all active agent sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            MCPTool(
                name="cancel_agent",
                description="Cancel a running agent session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to cancel",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            MCPTool(
                name="route_cognitive_task",
                description=(
                    "Route a cognitive task to the optimal AI model based on task requirements."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the cognitive task",
                        },
                        "cognitive_type": {
                            "type": "string",
                            "description": (
                                "Type of cognitive task: structured_reasoning, "
                                "creative_synthesis, mathematical_analysis, "
                                "cross_domain, meta_analysis, pattern_recognition"
                            ),
                            "enum": [
                                "structured_reasoning",
                                "creative_synthesis",
                                "mathematical_analysis",
                                "cross_domain",
                                "meta_analysis",
                                "pattern_recognition",
                            ],
                        },
                        "preferred_model": {
                            "type": "string",
                            "description": "Optional preferred model ID",
                        },
                    },
                    "required": ["task_description", "cognitive_type"],
                },
            ),
            MCPTool(
                name="compare_models",
                description="Compare two AI models across cognitive dimensions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_a": {
                            "type": "string",
                            "description": "First model ID",
                        },
                        "model_b": {
                            "type": "string",
                            "description": "Second model ID",
                        },
                    },
                    "required": ["model_a", "model_b"],
                },
            ),
            MCPTool(
                name="start_inquiry",
                description="Start a multi-perspective inquiry workflow on a topic.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to explore",
                        },
                        "workflow": {
                            "type": "string",
                            "description": "Workflow to use: expansive, quick, or creative",
                            "enum": ["expansive", "quick", "creative"],
                        },
                        "run_immediately": {
                            "type": "boolean",
                            "description": "Whether to run the workflow immediately",
                        },
                    },
                    "required": ["topic"],
                },
            ),
            MCPTool(
                name="inquiry_status",
                description="Get the status of an inquiry session.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Inquiry session ID",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
        ]

    def get_resources(self) -> list[MCPResource]:
        """Get available MCP resources."""
        # Base resources
        resources = [
            MCPResource(
                uri="titan://agents/types",
                name="Agent Types",
                description="List of available agent archetypes and their capabilities",
            ),
            MCPResource(
                uri="titan://agents/tools",
                name="Agent Tools",
                description="List of tools available to agents",
            ),
        ]

        # Add dynamic resources from resources module
        for resource_def in get_all_resources():
            resources.append(
                MCPResource(
                    uri=resource_def.uri,
                    name=resource_def.name,
                    description=resource_def.description,
                    mimeType=resource_def.mime_type,
                )
            )

        return resources

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request."""
        handler = self._handlers.get(request.method)

        if not handler:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}",
                },
            )

        try:
            result = await handler(request.params)
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            logger.exception(f"Error handling {request.method}: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e),
                },
            )

    # ========================================================================
    # Handlers
    # ========================================================================

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True

        # Load system awareness tools
        from tools.organvm_bridge import load_organvm_tools
        try:
            await load_organvm_tools()
        except Exception as e:
            logger.error(f"Failed to load OrganVM tools: {e}")

        # Start notification system
        await start_notifications()

        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "titan-mcp",
                "version": "0.2.0",
            },
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
        }

    async def _handle_shutdown(self, params: dict[str, Any]) -> None:
        """Handle shutdown request."""
        self._initialized = False

        # Stop notification system
        await stop_notifications()

        return None

    async def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": [t.to_dict() for t in self.get_tools()]}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "spawn_agent":
            session_id = await self._agent_manager.spawn_agent(
                agent_type=arguments["agent_type"],
                task=arguments["task"],
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "session_id": session_id,
                                "status": "running",
                                "message": f"Agent spawned with session {session_id}",
                            }
                        ),
                    }
                ],
            }

        elif name == "agent_status":
            session = self._agent_manager.get_session(arguments["session_id"])
            if not session:
                return {
                    "content": [{"type": "text", "text": "Session not found"}],
                    "isError": True,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "session_id": session.id,
                                "agent_type": session.agent_type,
                                "status": session.status,
                                "created_at": session.created_at.isoformat(),
                                "error": session.error,
                            }
                        ),
                    }
                ],
            }

        elif name == "agent_result":
            session = self._agent_manager.get_session(arguments["session_id"])
            if not session:
                return {
                    "content": [{"type": "text", "text": "Session not found"}],
                    "isError": True,
                }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "session_id": session.id,
                                "status": session.status,
                                "result": session.result,
                                "error": session.error,
                            },
                            default=str,
                        ),
                    }
                ],
            }

        elif name == "list_agents":
            sessions = self._agent_manager.list_sessions()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            [
                                {
                                    "session_id": s.id,
                                    "agent_type": s.agent_type,
                                    "status": s.status,
                                    "task": s.task[:100],
                                }
                                for s in sessions
                            ]
                        ),
                    }
                ],
            }

        elif name == "cancel_agent":
            success = await self._agent_manager.cancel_session(arguments["session_id"])
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "cancelled": success,
                                "session_id": arguments["session_id"],
                            }
                        ),
                    }
                ],
            }

        elif name == "route_cognitive_task":
            return await self._handle_route_cognitive_task(arguments)

        elif name == "compare_models":
            return await self._handle_compare_models(arguments)

        elif name == "start_inquiry":
            return await self._handle_start_inquiry(arguments)

        elif name == "inquiry_status":
            return await self._handle_inquiry_status(arguments)

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }

    async def _handle_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/list request."""
        return {"resources": [r.to_dict() for r in self.get_resources()]}

    async def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri_value = params.get("uri")
        if not isinstance(uri_value, str) or not uri_value:
            return {
                "contents": [
                    {
                        "uri": "",
                        "mimeType": "text/plain",
                        "text": "Missing required parameter: uri",
                    }
                ]
            }
        uri = uri_value

        if uri == "titan://agents/types":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "agent_types": [
                                    {
                                        "name": "researcher",
                                        "description": "Research and analyze information on topics",
                                        "capabilities": [
                                            "web_search",
                                            "document_analysis",
                                            "summarization",
                                        ],
                                    },
                                    {
                                        "name": "coder",
                                        "description": "Write, test, and review code",
                                        "capabilities": [
                                            "code_generation",
                                            "code_review",
                                            "testing",
                                        ],
                                    },
                                    {
                                        "name": "reviewer",
                                        "description": "Review code and documents for quality",
                                        "capabilities": ["code_review", "document_analysis"],
                                    },
                                    {
                                        "name": "orchestrator",
                                        "description": "Coordinate multi-agent workflows",
                                        "capabilities": ["planning", "coordination", "aggregation"],
                                    },
                                    {
                                        "name": "simple",
                                        "description": "Simple tool-using agent",
                                        "capabilities": ["tool_use"],
                                    },
                                ],
                            },
                            indent=2,
                        ),
                    }
                ],
            }

        elif uri == "titan://agents/tools":
            from tools.base import get_registry

            registry = get_registry()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "tools": [
                                    {
                                        "name": t.name,
                                        "description": t.description,
                                    }
                                    for t in registry.list()
                                ],
                            },
                            indent=2,
                        ),
                    }
                ],
            }

        # Try dynamic resources
        if uri.startswith("titan://"):
            try:
                data = await read_resource(uri)
                return format_resource_contents(uri, data)
            except ValueError:
                pass

        return {
            "contents": [{"uri": uri, "mimeType": "text/plain", "text": "Resource not found"}],
        }

    async def _handle_prompts_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompts/list request."""
        prompts = get_all_prompts()
        return {"prompts": [p.to_dict() for p in prompts]}

    async def _handle_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompts/get request."""
        name_value = params.get("name")
        if not isinstance(name_value, str) or not name_value:
            return {
                "description": "Missing required parameter: name",
                "messages": [],
            }
        name = name_value

        arguments_raw = params.get("arguments", {})
        arguments: dict[str, str] = {}
        if isinstance(arguments_raw, dict):
            arguments = {str(k): str(v) for k, v in arguments_raw.items()}

        prompt = get_prompt(name)
        if not prompt:
            return {
                "description": f"Unknown prompt: {name}",
                "messages": [],
            }

        try:
            messages = get_prompt_messages(name, arguments)
            return {
                "description": prompt.description,
                "messages": messages,
            }
        except ValueError as e:
            return {
                "description": str(e),
                "messages": [],
            }

    # ========================================================================
    # New Tool Handlers
    # ========================================================================

    async def _handle_route_cognitive_task(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle route_cognitive_task tool call."""
        try:
            from titan.workflows.cognitive_router import CognitiveTaskType, get_cognitive_router

            task_description = arguments["task_description"]
            cognitive_type_str = arguments["cognitive_type"]
            preferred_model = arguments.get("preferred_model")

            # Map string to enum
            cognitive_type = CognitiveTaskType(cognitive_type_str)

            router = get_cognitive_router()
            routing = await router.route_for_task(
                cognitive_type,
                preferred_model=preferred_model,
            )

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "recommended_model": routing.model_id,
                                "score": routing.score,
                                "reasoning": routing.reasoning,
                                "cognitive_type": cognitive_type_str,
                                "task": task_description[:200],
                            }
                        ),
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Route cognitive task failed: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def _handle_compare_models(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle compare_models tool call."""
        try:
            from titan.workflows.cognitive_router import MODEL_RANKINGS, CognitiveTaskType

            model_a = arguments["model_a"]
            model_b = arguments["model_b"]

            # Get traits for both models
            traits_a = MODEL_RANKINGS.get(model_a)
            traits_b = MODEL_RANKINGS.get(model_b)

            if not traits_a:
                return {
                    "content": [{"type": "text", "text": f"Model not found: {model_a}"}],
                    "isError": True,
                }

            if not traits_b:
                return {
                    "content": [{"type": "text", "text": f"Model not found: {model_b}"}],
                    "isError": True,
                }

            # Compare across dimensions
            dimensions = [
                CognitiveTaskType.STRUCTURED_REASONING,
                CognitiveTaskType.CREATIVE_SYNTHESIS,
                CognitiveTaskType.MATHEMATICAL_ANALYSIS,
                CognitiveTaskType.CROSS_DOMAIN,
                CognitiveTaskType.META_ANALYSIS,
                CognitiveTaskType.PATTERN_RECOGNITION,
            ]

            model_a_scores = {d.value: float(traits_a.get(d, 0.5)) for d in dimensions}
            model_b_scores = {d.value: float(traits_b.get(d, 0.5)) for d in dimensions}
            winner_by_dimension: dict[str, str] = {}

            for d in dimensions:
                key = d.value
                score_a = model_a_scores[key]
                score_b = model_b_scores[key]
                if score_a > score_b:
                    winner_by_dimension[key] = model_a
                elif score_b > score_a:
                    winner_by_dimension[key] = model_b
                else:
                    winner_by_dimension[key] = "tie"

            comparison: dict[str, Any] = {
                "model_a": {
                    "id": model_a,
                    "scores": model_a_scores,
                },
                "model_b": {
                    "id": model_b,
                    "scores": model_b_scores,
                },
                "dimensions": [d.value for d in dimensions],
                "winner_by_dimension": winner_by_dimension,
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(comparison, indent=2),
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Compare models failed: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def _handle_start_inquiry(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle start_inquiry tool call."""
        try:
            from titan.workflows.inquiry_config import get_workflow
            from titan.workflows.inquiry_engine import get_inquiry_engine

            topic = arguments["topic"]
            workflow_name = arguments.get("workflow", "expansive")
            run_immediately = arguments.get("run_immediately", False)

            workflow = get_workflow(workflow_name)
            if not workflow:
                return {
                    "content": [{"type": "text", "text": f"Unknown workflow: {workflow_name}"}],
                    "isError": True,
                }

            engine = get_inquiry_engine()
            session = await engine.start_inquiry(topic, workflow)

            # Notify
            await self._notification_manager.notify(
                NotificationType.INQUIRY_STARTED,
                {
                    "session_id": session.id,
                    "topic": topic[:200],
                    "workflow": workflow_name,
                },
            )

            result = {
                "session_id": session.id,
                "topic": topic,
                "workflow": workflow_name,
                "status": session.status.value,
                "total_stages": session.total_stages,
            }

            if run_immediately:
                # Run in background
                asyncio.create_task(engine.run_full_workflow(session))
                result["message"] = "Workflow started running in background"

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result),
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Start inquiry failed: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def _handle_inquiry_status(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle inquiry_status tool call."""
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            session_id = arguments["session_id"]
            engine = get_inquiry_engine()
            session = engine.get_session(session_id)

            if not session:
                return {
                    "content": [{"type": "text", "text": f"Session not found: {session_id}"}],
                    "isError": True,
                }

            result: dict[str, Any] = {
                "session_id": session.id,
                "topic": session.topic,
                "workflow": session.workflow.name,
                "status": session.status.value,
                "progress": session.progress,
                "current_stage": session.current_stage,
                "total_stages": session.total_stages,
                "stages_completed": len(session.results),
                "created_at": session.created_at.isoformat(),
            }

            if session.started_at:
                result["started_at"] = session.started_at.isoformat()

            if session.completed_at:
                result["completed_at"] = session.completed_at.isoformat()

            if session.error:
                result["error"] = session.error

            # Add stage summaries
            result["stages"] = [
                {
                    "name": r.stage_name,
                    "role": r.role,
                    "model": r.model_used,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                }
                for r in session.results
            ]

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ],
            }

        except Exception as e:
            logger.error(f"Inquiry status failed: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    async def run_stdio(self) -> None:
        """Run the server on stdin/stdout."""
        logger.info("Titan MCP Server starting on stdio...")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        loop = asyncio.get_event_loop()
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

        try:
            while True:
                # Read line (JSON-RPC message)
                line = await reader.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode())
                    request = MCPRequest(
                        jsonrpc=data.get("jsonrpc", "2.0"),
                        id=data.get("id"),
                        method=data.get("method", ""),
                        params=data.get("params", {}),
                    )

                    response = await self.handle_request(request)

                    # Write response
                    response_json = json.dumps(response.to_dict()) + "\n"
                    writer.write(response_json.encode())
                    await writer.drain()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Titan MCP Server shutting down")


def create_server() -> TitanMCPServer:
    """Create a new MCP server instance."""
    return TitanMCPServer()


def run_server() -> None:
    """Run the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    server = create_server()
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    run_server()
