"""
Titan Dashboard - FastAPI Application

Provides a web interface for managing and monitoring the agent swarm.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dashboard.models import (
    AgentCancelResponse,
    AgentResponse,
    AgentStateEnum,
    ErrorResponse,
    LearningStatsResponse,
    StatusResponse,
    TopologyAgentInfo,
    TopologyHistoryEntry,
    TopologyResponse,
    TopologySwitchResponse,
    TopologyTypeEnum,
)

logger = logging.getLogger("titan.dashboard")

# Try to import FastAPI, provide helpful error if missing
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn jinja2")


# Get package directory for templates
PACKAGE_DIR = Path(__file__).parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def _parse_datetime(value: object) -> datetime:
    """Parse optional ISO datetime values with a safe fallback."""
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.now()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


class TitanDashboard:
    """
    Titan Dashboard application.

    Manages the FastAPI application and integrates with Titan components.
    """

    def __init__(
        self,
        hive_mind: Any | None = None,
        topology_engine: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        """
        Initialize dashboard.

        Args:
            hive_mind: HiveMind instance for memory access
            topology_engine: TopologyEngine instance
            event_bus: EventBus instance for real-time updates
        """
        self.hive_mind = hive_mind
        self.topology_engine = topology_engine
        self.event_bus = event_bus

        self.app: FastAPI | None = None
        self.manager = ConnectionManager()

        # State tracking
        self._active_agents: dict[str, dict[str, Any]] = {}
        self._task_history: list[dict[str, Any]] = []
        self._topology_history: list[dict[str, Any]] = []

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi uvicorn jinja2"
            )

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan handler."""
            logger.info("Dashboard starting...")
            # Subscribe to events if event bus is available
            if self.event_bus:
                await self._setup_event_handlers()
            yield
            logger.info("Dashboard shutting down...")

        self.app = FastAPI(
            title="Titan Dashboard",
            description="Web dashboard for Agentic Titan swarm management",
            version="1.0.0",
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Mount static files
        if STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        # Setup templates
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

        # Register routes
        self._register_routes(templates)

        return self.app

    def _register_routes(self, templates: Any) -> None:
        """Register all routes."""
        if not self.app:
            return

        def render_template(
            request: Request,
            template_name: str,
            context: dict[str, Any],
        ) -> HTMLResponse:
            return cast(HTMLResponse, templates.TemplateResponse(request, template_name, context))

        # ====================================================================
        # HTML Routes
        # ====================================================================

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request) -> HTMLResponse:
            """Dashboard home page."""
            return render_template(
                request,
                "index.html",
                {
                    "request": request,
                    "title": "Titan Dashboard",
                    "active_agents": len(self._active_agents),
                    "current_topology": self._get_current_topology(),
                },
            )

        @self.app.get("/agents", response_class=HTMLResponse)
        async def agents_page(request: Request) -> HTMLResponse:
            """Agents management page."""
            return render_template(
                request,
                "agents.html",
                {
                    "request": request,
                    "title": "Agents",
                    "agents": list(self._active_agents.values()),
                },
            )

        @self.app.get("/topology", response_class=HTMLResponse)
        async def topology_page(request: Request) -> HTMLResponse:
            """Topology visualization page."""
            return render_template(
                request,
                "topology.html",
                {
                    "request": request,
                    "title": "Topology",
                    "current": self._get_current_topology(),
                    "history": self._topology_history[-20:],
                },
            )

        @self.app.get("/models", response_class=HTMLResponse)
        async def models_page(request: Request) -> HTMLResponse:
            """Model cognitive signatures page."""
            return render_template(
                request,
                "models.html",
                {
                    "request": request,
                    "title": "Model Signatures",
                },
            )

        @self.app.get("/inquiry", response_class=HTMLResponse)
        async def inquiry_page(request: Request) -> HTMLResponse:
            """Inquiry sessions page."""
            sessions = await self._get_inquiry_sessions()
            stats = self._get_inquiry_stats(sessions)
            return render_template(
                request,
                "inquiry.html",
                {
                    "request": request,
                    "title": "Inquiry Sessions",
                    "sessions": sessions,
                    "stats": stats,
                },
            )

        @self.app.get("/inquiry/{session_id}", response_class=HTMLResponse)
        async def inquiry_detail_page(request: Request, session_id: str) -> HTMLResponse:
            """Inquiry session detail page."""
            session, stages = await self._get_inquiry_detail(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            import json

            stages_json = json.dumps(
                [
                    {
                        "name": s["name"],
                        "emoji": s.get("emoji", ""),
                        "result": s.get("result"),
                    }
                    for s in stages
                ]
            )

            return render_template(
                request,
                "inquiry_detail.html",
                {
                    "request": request,
                    "title": f"Inquiry: {session.get('topic', '')[:30]}",
                    "session": session,
                    "stages": stages,
                    "stages_json": stages_json,
                },
            )

        @self.app.get("/analysis", response_class=HTMLResponse)
        async def analysis_page(request: Request) -> HTMLResponse:
            """Contradiction analysis page."""
            return render_template(
                request,
                "analysis.html",
                {
                    "request": request,
                    "title": "Contradiction Analysis",
                },
            )

        @self.app.get("/knowledge", response_class=HTMLResponse)
        async def knowledge_page(request: Request) -> HTMLResponse:
            """Knowledge graph browser page."""
            return render_template(
                request,
                "knowledge.html",
                {
                    "request": request,
                    "title": "Knowledge Graph",
                },
            )

        @self.app.get("/lexicon", response_class=HTMLResponse)
        async def lexicon_page(request: Request) -> HTMLResponse:
            """Organizational Body Lexicon visualization page."""
            return render_template(
                request,
                "lexicon.html",
                {
                    "request": request,
                    "title": "Body Lexicon",
                },
            )

        # ====================================================================
        # API Routes
        @self.app.get("/api/knowledge/lexicon")
        async def get_lexicon() -> dict[str, Any]:
            """Get organizational body lexicon."""
            try:
                import yaml

                with open("data/lexicon/seed_lexicon.yaml") as f:
                    loaded = yaml.safe_load(f)
                    return cast(dict[str, Any], loaded if isinstance(loaded, dict) else {})
            except Exception as e:
                logger.error(f"Error loading lexicon: {e}")
                return {}

        # ====================================================================

        @self.app.get("/api/status", response_model=StatusResponse)
        async def get_status() -> StatusResponse:
            """Get system status including active agents and current topology."""
            return StatusResponse(
                status="healthy",
                active_agents=len(self._active_agents),
                current_topology=TopologyTypeEnum(self._get_current_topology()),
                timestamp=datetime.now(),
            )

        @self.app.get("/api/agents", response_model=list[AgentResponse])
        async def get_agents() -> list[AgentResponse]:
            """Get all active agents with their current state and capabilities."""
            return [
                AgentResponse(
                    id=a["id"],
                    name=a.get("name", "unknown"),
                    role=a.get("role", "worker"),
                    state=AgentStateEnum(a.get("state", "running")),
                    joined_at=_parse_datetime(a.get("joined_at")),
                    capabilities=a.get("capabilities", []),
                )
                for a in self._active_agents.values()
            ]

        @self.app.get(
            "/api/agents/{agent_id}",
            response_model=AgentResponse,
            responses={404: {"model": ErrorResponse}},
        )
        async def get_agent(agent_id: str) -> AgentResponse:
            """Get specific agent details by ID."""
            if agent_id not in self._active_agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            a = self._active_agents[agent_id]
            return AgentResponse(
                id=a["id"],
                name=a.get("name", "unknown"),
                role=a.get("role", "worker"),
                state=AgentStateEnum(a.get("state", "running")),
                joined_at=_parse_datetime(a.get("joined_at")),
                capabilities=a.get("capabilities", []),
            )

        @self.app.post(
            "/api/agents/{agent_id}/cancel",
            response_model=AgentCancelResponse,
            responses={404: {"model": ErrorResponse}},
        )
        async def cancel_agent(agent_id: str) -> AgentCancelResponse:
            """Cancel a running agent by ID."""
            if agent_id not in self._active_agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            # In a real implementation, this would cancel the agent
            return AgentCancelResponse(status="cancelled", agent_id=agent_id)

        @self.app.get("/api/topology", response_model=TopologyResponse)
        async def get_topology() -> TopologyResponse:
            """Get current topology state including agents and recent history."""
            topology = self._get_current_topology()
            return TopologyResponse(
                current=TopologyTypeEnum(topology),
                agents=[
                    TopologyAgentInfo(id=a["id"], role=a.get("role", "worker"))
                    for a in self._active_agents.values()
                ],
                history=[
                    TopologyHistoryEntry.model_validate(
                        {
                            "from": TopologyTypeEnum(h["from"]) if h.get("from") else None,
                            "to": TopologyTypeEnum(h["to"]),
                            "timestamp": _parse_datetime(h.get("timestamp")),
                        }
                    )
                    for h in self._topology_history[-10:]
                ],
            )

        @self.app.post(
            "/api/topology/switch/{topology_type}",
            response_model=TopologySwitchResponse,
            responses={400: {"model": ErrorResponse}},
        )
        async def switch_topology(topology_type: TopologyTypeEnum) -> TopologySwitchResponse:
            """Switch to a different topology type. Migrates all agents to the new topology."""
            import time

            start_time = time.time()

            if self.topology_engine:
                try:
                    await self.topology_engine.switch_topology(topology_type.value)
                    duration_ms = (time.time() - start_time) * 1000
                    return TopologySwitchResponse(
                        status="success",
                        new_topology=topology_type,
                        agent_count=len(self._active_agents),
                        duration_ms=duration_ms,
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            else:
                # Mock response
                self._topology_history.append(
                    {
                        "from": self._get_current_topology(),
                        "to": topology_type.value,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                duration_ms = (time.time() - start_time) * 1000
                return TopologySwitchResponse(
                    status="success",
                    new_topology=topology_type,
                    agent_count=len(self._active_agents),
                    duration_ms=duration_ms,
                )

        @self.app.get("/api/inquiry/{session_id}/epistemic_signature")
        async def get_epistemic_signature(session_id: str) -> dict[str, Any]:
            """
            Get epistemic signature for an inquiry session.
            Calculates metrics based on stage results and content.
            """
            try:
                from titan.workflows.inquiry_config import CognitiveStyle
                from titan.workflows.inquiry_engine import get_inquiry_engine

                engine = get_inquiry_engine()
                session = engine.get_session(session_id)

                if not session or not session.results:
                    # Return default/empty data
                    return {
                        "labels": ["Logic", "Mythos", "Lateral", "Recursive", "Pattern"],
                        "datasets": [
                            {
                                "label": "Epistemic Signature",
                                "data": [0, 0, 0, 0, 0],
                                "fill": True,
                                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                                "borderColor": "rgb(54, 162, 235)",
                                "pointBackgroundColor": "rgb(54, 162, 235)",
                                "pointBorderColor": "#fff",
                                "pointHoverBackgroundColor": "#fff",
                                "pointHoverBorderColor": "rgb(54, 162, 235)",
                            }
                        ],
                    }

                # Calculate metrics from session results
                metrics = {
                    CognitiveStyle.STRUCTURED_REASONING.value: 0.0,
                    CognitiveStyle.CREATIVE_SYNTHESIS.value: 0.0,
                    CognitiveStyle.CROSS_DOMAIN.value: 0.0,
                    CognitiveStyle.META_ANALYSIS.value: 0.0,
                    CognitiveStyle.PATTERN_RECOGNITION.value: 0.0,
                }

                # Count occurrences and sum intensity
                counts = {k: 0 for k in metrics.keys()}

                for result in session.results:
                    # Infer style from role if metadata missing
                    style = result.metadata.get("cognitive_style")
                    if not style:
                        # Fallback mapping
                        role_map = {
                            "Logic AI": CognitiveStyle.STRUCTURED_REASONING.value,
                            "Mythos AI": CognitiveStyle.CREATIVE_SYNTHESIS.value,
                            "Bridge AI": CognitiveStyle.CROSS_DOMAIN.value,
                            "Meta AI": CognitiveStyle.META_ANALYSIS.value,
                            "Pattern AI": CognitiveStyle.PATTERN_RECOGNITION.value,
                            "Scope AI": CognitiveStyle.STRUCTURED_REASONING.value,
                        }
                        style = role_map.get(result.role, CognitiveStyle.STRUCTURED_REASONING.value)

                    if style in metrics:
                        # Basic intensity calc: presence + length factor
                        intensity = min(len(result.content.split()) / 500.0, 1.0)
                        metrics[style] += intensity
                        counts[style] += 1

                # Normalize (average intensity per style present, or cumulative?)
                # Radar charts usually show "strength" of dimension.
                # Let's use max observed intensity for each dimension to show peak capability

                data_values = [
                    metrics[CognitiveStyle.STRUCTURED_REASONING.value],
                    metrics[CognitiveStyle.CREATIVE_SYNTHESIS.value],
                    metrics[CognitiveStyle.CROSS_DOMAIN.value],
                    metrics[CognitiveStyle.META_ANALYSIS.value],
                    metrics[CognitiveStyle.PATTERN_RECOGNITION.value],
                ]

                # Normalize to 0-1 range for chart
                max_val = max(data_values) if data_values else 1.0
                if max_val > 0:
                    data_values = [v / max_val for v in data_values]

                return {
                    "labels": ["Logic", "Mythos", "Lateral", "Recursive", "Pattern"],
                    "datasets": [
                        {
                            "label": "Epistemic Signature",
                            "data": data_values,
                            "fill": True,
                            "backgroundColor": "rgba(54, 162, 235, 0.2)",
                            "borderColor": "rgb(54, 162, 235)",
                            "pointBackgroundColor": "rgb(54, 162, 235)",
                            "pointBorderColor": "#fff",
                            "pointHoverBackgroundColor": "#fff",
                            "pointHoverBorderColor": "rgb(54, 162, 235)",
                        }
                    ],
                }

            except Exception as e:
                logger.error(f"Error calculating epistemic signature: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/events")
        async def get_events(limit: int = 50) -> list[dict[str, Any]]:
            """Get recent events."""
            if self.event_bus:
                events = self.event_bus.get_history(limit=limit)
                return [e.to_dict() for e in events]
            return []

        @self.app.get("/api/learning/stats", response_model=LearningStatsResponse)
        async def get_learning_stats() -> LearningStatsResponse:
            """Get episodic learning statistics including per-topology performance."""
            # This would integrate with the EpisodicLearner
            return LearningStatsResponse(
                total_episodes=0,
                topologies={},
            )

        @self.app.get("/api/metrics")
        async def get_metrics() -> str:
            """Get Prometheus metrics."""
            try:
                from titan.metrics import get_metrics_text

                return get_metrics_text()
            except Exception as e:
                return f"# Error getting metrics: {e}"

        # ====================================================================
        # WebSocket Route
        # ====================================================================

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket for real-time updates."""
            await self.manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "subscribe":
                        # Handle subscription requests
                        pass

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)

    async def _setup_event_handlers(self) -> None:
        """Setup event handlers for real-time updates."""
        if not self.event_bus:
            return

        from hive.events import EventType

        async def on_agent_joined(event: Any) -> None:
            agent_data = event.payload
            self._active_agents[agent_data["agent_id"]] = {
                "id": agent_data["agent_id"],
                "name": agent_data.get("name", "unknown"),
                "state": "running",
                "joined_at": event.timestamp.isoformat(),
            }
            await self.manager.broadcast(
                {
                    "type": "agent_joined",
                    "data": agent_data,
                }
            )

        async def on_agent_left(event: Any) -> None:
            agent_id = event.payload.get("agent_id")
            if agent_id in self._active_agents:
                del self._active_agents[agent_id]
            await self.manager.broadcast(
                {
                    "type": "agent_left",
                    "data": event.payload,
                }
            )

        async def on_topology_changed(event: Any) -> None:
            self._topology_history.append(
                {
                    "from": event.payload.get("old_type"),
                    "to": event.payload.get("new_type"),
                    "timestamp": event.timestamp.isoformat(),
                }
            )
            await self.manager.broadcast(
                {
                    "type": "topology_changed",
                    "data": event.payload,
                }
            )

        self.event_bus.subscribe(EventType.AGENT_JOINED, on_agent_joined)
        self.event_bus.subscribe(EventType.AGENT_LEFT, on_agent_left)
        self.event_bus.subscribe(EventType.TOPOLOGY_CHANGED, on_topology_changed)

    def _get_current_topology(self) -> str:
        """Get current topology type."""
        if self.topology_engine and self.topology_engine.current_topology:
            return cast(str, self.topology_engine.current_topology.topology_type.value)
        if self._topology_history:
            return cast(str, self._topology_history[-1].get("to", "swarm"))
        return "swarm"

    async def _get_inquiry_sessions(self) -> list[dict[str, Any]]:
        """Get inquiry sessions for dashboard."""
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            engine = get_inquiry_engine()
            sessions = engine.list_sessions()
            return [s.to_dict() for s in sessions]
        except ImportError:
            return []
        except Exception as e:
            logger.warning(f"Error getting inquiry sessions: {e}")
            return []

    def _get_inquiry_stats(self, sessions: list[dict[str, Any]]) -> dict[str, int]:
        """Get inquiry statistics."""
        stats = {"total": 0, "running": 0, "completed": 0, "failed": 0}
        for s in sessions:
            stats["total"] += 1
            status = s.get("status", "pending")
            if status in stats:
                stats[status] += 1
        return stats

    async def _get_inquiry_detail(
        self, session_id: str
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Get inquiry session detail with stages."""
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            engine = get_inquiry_engine()
            session = engine.get_session(session_id)

            if not session:
                return None, []

            session_dict = session.to_dict()

            # Build stages with results
            stages = []
            for i, stage in enumerate(session.workflow.stages):
                stage_info: dict[str, Any] = {
                    "name": stage.name,
                    "role": stage.role,
                    "description": stage.description,
                    "emoji": stage.emoji,
                    "result": None,
                }
                # Find matching result
                for result in session.results:
                    if result.stage_index == i:
                        stage_info["result"] = result.to_dict()
                        break
                stages.append(stage_info)

            return session_dict, stages

        except ImportError:
            return None, []
        except Exception as e:
            logger.warning(f"Error getting inquiry detail: {e}")
            return None, []

    def register_agent(self, agent_id: str, name: str, role: str = "worker") -> None:
        """Register an agent with the dashboard."""
        self._active_agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "role": role,
            "state": "running",
            "joined_at": datetime.now().isoformat(),
        }

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the dashboard."""
        if agent_id in self._active_agents:
            del self._active_agents[agent_id]


def create_app(
    hive_mind: Any | None = None,
    topology_engine: Any | None = None,
    event_bus: Any | None = None,
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Args:
        hive_mind: HiveMind instance
        topology_engine: TopologyEngine instance
        event_bus: EventBus instance

    Returns:
        Configured FastAPI application
    """
    dashboard = TitanDashboard(
        hive_mind=hive_mind,
        topology_engine=topology_engine,
        event_bus=event_bus,
    )
    return dashboard.create_app()


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    **kwargs: Any,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        **kwargs: Additional uvicorn options
    """
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        return

    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload, **kwargs)
