"""
MCP Resources API - Resource handlers for Titan data.

Provides read-only access to Titan system state including:
- Learning statistics
- RLHF training stats
- Model cognitive signatures
- Topology state
- Recent hive events
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger("titan.mcp.resources")


class ResourceType(StrEnum):
    """Types of MCP resources."""

    LEARNING = "learning"
    MODELS = "models"
    TOPOLOGY = "topology"
    HIVE = "hive"
    INQUIRY = "inquiry"


@dataclass
class MCPResourceDefinition:
    """
    Definition of an MCP resource.

    Resources are read-only data endpoints that provide
    system state and statistics.
    """

    uri: str
    name: str
    description: str
    resource_type: ResourceType
    mime_type: str = "application/json"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


# =============================================================================
# Resource Definitions
# =============================================================================

LEARNING_STATS_RESOURCE = MCPResourceDefinition(
    uri="titan://learning/stats",
    name="Learning Statistics",
    description=(
        "Episodic learning system statistics including episode counts, "
        "topology performance, and learning rate"
    ),
    resource_type=ResourceType.LEARNING,
    metadata={"refresh_interval": 30},
)

RLHF_STATS_RESOURCE = MCPResourceDefinition(
    uri="titan://learning/rlhf/stats",
    name="RLHF Training Statistics",
    description=(
        "Reinforcement Learning from Human Feedback stats including preference "
        "pairs, reward model metrics, and A/B test results"
    ),
    resource_type=ResourceType.LEARNING,
    metadata={"refresh_interval": 60},
)

MODEL_SIGNATURES_RESOURCE = MCPResourceDefinition(
    uri="titan://models/signatures",
    name="Model Cognitive Signatures",
    description=(
        "Cognitive dimension scores for all registered models including "
        "structured reasoning, creative synthesis, and pattern recognition"
    ),
    resource_type=ResourceType.MODELS,
    metadata={"visualization": "radar"},
)

TOPOLOGY_CURRENT_RESOURCE = MCPResourceDefinition(
    uri="titan://topology/current",
    name="Current Topology",
    description=(
        "Active topology configuration including type, agents, connections, and performance metrics"
    ),
    resource_type=ResourceType.TOPOLOGY,
    metadata={"refresh_interval": 10},
)

HIVE_EVENTS_RESOURCE = MCPResourceDefinition(
    uri="titan://hive/events/recent",
    name="Recent Hive Events",
    description=(
        "Recent events from the hive event bus including agent state changes, "
        "topology transitions, and learning feedback"
    ),
    resource_type=ResourceType.HIVE,
    metadata={"max_events": 100, "refresh_interval": 5},
)

INQUIRY_SESSIONS_RESOURCE = MCPResourceDefinition(
    uri="titan://inquiry/sessions",
    name="Active Inquiry Sessions",
    description="Currently active and recent inquiry sessions with status and progress",
    resource_type=ResourceType.INQUIRY,
    metadata={"refresh_interval": 15},
)


# All resource definitions
_RESOURCES: dict[str, MCPResourceDefinition] = {
    "titan://learning/stats": LEARNING_STATS_RESOURCE,
    "titan://learning/rlhf/stats": RLHF_STATS_RESOURCE,
    "titan://models/signatures": MODEL_SIGNATURES_RESOURCE,
    "titan://topology/current": TOPOLOGY_CURRENT_RESOURCE,
    "titan://hive/events/recent": HIVE_EVENTS_RESOURCE,
    "titan://inquiry/sessions": INQUIRY_SESSIONS_RESOURCE,
}


# =============================================================================
# Resource Handlers
# =============================================================================


class ResourceHandler:
    """
    Handler for reading MCP resources.

    Provides async methods to fetch resource data from various
    Titan subsystems.
    """

    def __init__(self) -> None:
        """Initialize the resource handler."""
        self._handlers: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
            "titan://learning/stats": self._read_learning_stats,
            "titan://learning/rlhf/stats": self._read_rlhf_stats,
            "titan://models/signatures": self._read_model_signatures,
            "titan://topology/current": self._read_topology_current,
            "titan://hive/events/recent": self._read_hive_events,
            "titan://inquiry/sessions": self._read_inquiry_sessions,
        }

    async def read(self, uri: str) -> dict[str, Any]:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource contents dict

        Raises:
            ValueError: If URI not found
        """
        handler = self._handlers.get(uri)
        if not handler:
            raise ValueError(f"Unknown resource URI: {uri}")

        return await handler()

    async def _read_learning_stats(self) -> dict[str, Any]:
        """Read episodic learning statistics."""
        try:
            from hive.learning import get_episodic_learner

            learner = get_episodic_learner()
            stats = learner.get_statistics()

            return {
                "total_episodes": stats.get("total_episodes", 0),
                "completed_episodes": stats.get("completed_episodes", 0),
                "unique_profiles": stats.get("unique_profiles", 0),
                "learning_rate": stats.get("learning_rate", 0.1),
                "topology_stats": stats.get("topology_stats", {}),
                "timestamp": datetime.now().isoformat(),
            }
        except ImportError:
            logger.warning("Episodic learner not available")
            return self._empty_learning_stats()
        except Exception as e:
            logger.error(f"Error reading learning stats: {e}")
            return self._empty_learning_stats()

    def _empty_learning_stats(self) -> dict[str, Any]:
        """Return empty learning stats structure."""
        return {
            "total_episodes": 0,
            "completed_episodes": 0,
            "unique_profiles": 0,
            "learning_rate": 0.1,
            "topology_stats": {},
            "timestamp": datetime.now().isoformat(),
            "error": "Learning system not available",
        }

    async def _read_rlhf_stats(self) -> dict[str, Any]:
        """Read RLHF training statistics."""
        try:
            # Try to import RLHF components
            stats: dict[str, Any] = {
                "preference_pairs": {"total": 0, "by_source": {}},
                "reward_model": {"trained": False, "accuracy": 0.0},
                "dpo_trainer": {"runs": 0, "latest_loss": None},
                "ab_tests": {"active": 0, "completed": 0},
                "timestamp": datetime.now().isoformat(),
            }

            # Would need actual storage to get real counts.
            stats["preference_pairs"]["available"] = (
                importlib.util.find_spec("titan.learning.preference_pairs") is not None
            )
            stats["ab_tests"]["available"] = (
                importlib.util.find_spec("titan.learning.deployment") is not None
            )

            return stats

        except Exception as e:
            logger.error(f"Error reading RLHF stats: {e}")
            return {
                "preference_pairs": {"total": 0},
                "reward_model": {"trained": False},
                "dpo_trainer": {"runs": 0},
                "ab_tests": {"active": 0},
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def _read_model_signatures(self) -> dict[str, Any]:
        """Read model cognitive signatures."""
        try:
            from titan.workflows.cognitive_router import (
                MODEL_RANKINGS,
                CognitiveTaskType,
            )

            signatures = {}
            for model_id, traits in MODEL_RANKINGS.items():
                signatures[model_id] = {
                    "structured_reasoning": traits.get(CognitiveTaskType.STRUCTURED_REASONING, 0.5),
                    "creative_synthesis": traits.get(CognitiveTaskType.CREATIVE_SYNTHESIS, 0.5),
                    "mathematical_analysis": traits.get(
                        CognitiveTaskType.MATHEMATICAL_ANALYSIS, 0.5
                    ),
                    "cross_domain": traits.get(CognitiveTaskType.CROSS_DOMAIN, 0.5),
                    "meta_analysis": traits.get(CognitiveTaskType.META_ANALYSIS, 0.5),
                    "pattern_recognition": traits.get(CognitiveTaskType.PATTERN_RECOGNITION, 0.5),
                }

            return {
                "models": signatures,
                "dimensions": [
                    "structured_reasoning",
                    "creative_synthesis",
                    "mathematical_analysis",
                    "cross_domain",
                    "meta_analysis",
                    "pattern_recognition",
                ],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error reading model signatures: {e}")
            return {
                "models": {},
                "dimensions": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def _read_topology_current(self) -> dict[str, Any]:
        """Read current topology state."""
        try:
            from hive.topology import TopologyEngine, TopologyType

            engine = TopologyEngine()

            # Get active topology info
            topology_info: dict[str, Any] = {
                "type": "unknown",
                "agents": [],
                "connections": [],
                "metrics": {},
            }

            topo = engine.current_topology
            if topo is not None:
                topology_info["type"] = topo.topology_type.value
                nodes = topo.list_agents()
                topology_info["agents"] = [
                    {"id": node.agent_id, "name": node.name} for node in nodes
                ]
                topology_info["connections"] = [
                    {"from": node.agent_id, "to": neighbor}
                    for node in nodes
                    for neighbor in node.neighbors
                ]

            return {
                "topology": topology_info,
                "available_types": [t.value for t in TopologyType],
                "timestamp": datetime.now().isoformat(),
            }

        except ImportError:
            logger.warning("Topology engine not available")
            return self._empty_topology()
        except Exception as e:
            logger.error(f"Error reading topology: {e}")
            return self._empty_topology()

    def _empty_topology(self) -> dict[str, Any]:
        """Return empty topology structure."""
        return {
            "topology": {
                "type": "none",
                "agents": [],
                "connections": [],
                "metrics": {},
            },
            "available_types": [],
            "timestamp": datetime.now().isoformat(),
        }

    async def _read_hive_events(self) -> dict[str, Any]:
        """Read recent hive events."""
        try:
            from hive.events import get_event_bus

            event_bus = get_event_bus()
            events = event_bus.get_history(limit=100)

            return {
                "events": [
                    {
                        "type": e.event_type.value,
                        "source_id": e.source_id,
                        "timestamp": e.timestamp.isoformat(),
                        "payload": e.payload,
                    }
                    for e in events
                ],
                "total_count": len(events),
                "timestamp": datetime.now().isoformat(),
            }

        except ImportError:
            logger.warning("Event bus not available")
            return self._empty_events()
        except Exception as e:
            logger.error(f"Error reading events: {e}")
            return self._empty_events()

    def _empty_events(self) -> dict[str, Any]:
        """Return empty events structure."""
        return {
            "events": [],
            "total_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    async def _read_inquiry_sessions(self) -> dict[str, Any]:
        """Read active inquiry sessions."""
        try:
            from titan.workflows.inquiry_engine import get_inquiry_engine

            engine = get_inquiry_engine()
            sessions = engine.list_sessions()

            return {
                "sessions": [
                    {
                        "id": s.id,
                        "topic": s.topic[:100],
                        "workflow": s.workflow.name,
                        "status": s.status.value,
                        "progress": s.progress,
                        "current_stage": s.current_stage,
                        "total_stages": s.total_stages,
                        "created_at": s.created_at.isoformat(),
                    }
                    for s in sessions
                ],
                "total_count": len(sessions),
                "timestamp": datetime.now().isoformat(),
            }

        except ImportError:
            logger.warning("Inquiry engine not available")
            return self._empty_sessions()
        except Exception as e:
            logger.error(f"Error reading inquiry sessions: {e}")
            return self._empty_sessions()

    def _empty_sessions(self) -> dict[str, Any]:
        """Return empty sessions structure."""
        return {
            "sessions": [],
            "total_count": 0,
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# API Functions
# =============================================================================

# Singleton handler
_handler: ResourceHandler | None = None


def get_resource_handler() -> ResourceHandler:
    """Get the resource handler singleton."""
    global _handler
    if _handler is None:
        _handler = ResourceHandler()
    return _handler


def get_all_resources() -> list[MCPResourceDefinition]:
    """Get all resource definitions."""
    return list(_RESOURCES.values())


def get_resource_definition(uri: str) -> MCPResourceDefinition | None:
    """Get a resource definition by URI."""
    return _RESOURCES.get(uri)


def get_resources_by_type(resource_type: ResourceType) -> list[MCPResourceDefinition]:
    """Get resources filtered by type."""
    return [r for r in _RESOURCES.values() if r.resource_type == resource_type]


async def read_resource(uri: str) -> dict[str, Any]:
    """
    Read a resource by URI.

    Args:
        uri: Resource URI

    Returns:
        Resource contents as dict

    Raises:
        ValueError: If URI not found
    """
    handler = get_resource_handler()
    return await handler.read(uri)


def format_resource_contents(
    uri: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Format resource data for MCP response.

    Args:
        uri: Resource URI
        data: Resource data

    Returns:
        MCP-formatted contents dict
    """
    resource = _RESOURCES.get(uri)
    mime_type = resource.mime_type if resource else "application/json"

    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": mime_type,
                "text": json.dumps(data, indent=2, default=str),
            }
        ]
    }
