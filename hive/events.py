"""
Hive Mind - Event System

Provides event-based communication for topology transitions and agent coordination.
Enables graceful transitions between topologies with proper notification.

Integrates with PostgreSQL audit logging for durable event storage.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from titan.persistence.audit import AuditLogger

logger = logging.getLogger("titan.hive.events")


class EventType(StrEnum):
    """Types of events in the Hive Mind."""

    # Topology events
    TOPOLOGY_CHANGING = "topology.changing"  # Before switch starts
    TOPOLOGY_CHANGED = "topology.changed"  # After switch completes
    TOPOLOGY_MIGRATION_START = "topology.migration.start"
    TOPOLOGY_MIGRATION_COMPLETE = "topology.migration.complete"

    # Agent events
    AGENT_JOINING = "agent.joining"
    AGENT_JOINED = "agent.joined"
    AGENT_LEAVING = "agent.leaving"
    AGENT_LEFT = "agent.left"
    AGENT_MIGRATING = "agent.migrating"
    AGENT_MIGRATED = "agent.migrated"

    # Task events
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Learning events
    EPISODE_RECORDED = "learning.episode.recorded"
    MODEL_UPDATED = "learning.model.updated"

    # Stigmergy events
    PHEROMONE_DEPOSITED = "stigmergy.pheromone.deposited"
    PHEROMONE_DECAYED = "stigmergy.pheromone.decayed"
    TRAIL_FOLLOWED = "stigmergy.trail.followed"

    # Neighborhood events
    NEIGHBOR_CHANGED = "neighborhood.neighbor.changed"
    INFORMATION_PROPAGATED = "neighborhood.info.propagated"

    # Assembly events
    TERRITORY_CREATED = "assembly.territory.created"
    TERRITORY_DISSOLVED = "assembly.territory.dissolved"
    LINE_OF_FLIGHT = "assembly.line.of.flight"
    CAPTURE_ATTEMPTED = "assembly.capture.attempted"
    TERRITORIALIZED = "assembly.territorialized"
    DETERRITORIALIZED = "assembly.deterritorialized"

    # Governance events
    PROPOSAL_CREATED = "governance.proposal.created"
    VOTE_CAST = "governance.vote.cast"
    PROPOSAL_PASSED = "governance.proposal.passed"
    PROPOSAL_REJECTED = "governance.proposal.rejected"

    # Colony events
    COLONY_SIGNAL_EMITTED = "colony.signal.emitted"
    CASTE_ASSIGNED = "colony.caste.assigned"
    COLONY_TASK_CREATED = "colony.task.created"

    # Cell events
    CELL_APOPTOSIS = "cell.apoptosis"
    CELL_SIGNAL_SENT = "cell.signal.sent"
    CELL_DIFFERENTIATED = "cell.differentiated"

    # Criticality events (Phase 16)
    CRITICALITY_MEASURED = "criticality.measured"
    PHASE_TRANSITION_DETECTED = "criticality.phase_transition"
    CRITICALITY_INTERVENTION = "criticality.intervention"

    # Fission-fusion events (Phase 16)
    FISSION_INITIATED = "fission_fusion.fission.initiated"
    FISSION_COMPLETED = "fission_fusion.fission.completed"
    FUSION_INITIATED = "fission_fusion.fusion.initiated"
    FUSION_COMPLETED = "fission_fusion.fusion.completed"

    # Conflict detection events (#64)
    CONFLICT_DETECTED = "stigmergy.conflict.detected"

    # War machine events (Phase 16)
    SMOOTH_OPERATION = "machine.smooth"
    NOMADIZE_OPERATION = "machine.nomadize"
    DETERRITORIALIZE_OPERATION = "machine.deterritorialize"

    # Information center events (Phase 16)
    INFO_CENTER_CREATED = "info_center.created"
    INFO_CENTER_ELECTED = "info_center.elected"
    PATTERN_AGGREGATED = "info_center.pattern.aggregated"
    PATTERN_BROADCAST = "info_center.pattern.broadcast"
    GENERATION_ARCHIVED = "info_center.generation.archived"


@dataclass
class Event:
    """Represents an event in the system."""

    event_type: EventType
    payload: dict[str, Any]
    source_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_id": self.source_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Deserialize event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source_id=data.get("source_id"),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Central event bus for the Hive Mind.

    Supports:
    - Async event handlers
    - Multiple handlers per event type
    - Event filtering
    - Event history for debugging
    - PostgreSQL audit logging sink
    """

    def __init__(
        self,
        max_history: int = 1000,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._history: list[Event] = []
        self._max_history = max_history
        self._paused = False
        self._audit_logger = audit_logger

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to call when event occurs

        Returns:
            Unsubscribe function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type.value}")

        def unsubscribe() -> None:
            if handler in self._handlers.get(event_type, []):
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.value}")

        return unsubscribe

    def set_audit_logger(self, audit_logger: AuditLogger) -> None:
        """Set the audit logger for persistent event storage."""
        self._audit_logger = audit_logger

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        if self._paused:
            logger.debug(f"Event bus paused, skipping {event.event_type.value}")
            return

        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Persist to PostgreSQL audit log if available
        if self._audit_logger:
            await self._persist_to_audit_log(event)

        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug(f"No handlers for {event.event_type.value}")
            return

        logger.debug(f"Publishing {event.event_type.value} to {len(handlers)} handlers")

        # Run handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler error for {event.event_type.value}: {result}")

    async def _persist_to_audit_log(self, event: Event) -> None:
        """Persist event to PostgreSQL audit log."""
        audit_logger = self._audit_logger
        if audit_logger is None:
            return

        try:
            from titan.persistence.models import AuditEventType

            # Map hive events to audit event types
            event_type_map = {
                EventType.TOPOLOGY_CHANGED: AuditEventType.TOPOLOGY_CHANGED,
                EventType.TOPOLOGY_CHANGING: AuditEventType.TOPOLOGY_CHANGED,
                EventType.AGENT_JOINED: AuditEventType.AGENT_CREATED,
                EventType.AGENT_LEFT: AuditEventType.AGENT_COMPLETED,
                EventType.TASK_COMPLETED: AuditEventType.AGENT_COMPLETED,
                EventType.TASK_FAILED: AuditEventType.AGENT_FAILED,
            }

            audit_type = event_type_map.get(event.event_type)
            if audit_type:
                await audit_logger.log_event(
                    event_type=audit_type,
                    action=f"Hive event: {event.event_type.value}",
                    agent_id=event.source_id,
                    input_data=event.payload,
                    metadata={
                        "hive_event_id": event.event_id,
                        "hive_event_type": event.event_type.value,
                        **event.metadata,
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to persist event to audit log: {e}")

    async def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        source_id: str | None = None,
        **metadata: Any,
    ) -> Event:
        """
        Convenience method to create and publish an event.

        Args:
            event_type: Type of event
            payload: Event data
            source_id: ID of the event source
            **metadata: Additional metadata

        Returns:
            The created event
        """
        event = Event(
            event_type=event_type,
            payload=payload,
            source_id=source_id,
            metadata=metadata,
        )
        await self.publish(event)
        return event

    def pause(self) -> None:
        """Pause event processing."""
        self._paused = True
        logger.info("Event bus paused")

    def resume(self) -> None:
        """Resume event processing."""
        self._paused = False
        logger.info("Event bus resumed")

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum events to return

        Returns:
            List of events (most recent last)
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
        logger.debug("Event history cleared")


# Singleton event bus
_default_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the default event bus."""
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = EventBus()
    return _default_event_bus


# Topology-specific event helpers
async def emit_topology_changing(
    event_bus: EventBus,
    old_type: str,
    new_type: str,
    agent_count: int,
) -> Event:
    """Emit event before topology switch."""
    return await event_bus.emit(
        EventType.TOPOLOGY_CHANGING,
        {
            "old_type": old_type,
            "new_type": new_type,
            "agent_count": agent_count,
        },
        source_id="topology_engine",
    )


async def emit_topology_changed(
    event_bus: EventBus,
    old_type: str,
    new_type: str,
    agent_count: int,
    duration_ms: float,
) -> Event:
    """Emit event after topology switch."""
    return await event_bus.emit(
        EventType.TOPOLOGY_CHANGED,
        {
            "old_type": old_type,
            "new_type": new_type,
            "agent_count": agent_count,
            "duration_ms": duration_ms,
        },
        source_id="topology_engine",
    )


async def emit_agent_migrating(
    event_bus: EventBus,
    agent_id: str,
    from_topology: str,
    to_topology: str,
) -> Event:
    """Emit event when agent starts migration."""
    return await event_bus.emit(
        EventType.AGENT_MIGRATING,
        {
            "agent_id": agent_id,
            "from_topology": from_topology,
            "to_topology": to_topology,
        },
        source_id=agent_id,
    )


async def emit_agent_migrated(
    event_bus: EventBus,
    agent_id: str,
    new_topology: str,
    new_role: str | None,
) -> Event:
    """Emit event when agent completes migration."""
    return await event_bus.emit(
        EventType.AGENT_MIGRATED,
        {
            "agent_id": agent_id,
            "new_topology": new_topology,
            "new_role": new_role,
        },
        source_id=agent_id,
    )
