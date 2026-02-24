"""
MCP Notifications API - Real-time event notifications.

Provides a notification system that integrates with the Hive event bus
to deliver real-time updates to MCP clients about:
- Agent state changes
- Topology transitions
- Inquiry stage progress
- Learning feedback events
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

logger = logging.getLogger("titan.mcp.notifications")


class NotificationType(StrEnum):
    """Types of MCP notifications."""

    # Agent notifications
    AGENT_STATE_CHANGED = "agent/state"
    AGENT_SPAWNED = "agent/spawned"
    AGENT_COMPLETED = "agent/completed"
    AGENT_FAILED = "agent/failed"

    # Topology notifications
    TOPOLOGY_CHANGED = "topology/changed"
    TOPOLOGY_AGENT_ADDED = "topology/agent_added"
    TOPOLOGY_AGENT_REMOVED = "topology/agent_removed"

    # Inquiry notifications
    INQUIRY_STARTED = "inquiry/started"
    INQUIRY_STAGE_STARTED = "inquiry/stage"
    INQUIRY_STAGE_COMPLETED = "inquiry/stage_completed"
    INQUIRY_COMPLETED = "inquiry/completed"
    INQUIRY_FAILED = "inquiry/failed"
    INQUIRY_PAUSED = "inquiry/paused"
    INQUIRY_RESUMED = "inquiry/resumed"

    # Learning notifications
    LEARNING_FEEDBACK = "learning/feedback"
    LEARNING_EPISODE_RECORDED = "learning/episode"
    LEARNING_PREFERENCE_UPDATED = "learning/preference"

    # RLHF notifications
    RLHF_PAIR_CREATED = "rlhf/pair_created"
    RLHF_TRAINING_STARTED = "rlhf/training_started"
    RLHF_TRAINING_COMPLETED = "rlhf/training_completed"
    RLHF_AB_TEST_STARTED = "rlhf/ab_test_started"
    RLHF_AB_TEST_COMPLETED = "rlhf/ab_test_completed"

    # System notifications
    SYSTEM_ERROR = "system/error"
    SYSTEM_WARNING = "system/warning"


@dataclass
class MCPNotification:
    """
    An MCP notification message.

    Follows MCP notification format for JSON-RPC.
    """

    notification_type: NotificationType
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_jsonrpc(self) -> dict[str, Any]:
        """Convert to JSON-RPC notification format."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "type": self.notification_type.value,
                "data": self.data,
                "timestamp": self.timestamp.isoformat(),
                "id": self.id,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict."""
        return {
            "type": self.notification_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "id": self.id,
        }


# Type for notification handlers
NotificationHandler = Callable[[MCPNotification], Awaitable[None]]


class NotificationManager:
    """
    Manager for MCP notifications.

    Handles:
    - Subscriber registration
    - Event bus integration
    - Notification dispatch
    - Notification history
    """

    def __init__(
        self,
        history_limit: int = 1000,
        enable_event_bus: bool = True,
    ) -> None:
        """
        Initialize the notification manager.

        Args:
            history_limit: Maximum notifications to keep in history
            enable_event_bus: Whether to integrate with Hive event bus
        """
        self._subscribers: list[NotificationHandler] = []
        self._type_subscribers: dict[NotificationType, list[NotificationHandler]] = {}
        self._history: list[MCPNotification] = []
        self._history_limit = history_limit
        self._event_bus_enabled = enable_event_bus
        self._event_bus_connected = False
        self._running = False

        logger.info("Notification manager initialized")

    async def start(self) -> None:
        """Start the notification manager and connect to event bus."""
        if self._running:
            return

        self._running = True

        if self._event_bus_enabled:
            await self._connect_event_bus()

        logger.info("Notification manager started")

    async def stop(self) -> None:
        """Stop the notification manager."""
        self._running = False
        self._event_bus_connected = False
        logger.info("Notification manager stopped")

    async def _connect_event_bus(self) -> None:
        """Connect to the Hive event bus for automatic notifications."""
        try:
            from hive.events import EventType, get_event_bus

            event_bus = get_event_bus()

            # Map event types to notification types
            event_mappings: dict[EventType, NotificationType] = {
                EventType.AGENT_JOINED: NotificationType.AGENT_SPAWNED,
                EventType.AGENT_LEFT: NotificationType.AGENT_COMPLETED,
                EventType.TASK_FAILED: NotificationType.AGENT_FAILED,
                EventType.TOPOLOGY_CHANGED: NotificationType.TOPOLOGY_CHANGED,
                EventType.TASK_STARTED: NotificationType.INQUIRY_STARTED,
                EventType.TASK_COMPLETED: NotificationType.INQUIRY_COMPLETED,
            }

            # Subscribe to relevant events
            for event_type, notif_type in event_mappings.items():

                async def handler(event: Any, nt: NotificationType = notif_type) -> None:
                    await self.notify(
                        nt,
                        {
                            "source_id": event.source_id,
                            "payload": event.payload,
                        },
                    )

                event_bus.subscribe(event_type, handler)

            self._event_bus_connected = True
            logger.info("Connected to Hive event bus")

        except ImportError:
            logger.warning("Hive event bus not available")
        except Exception as e:
            logger.error(f"Failed to connect to event bus: {e}")

    def subscribe(
        self,
        handler: NotificationHandler,
        notification_types: list[NotificationType] | None = None,
    ) -> str:
        """
        Subscribe to notifications.

        Args:
            handler: Async function to receive notifications
            notification_types: Specific types to subscribe to (None = all)

        Returns:
            Subscription ID
        """
        sub_id = uuid4().hex[:8]

        if notification_types:
            for nt in notification_types:
                if nt not in self._type_subscribers:
                    self._type_subscribers[nt] = []
                self._type_subscribers[nt].append(handler)
        else:
            self._subscribers.append(handler)

        logger.debug(f"Subscriber {sub_id} registered")
        return sub_id

    def unsubscribe(self, handler: NotificationHandler) -> None:
        """
        Unsubscribe a handler from notifications.

        Args:
            handler: The handler to remove
        """
        if handler in self._subscribers:
            self._subscribers.remove(handler)

        for handlers in self._type_subscribers.values():
            if handler in handlers:
                handlers.remove(handler)

    async def notify(
        self,
        notification_type: NotificationType,
        data: dict[str, Any],
    ) -> MCPNotification:
        """
        Send a notification to all subscribers.

        Args:
            notification_type: Type of notification
            data: Notification data

        Returns:
            The created notification
        """
        notification = MCPNotification(
            notification_type=notification_type,
            data=data,
        )

        # Add to history
        self._history.append(notification)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

        # Dispatch to subscribers
        handlers = list(self._subscribers)

        # Add type-specific subscribers
        if notification_type in self._type_subscribers:
            handlers.extend(self._type_subscribers[notification_type])

        # Dispatch concurrently
        if handlers:
            tasks = [self._safe_dispatch(h, notification) for h in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(f"Notification {notification_type.value} sent to {len(handlers)} handlers")

        return notification

    async def _safe_dispatch(
        self,
        handler: NotificationHandler,
        notification: MCPNotification,
    ) -> None:
        """Safely dispatch notification to handler."""
        try:
            await handler(notification)
        except Exception as e:
            logger.error(f"Notification handler error: {e}")

    def get_history(
        self,
        notification_type: NotificationType | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[MCPNotification]:
        """
        Get notification history.

        Args:
            notification_type: Filter by type (None = all)
            limit: Maximum notifications to return
            since: Only notifications after this time

        Returns:
            List of notifications (newest first)
        """
        notifications = self._history.copy()

        # Filter by type
        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]

        # Filter by time
        if since:
            notifications = [n for n in notifications if n.timestamp > since]

        # Sort newest first and limit
        notifications.sort(key=lambda n: n.timestamp, reverse=True)
        return notifications[:limit]

    def clear_history(self) -> None:
        """Clear notification history."""
        self._history.clear()
        logger.info("Notification history cleared")

    # =========================================================================
    # Convenience Methods for Common Notifications
    # =========================================================================

    async def notify_agent_spawned(
        self,
        agent_id: str,
        agent_type: str,
        task: str,
    ) -> MCPNotification:
        """Notify that an agent was spawned."""
        return await self.notify(
            NotificationType.AGENT_SPAWNED,
            {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "task": task[:200],
            },
        )

    async def notify_agent_completed(
        self,
        agent_id: str,
        result: Any,
    ) -> MCPNotification:
        """Notify that an agent completed."""
        return await self.notify(
            NotificationType.AGENT_COMPLETED,
            {
                "agent_id": agent_id,
                "result_preview": str(result)[:500] if result else None,
            },
        )

    async def notify_agent_failed(
        self,
        agent_id: str,
        error: str,
    ) -> MCPNotification:
        """Notify that an agent failed."""
        return await self.notify(
            NotificationType.AGENT_FAILED,
            {
                "agent_id": agent_id,
                "error": error,
            },
        )

    async def notify_topology_changed(
        self,
        old_type: str,
        new_type: str,
        reason: str | None = None,
    ) -> MCPNotification:
        """Notify that topology changed."""
        return await self.notify(
            NotificationType.TOPOLOGY_CHANGED,
            {
                "old_type": old_type,
                "new_type": new_type,
                "reason": reason,
            },
        )

    async def notify_inquiry_stage(
        self,
        session_id: str,
        stage_index: int,
        stage_name: str,
        status: str,
        content: str | None = None,
    ) -> MCPNotification:
        """Notify inquiry stage progress."""
        notif_type = (
            NotificationType.INQUIRY_STAGE_COMPLETED
            if status == "completed"
            else NotificationType.INQUIRY_STAGE_STARTED
        )

        data = {
            "session_id": session_id,
            "stage_index": stage_index,
            "stage_name": stage_name,
            "status": status,
        }

        if content:
            data["content_preview"] = content[:500]

        return await self.notify(notif_type, data)

    async def notify_inquiry_completed(
        self,
        session_id: str,
        topic: str,
        stages_completed: int,
    ) -> MCPNotification:
        """Notify inquiry session completed."""
        return await self.notify(
            NotificationType.INQUIRY_COMPLETED,
            {
                "session_id": session_id,
                "topic": topic[:200],
                "stages_completed": stages_completed,
            },
        )

    async def notify_learning_feedback(
        self,
        session_id: str,
        feedback_type: str,
        score: float | None = None,
    ) -> MCPNotification:
        """Notify learning feedback received."""
        return await self.notify(
            NotificationType.LEARNING_FEEDBACK,
            {
                "session_id": session_id,
                "feedback_type": feedback_type,
                "score": score,
            },
        )

    async def notify_error(
        self,
        source: str,
        error: str,
        details: dict[str, Any] | None = None,
    ) -> MCPNotification:
        """Notify system error."""
        return await self.notify(
            NotificationType.SYSTEM_ERROR,
            {
                "source": source,
                "error": error,
                "details": details or {},
            },
        )


# =============================================================================
# Singleton and API Functions
# =============================================================================

_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get the notification manager singleton."""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager


async def start_notifications() -> None:
    """Start the notification system."""
    manager = get_notification_manager()
    await manager.start()


async def stop_notifications() -> None:
    """Stop the notification system."""
    manager = get_notification_manager()
    await manager.stop()


async def notify(
    notification_type: NotificationType,
    data: dict[str, Any],
) -> MCPNotification:
    """
    Send a notification.

    Args:
        notification_type: Type of notification
        data: Notification data

    Returns:
        The created notification
    """
    manager = get_notification_manager()
    return await manager.notify(notification_type, data)


def subscribe(
    handler: NotificationHandler,
    notification_types: list[NotificationType] | None = None,
) -> str:
    """
    Subscribe to notifications.

    Args:
        handler: Async function to receive notifications
        notification_types: Specific types (None = all)

    Returns:
        Subscription ID
    """
    manager = get_notification_manager()
    return manager.subscribe(handler, notification_types)


def get_notification_history(
    notification_type: NotificationType | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Get notification history as dicts.

    Args:
        notification_type: Filter by type
        limit: Maximum to return

    Returns:
        List of notification dicts
    """
    manager = get_notification_manager()
    notifications = manager.get_history(notification_type, limit)
    return [n.to_dict() for n in notifications]
