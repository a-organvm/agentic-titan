"""Tests for MCP notifications module."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from titan_mcp.notifications import (
    MCPNotification,
    NotificationManager,
    NotificationType,
    get_notification_history,
    get_notification_manager,
    notify,
    subscribe,
)


class TestMCPNotification:
    """Tests for MCPNotification dataclass."""

    def test_create_notification(self) -> None:
        """Test creating a notification."""
        notif = MCPNotification(
            notification_type=NotificationType.AGENT_SPAWNED,
            data={"agent_id": "test-123"},
        )

        assert notif.notification_type == NotificationType.AGENT_SPAWNED
        assert notif.data["agent_id"] == "test-123"
        assert notif.id is not None
        assert notif.timestamp is not None

    def test_to_jsonrpc(self) -> None:
        """Test converting to JSON-RPC format."""
        notif = MCPNotification(
            notification_type=NotificationType.TOPOLOGY_CHANGED,
            data={"old_type": "swarm", "new_type": "hierarchy"},
        )

        jsonrpc = notif.to_jsonrpc()

        assert jsonrpc["jsonrpc"] == "2.0"
        assert jsonrpc["method"] == "notifications/message"
        assert jsonrpc["params"]["type"] == "topology/changed"
        assert jsonrpc["params"]["data"]["old_type"] == "swarm"

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        notif = MCPNotification(
            notification_type=NotificationType.INQUIRY_STARTED,
            data={"session_id": "inq-123"},
        )

        d = notif.to_dict()

        assert d["type"] == "inquiry/started"
        assert d["data"]["session_id"] == "inq-123"
        assert "timestamp" in d
        assert "id" in d


class TestNotificationTypes:
    """Tests for notification type enum."""

    def test_agent_notification_types(self) -> None:
        """Test agent-related notification types."""
        assert NotificationType.AGENT_STATE_CHANGED.value == "agent/state"
        assert NotificationType.AGENT_SPAWNED.value == "agent/spawned"
        assert NotificationType.AGENT_COMPLETED.value == "agent/completed"
        assert NotificationType.AGENT_FAILED.value == "agent/failed"

    def test_inquiry_notification_types(self) -> None:
        """Test inquiry-related notification types."""
        assert NotificationType.INQUIRY_STARTED.value == "inquiry/started"
        assert NotificationType.INQUIRY_STAGE_STARTED.value == "inquiry/stage"
        assert NotificationType.INQUIRY_COMPLETED.value == "inquiry/completed"
        assert NotificationType.INQUIRY_PAUSED.value == "inquiry/paused"

    def test_learning_notification_types(self) -> None:
        """Test learning-related notification types."""
        assert NotificationType.LEARNING_FEEDBACK.value == "learning/feedback"
        assert NotificationType.RLHF_PAIR_CREATED.value == "rlhf/pair_created"


class TestNotificationManager:
    """Tests for NotificationManager class."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes correctly."""
        manager = NotificationManager(history_limit=100)

        assert len(manager._subscribers) == 0
        assert manager._history_limit == 100

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping manager."""
        manager = NotificationManager(enable_event_bus=False)

        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_subscribe_all_notifications(self) -> None:
        """Test subscribing to all notifications."""
        manager = NotificationManager(enable_event_bus=False)
        received: list[MCPNotification] = []

        async def handler(n: MCPNotification) -> None:
            received.append(n)

        sub_id = manager.subscribe(handler)
        assert sub_id is not None

        await manager.notify(NotificationType.AGENT_SPAWNED, {"test": "data"})

        assert len(received) == 1
        assert received[0].notification_type == NotificationType.AGENT_SPAWNED

    @pytest.mark.asyncio
    async def test_subscribe_specific_types(self) -> None:
        """Test subscribing to specific notification types."""
        manager = NotificationManager(enable_event_bus=False)
        received: list[MCPNotification] = []

        async def handler(n: MCPNotification) -> None:
            received.append(n)

        manager.subscribe(handler, [NotificationType.INQUIRY_STARTED])

        # Send different types
        await manager.notify(NotificationType.AGENT_SPAWNED, {"test": 1})
        await manager.notify(NotificationType.INQUIRY_STARTED, {"test": 2})
        await manager.notify(NotificationType.TOPOLOGY_CHANGED, {"test": 3})

        # Should only receive inquiry_started
        assert len(received) == 1
        assert received[0].notification_type == NotificationType.INQUIRY_STARTED

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """Test unsubscribing from notifications."""
        manager = NotificationManager(enable_event_bus=False)
        received: list[MCPNotification] = []

        async def handler(n: MCPNotification) -> None:
            received.append(n)

        manager.subscribe(handler)
        await manager.notify(NotificationType.AGENT_SPAWNED, {})
        assert len(received) == 1

        manager.unsubscribe(handler)
        await manager.notify(NotificationType.AGENT_SPAWNED, {})
        assert len(received) == 1  # No new notifications

    @pytest.mark.asyncio
    async def test_notification_history(self) -> None:
        """Test notification history tracking."""
        manager = NotificationManager(history_limit=10, enable_event_bus=False)

        for i in range(5):
            await manager.notify(NotificationType.AGENT_SPAWNED, {"i": i})

        history = manager.get_history()
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_history_limit(self) -> None:
        """Test history respects limit."""
        manager = NotificationManager(history_limit=3, enable_event_bus=False)

        for i in range(5):
            await manager.notify(NotificationType.AGENT_SPAWNED, {"i": i})

        history = manager.get_history()
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_history_filter_by_type(self) -> None:
        """Test filtering history by type."""
        manager = NotificationManager(enable_event_bus=False)

        await manager.notify(NotificationType.AGENT_SPAWNED, {})
        await manager.notify(NotificationType.INQUIRY_STARTED, {})
        await manager.notify(NotificationType.AGENT_COMPLETED, {})

        history = manager.get_history(NotificationType.AGENT_SPAWNED)
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_history_filter_by_time(self) -> None:
        """Test filtering history by time."""
        manager = NotificationManager(enable_event_bus=False)

        await manager.notify(NotificationType.AGENT_SPAWNED, {})
        cutoff = datetime.now()
        await asyncio.sleep(0.01)
        await manager.notify(NotificationType.AGENT_COMPLETED, {})

        history = manager.get_history(since=cutoff)
        assert len(history) == 1
        assert history[0].notification_type == NotificationType.AGENT_COMPLETED


class TestConvenienceMethods:
    """Tests for convenience notification methods."""

    @pytest.mark.asyncio
    async def test_notify_agent_spawned(self) -> None:
        """Test convenience method for agent spawned."""
        manager = NotificationManager(enable_event_bus=False)

        notif = await manager.notify_agent_spawned(
            agent_id="agent-123",
            agent_type="researcher",
            task="Research AI",
        )

        assert notif.notification_type == NotificationType.AGENT_SPAWNED
        assert notif.data["agent_id"] == "agent-123"
        assert notif.data["agent_type"] == "researcher"

    @pytest.mark.asyncio
    async def test_notify_topology_changed(self) -> None:
        """Test convenience method for topology changed."""
        manager = NotificationManager(enable_event_bus=False)

        notif = await manager.notify_topology_changed(
            old_type="swarm",
            new_type="hierarchy",
            reason="task complexity",
        )

        assert notif.notification_type == NotificationType.TOPOLOGY_CHANGED
        assert notif.data["old_type"] == "swarm"
        assert notif.data["new_type"] == "hierarchy"

    @pytest.mark.asyncio
    async def test_notify_inquiry_stage(self) -> None:
        """Test convenience method for inquiry stage."""
        manager = NotificationManager(enable_event_bus=False)

        notif = await manager.notify_inquiry_stage(
            session_id="inq-123",
            stage_index=2,
            stage_name="Logical Analysis",
            status="completed",
            content="Analysis results...",
        )

        assert notif.notification_type == NotificationType.INQUIRY_STAGE_COMPLETED
        assert notif.data["stage_name"] == "Logical Analysis"

    @pytest.mark.asyncio
    async def test_notify_error(self) -> None:
        """Test convenience method for errors."""
        manager = NotificationManager(enable_event_bus=False)

        notif = await manager.notify_error(
            source="inquiry_engine",
            error="Stage execution failed",
            details={"stage": 3},
        )

        assert notif.notification_type == NotificationType.SYSTEM_ERROR
        assert notif.data["source"] == "inquiry_engine"


class TestGlobalFunctions:
    """Tests for module-level functions."""

    @pytest.mark.asyncio
    async def test_notify_function(self) -> None:
        """Test the global notify function."""
        notif = await notify(NotificationType.AGENT_SPAWNED, {"test": "data"})

        assert notif.notification_type == NotificationType.AGENT_SPAWNED

    def test_subscribe_function(self) -> None:
        """Test the global subscribe function."""

        async def handler(n: MCPNotification) -> None:
            pass

        sub_id = subscribe(handler)
        assert sub_id is not None

    def test_get_notification_history_function(self) -> None:
        """Test the global get_notification_history function."""
        history = get_notification_history(limit=10)

        assert isinstance(history, list)

    def test_get_notification_manager_singleton(self) -> None:
        """Test singleton behavior of notification manager."""
        manager1 = get_notification_manager()
        manager2 = get_notification_manager()

        assert manager1 is manager2
