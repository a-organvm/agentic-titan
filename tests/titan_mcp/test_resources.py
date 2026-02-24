"""Tests for MCP resources module."""

from __future__ import annotations

import pytest

from titan_mcp.resources import (
    LEARNING_STATS_RESOURCE,
    MODEL_SIGNATURES_RESOURCE,
    MCPResourceDefinition,
    ResourceHandler,
    ResourceType,
    format_resource_contents,
    get_all_resources,
    get_resource_definition,
    get_resource_handler,
    get_resources_by_type,
    read_resource,
)


class TestMCPResourceDefinition:
    """Tests for MCPResourceDefinition dataclass."""

    def test_create_resource(self) -> None:
        """Test creating a resource definition."""
        resource = MCPResourceDefinition(
            uri="titan://test/resource",
            name="Test Resource",
            description="A test resource",
            resource_type=ResourceType.LEARNING,
        )

        assert resource.uri == "titan://test/resource"
        assert resource.name == "Test Resource"
        assert resource.mime_type == "application/json"

    def test_to_dict(self) -> None:
        """Test converting resource to dict."""
        resource = MCPResourceDefinition(
            uri="titan://test",
            name="Test",
            description="Test desc",
            resource_type=ResourceType.MODELS,
            mime_type="text/plain",
        )

        d = resource.to_dict()

        assert d["uri"] == "titan://test"
        assert d["name"] == "Test"
        assert d["description"] == "Test desc"
        assert d["mimeType"] == "text/plain"


class TestResourceDefinitions:
    """Tests for predefined resource definitions."""

    def test_learning_stats_resource(self) -> None:
        """Test learning stats resource definition."""
        assert LEARNING_STATS_RESOURCE.uri == "titan://learning/stats"
        assert LEARNING_STATS_RESOURCE.resource_type == ResourceType.LEARNING

    def test_model_signatures_resource(self) -> None:
        """Test model signatures resource definition."""
        assert MODEL_SIGNATURES_RESOURCE.uri == "titan://models/signatures"
        assert MODEL_SIGNATURES_RESOURCE.resource_type == ResourceType.MODELS
        assert MODEL_SIGNATURES_RESOURCE.metadata.get("visualization") == "radar"


class TestResourceRegistry:
    """Tests for resource registry functions."""

    def test_get_all_resources(self) -> None:
        """Test getting all resources."""
        resources = get_all_resources()

        assert len(resources) >= 6
        uris = [r.uri for r in resources]
        assert "titan://learning/stats" in uris
        assert "titan://models/signatures" in uris
        assert "titan://topology/current" in uris

    def test_get_resource_definition(self) -> None:
        """Test getting resource by URI."""
        resource = get_resource_definition("titan://learning/stats")

        assert resource is not None
        assert resource.name == "Learning Statistics"

    def test_get_resource_unknown_returns_none(self) -> None:
        """Test getting unknown resource returns None."""
        resource = get_resource_definition("titan://nonexistent")

        assert resource is None

    def test_get_resources_by_type(self) -> None:
        """Test filtering resources by type."""
        learning_resources = get_resources_by_type(ResourceType.LEARNING)

        assert len(learning_resources) >= 2
        for r in learning_resources:
            assert r.resource_type == ResourceType.LEARNING


class TestResourceHandler:
    """Tests for ResourceHandler class."""

    def test_handler_initialization(self) -> None:
        """Test handler initializes with handlers for all resources."""
        handler = ResourceHandler()

        assert "titan://learning/stats" in handler._handlers
        assert "titan://models/signatures" in handler._handlers

    @pytest.mark.asyncio
    async def test_read_learning_stats(self) -> None:
        """Test reading learning stats resource."""
        handler = get_resource_handler()
        data = await handler.read("titan://learning/stats")

        assert "total_episodes" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_read_rlhf_stats(self) -> None:
        """Test reading RLHF stats resource."""
        handler = get_resource_handler()
        data = await handler.read("titan://learning/rlhf/stats")

        assert "preference_pairs" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_read_model_signatures(self) -> None:
        """Test reading model signatures resource."""
        handler = get_resource_handler()
        data = await handler.read("titan://models/signatures")

        assert "models" in data
        assert "dimensions" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_read_topology_current(self) -> None:
        """Test reading current topology resource."""
        handler = get_resource_handler()
        data = await handler.read("titan://topology/current")

        assert "topology" in data
        assert "available_types" in data

    @pytest.mark.asyncio
    async def test_read_hive_events(self) -> None:
        """Test reading hive events resource."""
        handler = get_resource_handler()
        data = await handler.read("titan://hive/events/recent")

        assert "events" in data
        assert "total_count" in data

    @pytest.mark.asyncio
    async def test_read_unknown_resource_raises(self) -> None:
        """Test reading unknown resource raises ValueError."""
        handler = get_resource_handler()

        with pytest.raises(ValueError, match="Unknown resource URI"):
            await handler.read("titan://nonexistent")


class TestResourceFormatting:
    """Tests for resource content formatting."""

    def test_format_resource_contents(self) -> None:
        """Test formatting resource data for MCP response."""
        data = {"test": "value", "count": 42}
        formatted = format_resource_contents("titan://test", data)

        assert "contents" in formatted
        assert len(formatted["contents"]) == 1
        assert formatted["contents"][0]["uri"] == "titan://test"
        assert formatted["contents"][0]["mimeType"] == "application/json"
        assert "test" in formatted["contents"][0]["text"]

    def test_format_uses_resource_mime_type(self) -> None:
        """Test formatting uses resource's mime type."""
        # The format function looks up the resource definition
        data = {"episodes": 10}
        formatted = format_resource_contents("titan://learning/stats", data)

        assert formatted["contents"][0]["mimeType"] == "application/json"


@pytest.mark.asyncio
async def test_read_resource_function() -> None:
    """Test the convenience read_resource function."""
    data = await read_resource("titan://learning/stats")

    assert "total_episodes" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_read_inquiry_sessions() -> None:
    """Test reading inquiry sessions resource."""
    handler = get_resource_handler()
    data = await handler.read("titan://inquiry/sessions")

    assert "sessions" in data
    assert "total_count" in data
    assert "timestamp" in data
