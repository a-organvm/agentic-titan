"""
Tests for Titan MCP Server.

Tests JSON-RPC protocol handling and agent management.
"""

import json

import pytest

from titan_mcp.server import (
    AgentManager,
    MCPRequest,
    TitanMCPServer,
)

# ============================================================================
# Server Tests
# ============================================================================


@pytest.fixture
def server() -> TitanMCPServer:
    """Create a fresh server instance."""
    return TitanMCPServer()


class TestMCPProtocol:
    """Test MCP protocol handling."""

    @pytest.mark.asyncio
    async def test_initialize(self, server: TitanMCPServer) -> None:
        """Test initialize handshake."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={"protocolVersion": "2024-11-05"},
        )

        response = await server.handle_request(request)

        assert response.id == 1
        assert response.error is None
        assert response.result["protocolVersion"] == "2024-11-05"
        assert response.result["serverInfo"]["name"] == "titan-mcp"
        assert "tools" in response.result["capabilities"]

    @pytest.mark.asyncio
    async def test_tools_list(self, server: TitanMCPServer) -> None:
        """Test tools/list returns all tools."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=2,
            method="tools/list",
            params={},
        )

        response = await server.handle_request(request)

        assert response.error is None
        tools = response.result["tools"]
        tool_names = {t["name"] for t in tools}

        assert "spawn_agent" in tool_names
        assert "agent_status" in tool_names
        assert "agent_result" in tool_names
        assert "list_agents" in tool_names
        assert "cancel_agent" in tool_names

    @pytest.mark.asyncio
    async def test_resources_list(self, server: TitanMCPServer) -> None:
        """Test resources/list returns all resources."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=3,
            method="resources/list",
            params={},
        )

        response = await server.handle_request(request)

        assert response.error is None
        resources = response.result["resources"]
        uris = {r["uri"] for r in resources}

        assert "titan://agents/types" in uris
        assert "titan://agents/tools" in uris

    @pytest.mark.asyncio
    async def test_resources_read_types(self, server: TitanMCPServer) -> None:
        """Test reading agent types resource."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=4,
            method="resources/read",
            params={"uri": "titan://agents/types"},
        )

        response = await server.handle_request(request)

        assert response.error is None
        contents = response.result["contents"]
        assert len(contents) == 1

        data = json.loads(contents[0]["text"])
        type_names = {t["name"] for t in data["agent_types"]}

        assert "researcher" in type_names
        assert "coder" in type_names
        assert "reviewer" in type_names
        assert "orchestrator" in type_names

    @pytest.mark.asyncio
    async def test_unknown_method(self, server: TitanMCPServer) -> None:
        """Test unknown method returns error."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=5,
            method="unknown/method",
            params={},
        )

        response = await server.handle_request(request)

        assert response.error is not None
        assert response.error["code"] == -32601
        assert "not found" in response.error["message"].lower()


class TestToolsCalls:
    """Test tools/call operations."""

    @pytest.mark.asyncio
    async def test_spawn_agent(self, server: TitanMCPServer) -> None:
        """Test spawning an agent."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=10,
            method="tools/call",
            params={
                "name": "spawn_agent",
                "arguments": {
                    "agent_type": "simple",
                    "task": "Test task for unit test",
                },
            },
        )

        response = await server.handle_request(request)

        assert response.error is None
        content = json.loads(response.result["content"][0]["text"])
        assert "session_id" in content
        assert content["status"] == "running"

    @pytest.mark.asyncio
    async def test_list_agents(self, server: TitanMCPServer) -> None:
        """Test listing agents."""
        # First spawn an agent
        spawn_req = MCPRequest(
            jsonrpc="2.0",
            id=11,
            method="tools/call",
            params={
                "name": "spawn_agent",
                "arguments": {"agent_type": "simple", "task": "Test"},
            },
        )
        await server.handle_request(spawn_req)

        # Then list
        list_req = MCPRequest(
            jsonrpc="2.0",
            id=12,
            method="tools/call",
            params={"name": "list_agents", "arguments": {}},
        )
        response = await server.handle_request(list_req)

        assert response.error is None
        agents = json.loads(response.result["content"][0]["text"])
        assert len(agents) >= 1

    @pytest.mark.asyncio
    async def test_agent_status(self, server: TitanMCPServer) -> None:
        """Test checking agent status."""
        # Spawn
        spawn_req = MCPRequest(
            jsonrpc="2.0",
            id=13,
            method="tools/call",
            params={
                "name": "spawn_agent",
                "arguments": {"agent_type": "simple", "task": "Test"},
            },
        )
        spawn_resp = await server.handle_request(spawn_req)
        session_id = json.loads(spawn_resp.result["content"][0]["text"])["session_id"]

        # Check status
        status_req = MCPRequest(
            jsonrpc="2.0",
            id=14,
            method="tools/call",
            params={
                "name": "agent_status",
                "arguments": {"session_id": session_id},
            },
        )
        response = await server.handle_request(status_req)

        assert response.error is None
        status = json.loads(response.result["content"][0]["text"])
        assert status["session_id"] == session_id
        assert status["agent_type"] == "simple"
        assert status["status"] in ["running", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_cancel_agent(self, server: TitanMCPServer) -> None:
        """Test cancelling an agent."""
        # Spawn
        spawn_req = MCPRequest(
            jsonrpc="2.0",
            id=15,
            method="tools/call",
            params={
                "name": "spawn_agent",
                "arguments": {"agent_type": "simple", "task": "Long task"},
            },
        )
        spawn_resp = await server.handle_request(spawn_req)
        session_id = json.loads(spawn_resp.result["content"][0]["text"])["session_id"]

        # Cancel
        cancel_req = MCPRequest(
            jsonrpc="2.0",
            id=16,
            method="tools/call",
            params={
                "name": "cancel_agent",
                "arguments": {"session_id": session_id},
            },
        )
        response = await server.handle_request(cancel_req)

        assert response.error is None
        result = json.loads(response.result["content"][0]["text"])
        # May or may not be cancelled depending on timing
        assert "cancelled" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, server: TitanMCPServer) -> None:
        """Test calling unknown tool."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=17,
            method="tools/call",
            params={"name": "nonexistent_tool", "arguments": {}},
        )

        response = await server.handle_request(request)

        assert "isError" in response.result
        assert response.result["isError"] is True

    @pytest.mark.asyncio
    async def test_agent_status_not_found(self, server: TitanMCPServer) -> None:
        """Test status for nonexistent session."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=18,
            method="tools/call",
            params={
                "name": "agent_status",
                "arguments": {"session_id": "sess_nonexistent"},
            },
        )

        response = await server.handle_request(request)

        assert "isError" in response.result
        assert response.result["isError"] is True


class TestAgentManager:
    """Test AgentManager directly."""

    @pytest.fixture
    def manager(self) -> AgentManager:
        return AgentManager()

    @pytest.mark.asyncio
    async def test_spawn_returns_session_id(self, manager: AgentManager) -> None:
        """Test spawn returns valid session ID."""
        session_id = await manager.spawn_agent("simple", "Test task")

        assert session_id.startswith("sess_")
        assert len(session_id) == 13  # sess_ + 8 hex chars

    @pytest.mark.asyncio
    async def test_get_session(self, manager: AgentManager) -> None:
        """Test retrieving session."""
        session_id = await manager.spawn_agent("simple", "Test")
        session = manager.get_session(session_id)

        assert session is not None
        assert session.id == session_id
        assert session.agent_type == "simple"
        assert session.task == "Test"

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager: AgentManager) -> None:
        """Test listing all sessions."""
        await manager.spawn_agent("simple", "Task 1")
        await manager.spawn_agent("researcher", "Task 2")

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        types = {s.agent_type for s in sessions}
        assert "simple" in types
        assert "researcher" in types


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCPIntegration:
    """Integration tests for complete MCP workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, server: TitanMCPServer) -> None:
        """Test complete spawn → status → result workflow."""
        # Initialize
        init_resp = await server.handle_request(
            MCPRequest(
                jsonrpc="2.0",
                id=100,
                method="initialize",
                params={},
            )
        )
        assert init_resp.error is None

        # Spawn
        spawn_resp = await server.handle_request(
            MCPRequest(
                jsonrpc="2.0",
                id=101,
                method="tools/call",
                params={
                    "name": "spawn_agent",
                    "arguments": {"agent_type": "simple", "task": "Quick test"},
                },
            )
        )
        session_id = json.loads(spawn_resp.result["content"][0]["text"])["session_id"]

        # Check status
        status_resp = await server.handle_request(
            MCPRequest(
                jsonrpc="2.0",
                id=102,
                method="tools/call",
                params={
                    "name": "agent_status",
                    "arguments": {"session_id": session_id},
                },
            )
        )
        status = json.loads(status_resp.result["content"][0]["text"])
        assert status["session_id"] == session_id

        # Get result (may be running still)
        result_resp = await server.handle_request(
            MCPRequest(
                jsonrpc="2.0",
                id=103,
                method="tools/call",
                params={
                    "name": "agent_result",
                    "arguments": {"session_id": session_id},
                },
            )
        )
        assert result_resp.error is None

    @pytest.mark.asyncio
    async def test_response_format(self, server: TitanMCPServer) -> None:
        """Test response follows JSON-RPC 2.0 format."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=200,
            method="tools/list",
            params={},
        )

        response = await server.handle_request(request)
        response_dict = response.to_dict()

        assert response_dict["jsonrpc"] == "2.0"
        assert response_dict["id"] == 200
        assert "result" in response_dict or "error" in response_dict
