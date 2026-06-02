from __future__ import annotations

import pytest

from gateway import http_gateway


def test_task_result_serializes_public_aliases() -> None:
    result = http_gateway.TaskResult(
        taskId="task-1",
        status="completed",
        result="done",
        model="test-model",
        agentType="researcher",
    )

    assert result.model_dump(by_alias=True) == {
        "taskId": "task-1",
        "status": "completed",
        "result": "done",
        "model": "test-model",
        "agentType": "researcher",
    }


@pytest.mark.asyncio
async def test_call_tool_uses_mcp_json_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        error = None
        result = {"content": [{"type": "text", "text": "ok"}]}

    class FakeServer:
        async def handle_request(self, request):
            captured["request"] = request
            return FakeResponse()

    monkeypatch.setattr(http_gateway, "_mcp_server", FakeServer())

    result = await http_gateway._call_tool("route_cognitive_task", {"task_description": "x"})

    assert result == {"content": [{"type": "text", "text": "ok"}]}
    assert captured["request"].jsonrpc == "2.0"
    assert captured["request"].method == "tools/call"
    assert captured["request"].params == {
        "name": "route_cognitive_task",
        "arguments": {"task_description": "x"},
    }


@pytest.mark.asyncio
async def test_read_resource_uses_topology_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        error = None
        result = {"contents": [{"text": "{}"}]}

    class FakeServer:
        async def handle_request(self, request):
            captured["request"] = request
            return FakeResponse()

    monkeypatch.setattr(http_gateway, "_mcp_server", FakeServer())

    result = await http_gateway._read_resource("titan://topology/current")

    assert result == {"contents": [{"text": "{}"}]}
    assert captured["request"].jsonrpc == "2.0"
    assert captured["request"].method == "resources/read"
    assert captured["request"].params == {"uri": "titan://topology/current"}
