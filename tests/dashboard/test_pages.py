from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.app import TitanDashboard


def test_dashboard_pages_render() -> None:
    client = TestClient(TitanDashboard().create_app())

    for path in ["/", "/inquiry", "/models", "/knowledge", "/analysis"]:
        response = client.get(path)

        assert response.status_code == 200
        assert "<html" in response.text
