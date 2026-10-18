from fastapi.testclient import TestClient

from insightforge.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()
    assert body["application"] == "InsightForge Enterprise"
    assert body["status"] == "running"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
