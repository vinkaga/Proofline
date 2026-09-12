from app import app
from fastapi.testclient import TestClient


def test_authenticated_http_host_returns_only_the_callers_scope() -> None:
    response = TestClient(app).get(
        "/search",
        params={"query": "rollout approval"},
        headers={"x-principal": "ana"},
    )

    assert response.status_code == 200
    assert response.json() == {"resource_ids": ["document:acme-rollout"]}


def test_query_parameters_cannot_widen_the_authenticated_scope() -> None:
    response = TestClient(app).get(
        "/search",
        params={"query": "rollout approval", "resource_id": "document:beta-rollout"},
        headers={"x-principal": "ana"},
    )

    assert response.status_code == 200
    assert response.json() == {"resource_ids": ["document:acme-rollout"]}


def test_unknown_principal_is_rejected_before_retrieval() -> None:
    response = TestClient(app).get(
        "/search", params={"query": "beta rollout"}, headers={"x-principal": "unknown"}
    )

    assert response.status_code == 401


def test_follow_up_preserves_scope_and_rejects_scope_bearing_planner_input() -> None:
    client = TestClient(app)
    headers = {"x-principal": "ana"}

    allowed = client.post(
        "/search/follow-up", json={"query": "release-manager approval"}, headers=headers
    )
    rejected = client.post(
        "/search/follow-up",
        json={"query": "beta rollout", "resource_id": "document:beta-rollout"},
        headers=headers,
    )

    assert allowed.json()["resource_ids"] == ["document:acme-rollout"]
    assert allowed.json()["parent_scope_id"] is not None
    assert rejected.status_code == 400
    assert "resource_id" in rejected.json()["detail"]
