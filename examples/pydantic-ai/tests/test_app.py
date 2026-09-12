import asyncio

import app
import pytest
from app import agent, follow_up
from proofline_example_host import TrustedRequest
from pydantic_ai.models.test import TestModel

from proofline import ProposedStepError


def test_registered_pydantic_ai_tool_receives_only_trusted_dependencies(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def retrieve_for_host(request: TrustedRequest, query: str) -> list[str]:
        received["request"] = request
        received["query"] = query
        return ["document:acme-rollout"]

    monkeypatch.setattr(app, "retrieve_for_host", retrieve_for_host)
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    result = asyncio.run(
        agent.run(
            "retrieve rollout evidence",
            deps=request,
            model=TestModel(call_tools=["retrieve_evidence"]),
        )
    )

    assert received["request"] is request
    assert isinstance(received["query"], str)
    assert "document:acme-rollout" in result.output


def test_pydantic_ai_host_rejects_scope_bearing_planner_output() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))

    with pytest.raises(ProposedStepError, match="resource_id"):
        asyncio.run(
            follow_up(
                request,
                {"query": "beta rollout", "resource_id": "document:beta-rollout"},
            )
        )
