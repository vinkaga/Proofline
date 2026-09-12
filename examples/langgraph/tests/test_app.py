import asyncio

import pytest
from app import run
from proofline_example_host import TrustedRequest

from proofline import ProposedStepError


def test_graph_node_cannot_expand_the_captured_request_scope() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    result = asyncio.run(run(request, "rollout approval"))

    assert result.resource_ids == ("document:acme-rollout",)

    denied = asyncio.run(run(request, "beta rollout"))

    assert denied.resource_ids == ()


def test_graph_host_rejects_scope_bearing_planner_output() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))

    with pytest.raises(ProposedStepError, match="resource_id"):
        asyncio.run(
            run(
                request,
                "rollout approval",
                planner_output={"query": "beta rollout", "resource_id": "document:beta-rollout"},
            )
        )


def test_graph_follow_up_has_a_child_scope() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))

    result = asyncio.run(
        run(request, "rollout approval", planner_output={"query": "release-manager approval"})
    )

    assert result.resource_ids == ("document:acme-rollout",)
    assert result.parent_scope_id is not None
