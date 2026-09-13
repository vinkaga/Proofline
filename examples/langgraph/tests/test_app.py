import asyncio

import pytest
from app import build_graph, run
from langgraph.checkpoint.memory import InMemorySaver
from scopeanchor_example_host import TrustedRequest

from scopeanchor import ProposedStepError


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


def test_graph_can_checkpoint_and_restore_scope_state_between_nodes() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    graph = build_graph(
        request,
        checkpoint_binding={"principal": "user:ana", "task_id": "task-123"},
        checkpointer=InMemorySaver(),
    )

    async def run_checkpointed() -> dict[str, object]:
        return await graph.ainvoke(
            {
                "query": "rollout approval",
                "planner_output": {"query": "release-manager approval"},
                "resource_ids": (),
                "scope_checkpoint": None,
                "scope_id": None,
                "parent_scope_id": None,
            },
            config={"configurable": {"thread_id": "task-123"}},
        )

    state = asyncio.run(run_checkpointed())

    assert state["resource_ids"] == ("document:acme-rollout",)
    assert state["parent_scope_id"] is not None
    assert isinstance(state["scope_checkpoint"], dict)
    assert "initial" not in state
