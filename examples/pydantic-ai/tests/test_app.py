import asyncio

import app
import pytest
from app import PydanticRunDependencies, agent, follow_up
from proofline_example_host import TrustedRequest, start_retrieval_run
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from proofline import ProposedStepError, ScopeError


def test_registered_pydantic_ai_tool_receives_only_trusted_dependencies(monkeypatch) -> None:
    received: dict[str, object] = {}

    async def retrieve_for_host(deps: PydanticRunDependencies, query: str) -> list[str]:
        received["request"] = deps.request
        received["query"] = query
        return ["document:acme-rollout"]

    monkeypatch.setattr(app, "retrieve_for_host", retrieve_for_host)
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    deps = PydanticRunDependencies.from_request(request)
    model = TestModel(call_tools=["retrieve_evidence"])
    result = asyncio.run(
        agent.run(
            "retrieve rollout evidence",
            deps=deps,
            model=model,
        )
    )

    assert received["request"] is request
    assert isinstance(received["query"], str)
    assert "document:acme-rollout" in result.output
    assert model.last_model_request_parameters is not None
    tool = model.last_model_request_parameters.function_tools[0]
    assert tool.name == "retrieve_evidence"
    assert set(tool.parameters_json_schema["properties"]) == {"query"}
    assert tool.sequential


def test_pydantic_ai_host_rejects_scope_bearing_planner_output() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))

    with pytest.raises(ProposedStepError, match="resource_id"):
        asyncio.run(
            follow_up(
                request,
                {"query": "beta rollout", "resource_id": "document:beta-rollout"},
        )
    )


def test_actual_pydantic_ai_tool_rounds_retain_scope_lineage_and_budget() -> None:
    """Successive model calls use one root and one inherited child scope."""

    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    deps = PydanticRunDependencies.from_request(request)
    model_turn = 0

    async def model(messages, info):  # noqa: ANN001, ARG001
        nonlocal model_turn
        model_turn += 1
        if model_turn == 1:
            return ModelResponse(
                parts=[ToolCallPart("retrieve_evidence", {"query": "rollout approval"})]
            )
        if model_turn == 2:
            return ModelResponse(
                parts=[ToolCallPart("retrieve_evidence", {"query": "release manager"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    result = asyncio.run(agent.run("find rollout evidence", deps=deps, model=FunctionModel(model)))

    assert result.output == "done"
    assert model_turn == 3
    assert len(deps.retrieval_run.history) == 2
    root, follow_up_result = deps.retrieval_run.history
    assert root.scope.parent_scope_id is None
    assert root.scope.follow_up_count == 0
    assert follow_up_result.scope.parent_scope_id == root.scope.scope_id
    assert follow_up_result.scope.follow_up_count == 1
    assert follow_up_result.scope.filters == root.scope.filters

    with pytest.raises(ScopeError, match="exhausted"):
        asyncio.run(deps.retrieval_run.retrieve("another follow-up"))


def test_retrieval_run_serializes_parallel_calls_without_new_roots() -> None:
    request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
    run = start_retrieval_run(request)

    async def invoke_in_parallel():
        return await asyncio.gather(
            run.retrieve("rollout approval"),
            run.retrieve("release manager"),
        )

    root, follow_up_result = asyncio.run(invoke_in_parallel())

    assert len(run.history) == 2
    assert root.scope.parent_scope_id is None
    assert follow_up_result.scope.parent_scope_id == root.scope.scope_id
