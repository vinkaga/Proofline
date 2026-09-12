# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""LangGraph runs a host-controlled retrieval node under trusted context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from proofline_example_host import Evidence, TrustedRequest, retriever

from proofline import ProposedRetrievalStep, ScopedResults


class RetrievalState(TypedDict):
    query: str
    planner_output: dict[str, object] | None
    resource_ids: tuple[str, ...]
    initial: ScopedResults[Evidence] | None
    scope_id: str | None
    parent_scope_id: str | None


@dataclass(frozen=True, slots=True)
class GraphResult:
    """The evidence IDs and scope lineage from one graph execution."""

    resource_ids: tuple[str, ...]
    scope_id: str
    parent_scope_id: str | None


def build_graph(request: TrustedRequest):
    """Capture trusted request context outside graph/model-visible state."""

    boundary = retriever()

    async def retrieve_node(state: RetrievalState) -> dict[str, object]:
        initial = await boundary.search(state["query"], context=request)
        return {
            "initial": initial,
            "resource_ids": tuple(item.resource_id for item in initial.items),
            "scope_id": initial.scope.scope_id,
            "parent_scope_id": initial.scope.parent_scope_id,
        }

    def after_initial(state: RetrievalState) -> str:
        return "follow" if state["planner_output"] is not None else "end"

    async def follow_node(state: RetrievalState) -> dict[str, object]:
        initial = state["initial"]
        planner_output = state["planner_output"]
        if initial is None or planner_output is None:
            raise RuntimeError("follow-up node requires initial results and a planner proposal")
        step = ProposedRetrievalStep.from_untrusted(planner_output)
        follow_up = await boundary.follow_proposed(initial, step)
        return {
            "resource_ids": tuple(item.resource_id for item in follow_up.items),
            "scope_id": follow_up.scope.scope_id,
            "parent_scope_id": follow_up.scope.parent_scope_id,
        }

    return (
        StateGraph(RetrievalState)
        .add_node("retrieve", retrieve_node)
        .add_node("follow", follow_node)
        .add_edge(START, "retrieve")
        .add_conditional_edges("retrieve", after_initial, {"follow": "follow", "end": END})
        .add_edge("follow", END)
        .compile()
    )


async def run(
    request: TrustedRequest,
    query: str,
    *,
    planner_output: dict[str, object] | None = None,
) -> GraphResult:
    """Run one host-controlled graph path, optionally including one planned hop."""

    state = await build_graph(request).ainvoke(
        {
            "query": query,
            "planner_output": planner_output,
            "resource_ids": (),
            "initial": None,
            "scope_id": None,
            "parent_scope_id": None,
        }
    )
    scope_id = state["scope_id"]
    if scope_id is None:
        raise RuntimeError("retrieval graph completed without a scope")
    return GraphResult(
        resource_ids=state["resource_ids"],
        scope_id=scope_id,
        parent_scope_id=state["parent_scope_id"],
    )
