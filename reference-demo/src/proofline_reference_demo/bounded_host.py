# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""A deliberately small host flow that exercises Proofline's retrieval boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from proofline import ProposedRetrievalStep, ScopeFilters

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal, RequestMode, RetrievalCandidate
from proofline_reference_demo.permission_mcp import check_access_via_mcp
from proofline_reference_demo.request_routing import classify_request
from proofline_reference_demo.response_composition import ComposedResponse, compose_response
from proofline_reference_demo.scoped_fixture import DemoRequestContext, build_scoped_fixture


@dataclass(frozen=True, slots=True)
class BoundedHostTrace:
    """Inspectable output of one bounded host request."""

    request_mode: RequestMode
    state_transitions: tuple[str, ...]
    answer: str
    abstained: bool
    candidate_chunk_ids: tuple[str, ...]
    citation_chunk_ids: tuple[str, ...]
    retrieval_hop_count: int
    tool_calls: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe data for the CLI."""

        return asdict(self)


async def run_bounded_host(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
    query: str,
    relation: str = "viewer",
    resource_id: str | None = None,
) -> BoundedHostTrace:
    """Classify, authorize or retrieve, then cite or abstain with a fixed budget."""

    mode = classify_request(query)
    transitions = ["classified"]
    if mode is RequestMode.PERMISSION:
        if resource_id is None:
            return _abstention_trace(mode, transitions + ["missing_permission_target"])
        allowed = await check_access_via_mcp(
            authorization,
            principal=principal,
            tenant_id=tenant_id,
            relation=relation,
            resource_id=resource_id,
        )
        transitions.extend(("checked_access", "allowed" if allowed else "denied"))
        return BoundedHostTrace(
            request_mode=mode,
            state_transitions=tuple(transitions),
            answer="Access is allowed." if allowed else "Access is denied.",
            abstained=False,
            candidate_chunk_ids=(),
            citation_chunk_ids=(),
            retrieval_hop_count=0,
            tool_calls=("check_access",),
        )

    retrieval_hop_count = 0

    def count_retrieval(_: str, __: ScopeFilters) -> None:
        nonlocal retrieval_hop_count
        retrieval_hop_count += 1

    retriever = build_scoped_fixture(
        authorization,
        max_follow_ups=1,
        on_retrieval=count_retrieval,
    )
    initial = await retriever.search(
        query,
        context=DemoRequestContext(principal=principal, tenant_id=tenant_id),
    )
    transitions.append("retrieved")
    candidates = initial.items
    scope_ids = [initial.scope.scope_id]

    if mode is RequestMode.TENANT_KNOWLEDGE and _requires_fixture_follow_up(query, candidates):
        proposed = ProposedRetrievalStep.from_untrusted(
            {
                "query": "public release approval policy",
                "parent_step_id": initial.scope.scope_id,
            }
        )
        follow_up = await retriever.follow_proposed(initial, proposed)
        candidates = _unique_candidates(candidates + follow_up.items)
        scope_ids.append(follow_up.scope.scope_id)
        transitions.append("followed_up")

    response = compose_response(query, candidates)
    transitions.append("abstained" if response.abstained else "composed")
    return _response_trace(
        mode,
        transitions,
        response,
        candidates,
        retrieval_hop_count,
        scope_ids,
    )


def _requires_fixture_follow_up(query: str, candidates: tuple[RetrievalCandidate, ...]) -> bool:
    return "rollout" in query.lower() and any(
        candidate.chunk_id == "chunk:acme-rollout" for candidate in candidates
    )


def _unique_candidates(
    candidates: tuple[RetrievalCandidate, ...],
) -> tuple[RetrievalCandidate, ...]:
    seen: set[str] = set()
    unique: list[RetrievalCandidate] = []
    for candidate in candidates:
        if candidate.chunk_id not in seen:
            seen.add(candidate.chunk_id)
            unique.append(candidate)
    return tuple(unique)


def _response_trace(
    mode: RequestMode,
    transitions: list[str],
    response: ComposedResponse,
    candidates: tuple[RetrievalCandidate, ...],
    retrieval_hop_count: int,
    scope_ids: list[str],
) -> BoundedHostTrace:
    return BoundedHostTrace(
        request_mode=mode,
        state_transitions=tuple(transitions),
        answer=response.text,
        abstained=response.abstained,
        candidate_chunk_ids=tuple(candidate.chunk_id for candidate in candidates),
        citation_chunk_ids=tuple(citation.chunk_id for citation in response.citations),
        retrieval_hop_count=retrieval_hop_count,
        scope_ids=tuple(scope_ids),
    )


def _abstention_trace(mode: RequestMode, transitions: list[str]) -> BoundedHostTrace:
    response = compose_response("", ())
    return _response_trace(mode, transitions + ["abstained"], response, (), 0, [])
