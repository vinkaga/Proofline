# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""A deliberately small host flow that exercises ScopeAnchor's retrieval boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scopeanchor import ProposedRetrievalStep, ScopeFilters

from scopeanchor_reference_demo.authorization import AuthorizationAdapter
from scopeanchor_reference_demo.domain import Principal, RequestMode, RetrievalCandidate
from scopeanchor_reference_demo.permission_mcp import check_access_via_mcp
from scopeanchor_reference_demo.request_routing import classify_request
from scopeanchor_reference_demo.response_composition import ComposedResponse, compose_response
from scopeanchor_reference_demo.retrieval import AccessGatedBm25Retriever, DocumentChunk
from scopeanchor_reference_demo.scoped_fixture import (
    DemoRequestContext,
    build_scoped_fixture,
    build_scoped_retriever,
)
from scopeanchor_reference_demo.tracing import trace_operation
from scopeanchor_reference_demo.vertical_slice import vertical_slice_chunks


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
    tool_arguments: tuple[dict[str, str], ...] = ()
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
    chunks: tuple[DocumentChunk, ...] | None = None,
    request_mode: RequestMode | None = None,
) -> BoundedHostTrace:
    """Classify, authorize or retrieve, then cite or abstain with a fixed budget."""

    with trace_operation(
        "scopeanchor.request",
        {
            "enduser.id": principal.id,
            "scopeanchor.tenant_id": tenant_id,
            "scopeanchor.query_length": len(query),
        },
    ) as request_span:
        result = await _run_bounded_host(
            authorization,
            principal=principal,
            tenant_id=tenant_id,
            query=query,
            relation=relation,
            resource_id=resource_id,
            chunks=chunks,
            request_mode=request_mode,
        )
        request_span.set_attribute("scopeanchor.request_mode", result.request_mode.value)
        request_span.set_attribute("scopeanchor.scope.count", len(result.scope_ids))
        if result.scope_ids:
            request_span.set_attribute("scopeanchor.scope.ids", result.scope_ids)
        request_span.set_attribute("scopeanchor.candidate_count", len(result.candidate_chunk_ids))
        request_span.set_attribute("scopeanchor.abstained", result.abstained)
        return result


async def _run_bounded_host(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
    query: str,
    relation: str = "viewer",
    resource_id: str | None = None,
    chunks: tuple[DocumentChunk, ...] | None = None,
    request_mode: RequestMode | None = None,
) -> BoundedHostTrace:
    """Execute one request inside the request span created by the public entry point."""

    with trace_operation("scopeanchor.request.classify", {"scopeanchor.query_length": len(query)}):
        mode = request_mode or classify_request(query)
    transitions = ["classified"]
    if mode is RequestMode.PERMISSION:
        if resource_id is None:
            return _abstention_trace(mode, transitions + ["missing_permission_target"])
        with trace_operation(
            "scopeanchor.mcp.check_access",
            {"scopeanchor.relation": relation, "enduser.id": principal.id},
        ):
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
            tool_arguments=(
                {
                    "tenant_id": tenant_id,
                    "relation": relation,
                    "resource_id": resource_id,
                },
            ),
        )

    retrieval_hop_count = 0

    def count_retrieval(_: str, __: ScopeFilters) -> None:
        nonlocal retrieval_hop_count
        retrieval_hop_count += 1

    if mode is RequestMode.PUBLIC_DOCUMENTATION:
        with trace_operation(
            "scopeanchor.retrieval", {"scopeanchor.hop": 0, "scopeanchor.public_corpus": True}
        ) as span:
            candidates = _retrieve_public_documentation(query, chunks or vertical_slice_chunks())
            span.set_attribute("scopeanchor.candidate_count", len(candidates))
        response = compose_response(query, candidates)
        return _response_trace(
            mode,
            ["classified", "retrieved", "composed"],
            response,
            candidates,
            1,
            [],
        )

    retriever = (
        build_scoped_fixture(authorization, max_follow_ups=1, on_retrieval=count_retrieval)
        if chunks is None
        else build_scoped_retriever(
            chunks,
            authorization,
            max_follow_ups=1,
            on_retrieval=count_retrieval,
            ranker=AccessGatedBm25Retriever.rank_for_answer,
        )
    )
    with trace_operation(
        "scopeanchor.retrieval",
        {"scopeanchor.hop": 0, "enduser.id": principal.id},
    ) as span:
        initial = await retriever.search(
            query,
            context=DemoRequestContext(principal=principal, tenant_id=tenant_id),
        )
        span.set_attribute("scopeanchor.scope.id", initial.scope.scope_id)
        span.set_attribute("scopeanchor.candidate_count", len(initial.items))
    transitions.append("retrieved")
    candidates = initial.items
    scope_ids = [initial.scope.scope_id]

    if mode is RequestMode.TENANT_KNOWLEDGE and _requires_fixture_follow_up(query, candidates):
        with trace_operation("scopeanchor.planner.proposal", {"scopeanchor.hop": 1}):
            proposed = ProposedRetrievalStep.from_untrusted(
                {
                    "query": "public release approval policy",
                    "parent_step_id": initial.scope.scope_id,
                }
            )
        with trace_operation(
            "scopeanchor.retrieval",
            {"scopeanchor.hop": 1, "scopeanchor.parent_scope_id": initial.scope.scope_id},
        ) as span:
            follow_up = await retriever.follow_proposed(initial, proposed)
            span.set_attribute("scopeanchor.scope.id", follow_up.scope.scope_id)
            span.set_attribute("scopeanchor.candidate_count", len(follow_up.items))
        candidates = _unique_candidates(candidates + follow_up.items)
        scope_ids.append(follow_up.scope.scope_id)
        transitions.append("followed_up")

    with trace_operation(
        "scopeanchor.response.compose", {"scopeanchor.candidate_count": len(candidates)}
    ):
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


def _retrieve_public_documentation(
    query: str,
    chunks: tuple[DocumentChunk, ...],
) -> tuple[RetrievalCandidate, ...]:
    """Search only the explicit public corpus, independent of tenant authority.

    Public routing deliberately does not resolve an access scope or reuse the
    tenant retriever.  This keeps protected evidence out of a public response,
    cache, and citation set even when the current principal could access it.
    """

    public_chunks = tuple(chunk for chunk in chunks if chunk.is_public)
    return AccessGatedBm25Retriever.rank_for_answer(query, public_chunks, 10)


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
