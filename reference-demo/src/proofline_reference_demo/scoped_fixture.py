# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Route the lexical fixture through the Proofline scoped-retrieval library."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from proofline import (
    RetrievalScope,
    ScopedRetriever,
    ScopeFilters,
    matches_scope_filters,
    scoped,
    validate_scope_filter_fields,
)

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal, RetrievalCandidate
from proofline_reference_demo.retrieval import AccessGatedBm25Retriever, DocumentChunk
from proofline_reference_demo.tracing import trace_operation
from proofline_reference_demo.vertical_slice import vertical_slice_chunks

ChunkRanker = Callable[[str, tuple[DocumentChunk, ...], int], tuple[RetrievalCandidate, ...]]


@dataclass(frozen=True, slots=True)
class DemoRequestContext:
    """Trusted host context for one tenant-scoped fixture request."""

    principal: Principal
    tenant_id: str


def build_scoped_fixture(
    authorization: AuthorizationAdapter,
    *,
    max_follow_ups: int | None = None,
    on_retrieval: Callable[[str, ScopeFilters], None] | None = None,
    ranker: ChunkRanker | None = None,
) -> ScopedRetriever[DemoRequestContext, RetrievalCandidate]:
    """Build a demo retriever that resolves authorization before every search."""

    return build_scoped_retriever(
        vertical_slice_chunks(),
        authorization,
        max_follow_ups=max_follow_ups,
        on_retrieval=on_retrieval,
        ranker=ranker,
    )


def build_scoped_retriever(
    chunks: tuple[DocumentChunk, ...],
    authorization: AuthorizationAdapter,
    *,
    max_follow_ups: int | None = None,
    on_retrieval: Callable[[str, ScopeFilters], None] | None = None,
    ranker: ChunkRanker | None = None,
) -> ScopedRetriever[DemoRequestContext, RetrievalCandidate]:
    """Wrap a corpus backend with the reference adapter's full filter contract."""

    supported_filter_fields = frozenset({"tenant_id", "resource_id"})

    async def resolve_scope(context: DemoRequestContext) -> RetrievalScope:
        with trace_operation(
            "proofline.authorization.resolve_scope",
            {"enduser.id": context.principal.id},
        ):
            access_scope = await authorization.list_permitted_resources(
                context.principal, context.tenant_id
            )
        return RetrievalScope.root(
            principal=context.principal.id,
            filters={
                # Public chunks are included as explicit allowlist entries,
                # rather than bypassing the scope's conjunctive filters.
                "tenant_id": [None, context.tenant_id],
                "resource_id": [
                    *access_scope.resource_ids,
                    *(chunk.resource_id for chunk in chunks if chunk.is_public),
                ],
            },
            policy_version="reference-demo",
            max_follow_ups=max_follow_ups,
        )

    async def search(
        query: str,
        *,
        filters: ScopeFilters,
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if on_retrieval is not None:
            on_retrieval(query, filters)
        validate_scope_filter_fields(filters, supported_fields=supported_filter_fields)
        permitted_chunks = tuple(
            chunk
            for chunk in chunks
            if matches_scope_filters(
                {"tenant_id": chunk.tenant_id, "resource_id": chunk.resource_id},
                filters,
                supported_fields=supported_filter_fields,
            )
        )
        return (ranker or AccessGatedBm25Retriever._rank)(query, permitted_chunks, limit)

    return scoped(search, resolve_scope=resolve_scope)
