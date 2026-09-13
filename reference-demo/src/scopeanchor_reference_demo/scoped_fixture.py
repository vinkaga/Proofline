# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Route the lexical fixture through the ScopeAnchor scoped-retrieval library."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from scopeanchor import (
    RetrievalScope,
    ScopedRetriever,
    ScopeFilters,
    ScopeResolver,
    matches_scope_filters,
    scoped,
    validate_scope_filter_fields,
)

from scopeanchor_reference_demo.authorization import AuthorizationAdapter
from scopeanchor_reference_demo.dense_retrieval import QdrantDenseRetriever
from scopeanchor_reference_demo.domain import Principal, RetrievalCandidate
from scopeanchor_reference_demo.retrieval import AccessGatedBm25Retriever, DocumentChunk
from scopeanchor_reference_demo.tracing import trace_operation
from scopeanchor_reference_demo.vertical_slice import vertical_slice_chunks

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
    resolve_scope = _scope_resolver(
        authorization,
        public_resource_ids=(chunk.resource_id for chunk in chunks if chunk.is_public),
        public_tenant_id=None,
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


def build_scoped_qdrant_retriever(
    dense_retriever: QdrantDenseRetriever,
    authorization: AuthorizationAdapter,
    *,
    public_resource_ids: Iterable[str],
    max_follow_ups: int | None = None,
) -> ScopedRetriever[DemoRequestContext, RetrievalCandidate]:
    """Wrap the Qdrant adapter with the same trusted scope-resolution boundary.

    Qdrant stores public chunks with an empty-string ``tenant_id`` payload,
    while the in-memory fixture represents that value as ``None``. Both paths
    resolve the same resource grants, then hand the complete conjunctive scope
    to their backend through ScopeAnchor's public ``FilteredSearch`` contract.
    """

    resolve_scope = _scope_resolver(
        authorization,
        public_resource_ids=public_resource_ids,
        public_tenant_id="",
        max_follow_ups=max_follow_ups,
    )
    return scoped(dense_retriever.search, resolve_scope=resolve_scope)


def _scope_resolver(
    authorization: AuthorizationAdapter,
    *,
    public_resource_ids: Iterable[str],
    public_tenant_id: str | None,
    max_follow_ups: int | None,
) -> ScopeResolver[DemoRequestContext]:
    """Build trusted scope resolution shared by in-memory and Qdrant adapters."""

    if isinstance(public_resource_ids, str):
        raise ValueError("public_resource_ids must be an iterable of resource IDs, not a string")
    public_ids = tuple(public_resource_ids)
    if any(not isinstance(resource_id, str) or not resource_id for resource_id in public_ids):
        raise ValueError("public_resource_ids must contain non-empty strings")

    async def resolve_scope(context: DemoRequestContext) -> RetrievalScope:
        with trace_operation(
            "scopeanchor.authorization.resolve_scope",
            {"enduser.id": context.principal.id, "scopeanchor.tenant_id": context.tenant_id},
        ):
            access_scope = await authorization.list_permitted_resources(
                context.principal, context.tenant_id
            )
        return RetrievalScope.root(
            principal=context.principal.id,
            filters={
                # Public chunks remain ordinary allowlist entries; neither
                # backend receives an authorization-bypassing public path.
                "tenant_id": [public_tenant_id, context.tenant_id],
                "resource_id": [*access_scope.resource_ids, *public_ids],
            },
            policy_version="reference-demo",
            max_follow_ups=max_follow_ups,
        )

    return resolve_scope
