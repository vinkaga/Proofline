# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Route the lexical fixture through the Proofline scoped-retrieval library."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from proofline import RetrievalScope, ScopedRetriever, ScopeFilters, scoped

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal, RetrievalCandidate
from proofline_reference_demo.retrieval import AccessGatedBm25Retriever
from proofline_reference_demo.tracing import trace_operation
from proofline_reference_demo.vertical_slice import vertical_slice_chunks


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
) -> ScopedRetriever[DemoRequestContext, RetrievalCandidate]:
    """Build a demo retriever that resolves authorization before every search."""

    chunks = vertical_slice_chunks()

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
                "tenant_id": [context.tenant_id],
                "resource_id": list(access_scope.resource_ids),
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
        tenant_ids = filters.get("tenant_id", frozenset())
        resource_ids = filters.get("resource_id", frozenset())
        permitted_chunks = tuple(
            chunk
            for chunk in chunks
            if chunk.is_public
            or (chunk.tenant_id in tenant_ids and chunk.resource_id in resource_ids)
        )
        return AccessGatedBm25Retriever._rank(query, permitted_chunks, limit)

    return scoped(search, resolve_scope=resolve_scope)
