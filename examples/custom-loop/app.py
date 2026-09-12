# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""A minimal host loop: trusted context creates scope; planner output creates queries only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from proofline import (
    ProposedRetrievalStep,
    RetrievalScope,
    ScopedRetriever,
    matches_scope_filters,
    scoped,
    validate_scope_filter_fields,
)
from proofline.scope import ScopeFilters


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Trusted host-authentication output; never constructed from model output."""

    principal: str
    authorized_resource_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class Document:
    resource_id: str
    text: str


DOCUMENTS = (
    Document("document:public-policy", "All production rollouts require approval."),
    Document(
        "document:acme-rollout", "Acme rollout requires release-manager approval."
    ),
    Document("document:beta-rollout", "Beta rollout contains protected details."),
)


def build_retriever() -> ScopedRetriever[RequestContext, Document]:
    """Wrap the application's existing search callable at its authority boundary."""

    def resolve_scope(context: RequestContext) -> RetrievalScope:
        return RetrievalScope.root(
            principal=context.principal,
            filters={"resource_id": context.authorized_resource_ids},
            max_follow_ups=1,
        )

    def search(
        query: str, *, filters: ScopeFilters, limit: int
    ) -> tuple[Document, ...]:
        supported_filter_fields = frozenset({"resource_id"})
        validate_scope_filter_fields(filters, supported_fields=supported_filter_fields)
        query_words = frozenset(query.lower().split())
        matches = tuple(
            document
            for document in DOCUMENTS
            if matches_scope_filters(
                {"resource_id": document.resource_id},
                filters,
                supported_fields=supported_filter_fields,
            )
            and query_words.intersection(document.text.lower().replace(".", "").split())
        )
        return matches[:limit]

    return scoped(search, resolve_scope=resolve_scope)


async def retrieve(
    query: str,
    *,
    context: RequestContext,
    planner_output: Mapping[str, object] | None = None,
) -> tuple[Document, ...]:
    """Run initial retrieval and, optionally, one data-only planned follow-up.

    ``planner_output`` is parsed at the trust boundary. A ``resource_id``,
    tenant, principal, or raw filter is rejected rather than ignored.
    """

    retriever = build_retriever()
    initial = await retriever.search(query, context=context)
    if planner_output is None:
        return initial.items
    proposal = ProposedRetrievalStep.from_untrusted(planner_output)
    follow_up = await retriever.follow_proposed(initial, proposal)
    return follow_up.items
