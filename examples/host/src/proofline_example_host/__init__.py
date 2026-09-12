# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Shared host fixture for the runnable interoperability examples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from proofline import ProposedRetrievalStep, RetrievalScope, ScopedResults, ScopedRetriever, scoped
from proofline.scope import ScopeFilters


@dataclass(frozen=True, slots=True)
class TrustedRequest:
    principal: str
    resource_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class Evidence:
    resource_id: str
    text: str


_EVIDENCE = (
    Evidence("document:acme-rollout", "Acme requires release-manager approval."),
    Evidence("document:beta-rollout", "Beta has protected rollout details."),
)


def retriever() -> ScopedRetriever[TrustedRequest, Evidence]:
    """Build the common host boundary; frameworks never supply raw filters."""

    def resolve(request: TrustedRequest) -> RetrievalScope:
        return RetrievalScope.root(
            principal=request.principal,
            filters={"resource_id": request.resource_ids},
            max_follow_ups=1,
        )

    def search(query: str, *, filters: ScopeFilters, limit: int) -> tuple[Evidence, ...]:
        terms = frozenset(query.lower().replace(".", "").split())
        return tuple(
            evidence
            for evidence in _EVIDENCE
            if evidence.resource_id in filters["resource_id"]
            and terms.intersection(evidence.text.lower().replace(".", "").split())
        )[:limit]

    return scoped(search, resolve_scope=resolve)


async def retrieve(
    request: TrustedRequest, query: str, proposal: Mapping[str, object] | None = None
) -> tuple[Evidence, ...]:
    """The only common flow examples use for initial and planned retrieval."""

    return (await retrieve_results(request, query, proposal=proposal)).items


async def retrieve_results(
    request: TrustedRequest,
    query: str,
    proposal: Mapping[str, object] | None = None,
) -> ScopedResults[Evidence]:
    """Return evidence with scope lineage for hosts that expose an audited hop."""

    boundary = retriever()
    initial = await boundary.search(query, context=request)
    if proposal is None:
        return initial
    step = ProposedRetrievalStep.from_untrusted(proposal)
    return await boundary.follow_proposed(initial, step)
