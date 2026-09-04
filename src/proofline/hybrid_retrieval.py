# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Fuse independently access-filtered lexical and dense rankings with RRF."""

import asyncio
from collections import defaultdict

from proofline.domain import Principal, RetrievalCandidate
from proofline.retrieval import AccessGatedRetriever, RetrievalResult


class HybridRrfRetriever:
    """Apply reciprocal-rank fusion only after both retrievers enforce access."""

    def __init__(
        self,
        lexical: AccessGatedRetriever,
        dense: AccessGatedRetriever,
        *,
        rrf_k: int = 60,
        candidate_limit: int = 20,
    ) -> None:
        if rrf_k < 1 or candidate_limit < 1:
            raise ValueError("rrf_k and candidate_limit must be positive")
        self._lexical = lexical
        self._dense = dense
        self._rrf_k = rrf_k
        self._candidate_limit = candidate_limit

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult:
        pool_limit = max(limit, self._candidate_limit)
        lexical, dense = await asyncio.gather(
            self._lexical.search_public(query, pool_limit),
            self._dense.search_public(query, pool_limit),
        )
        if lexical.access_scope is not None or dense.access_scope is not None:
            raise ValueError("public retrievers must not resolve a tenant scope")
        return RetrievalResult(None, self._fuse(lexical.candidates, dense.candidates, limit))

    async def search_tenant(
        self, principal: Principal, tenant_id: str, query: str, limit: int = 5
    ) -> RetrievalResult:
        pool_limit = max(limit, self._candidate_limit)
        lexical, dense = await asyncio.gather(
            self._lexical.search_tenant(principal, tenant_id, query, pool_limit),
            self._dense.search_tenant(principal, tenant_id, query, pool_limit),
        )
        if lexical.access_scope != dense.access_scope or lexical.access_scope is None:
            raise ValueError("hybrid retrievers resolved inconsistent access scopes")
        return RetrievalResult(
            lexical.access_scope, self._fuse(lexical.candidates, dense.candidates, limit)
        )

    def _fuse(
        self,
        lexical: tuple[RetrievalCandidate, ...],
        dense: tuple[RetrievalCandidate, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if limit < 1:
            return ()
        scores: defaultdict[str, float] = defaultdict(float)
        candidates: dict[str, RetrievalCandidate] = {}
        for ranking in (lexical, dense):
            for candidate in ranking:
                prior = candidates.setdefault(candidate.chunk_id, candidate)
                if _identity(prior) != _identity(candidate):
                    raise ValueError(
                        f"retrievers disagreed on chunk provenance: {candidate.chunk_id}"
                    )
                scores[candidate.chunk_id] += 1 / (self._rrf_k + candidate.rank)
        ranked = sorted(
            candidates.values(), key=lambda item: (-scores[item.chunk_id], item.chunk_id)
        )[:limit]
        return tuple(
            candidate.model_copy(update={"rank": rank, "score": scores[candidate.chunk_id]})
            for rank, candidate in enumerate(ranked, start=1)
        )


def _identity(candidate: RetrievalCandidate) -> tuple[str, str, str | None, str, str, str]:
    return (
        candidate.resource_id,
        candidate.document_id,
        candidate.tenant_id,
        candidate.source_url,
        candidate.source_revision,
        candidate.chunk_id,
    )
