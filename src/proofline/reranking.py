# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Fixed-candidate reranking that cannot bypass retrieval access controls."""

import re
from typing import Protocol

from proofline.domain import Principal, RetrievalCandidate
from proofline.retrieval import AccessGatedRetriever, DocumentChunk, RetrievalResult

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class Reranker(Protocol):
    """Reorder only candidates supplied by an already access-filtered retriever."""

    def rerank(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[RetrievalCandidate, ...]: ...


class TokenCoverageReranker:
    """Prioritize fixed candidates covering more unique query terms.

    This transparent control tests whether a second-stage ordering signal helps
    documented lexical ranking failures; it never creates a new candidate.
    """

    def __init__(self, chunks: tuple[DocumentChunk, ...]) -> None:
        self._content = {
            chunk.id: frozenset(_TOKEN_PATTERN.findall(chunk.content.lower())) for chunk in chunks
        }

    def rerank(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[RetrievalCandidate, ...]:
        query_tokens = frozenset(_TOKEN_PATTERN.findall(query.lower()))
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -len(query_tokens & self._content.get(candidate.chunk_id, frozenset())),
                candidate.rank,
            ),
        )
        return tuple(
            candidate.model_copy(
                update={
                    "rank": rank,
                    "score": float(
                        len(query_tokens & self._content.get(candidate.chunk_id, frozenset()))
                    ),
                }
            )
            for rank, candidate in enumerate(ranked, start=1)
        )


class RerankingRetriever:
    """Enforce that a reranker returns a permutation of fixed candidates only."""

    def __init__(self, retriever: AccessGatedRetriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult:
        return self._rerank(query, await self._retriever.search_public(query, limit))

    async def search_tenant(
        self, principal: Principal, tenant_id: str, query: str, limit: int = 5
    ) -> RetrievalResult:
        result = await self._retriever.search_tenant(principal, tenant_id, query, limit)
        return self._rerank(query, result)

    def _rerank(self, query: str, result: RetrievalResult) -> RetrievalResult:
        candidates = self._reranker.rerank(query, result.candidates)
        if {candidate.chunk_id for candidate in candidates} != {
            candidate.chunk_id for candidate in result.candidates
        } or len(candidates) != len(result.candidates):
            raise ValueError("reranker must return each fixed candidate exactly once")
        return RetrievalResult(result.access_scope, candidates)
