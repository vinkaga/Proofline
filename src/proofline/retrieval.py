# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Implement the first proof of access-gated retrieval with lexical BM25.

This module deliberately resolves authorization scope before scoring documents.
It therefore returns only permitted candidates, making the absence of protected
chunks from traces and eventual model context a testable property rather than a
post-processing convention.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from proofline.authorization import AuthorizationAdapter
from proofline.domain import AccessScope, Principal, RetrievalCandidate

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A chunk with the minimum metadata needed for access-filtered retrieval."""

    id: str
    resource_id: str
    tenant_id: str | None
    content: str
    is_public: bool = False
    source_revision: str = ""
    source_url: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Only permitted candidates and their resolved authorization scope."""

    access_scope: AccessScope | None
    candidates: tuple[RetrievalCandidate, ...]


class AccessGatedBm25Retriever:
    """Scores only chunks that authorization permitted before retrieval began."""

    def __init__(
        self, chunks: tuple[DocumentChunk, ...], authorization: AuthorizationAdapter
    ) -> None:
        self._chunks = chunks
        self._authorization = authorization

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult:
        public_chunks = tuple(chunk for chunk in self._chunks if chunk.is_public)
        return RetrievalResult(
            access_scope=None, candidates=self._rank(query, public_chunks, limit)
        )

    async def search_tenant(
        self,
        principal: Principal,
        tenant_id: str,
        query: str,
        limit: int = 5,
    ) -> RetrievalResult:
        scope = await self._authorization.list_permitted_resources(principal, tenant_id)
        permitted_chunks = tuple(
            chunk
            for chunk in self._chunks
            if chunk.is_public
            or (chunk.tenant_id == tenant_id and chunk.resource_id in scope.resource_ids)
        )
        return RetrievalResult(
            access_scope=scope, candidates=self._rank(query, permitted_chunks, limit)
        )

    @staticmethod
    def _rank(
        query: str, chunks: tuple[DocumentChunk, ...], limit: int
    ) -> tuple[RetrievalCandidate, ...]:
        query_tokens = _tokens(query)
        if not query_tokens or not chunks or limit < 1:
            return ()

        documents = tuple(_tokens(chunk.content) for chunk in chunks)
        document_frequency = Counter(token for document in documents for token in set(document))
        average_length = sum(len(document) for document in documents) / len(documents)
        scored = [
            (
                chunk,
                _bm25_score(
                    query_tokens, document, document_frequency, len(documents), average_length
                ),
            )
            for chunk, document in zip(chunks, documents, strict=True)
        ]
        ranked = sorted(
            (item for item in scored if item[1] > 0),
            key=lambda item: (-item[1], item[0].id),
        )[:limit]
        return tuple(
            RetrievalCandidate(
                chunk_id=chunk.id,
                resource_id=chunk.resource_id,
                rank=rank,
                score=score,
            )
            for rank, (chunk, score) in enumerate(ranked, start=1)
        )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _bm25_score(
    query_tokens: tuple[str, ...],
    document: tuple[str, ...],
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
) -> float:
    frequencies = Counter(document)
    score = 0.0
    for token in query_tokens:
        frequency = frequencies[token]
        if not frequency:
            continue
        inverse_frequency = math.log(
            1
            + (document_count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
        )
        score += (
            inverse_frequency
            * (frequency * 2.2)
            / (frequency + 1.2 * (1 - 0.75 + 0.75 * len(document) / average_length))
        )
    return score
