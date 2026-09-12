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
from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import AccessScope, Principal, RetrievalCandidate

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_COMPOUND_IDENTIFIER = re.compile(r"\b[A-Z]?[a-z]+[A-Z][a-z]+\b")
_EXPLICIT_RESOURCE = re.compile(r"\bdocument:[a-z0-9_-]+\b", re.IGNORECASE)
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "every",
        "explain",
        "for",
        "how",
        "i",
        "is",
        "it",
        "may",
        "of",
        "open",
        "or",
        "say",
        "the",
        "to",
        "fga",
        "what",
        "which",
        "without",
    }
)


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
    document_id: str = ""
    search_context: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Only permitted candidates and their resolved authorization scope."""

    access_scope: AccessScope | None
    candidates: tuple[RetrievalCandidate, ...]


class AccessGatedRetriever(Protocol):
    """Shared contract for retrieval methods that enforce access before ranking."""

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult: ...

    async def search_tenant(
        self, principal: Principal, tenant_id: str, query: str, limit: int = 5
    ) -> RetrievalResult: ...


def is_permitted_tenant_chunk(
    chunk: DocumentChunk,
    *,
    tenant_id: str,
    permitted_resource_ids: Collection[str],
) -> bool:
    """Apply the demo's public-or-authorized-tenant retrieval policy.

    The caller must resolve ``permitted_resource_ids`` from authorization before
    using this predicate. It intentionally treats public chunks as visible to a
    tenant-scoped request and requires both tenant and resource matches for a
    protected chunk.
    """

    return chunk.is_public or (
        chunk.tenant_id == tenant_id and chunk.resource_id in permitted_resource_ids
    )


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
            if is_permitted_tenant_chunk(
                chunk,
                tenant_id=tenant_id,
                permitted_resource_ids=scope.resource_ids,
            )
        )
        return RetrievalResult(
            access_scope=scope, candidates=self._rank(query, permitted_chunks, limit)
        )

    @staticmethod
    def _rank(
        query: str, chunks: tuple[DocumentChunk, ...], limit: int
    ) -> tuple[RetrievalCandidate, ...]:
        """Return the benchmark's plain BM25 ranking without host answer policy."""

        query_tokens = tuple(_TOKEN_PATTERN.findall(query.lower()))
        if not query_tokens or not chunks or limit < 1:
            return ()
        documents = tuple(tuple(_TOKEN_PATTERN.findall(chunk.content.lower())) for chunk in chunks)
        return _rank_documents(query_tokens, chunks, documents, limit)

    @staticmethod
    def rank_for_answer(
        query: str, chunks: tuple[DocumentChunk, ...], limit: int
    ) -> tuple[RetrievalCandidate, ...]:
        """Apply the bounded host's stricter grounding policy after ACL filtering.

        This is intentionally separate from :meth:`_rank`: benchmark utility
        measurement must not be altered by a product host's abstention policy.
        """

        grounding_tokens = _query_tokens(query)
        query_tokens = _expand_query_tokens(grounding_tokens)
        if not grounding_tokens or not chunks or limit < 1:
            return ()

        # Index stable document and resource identifiers alongside source text.
        # They are part of the corpus contract, not generated answer text, and
        # make identifier-bearing requests retrievable without weakening ACLs.
        documents = tuple(
            _tokens(
                f"{chunk.document_id} {chunk.resource_id} {chunk.search_context} {chunk.content}"
            )
            for chunk in chunks
        )
        scored = _score_documents(query_tokens, chunks, documents)
        minimum_matches = max(1, (len(set(grounding_tokens)) + 1) // 2)
        requested_resources = frozenset(
            match.group().lower() for match in _EXPLICIT_RESOURCE.finditer(query)
        )
        compound_identifiers = frozenset(
            match.group().lower() for match in _COMPOUND_IDENTIFIER.finditer(query)
        )
        # A single incidental word is not sufficient grounding for an answer.
        # This removes lexical false positives such as an unsupported question
        # sharing only "OpenFGA" with documentation, while retaining concise
        # identifier queries (whose identifier expands into meaningful tokens).
        ranked = sorted(
            (
                item
                for item, document in zip(scored, documents, strict=True)
                if item[1] > 0 and len(set(query_tokens) & set(document)) >= minimum_matches
                and (
                    not requested_resources
                    or item[0].resource_id.lower() in requested_resources
                )
                and all(
                    identifier in _candidate_search_text(item[0]).lower()
                    for identifier in compound_identifiers
                )
            ),
            key=lambda item: (-item[1], item[0].id),
        )[:limit]
        return _candidates(ranked)


def _rank_documents(
    query_tokens: tuple[str, ...],
    chunks: tuple[DocumentChunk, ...],
    documents: tuple[tuple[str, ...], ...],
    limit: int,
) -> tuple[RetrievalCandidate, ...]:
    ranked = sorted(
        (item for item in _score_documents(query_tokens, chunks, documents) if item[1] > 0),
        key=lambda item: (-item[1], item[0].id),
    )[:limit]
    return _candidates(ranked)


def _score_documents(
    query_tokens: tuple[str, ...],
    chunks: tuple[DocumentChunk, ...],
    documents: tuple[tuple[str, ...], ...],
) -> list[tuple[DocumentChunk, float]]:
    document_frequency = Counter(token for document in documents for token in set(document))
    average_length = sum(len(document) for document in documents) / len(documents)
    return [
        (
            chunk,
            _bm25_score(query_tokens, document, document_frequency, len(documents), average_length),
        )
        for chunk, document in zip(chunks, documents, strict=True)
    ]


def _candidates(ranked: list[tuple[DocumentChunk, float]]) -> tuple[RetrievalCandidate, ...]:
    return tuple(
        RetrievalCandidate(
            chunk_id=chunk.id,
            resource_id=chunk.resource_id,
            rank=rank,
            score=score,
            document_id=chunk.document_id,
            tenant_id=chunk.tenant_id,
            source_url=chunk.source_url,
            source_revision=chunk.source_revision,
        )
        for rank, (chunk, score) in enumerate(ranked, start=1)
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(_CAMEL_CASE_BOUNDARY.sub(" ", text).lower()))


def _query_tokens(query: str) -> tuple[str, ...]:
    """Keep answer-bearing terms; discard syntax words that cause false grounding."""

    return tuple(token for token in _tokens(query) if token not in _QUERY_STOP_WORDS)


def _expand_query_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Add a small, documented vocabulary bridge for the selected corpus.

    The host remains lexical and deterministic, but OpenFGA users commonly ask
    for the Check and ListObjects APIs by their outcomes rather than their API
    names. These expansions are retrieval terms only; they cannot widen the
    authorized corpus selected before ranking.
    """

    token_set = frozenset(tokens)
    expansions: tuple[str, ...] = ()
    if (
        {"find", "view"}.issubset(token_set)
        or {"enumerate", "objects"}.issubset(token_set)
        or "accessible" in token_set
        or "visible" in token_set
    ):
        expansions += ("list", "objects")
    if "decision" in token_set or {"relation", "object"}.issubset(token_set):
        expansions += ("check",)
    if {"model", "configured"}.issubset(token_set) or {"model", "written"}.issubset(token_set):
        expansions += ("configuration", "language")
    if "concepts" in token_set:
        expansions += ("fine", "grained", "permissions")
    return tokens + expansions


def _candidate_search_text(chunk: DocumentChunk) -> str:
    """Use the same stable, non-answer-bearing fields as the lexical index."""

    return f"{chunk.document_id} {chunk.resource_id} {chunk.search_context} {chunk.content}"


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
