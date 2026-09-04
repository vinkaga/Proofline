# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Prove Qdrant filters protected points before dense candidates are returned."""

import warnings

import pytest
from qdrant_client import QdrantClient

from proofline.authorization import StaticAuthorizationAdapter
from proofline.dense_retrieval import QdrantDenseRetriever, TokenHashEmbeddingProvider, _candidate
from proofline.domain import Principal, ScopedResource
from proofline.retrieval import DocumentChunk


@pytest.fixture
def dense_retriever() -> QdrantDenseRetriever:
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )
    retriever = QdrantDenseRetriever(
        QdrantClient(":memory:"),
        "dense-test",
        TokenHashEmbeddingProvider(dimensions=16),
        authorization,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        metadata = retriever.index(
            (
                DocumentChunk(
                    "chunk:public",
                    "document:public",
                    None,
                    "Public rollout policy",
                    True,
                    "revision",
                    "https://example.test/public",
                    "public",
                ),
                DocumentChunk(
                    "chunk:acme",
                    "document:acme-rollout",
                    "tenant:acme",
                    "Acme production rollout guide",
                    source_revision="revision",
                    source_url="https://example.test/acme",
                    document_id="acme",
                ),
                DocumentChunk(
                    "chunk:beta",
                    "document:beta-rollout",
                    "tenant:beta",
                    "Beta production rollout guide secret",
                    source_revision="revision",
                    source_url="https://example.test/beta",
                    document_id="beta",
                ),
            )
        )
    assert metadata.chunk_count == 3
    assert metadata.dimensions == 16
    return retriever


@pytest.mark.asyncio
async def test_dense_tenant_search_filters_cross_tenant_and_unpermitted_chunks(
    dense_retriever: QdrantDenseRetriever,
) -> None:
    result = await dense_retriever.search_tenant(
        Principal(id="user:ana"), "tenant:acme", "production rollout secret"
    )

    assert result.access_scope is not None
    assert result.access_scope.resource_ids == ("document:acme-rollout",)
    assert {candidate.chunk_id for candidate in result.candidates} == {"chunk:public", "chunk:acme"}
    assert all(candidate.source_revision == "revision" for candidate in result.candidates)


@pytest.mark.asyncio
async def test_dense_public_search_returns_only_public_chunks(
    dense_retriever: QdrantDenseRetriever,
) -> None:
    result = await dense_retriever.search_public("production rollout")

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk:public"]


def test_dense_index_rejects_existing_collection() -> None:
    client = QdrantClient(":memory:")
    retriever = QdrantDenseRetriever(
        client,
        "existing",
        TokenHashEmbeddingProvider(dimensions=16),
        StaticAuthorizationAdapter({}),
    )
    chunks = (
        DocumentChunk(
            "chunk:public",
            "document:public",
            None,
            "Public policy",
            True,
            "revision",
            "https://example.test/public",
            "public",
        ),
    )

    retriever.index(chunks)

    with pytest.raises(ValueError, match="already exists"):
        retriever.index(chunks)

    metadata = retriever.index(chunks, recreate=True)

    assert metadata.collection_name == "existing"


def test_dense_index_leaves_no_collection_when_embedding_fails() -> None:
    class InvalidProvider(TokenHashEmbeddingProvider):
        def embed_documents(self, texts):  # noqa: ANN001
            return ()

    client = QdrantClient(":memory:")
    retriever = QdrantDenseRetriever(
        client,
        "invalid",
        InvalidProvider(dimensions=16),
        StaticAuthorizationAdapter({}),
    )
    chunks = (
        DocumentChunk(
            "chunk:public",
            "document:public",
            None,
            "Public policy",
            True,
            "revision",
            "https://example.test/public",
            "public",
        ),
    )

    with pytest.raises(ValueError, match="wrong number"):
        retriever.index(chunks)

    assert not client.collection_exists("invalid")


def test_dense_candidate_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _candidate(None, score=1, rank=1)
