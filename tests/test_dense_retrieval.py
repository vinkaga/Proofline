# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Prove Qdrant filters protected points before dense candidates are returned."""

import json
import warnings
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest
from qdrant_client import QdrantClient

import proofline.dense_retrieval as dense_retrieval
from proofline.authorization import StaticAuthorizationAdapter
from proofline.dense_retrieval import (
    OpenAiEmbeddingProvider,
    QdrantDenseRetriever,
    TokenHashEmbeddingProvider,
    _candidate,
)
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


def test_dense_index_batches_upserts_for_large_vector_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(":memory:")
    retriever = QdrantDenseRetriever(
        client,
        "batched",
        TokenHashEmbeddingProvider(dimensions=16),
        StaticAuthorizationAdapter({}),
    )
    chunks = tuple(
        DocumentChunk(
            id=f"chunk:{index}",
            resource_id=f"document:{index}",
            tenant_id=None,
            content=f"Public policy {index}",
            is_public=True,
            source_revision="revision",
            source_url=f"https://example.test/{index}",
            document_id=f"document-{index}",
        )
        for index in range(129)
    )
    batch_sizes: list[int] = []
    upsert = client.upsert

    def recording_upsert(*args: object, **kwargs: object) -> object:
        points = kwargs["points"]
        assert isinstance(points, list)
        batch_sizes.append(len(points))
        return upsert(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client, "upsert", recording_upsert)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        retriever.index(chunks)

    assert batch_sizes == [128, 1]


def test_dense_candidate_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        _candidate(None, score=1, rank=1)


def test_openai_embedding_provider_tracks_usage_without_exposing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, count: int) -> None:
            self._count = count

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            vectors = [
                {"index": index, "embedding": [float(index + 1), float(index + 2)]}
                for index in reversed(range(self._count))
            ]
            return json.dumps(
                {
                    "data": vectors,
                    "usage": {"total_tokens": 10},
                }
            ).encode()

    inputs: list[list[str]] = []

    def fake_urlopen(request, timeout: int):  # noqa: ANN001
        assert request.full_url == "https://api.openai.com/v1/embeddings"
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == 30
        body = json.loads(request.data)
        assert body["model"] == "text-embedding-3-small"
        inputs.append(body["input"])
        return Response(len(body["input"]))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(dense_retrieval, "urlopen", fake_urlopen)
    provider = OpenAiEmbeddingProvider()

    assert provider.embed_documents(("first", "second")) == ((1.0, 2.0), (2.0, 3.0))
    assert provider.embed_query("question") == (1.0, 2.0)
    assert inputs == [["first", "second"], ["question"]]
    assert provider.estimated_embedding_cost_usd == pytest.approx(0.0000002)
    assert provider.estimated_query_cost_usd == pytest.approx(0.0000002)


def test_openai_embedding_provider_requires_a_supported_model_and_credential(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAiEmbeddingProvider()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="unsupported OpenAI"):
        OpenAiEmbeddingProvider("not-an-embedding-model")


def test_openai_embedding_provider_batches_documents_and_explains_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Response:
        def __init__(self, count: int) -> None:
            self._count = count

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": [
                        {"index": index, "embedding": [1.0, 2.0]}
                        for index in range(self._count)
                    ],
                    "usage": {"total_tokens": self._count},
                }
            ).encode()

    def fake_urlopen(request, timeout: int):  # noqa: ANN001
        body = json.loads(request.data)
        calls.append(body["input"])
        return Response(len(body["input"]))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(dense_retrieval, "urlopen", fake_urlopen)
    provider = OpenAiEmbeddingProvider()

    assert len(provider.embed_documents(tuple(str(index) for index in range(129)))) == 129
    assert [len(batch) for batch in calls] == [128, 1]

    def rate_limited(*_args: object, **_kwargs: object) -> None:
        raise HTTPError(
            "https://api.openai.com/v1/embeddings",
            429,
            "Too Many Requests",
            Message(),
            BytesIO(b'{"error":{"message":"insufficient quota"}}'),
        )

    monkeypatch.setattr(dense_retrieval, "urlopen", rate_limited)
    with pytest.raises(RuntimeError, match="429: insufficient quota"):
        provider.embed_query("question")
