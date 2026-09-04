# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Qdrant-backed dense retrieval with filters applied before candidate return."""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from proofline.authorization import AuthorizationAdapter
from proofline.domain import Principal, RetrievalCandidate
from proofline.retrieval import DocumentChunk, RetrievalResult
from proofline.settings import load_settings

if TYPE_CHECKING:
    from proofline.lexical_evaluation import CaseMeasurement, LexicalBaselineMeasurement

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class EmbeddingProvider(Protocol):
    """Provider contract shared by local and API-backed embedding implementations."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def estimated_embedding_cost_usd(self) -> float: ...

    @property
    def estimated_query_cost_usd(self) -> float: ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class TokenHashEmbeddingProvider:
    """A deterministic, zero-cost local embedding baseline.

    This provider is intentionally a reproducible dense-vector control, not a
    claim of semantic quality. It lets Qdrant filter and indexing behavior be
    evaluated without downloading a model or requiring credentials.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 2:
            raise ValueError("embedding dimensions must be at least two")
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"token-hash-v1/{self._dimensions}d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def estimated_embedding_cost_usd(self) -> float:
        return 0.0

    @property
    def estimated_query_cost_usd(self) -> float:
        return 0.0

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed_query(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        for token, frequency in Counter(_TOKEN_PATTERN.findall(text.lower())).items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1 if digest[4] & 1 else -1
            vector[bucket] += sign * frequency
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector) if norm else tuple(vector)


class FastEmbedEmbeddingProvider:
    """Optional local learned embeddings supplied by FastEmbed without an API credential.

    FastEmbed is intentionally optional because ONNX Runtime does not publish
    wheels for every supported Python/platform combination. Install it in an
    environment with a compatible runtime before selecting this provider.
    """

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "install the optional FastEmbed extra with `uv sync --extra fastembed` "
                "in an environment supported by ONNX Runtime"
            ) from error
        self._model_id = model_id
        self._dimensions: int | None = None
        self._model = TextEmbedding(model_name=model_id)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("embed documents before reading learned-model dimensions")
        return self._dimensions

    @property
    def estimated_embedding_cost_usd(self) -> float:
        return 0.0

    @property
    def estimated_query_cost_usd(self) -> float:
        return 0.0

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return self._embed(texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed((text,))[0]

    def _embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors = tuple(
            tuple(float(value) for value in vector) for vector in self._model.embed(texts)
        )
        dimensions = len(vectors[0]) if vectors else 0
        if dimensions < 2 or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("embedding model returned an unexpected vector dimension")
        if self._dimensions is not None and self._dimensions != dimensions:
            raise ValueError("embedding model dimensions changed between calls")
        self._dimensions = dimensions
        return vectors


class OpenAiEmbeddingProvider:
    """OpenAI embeddings provider that reads its API key from the environment."""

    def __init__(self, model_id: str = "text-embedding-3-small") -> None:
        api_key = load_settings().openai_api_key
        if api_key is None:
            raise RuntimeError("set OPENAI_API_KEY before selecting an OpenAI embedding model")
        self._api_key = api_key.get_secret_value()
        self._model_id = model_id
        self._input_token_price = _openai_embedding_price(model_id)
        self._dimensions: int | None = None
        self._document_tokens = 0
        self._query_tokens = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("embed documents before reading OpenAI embedding dimensions")
        return self._dimensions

    @property
    def estimated_embedding_cost_usd(self) -> float:
        return self._document_tokens * self._input_token_price / 1_000_000

    @property
    def estimated_query_cost_usd(self) -> float:
        return self._query_tokens * self._input_token_price / 1_000_000

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            vector
            for batch in _batches(texts, size=128)
            for vector in self._embed(batch, document_request=True)
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed((text,), document_request=False)[0]

    def _embed(
        self, texts: Sequence[str], *, document_request: bool
    ) -> tuple[tuple[float, ...], ...]:
        request = Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self._model_id, "input": list(texts)}).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed OpenAI endpoint
                payload = json.loads(response.read())
        except HTTPError as error:
            detail = _openai_error_detail(error)
            raise RuntimeError(
                f"OpenAI embeddings request failed with HTTP {error.code}{detail}"
            ) from error
        except URLError as error:
            raise RuntimeError("OpenAI embeddings request could not reach the API") from error
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings response did not contain data")
        entries: list[tuple[int, list[float]]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("index"), int)
                or not isinstance(embedding, list)
                or any(not isinstance(value, (int, float)) for value in embedding)
            ):
                raise RuntimeError("OpenAI embeddings response contained an invalid vector")
            entries.append((item["index"], [float(value) for value in embedding]))
        vectors = tuple(
            tuple(embedding) for _, embedding in sorted(entries)
        )
        dimensions = _validate_vectors(vectors, len(texts))
        if self._dimensions is not None and self._dimensions != dimensions:
            raise RuntimeError("OpenAI embedding dimensions changed between calls")
        self._dimensions = dimensions
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            if document_request:
                self._document_tokens += usage["total_tokens"]
            else:
                self._query_tokens += usage["total_tokens"]
        return vectors


def _openai_embedding_price(model_id: str) -> float:
    """Return the published per-million-input-token price for supported OpenAI models."""

    prices = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }
    try:
        return prices[model_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported OpenAI embedding model for cost accounting: {model_id}"
        ) from error


def _batches(texts: Sequence[str], *, size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(texts), size):
        yield texts[start : start + size]


def _openai_error_detail(error: HTTPError) -> str:
    """Return a concise, non-sensitive explanation from an OpenAI error body."""

    try:
        payload = json.loads(error.read(512).decode())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    error_payload = payload.get("error") if isinstance(payload, dict) else None
    message = error_payload.get("message") if isinstance(error_payload, dict) else None
    return f": {message}" if isinstance(message, str) else ""


@dataclass(frozen=True, slots=True)
class DenseIndexMetadata:
    """Reproducibility metadata recorded alongside a dense evaluation report."""

    collection_name: str
    chunk_count: int
    dimensions: int
    embedding_model: str
    corpus_revision: str
    estimated_vector_index_bytes: int
    estimated_embedding_cost_usd: float = 0.0
    estimated_query_cost_usd: float = 0.0


class QdrantDenseRetriever:
    """Retrieve only Qdrant points admitted by an access-derived payload filter."""

    _UPSERT_BATCH_SIZE = 128

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        authorization: AuthorizationAdapter,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._authorization = authorization

    def index(
        self, chunks: tuple[DocumentChunk, ...], *, recreate: bool = False
    ) -> DenseIndexMetadata:
        """Create a fresh collection and its access-metadata payload indexes."""

        if not chunks:
            raise ValueError("dense index requires at least one chunk")
        revisions = {chunk.source_revision for chunk in chunks}
        if len(revisions) != 1 or not next(iter(revisions)):
            raise ValueError("dense index requires exactly one non-empty corpus revision")
        vectors = self._embedding_provider.embed_documents([chunk.content for chunk in chunks])
        dimensions = _validate_vectors(vectors, len(chunks))
        if self._client.collection_exists(self._collection_name):
            if not recreate:
                raise ValueError(f"collection already exists: {self._collection_name}")
            self._client.delete_collection(self._collection_name)
        created = False
        try:
            self._client.create_collection(
                self._collection_name,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )
            created = True
            for field in ("visibility", "tenant_id", "resource_id"):
                self._client.create_payload_index(
                    self._collection_name, field, field_schema=PayloadSchemaType.KEYWORD
                )
            for start in range(0, len(chunks), self._UPSERT_BATCH_SIZE):
                self._client.upsert(
                    self._collection_name,
                    points=[
                        PointStruct(id=index, vector=list(vector), payload=_payload(chunk))
                        for index, (chunk, vector) in enumerate(
                            zip(
                                chunks[start : start + self._UPSERT_BATCH_SIZE],
                                vectors[start : start + self._UPSERT_BATCH_SIZE],
                                strict=True,
                            ),
                            start=start,
                        )
                    ],
                )
        except Exception:
            if created:
                self._client.delete_collection(self._collection_name)
            raise
        return DenseIndexMetadata(
            collection_name=self._collection_name,
            chunk_count=len(chunks),
            dimensions=dimensions,
            embedding_model=self._embedding_provider.model_id,
            corpus_revision=next(iter(revisions)),
            estimated_vector_index_bytes=len(chunks) * dimensions * 4,
            estimated_embedding_cost_usd=self._embedding_provider.estimated_embedding_cost_usd,
            estimated_query_cost_usd=self._embedding_provider.estimated_query_cost_usd,
        )

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult:
        return RetrievalResult(None, self._search(query, _public_filter(), limit))

    async def search_tenant(
        self, principal: Principal, tenant_id: str, query: str, limit: int = 5
    ) -> RetrievalResult:
        scope = await self._authorization.list_permitted_resources(principal, tenant_id)
        candidates = self._search(query, _tenant_filter(tenant_id, scope.resource_ids), limit)
        return RetrievalResult(scope, candidates)

    def _search(
        self, query: str, query_filter: Filter, limit: int
    ) -> tuple[RetrievalCandidate, ...]:
        if limit < 1:
            return ()
        response = self._client.query_points(
            self._collection_name,
            query=list(self._embedding_provider.embed_query(query)),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return tuple(
            _candidate(point.payload, point.score, rank)
            for rank, point in enumerate(response.points, start=1)
        )


def _payload(chunk: DocumentChunk) -> dict[str, str]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "resource_id": chunk.resource_id,
        "tenant_id": chunk.tenant_id or "",
        "visibility": "public" if chunk.is_public else "tenant",
        "source_url": chunk.source_url,
        "source_revision": chunk.source_revision,
    }


def _validate_vectors(vectors: Sequence[Sequence[float]], expected_count: int) -> int:
    if len(vectors) != expected_count:
        raise ValueError("embedding provider returned the wrong number of vectors")
    dimensions = len(vectors[0]) if vectors else 0
    if dimensions < 2 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("embedding provider returned invalid vector dimensions")
    return dimensions


def _public_filter() -> Filter:
    return Filter(must=[FieldCondition(key="visibility", match=MatchValue(value="public"))])


def _tenant_filter(tenant_id: str, resource_ids: tuple[str, ...]) -> Filter:
    public = _public_filter()
    if not resource_ids:
        return public
    permitted = Filter(
        must=[
            FieldCondition(key="visibility", match=MatchValue(value="tenant")),
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            FieldCondition(key="resource_id", match=MatchAny(any=list(resource_ids))),
        ]
    )
    return Filter(should=[public, permitted])


def _candidate(payload: object, score: float, rank: int) -> RetrievalCandidate:
    if not isinstance(payload, dict):
        raise ValueError("Qdrant point payload must be a mapping")
    values = payload
    tenant_id = _payload_string(values, "tenant_id") or None
    return RetrievalCandidate(
        chunk_id=_payload_string(values, "chunk_id"),
        document_id=_payload_string(values, "document_id"),
        resource_id=_payload_string(values, "resource_id"),
        tenant_id=tenant_id,
        source_url=_payload_string(values, "source_url"),
        source_revision=_payload_string(values, "source_revision"),
        score=score,
        rank=rank,
    )


def _payload_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Qdrant point payload has no string {field!r}")
    return value


def write_dense_comparison_report(
    dense: "LexicalBaselineMeasurement",
    lexical: "LexicalBaselineMeasurement",
    index: DenseIndexMetadata,
    output: Path,
) -> None:
    """Write dense-versus-lexical results with configuration and changed cases."""

    lexical_by_id = {case.case_id: case for case in lexical.cases}
    changed = [
        case.case_id
        for case in dense.cases
        if case.relevant_identifiers
        and _retrieved_relevance(case) != _retrieved_relevance(lexical_by_id[case.case_id])
    ]
    lines = [
        "# Dense retrieval baseline",
        "",
        f"Embedding model: `{index.embedding_model}` · dimensions: {index.dimensions}",
        f"Corpus revision: `{index.corpus_revision}` · indexed chunks: {index.chunk_count}",
        (
            f"Estimated vector index size: {index.estimated_vector_index_bytes:,} bytes "
            "(float32 vectors)"
        ),
        (
            f"Estimated embedding cost: ${index.estimated_embedding_cost_usd:.6f} · "
            f"query cost: ${index.estimated_query_cost_usd:.6f}"
        ),
        "",
        "| Method | Recall@k | MRR | nDCG@k | Exposure | p50 latency | p95 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_row("Lexical", lexical),
        _metric_row("Dense", dense),
        "",
        "## Changed evidence-retrieval cases",
        "",
    ]
    lines.extend(f"- `{case_id}`" for case_id in changed)
    if not changed:
        lines.append("- None")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def _retrieved_relevance(case: "CaseMeasurement") -> bool:
    return bool(
        case.relevant_identifiers & {identifier for _, identifier in case.ranked_identifiers}
    )


def _metric_row(name: str, measurement: "LexicalBaselineMeasurement") -> str:
    return (
        f"| {name} | {measurement.recall_at_k:.3f} | {measurement.mrr:.3f} | "
        f"{measurement.ndcg_at_k:.3f} | {measurement.unauthorized_exposure_rate:.3f} | "
        f"{measurement.p50_latency_ms:.2f} ms | {measurement.p95_latency_ms:.2f} ms |"
    )
