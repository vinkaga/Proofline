# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify RRF and reranking preserve the access-filtered candidate universe."""

import pytest

from proofline.domain import AccessScope, Principal, RetrievalCandidate, ScopedResource
from proofline.evaluation_data import EvaluationSuite
from proofline.hybrid_retrieval import HybridRrfRetriever
from proofline.lexical_evaluation import evaluate_lexical_baseline
from proofline.reranking import RerankingRetriever, TokenCoverageReranker
from proofline.retrieval import DocumentChunk, RetrievalResult
from proofline.retrieval_comparison import write_method_comparison_report


class StaticRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    async def search_public(self, query: str, limit: int = 5) -> RetrievalResult:  # noqa: ARG002
        return self._result

    async def search_tenant(
        self, principal: Principal, tenant_id: str, query: str, limit: int = 5  # noqa: ARG002
    ) -> RetrievalResult:
        return self._result


def _candidate(chunk_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        resource_id=f"document:{chunk_id}",
        document_id=chunk_id,
        rank=rank,
        score=1,
        source_url="https://example.test/source",
        source_revision="revision",
    )


@pytest.mark.asyncio
async def test_rrf_fuses_fixed_candidates_and_preserves_scope() -> None:
    scope = AccessScope(
        tenant_id="tenant:acme",
        resources=(ScopedResource(tenant_id="tenant:acme", resource_id="document:one"),),
    )
    lexical = StaticRetriever(RetrievalResult(scope, (_candidate("one", 1), _candidate("two", 2))))
    dense = StaticRetriever(RetrievalResult(scope, (_candidate("two", 1), _candidate("one", 2))))

    result = await HybridRrfRetriever(lexical, dense).search_tenant(
        Principal(id="user:ana"), "tenant:acme", "query"
    )

    assert result.access_scope == scope
    assert {candidate.chunk_id for candidate in result.candidates} == {"one", "two"}
    assert all(candidate.score > 0 for candidate in result.candidates)


@pytest.mark.asyncio
async def test_reranker_cannot_add_unauthorized_candidate() -> None:
    source = StaticRetriever(RetrievalResult(None, (_candidate("one", 1), _candidate("two", 2))))
    chunks = (
        DocumentChunk("one", "document:one", None, "release approval", True),
        DocumentChunk("two", "document:two", None, "release", True),
    )

    result = await RerankingRetriever(source, TokenCoverageReranker(chunks)).search_public(
        "release approval"
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["one", "two"]


@pytest.mark.asyncio
async def test_comparison_report_classifies_rank_changes(tmp_path) -> None:
    suite = EvaluationSuite.model_validate(
        {
            "version": "test",
            "cases": [
                {
                    "id": "case",
                    "mode": "public_documentation",
                    "principal": "user:ana",
                    "query": "one",
                    "expected": "cited_answer",
                    "required_sources": ["one"],
                }
            ],
        }
    )
    lexical = await evaluate_lexical_baseline(
        StaticRetriever(RetrievalResult(None, (_candidate("two", 1), _candidate("one", 2)))),
        suite,
        "revision",
    )
    improved = await evaluate_lexical_baseline(
        StaticRetriever(RetrievalResult(None, (_candidate("one", 1), _candidate("two", 2)))),
        suite,
        "revision",
    )
    from proofline.dense_retrieval import DenseIndexMetadata

    output = tmp_path / "comparison.md"
    write_method_comparison_report(
        {"lexical": lexical, "hybrid": improved},
        DenseIndexMetadata("test", 2, 16, "model", "revision", 128),
        output,
    )

    assert "improved: first relevant rank 2 → 1" in output.read_text()
    assert "Preferred configuration" in output.read_text()
