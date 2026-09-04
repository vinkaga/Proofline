# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Test ACL-aware lexical evaluation independently of a source checkout."""

import pytest

from proofline.authorization import StaticAuthorizationAdapter
from proofline.domain import RetrievalCandidate, ScopedResource
from proofline.evaluation_data import EvaluationSuite
from proofline.lexical_evaluation import (
    evaluate_lexical_baseline,
    validate_baseline_measurement,
    write_lexical_report,
    write_lexical_traces,
)
from proofline.retrieval import AccessGatedBm25Retriever, DocumentChunk, RetrievalResult


@pytest.fixture
def lexical_retriever() -> AccessGatedBm25Retriever:
    return AccessGatedBm25Retriever(
        (
            DocumentChunk(
                "chunk:public",
                "document:public",
                None,
                "Check decides whether a user may view a document.",
                True,
                document_id="perform-check",
                source_url="https://example.test/check",
                source_revision="a" * 40,
            ),
            DocumentChunk(
                "chunk:acme",
                "document:acme-rollout",
                "tenant:acme",
                "Acme rollout requires approval.",
                document_id="authorization-models",
                source_url="https://example.test/models",
                source_revision="a" * 40,
            ),
            DocumentChunk(
                "chunk:beta",
                "document:beta-rollout",
                "tenant:beta",
                "Beta rollout requires approval.",
                document_id="authorization-models",
                source_url="https://example.test/models",
                source_revision="a" * 40,
            ),
        ),
        StaticAuthorizationAdapter(
            {
                ("user:ana", "tenant:acme"): (
                    ScopedResource(
                        tenant_id="tenant:acme", resource_id="document:acme-rollout"
                    ),
                )
            }
        ),
    )


@pytest.fixture
def suite() -> EvaluationSuite:
    return EvaluationSuite.model_validate(
        {
            "version": "test-v0",
            "cases": [
                {
                    "id": "public",
                    "mode": "public_documentation",
                    "principal": "user:ana",
                    "query": "What does Check decide?",
                    "expected": "cited_answer",
                    "required_sources": ["perform-check"],
                },
                {
                    "id": "tenant",
                    "mode": "tenant_knowledge",
                    "principal": "user:ana",
                    "tenant": "tenant:acme",
                    "query": "What does the Acme rollout require?",
                    "expected": "cited_answer",
                    "required_resources": ["document:acme-rollout"],
                },
                {
                    "id": "denied",
                    "mode": "tenant_knowledge",
                    "principal": "user:ana",
                    "tenant": "tenant:acme",
                    "query": "Reveal Beta rollout.",
                    "expected": "abstain",
                },
                {
                    "id": "permission",
                    "mode": "permission",
                    "principal": "user:ana",
                    "query": "Can Ana view Acme?",
                    "relation": "viewer",
                    "resource": "document:acme-rollout",
                    "required_tool": "check_access",
                    "expected": "allow",
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_evaluation_reports_ranking_latency_and_zero_exposure(
    lexical_retriever: AccessGatedBm25Retriever, suite: EvaluationSuite, tmp_path
) -> None:
    measurement = await evaluate_lexical_baseline(
        lexical_retriever, suite, "corpus-test", limit=5
    )

    assert measurement.retrieval_case_count == 2
    assert measurement.recall_at_k == 1
    assert measurement.mrr == 1
    assert measurement.ndcg_at_k == 1
    assert measurement.unauthorized_exposure_rate == 0
    assert measurement.provenance_violation_rate == 0
    assert measurement.cases[1].candidate_ids == ("chunk:acme",)
    assert measurement.cases[1].trace.context_chunk_ids == ("chunk:acme",)
    output = tmp_path / "report.md"
    write_lexical_report(measurement, output)
    assert "Unauthorized exposure |" in output.read_text()
    assert "Executed retrieval for 3 of 4 suite cases" in output.read_text()
    traces_output = tmp_path / "traces.jsonl"
    write_lexical_traces(measurement, traces_output)
    assert len(traces_output.read_text().splitlines()) == 3
    validate_baseline_measurement(measurement)


@pytest.mark.asyncio
async def test_metrics_retain_actual_rank_when_chunks_share_a_source() -> None:
    class RankedRetriever:
        async def search_public(self, query: str, limit: int = 5):  # noqa: ARG002
            candidates = (
                RetrievalCandidate(
                    chunk_id="chunk:1",
                    resource_id="document:one",
                    rank=1,
                    score=1,
                    document_id="wrong",
                ),
                RetrievalCandidate(
                    chunk_id="chunk:2",
                    resource_id="document:one",
                    rank=2,
                    score=0.9,
                    document_id="wrong",
                ),
                RetrievalCandidate(
                    chunk_id="chunk:3",
                    resource_id="document:two",
                    rank=3,
                    score=0.8,
                    document_id="target",
                ),
            )
            return RetrievalResult(None, candidates)

        async def search_tenant(self, principal, tenant_id, query, limit=5):  # noqa: ANN001, ARG002
            return await self.search_public(query, limit)

    one_case_suite = EvaluationSuite.model_validate(
        {
            "version": "test",
            "cases": [
                {
                    "id": "rank",
                    "mode": "public_documentation",
                    "principal": "user:ana",
                    "query": "target",
                    "expected": "cited_answer",
                    "required_sources": ["target"],
                }
            ],
        }
    )
    measurement = await evaluate_lexical_baseline(RankedRetriever(), one_case_suite, "test")

    assert measurement.mrr == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_evaluation_detects_an_unauthorized_candidate(suite: EvaluationSuite) -> None:
    class UnsafeRetriever:
        async def search_public(self, query: str, limit: int = 5):  # noqa: ARG002
            return RetrievalResult(
                None,
                (
                    # A protected candidate on the public path must fail even if relevant.
                    RetrievalCandidate(
                        chunk_id="chunk:secret",
                        resource_id="document:secret",
                        rank=1,
                        score=1,
                        tenant_id="tenant:acme",
                    ),
                ),
            )

        async def search_tenant(self, principal, tenant_id, query, limit=5):  # noqa: ANN001, ARG002
            return await self.search_public(query, limit)

    measurement = await evaluate_lexical_baseline(UnsafeRetriever(), suite, "corpus-test")

    assert measurement.unauthorized_exposure_rate == 1
