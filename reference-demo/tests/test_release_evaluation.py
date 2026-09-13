# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import pytest

import proofline_reference_demo.release_evaluation as release_evaluation
from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.bounded_host import BoundedHostTrace
from proofline_reference_demo.domain import RequestMode, ScopedResource
from proofline_reference_demo.evaluation_data import EvaluationSuite
from proofline_reference_demo.release_evaluation import (
    evaluate_release_suite,
    validate_release_evaluation,
)
from proofline_reference_demo.retrieval import DocumentChunk


@pytest.mark.asyncio
async def test_release_evaluation_executes_public_tenant_and_permission_paths() -> None:
    chunks = (
        DocumentChunk(
            "public",
            "document:public",
            None,
            "Check access documentation",
            is_public=True,
            document_id="check",
            source_url="https://example.test/check",
            source_revision="revision",
        ),
        DocumentChunk(
            "acme",
            "document:acme",
            "tenant:acme",
            "Acme rollout approval",
            document_id="acme",
            source_url="https://example.test/acme",
            source_revision="revision",
        ),
    )
    suite = EvaluationSuite.model_validate(
        {
            "version": "test",
            "cases": [
                {
                    "id": "public",
                    "mode": "public_documentation",
                    "principal": "user:ana",
                    "query": "Check access",
                    "expected": "cited_answer",
                    "required_sources": ["check"],
                },
                {
                    "id": "tenant",
                    "mode": "tenant_knowledge",
                    "principal": "user:ana",
                    "tenant": "tenant:acme",
                    "query": "Acme rollout",
                    "expected": "cited_answer",
                    "required_resources": ["document:acme"],
                },
                {
                    "id": "permission",
                    "mode": "permission",
                    "principal": "user:ana",
                    "tenant": "tenant:acme",
                    "relation": "viewer",
                    "resource": "document:acme",
                    "query": "Can Ana view Acme?",
                    "expected": "allow",
                    "required_tool": "check_access",
                },
            ],
        }
    )
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme"),
            )
        }
    )

    report = await evaluate_release_suite(authorization, chunks, suite)

    assert report.passed
    validate_release_evaluation(report)


@pytest.mark.asyncio
async def test_release_gate_independently_rejects_unauthorized_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forged trace must not pass merely because its IDs exist in the corpus."""

    chunks = (
        DocumentChunk(
            "acme",
            "document:acme",
            "tenant:acme",
            "Authorized rollout",
            document_id="acme",
            source_url="https://example.test/acme",
            source_revision="revision",
        ),
        DocumentChunk(
            "secret",
            "document:secret",
            "tenant:acme",
            "Unauthorized incident notes",
            document_id="secret",
            source_url="https://example.test/secret",
            source_revision="revision",
        ),
    )
    suite = EvaluationSuite.model_validate(
        {
            "version": "test",
            "cases": [
                {
                    "id": "tenant",
                    "mode": "tenant_knowledge",
                    "principal": "user:ana",
                    "tenant": "tenant:acme",
                    "query": "Authorized rollout",
                    "expected": "cited_answer",
                    "required_resources": ["document:acme"],
                }
            ],
        }
    )
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme"),
            )
        }
    )

    async def leaked_host(authorization, **kwargs):  # noqa: ANN001, ANN003
        return BoundedHostTrace(
            request_mode=RequestMode.TENANT_KNOWLEDGE,
            state_transitions=("classified", "retrieved", "composed"),
            answer="Retrieved permitted evidence for: Authorized rollout",
            abstained=False,
            candidate_chunk_ids=("secret",),
            citation_chunk_ids=("secret",),
            retrieval_hop_count=1,
            scope_ids=("scope:root",),
        )

    monkeypatch.setattr(release_evaluation, "run_bounded_host", leaked_host)
    report = await evaluate_release_suite(authorization, chunks, suite)

    assert "tenant request returned unauthorized evidence" in report.cases[0].violations
    with pytest.raises(ValueError, match="unauthorized evidence"):
        validate_release_evaluation(report)


@pytest.mark.asyncio
async def test_release_gate_enforces_citation_and_retrieval_hop_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = (
        DocumentChunk(
            "public",
            "document:public",
            None,
            "Public Check documentation",
            is_public=True,
            document_id="check",
            source_url="https://example.test/check",
            source_revision="revision",
        ),
    )
    suite = EvaluationSuite.model_validate(
        {
            "version": "test",
            "cases": [
                {
                    "id": "public",
                    "mode": "public_documentation",
                    "principal": "user:ana",
                    "query": "Check",
                    "expected": "cited_answer",
                    "required_sources": ["check"],
                }
            ],
        }
    )

    async def over_budget_host(authorization, **kwargs):  # noqa: ANN001, ANN003
        return BoundedHostTrace(
            request_mode=RequestMode.PUBLIC_DOCUMENTATION,
            state_transitions=("classified", "retrieved", "composed"),
            answer="Retrieved permitted evidence for: Check",
            abstained=False,
            candidate_chunk_ids=("public",),
            citation_chunk_ids=(),
            retrieval_hop_count=2,
        )

    monkeypatch.setattr(release_evaluation, "run_bounded_host", over_budget_host)
    report = await evaluate_release_suite(StaticAuthorizationAdapter({}), chunks, suite)

    assert set(report.cases[0].violations) >= {
        "expected cited evidence",
        "retrieval hop budget exceeded",
        "required source was not cited",
    }
