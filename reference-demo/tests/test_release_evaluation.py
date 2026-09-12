# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import pytest

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import ScopedResource
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
