# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify the typed records that make Proofline traces and evaluation cases auditable."""

import pytest
from pydantic import HttpUrl, TypeAdapter, ValidationError

from proofline.domain import (
    AccessScope,
    Citation,
    EvaluationCase,
    InteractionTrace,
    Principal,
    RequestMode,
    RetrievalCandidate,
    ToolCall,
)


def test_domain_models_capture_a_permitted_interaction() -> None:
    principal = Principal(id="user:ana")
    scope = AccessScope(
        tenant_id="tenant:acme",
        resource_ids=("document:rollout-guide",),
    )
    candidate = RetrievalCandidate(
        chunk_id="chunk:rollout:1",
        resource_id="document:rollout-guide",
        rank=1,
        score=0.98,
    )
    citation = Citation(
        chunk_id=candidate.chunk_id,
        source_url=TypeAdapter(HttpUrl).validate_python("https://example.test/rollout"),
        source_revision="abc123",
    )
    trace = InteractionTrace(
        case_id="tenant-rollout-allowed",
        request_mode=RequestMode.TENANT_KNOWLEDGE,
        principal=principal,
        access_scope=scope,
        candidates=(candidate,),
        tool_calls=(
            ToolCall(
                name="check_access",
                arguments={"user": principal.id},
                result={"allowed": True},
            ),
        ),
        citations=(citation,),
    )

    assert trace.access_scope == scope
    assert trace.candidates[0].rank == 1
    assert trace.citations[0].source_url.host == "example.test"


def test_evaluation_case_defaults_to_no_abstention_or_tool() -> None:
    case = EvaluationCase(
        id="public-relation",
        request_mode=RequestMode.PUBLIC_DOCUMENTATION,
        principal=Principal(id="user:ana"),
        query="How does a relation work?",
    )

    assert case.expected_abstention is False
    assert case.required_tool is None


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (Principal, {"id": "ana"}),
        (AccessScope, {"tenant_id": "acme"}),
        (
            RetrievalCandidate,
            {
                "chunk_id": "chunk:1",
                "resource_id": "document:1",
                "rank": 0,
                "score": 0.0,
            },
        ),
    ],
)
def test_invalid_domain_identifiers_are_rejected(
    factory: object,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        factory(**kwargs)  # type: ignore[operator]
