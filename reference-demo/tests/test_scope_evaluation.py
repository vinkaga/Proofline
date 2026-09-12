# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify the protected scope-propagation release-gate contract."""

from dataclasses import replace

import pytest
from proofline import ScopedRetriever

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import RetrievalCandidate
from proofline_reference_demo.openfga_fixture import load_static_permissions
from proofline_reference_demo.scope_evaluation import (
    ScopeGateError,
    evaluate_scope_propagation,
    validate_scope_propagation,
)


@pytest.mark.asyncio
async def test_scope_propagation_gate_distinguishes_the_three_configurations() -> None:
    report = await evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))

    insecure, acl_only, scoped = report.configurations
    assert report.passed
    assert report.failures == ()
    assert insecure.planner_accepted_scope_input
    assert insecure.unauthorized_exposure
    assert acl_only.planner_accepted_scope_input
    assert not acl_only.unauthorized_exposure
    assert scoped.rejected_scope_fields == ("resource_id",)
    assert not scoped.planner_accepted_scope_input
    assert not scoped.unauthorized_exposure
    assert scoped.retrieval_hop_count == 1
    assert not insecure.scope_lineage_complete
    assert not acl_only.scope_lineage_complete


@pytest.mark.asyncio
async def test_scope_propagation_gate_records_clean_benign_and_poisoned_traces() -> None:
    report = await evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))

    clean, benign, poisoned = report.traces
    assert (clean.scenario, benign.scenario, poisoned.scenario) == (
        "clean",
        "benign",
        "poisoned",
    )
    assert clean.scopes[1].parent_scope_id == clean.scopes[0].scope_id
    assert benign.scopes[1].parent_scope_id == benign.scopes[0].scope_id
    assert benign.proposal_source_chunk_id == "chunk:public-security-guidance"
    assert poisoned.proposal_source_chunk_id == "chunk:acme-rollout"

    repeated = await evaluate_scope_propagation(
        StaticAuthorizationAdapter(load_static_permissions())
    )
    assert report.as_dict() == repeated.as_dict()


@pytest.mark.asyncio
async def test_scope_propagation_gate_detects_an_actual_follow_up_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_follow_proposed = ScopedRetriever.follow_proposed

    async def leaking_follow_proposed(self, previous, proposed, *, limit=10):  # noqa: ANN001
        result = await original_follow_proposed(self, previous, proposed, limit=limit)
        return replace(
            result,
            items=(
                *result.items,
                RetrievalCandidate(
                    chunk_id="chunk:beta-rollout",
                    resource_id="document:beta-rollout",
                    tenant_id="tenant:beta",
                    rank=len(result.items) + 1,
                    score=0.0,
                ),
            ),
        )

    monkeypatch.setattr(ScopedRetriever, "follow_proposed", leaking_follow_proposed)
    report = await evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))

    with pytest.raises(ScopeGateError, match="unauthorized evidence was exposed"):
        validate_scope_propagation(report)

    scoped = report.configurations[-1]
    assert not report.passed
    assert scoped.unauthorized_exposure
    assert report.failures[-1].trace_scenarios == ("clean", "benign")
