# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify the protected scope-propagation release-gate contract."""

from dataclasses import replace

import pytest

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.openfga_fixture import load_static_permissions
from proofline_reference_demo.scope_evaluation import (
    GateFailure,
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
async def test_scope_propagation_gate_fails_for_an_introduced_exposure_regression() -> None:
    report = await evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))
    scoped = replace(report.configurations[-1], unauthorized_exposure=True)
    regressed = replace(
        report,
        configurations=(*report.configurations[:-1], scoped),
        passed=False,
        failures=(
            GateFailure("access-isolation", "unauthorized evidence was exposed", ("poisoned",)),
        ),
    )

    with pytest.raises(ScopeGateError, match="unauthorized evidence was exposed"):
        validate_scope_propagation(regressed)
