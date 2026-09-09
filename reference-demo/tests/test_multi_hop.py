# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import Principal, ScopedResource
from proofline_reference_demo.multi_hop import (
    run_benign_two_hop,
    run_clean_two_hop,
    run_poisoned_two_hop,
)


def _authorization() -> StaticAuthorizationAdapter:
    return StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )


def test_clean_two_hop_preserves_scope_and_returns_permitted_evidence() -> None:
    trace = asyncio.run(
        run_clean_two_hop(
            _authorization(), principal=Principal(id="user:ana"), tenant_id="tenant:acme"
        )
    )

    assert trace.scenario == "clean"
    assert "chunk:acme-rollout" in trace.initial_candidate_ids
    assert "chunk:public-policy" in trace.follow_up_candidate_ids
    assert trace.retrieval_hop_count == 2
    assert len(trace.scopes) == 2
    assert trace.scopes[1].parent_scope_id == trace.scopes[0].scope_id
    assert trace.scopes[1].filter_counts == trace.scopes[0].filter_counts
    assert trace.scopes[1].follow_up_count == 1
    assert trace.rejected_fields == ()
    assert "chunk:beta-rollout" not in trace.initial_candidate_ids
    assert "chunk:beta-rollout" not in trace.follow_up_candidate_ids


def test_poisoned_two_hop_is_rejected_before_follow_up_retrieval() -> None:
    trace = asyncio.run(
        run_poisoned_two_hop(
            _authorization(), principal=Principal(id="user:ana"), tenant_id="tenant:acme"
        )
    )

    assert trace.scenario == "poisoned"
    assert "chunk:acme-rollout" in trace.initial_candidate_ids
    assert trace.follow_up_candidate_ids == ()
    assert trace.retrieval_hop_count == 1
    assert len(trace.scopes) == 1
    assert trace.proposal_source_chunk_id == "chunk:acme-rollout"
    assert trace.rejected_fields == ("resource_id",)


def test_benign_security_discussion_can_propose_a_data_only_follow_up() -> None:
    trace = asyncio.run(
        run_benign_two_hop(
            _authorization(), principal=Principal(id="user:ana"), tenant_id="tenant:acme"
        )
    )

    assert trace.scenario == "benign"
    assert trace.proposal_source_chunk_id == "chunk:public-security-guidance"
    assert "chunk:public-security-guidance" in trace.initial_candidate_ids
    assert "chunk:public-policy" in trace.follow_up_candidate_ids
    assert trace.retrieval_hop_count == 2
