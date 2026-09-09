# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import Principal, ScopedResource
from proofline_reference_demo.scoped_fixture import DemoRequestContext, build_scoped_fixture


def test_scoped_fixture_uses_core_scope_filters_before_ranking() -> None:
    retriever = build_scoped_fixture(
        StaticAuthorizationAdapter(
            {
                ("user:ana", "tenant:acme"): (
                    ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
                )
            }
        )
    )

    results = asyncio.run(
        retriever.search(
            "rollout approval",
            context=DemoRequestContext(principal=Principal(id="user:ana"), tenant_id="tenant:acme"),
        )
    )

    # Public material remains visible, but another tenant's material cannot
    # enter the candidate set before lexical ranking.
    assert {candidate.resource_id for candidate in results.items} == {
        "document:acme-rollout",
        "document:public-policy",
        "document:public-security-guidance",
    }
    assert results.scope.filters["resource_id"] == frozenset({"document:acme-rollout"})
