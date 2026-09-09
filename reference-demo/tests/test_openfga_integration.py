# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Exercise the checked-in policy against a real local OpenFGA server."""

import asyncio
import json
import os

import pytest
from typer.testing import CliRunner

from proofline_reference_demo.bounded_host import run_bounded_host
from proofline_reference_demo.cli import app
from proofline_reference_demo.domain import Principal
from proofline_reference_demo.multi_hop import run_clean_two_hop, run_poisoned_two_hop
from proofline_reference_demo.openfga_fixture import provision_openfga
from proofline_reference_demo.scoped_fixture import DemoRequestContext, build_scoped_fixture

OPENFGA_URL = os.environ.get("OPENFGA_URL")
runner = CliRunner()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not OPENFGA_URL, reason="set OPENFGA_URL to run OpenFGA integration tests"),
]


@pytest.mark.asyncio
async def test_checked_in_model_enforces_tenant_membership_and_scope() -> None:
    provisioned = await provision_openfga(OPENFGA_URL or "")
    try:
        scope = await provisioned.adapter.list_permitted_resources(
            Principal(id="user:ana"), "tenant:acme"
        )

        assert scope.resource_ids == ("document:acme-rollout",)
        beta_scope = await provisioned.adapter.list_permitted_resources(
            Principal(id="user:ana"), "tenant:beta"
        )
        assert beta_scope.resource_ids == ()
        assert await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:acme-rollout", "tenant:acme"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:beta-rollout", "tenant:beta"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:acme-secret", "tenant:acme"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:unknown", "tenant:acme"
        )
        assert await provisioned.adapter.check_access(
            Principal(id="user:carla"), "viewer", "document:acme-secret", "tenant:acme"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:carla"), "viewer", "document:acme-secret", "tenant:beta"
        )
        results = await build_scoped_fixture(provisioned.adapter).search(
            "release approval incident",
            context=DemoRequestContext(
                principal=Principal(id="user:ana"), tenant_id="tenant:acme"
            ),
        )
        assert {candidate.chunk_id for candidate in results.items} == {
            "chunk:public-policy",
            "chunk:public-security-guidance",
            "chunk:acme-rollout",
        }
        assert results.scope.filters["resource_id"] == frozenset({"document:acme-rollout"})
        clean_trace = await run_clean_two_hop(
            provisioned.adapter,
            principal=Principal(id="user:ana"),
            tenant_id="tenant:acme",
        )
        poisoned_trace = await run_poisoned_two_hop(
            provisioned.adapter,
            principal=Principal(id="user:ana"),
            tenant_id="tenant:acme",
        )
        assert clean_trace.retrieval_hop_count == 2
        assert poisoned_trace.retrieval_hop_count == 1
        assert poisoned_trace.rejected_fields == ("resource_id",)
        denied_host_trace = await run_bounded_host(
            provisioned.adapter,
            principal=Principal(id="user:ana"),
            tenant_id="tenant:beta",
            query="Can Ana view the Beta rollout?",
            resource_id="document:beta-rollout",
        )
        assert denied_host_trace.answer == "Access is denied."
        assert denied_host_trace.tool_calls == ("check_access",)
    finally:
        await provisioned.delete()


@pytest.mark.asyncio
async def test_demo_check_access_uses_openfga() -> None:
    result = await asyncio.to_thread(
        runner.invoke,
        app,
        [
            "demo-check-access",
            "--principal",
            "user:ana",
            "--tenant",
            "tenant:acme",
            "--resource",
            "document:acme-rollout",
        ],
        env={"OPENFGA_URL": OPENFGA_URL or ""},
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"allowed": True}
