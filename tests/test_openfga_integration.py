# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Exercise the checked-in policy against a real local OpenFGA server."""

import asyncio
import json
import os

import pytest
from typer.testing import CliRunner

from proofline.cli import app
from proofline.domain import Principal
from proofline.openfga_fixture import provision_openfga
from proofline.vertical_slice import build_vertical_slice

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
        assert await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:acme-rollout", "tenant:acme"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:beta-rollout", "tenant:beta"
        )
        assert not await provisioned.adapter.check_access(
            Principal(id="user:ana"), "viewer", "document:acme-secret", "tenant:acme"
        )
        assert await provisioned.adapter.check_access(
            Principal(id="user:carla"), "viewer", "document:acme-secret", "tenant:acme"
        )
        result = await build_vertical_slice(provisioned.adapter).search_tenant(
            Principal(id="user:ana"), "tenant:acme", "release approval incident"
        )
        assert {candidate.chunk_id for candidate in result.candidates} == {
            "chunk:public-policy",
            "chunk:acme-rollout",
        }
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
