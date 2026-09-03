# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify OpenFGA request construction without requiring a running service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proofline.authorization import OpenFgaAuthorizationAdapter
from proofline.domain import Principal


@pytest.mark.asyncio
async def test_list_objects_builds_a_viewer_request_and_maps_sorted_scope() -> None:
    client = SimpleNamespace(
        list_objects=AsyncMock(return_value=SimpleNamespace(objects=["document:z", "document:a"]))
    )
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    scope = await adapter.list_permitted_resources(Principal(id="user:ana"), "tenant:acme")

    request = client.list_objects.await_args.args[0]
    assert request.user == "user:ana"
    assert request.relation == "viewer"
    assert request.type == "document"
    assert scope.tenant_id == "tenant:acme"
    assert scope.resource_ids == ("document:a", "document:z")


@pytest.mark.asyncio
async def test_check_builds_request_and_maps_allowed_response() -> None:
    client = SimpleNamespace(check=AsyncMock(return_value=SimpleNamespace(allowed=True)))
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    allowed = await adapter.check_access(
        Principal(id="user:ana"), "viewer", "document:acme-rollout", "tenant:acme"
    )

    request = client.check.await_args.args[0]
    assert request.user == "user:ana"
    assert request.relation == "viewer"
    assert request.object == "document:acme-rollout"
    assert allowed is True
