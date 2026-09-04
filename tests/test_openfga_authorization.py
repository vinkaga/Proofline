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
        list_objects=AsyncMock(return_value=SimpleNamespace(objects=["document:z", "document:a"])),
        read=AsyncMock(
            return_value=SimpleNamespace(
                tuples=[
                    SimpleNamespace(key=SimpleNamespace(object="document:a")),
                    SimpleNamespace(key=SimpleNamespace(object="document:not-allowed")),
                    SimpleNamespace(key=SimpleNamespace(object="folder:acme")),
                ],
                continuation_token=None,
            )
        ),
    )
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    scope = await adapter.list_permitted_resources(Principal(id="user:ana"), "tenant:acme")

    request = client.list_objects.await_args.args[0]
    assert request.user == "user:ana"
    assert request.relation == "viewer"
    assert request.type == "document"
    tenant_request, options = client.read.await_args.args
    assert tenant_request.user == "tenant:acme"
    assert tenant_request.relation == "tenant"
    assert tenant_request.object == "document:"
    assert options == {"page_size": 100}
    assert scope.tenant_id == "tenant:acme"
    assert scope.resource_ids == ("document:a",)


@pytest.mark.asyncio
async def test_list_objects_follows_tenant_tuple_pages() -> None:
    client = SimpleNamespace(
        list_objects=AsyncMock(return_value=SimpleNamespace(objects=["document:a", "document:b"])),
        read=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    tuples=[SimpleNamespace(key=SimpleNamespace(object="document:a"))],
                    continuation_token="next-page",
                ),
                SimpleNamespace(
                    tuples=[SimpleNamespace(key=SimpleNamespace(object="document:b"))],
                    continuation_token=None,
                ),
            ]
        ),
    )
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    scope = await adapter.list_permitted_resources(Principal(id="user:ana"), "tenant:acme")

    assert scope.resource_ids == ("document:a", "document:b")
    assert client.read.await_args_list[1].args[1] == {
        "page_size": 100,
        "continuation_token": "next-page",
    }


@pytest.mark.asyncio
async def test_check_builds_request_and_maps_allowed_response() -> None:
    client = SimpleNamespace(
        check=AsyncMock(return_value=SimpleNamespace(allowed=True)),
        read=AsyncMock(
            return_value=SimpleNamespace(
                tuples=[
                    SimpleNamespace(
                        key=SimpleNamespace(
                            user="tenant:acme",
                            relation="tenant",
                            object="document:acme-rollout",
                        )
                    )
                ]
            )
        ),
    )
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    allowed = await adapter.check_access(
        Principal(id="user:ana"), "viewer", "document:acme-rollout", "tenant:acme"
    )

    request = client.check.await_args.args[0]
    assert request.user == "user:ana"
    assert request.relation == "viewer"
    assert request.object == "document:acme-rollout"
    assert allowed is True


@pytest.mark.asyncio
async def test_check_denies_a_resource_outside_the_requested_tenant() -> None:
    client = SimpleNamespace(
        check=AsyncMock(return_value=SimpleNamespace(allowed=True)),
        read=AsyncMock(return_value=SimpleNamespace(tuples=[])),
    )
    adapter = OpenFgaAuthorizationAdapter(client)  # type: ignore[arg-type]

    allowed = await adapter.check_access(
        Principal(id="user:carla"), "viewer", "document:acme-secret", "tenant:beta"
    )

    assert not allowed
    client.check.assert_not_awaited()
