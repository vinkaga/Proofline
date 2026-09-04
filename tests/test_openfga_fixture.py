# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify OpenFGA fixture provisioning behavior without a live server."""

import asyncio
import json
from typing import cast

import pytest

import proofline.openfga_fixture as fixture
from proofline.authorization import OpenFgaAuthorizationAdapter
from proofline.domain import ScopedResource


class FakeClient:
    async def close(self) -> None:
        pass


def test_provision_openfga_writes_checked_in_model_and_tuples(tmp_path, monkeypatch) -> None:
    (tmp_path / "model.json").write_text(
        json.dumps({"schema_version": "1.1", "type_definitions": []})
    )
    (tmp_path / "tuples.yaml").write_text("tuples: []\n")
    responses: list[dict[str, object]] = [
        {"id": "store-id"},
        {"authorization_model_id": "model-id"},
        {},
    ]
    calls: list[tuple[str, str]] = []

    def fake_request(
        server_url: str, method: str, path: str, payload: object = None
    ) -> dict[str, object]:
        del server_url, payload
        calls.append((method, path))
        return responses.pop(0)

    monkeypatch.setattr(fixture, "_request", fake_request)
    monkeypatch.setattr(fixture, "OpenFgaClient", lambda configuration: FakeClient())

    provisioned = asyncio.run(fixture.provision_openfga("http://localhost:8080/", tmp_path))

    assert provisioned.store_id == "store-id"
    assert calls == [
        ("POST", "/stores"),
        ("POST", "/stores/store-id/authorization-models"),
        ("POST", "/stores/store-id/write"),
    ]


def test_provisioned_store_is_deleted_and_invalid_responses_fail(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(
        server_url: str, method: str, path: str, payload: object = None
    ) -> dict[str, object]:
        del server_url, payload
        calls.append((method, path))
        return {}

    monkeypatch.setattr(fixture, "_request", fake_request)
    provisioned = fixture.ProvisionedOpenFga(
        adapter=cast(OpenFgaAuthorizationAdapter, object()),
        client=cast(fixture.OpenFgaClient, FakeClient()),
        server_url="http://test",
        store_id="store",
    )
    asyncio.run(provisioned.delete())

    assert calls == [("DELETE", "/stores/store")]
    with pytest.raises(ValueError, match="string 'id'"):
        fixture._required_string({}, "id")


def test_static_permissions_are_derived_from_versioned_tuples(tmp_path) -> None:
    (tmp_path / "tuples.yaml").write_text(
        """tuples:
  - user: tenant:acme
    relation: tenant
    object: document:rollout
  - user: user:ana
    relation: direct_viewer
    object: document:rollout
"""
    )

    permissions = fixture.load_static_permissions(tmp_path)

    assert permissions == {
        ("user:ana", "tenant:acme"): (
            ScopedResource(tenant_id="tenant:acme", resource_id="document:rollout"),
        )
    }
