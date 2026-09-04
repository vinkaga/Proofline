# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provide the sole authorization boundary used by protected retrieval.

Retrieval asks this module for permitted resources before it scores tenant
content, and uses it for authoritative allow or deny decisions. The static
adapter makes boundary cases deterministic in unit tests. The OpenFGA adapter
keeps the production-shaped decision path separate from the LLM and corpus.
"""

from collections.abc import Mapping
from typing import Protocol

from openfga_sdk import OpenFgaClient
from openfga_sdk.client.models.check_request import ClientCheckRequest
from openfga_sdk.client.models.list_objects_request import ClientListObjectsRequest

from proofline.domain import AccessScope, Principal, ScopedResource


class AuthorizationAdapter(Protocol):
    """The minimal authorization contract used by retrieval and agent code."""

    async def list_permitted_resources(
        self,
        principal: Principal,
        tenant_id: str,
    ) -> AccessScope: ...

    async def check_access(
        self,
        principal: Principal,
        relation: str,
        resource_id: str,
        tenant_id: str,
    ) -> bool: ...


class StaticAuthorizationAdapter:
    """A deterministic adapter for unit tests and fixture-driven evaluation."""

    def __init__(self, permissions: Mapping[tuple[str, str], tuple[ScopedResource, ...]]) -> None:
        self._permissions = permissions

    async def list_permitted_resources(
        self,
        principal: Principal,
        tenant_id: str,
    ) -> AccessScope:
        return AccessScope(
            tenant_id=tenant_id,
            resources=tuple(
                sorted(
                    self._permissions.get((principal.id, tenant_id), ()),
                    key=lambda resource: resource.resource_id,
                )
            ),
        )

    async def check_access(
        self,
        principal: Principal,
        relation: str,
        resource_id: str,
        tenant_id: str,
    ) -> bool:
        if relation != "viewer":
            return False
        resource = ScopedResource(tenant_id=tenant_id, resource_id=resource_id)
        return resource in self._permissions.get((principal.id, tenant_id), ())


class OpenFgaAuthorizationAdapter:
    """Adapter that uses OpenFGA as the authoritative policy decision point."""

    def __init__(self, client: OpenFgaClient, resource_type: str = "document") -> None:
        self._client = client
        self._resource_type = resource_type

    async def list_permitted_resources(
        self,
        principal: Principal,
        tenant_id: str,
    ) -> AccessScope:
        response = await self._client.list_objects(
            ClientListObjectsRequest(
                user=principal.id,
                relation="viewer",
                type=self._resource_type,
            )
        )
        return AccessScope(
            tenant_id=tenant_id,
            resources=tuple(
                ScopedResource(tenant_id=tenant_id, resource_id=resource_id)
                for resource_id in sorted(response.objects)
            ),
        )

    async def check_access(
        self,
        principal: Principal,
        relation: str,
        resource_id: str,
        tenant_id: str,
    ) -> bool:
        # Tenant membership is enforced by the OpenFGA model; document IDs are global.
        del tenant_id
        response = await self._client.check(
            ClientCheckRequest(
                user=principal.id,
                relation=relation,
                object=resource_id,
            )
        )
        return response.allowed
