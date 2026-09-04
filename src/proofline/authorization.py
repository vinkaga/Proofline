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
from openfga_sdk.models.read_request_tuple_key import ReadRequestTupleKey

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
        allowed = await self._client.list_objects(
            ClientListObjectsRequest(
                user=principal.id,
                relation="viewer",
                type=self._resource_type,
            )
        )
        tenant_resources = await self._list_tenant_resources(tenant_id)
        return AccessScope(
            tenant_id=tenant_id,
            resources=tuple(
                ScopedResource(tenant_id=tenant_id, resource_id=resource_id)
                for resource_id in sorted(set(allowed.objects) & tenant_resources)
            ),
        )

    async def _list_tenant_resources(self, tenant_id: str) -> set[str]:
        """Read every document explicitly assigned to one tenant.

        ``ListObjects`` answers for a principal across all tenants. Intersecting
        that result with the policy tuples for this tenant keeps the returned
        ``AccessScope`` tenant-qualified before retrieval sees it.
        """

        resources: set[str] = set()
        continuation_token: str | None = None
        while True:
            # The SDK consumes pagination keys from this mapping, so each request
            # needs its own options object.
            options: dict[str, int | str | dict[str, int | str]] = {"page_size": 100}
            if continuation_token:
                options["continuation_token"] = continuation_token
            response = await self._client.read(
                ReadRequestTupleKey(
                    user=tenant_id,
                    relation="tenant",
                    object=f"{self._resource_type}:",
                ),
                options,
            )
            resources.update(
                item.key.object
                for item in response.tuples or ()
                if item.key is not None
                and item.key.object is not None
                and item.key.object.startswith(f"{self._resource_type}:")
            )
            if not response.continuation_token:
                return resources
            continuation_token = response.continuation_token

    async def check_access(
        self,
        principal: Principal,
        relation: str,
        resource_id: str,
        tenant_id: str,
    ) -> bool:
        if not await self._tenant_contains_resource(tenant_id, resource_id):
            return False
        response = await self._client.check(
            ClientCheckRequest(
                user=principal.id,
                relation=relation,
                object=resource_id,
            )
        )
        return response.allowed

    async def _tenant_contains_resource(self, tenant_id: str, resource_id: str) -> bool:
        """Confirm that an explicit permission request names the right tenant."""

        response = await self._client.read(
            ReadRequestTupleKey(
                user=tenant_id,
                relation="tenant",
                object=resource_id,
            )
        )
        return any(
            item.key is not None
            and item.key.object == resource_id
            and item.key.relation == "tenant"
            and item.key.user == tenant_id
            for item in response.tuples or ()
        )
