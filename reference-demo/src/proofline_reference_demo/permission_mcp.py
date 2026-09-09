# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""The reference host's narrow MCP boundary for permission decisions."""

from __future__ import annotations

from typing import Any, cast

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult
from pydantic import BaseModel

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal


class CheckAccessResult(BaseModel):
    """Typed result returned by the authoritative permission tool."""

    allowed: bool


def build_permission_server(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
) -> MCPServer[Any]:
    """Build a context-bound MCP server exposing only ``check_access``.

    Principal and tenant originate in trusted host code, not tool arguments.
    """

    server: MCPServer[Any] = MCPServer("proofline-reference-demo")

    @server.tool(
        name="check_access",
        description="Check whether a principal may access one tenant-qualified resource.",
        structured_output=True,
    )
    async def check_access(
        relation: str,
        resource_id: str,
    ) -> CheckAccessResult:
        allowed = await authorization.check_access(
            principal, relation, resource_id, tenant_id
        )
        return CheckAccessResult(allowed=allowed)

    return server


async def check_access_via_mcp(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
    relation: str,
    resource_id: str,
) -> bool:
    """Invoke the local MCP tool rather than calling authorization directly."""

    result = await build_permission_server(
        authorization,
        principal=principal,
        tenant_id=tenant_id,
    ).call_tool(
        "check_access",
        {
            "relation": relation,
            "resource_id": resource_id,
        },
    )
    if not isinstance(result, CallToolResult) or result.is_error:
        raise RuntimeError("MCP check_access invocation failed")
    structured = cast(dict[str, object], result.structured_content)
    allowed = structured.get("allowed")
    if not isinstance(allowed, bool):
        raise RuntimeError("MCP check_access returned an invalid result")
    return allowed
