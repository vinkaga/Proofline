# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.bounded_host import run_bounded_host
from proofline_reference_demo.domain import Principal, RequestMode, ScopedResource
from proofline_reference_demo.permission_mcp import build_permission_server, check_access_via_mcp
from proofline_reference_demo.request_routing import classify_request


def _authorization() -> StaticAuthorizationAdapter:
    return StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )


def test_classifier_routes_the_three_supported_request_modes() -> None:
    assert classify_request("Can Ana view the rollout?") is RequestMode.PERMISSION
    assert classify_request("What is ListObjects?") is RequestMode.PUBLIC_DOCUMENTATION
    assert (
        classify_request("What approval does Acme need for rollout?")
        is RequestMode.TENANT_KNOWLEDGE
    )


def test_mcp_server_exposes_only_typed_check_access() -> None:
    server = build_permission_server(
        _authorization(),
        principal=Principal(id="user:ana"),
        tenant_id="tenant:acme",
    )
    tools = asyncio.run(server.list_tools())
    allowed = asyncio.run(
        check_access_via_mcp(
            _authorization(),
            principal=Principal(id="user:ana"),
            tenant_id="tenant:acme",
            relation="viewer",
            resource_id="document:acme-rollout",
        )
    )

    assert [tool.name for tool in tools] == ["check_access"]
    assert set(tools[0].input_schema["properties"]) == {"relation", "resource_id"}
    assert allowed


def test_bounded_host_composes_only_retrieved_permitted_evidence() -> None:
    trace = asyncio.run(
        run_bounded_host(
            _authorization(),
            principal=Principal(id="user:ana"),
            tenant_id="tenant:acme",
            query="What approval does Acme need for rollout?",
        )
    )

    assert trace.request_mode is RequestMode.TENANT_KNOWLEDGE
    assert trace.retrieval_hop_count == 2
    assert trace.citation_chunk_ids[0] in trace.candidate_chunk_ids
    assert "chunk:beta-rollout" not in trace.candidate_chunk_ids
    assert not trace.abstained


def test_bounded_host_abstains_when_permitted_retrieval_has_no_evidence() -> None:
    trace = asyncio.run(
        run_bounded_host(
            _authorization(),
            principal=Principal(id="user:ana"),
            tenant_id="tenant:acme",
            query="Explain quantum banana topology",
        )
    )

    assert trace.abstained
    assert trace.retrieval_hop_count == 1
    assert trace.citation_chunk_ids == ()


def test_bounded_host_uses_mcp_for_denied_permission_without_retrieval() -> None:
    trace = asyncio.run(
        run_bounded_host(
            _authorization(),
            principal=Principal(id="user:ana"),
            tenant_id="tenant:beta",
            query="Can Ana view the Beta rollout?",
            resource_id="document:beta-rollout",
        )
    )

    assert trace.request_mode is RequestMode.PERMISSION
    assert trace.answer == "Access is denied."
    assert trace.tool_calls == ("check_access",)
    assert trace.retrieval_hop_count == 0
