# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Prove that lexical retrieval excludes unauthorized evidence before ranking."""

import pytest

from proofline.authorization import StaticAuthorizationAdapter
from proofline.domain import Principal, ScopedResource
from proofline.retrieval import AccessGatedBm25Retriever, DocumentChunk
from proofline.tracing import trace_tenant_retrieval


@pytest.fixture
def retriever() -> AccessGatedBm25Retriever:
    chunks = (
        DocumentChunk(
            "chunk:public", "document:public-policy", None, "Public rollout policy", True
        ),
        DocumentChunk(
            "chunk:acme", "document:acme-rollout", "tenant:acme", "Acme production rollout guide"
        ),
        DocumentChunk(
            "chunk:beta", "document:beta-rollout", "tenant:beta", "Beta production rollout guide"
        ),
        DocumentChunk(
            "chunk:acme-child", "document:acme-secret", "tenant:acme", "Acme secret incident notes"
        ),
    )
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )
    return AccessGatedBm25Retriever(chunks, authorization)


@pytest.mark.asyncio
async def test_tenant_retrieval_never_returns_cross_tenant_or_unpermitted_chunks(
    retriever: AccessGatedBm25Retriever,
) -> None:
    result = await retriever.search_tenant(
        Principal(id="user:ana"),
        "tenant:acme",
        "production rollout secret incident",
    )

    assert result.access_scope is not None
    assert result.access_scope.resource_ids == ("document:acme-rollout",)
    assert {candidate.chunk_id for candidate in result.candidates} == {"chunk:public", "chunk:acme"}


@pytest.mark.asyncio
async def test_denied_principal_only_receives_public_evidence(
    retriever: AccessGatedBm25Retriever,
) -> None:
    result = await retriever.search_tenant(
        Principal(id="user:bob"),
        "tenant:acme",
        "production rollout",
    )

    assert result.access_scope is not None
    assert result.access_scope.resource_ids == ()
    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk:public"]


@pytest.mark.asyncio
async def test_public_search_cannot_return_tenant_chunks(
    retriever: AccessGatedBm25Retriever,
) -> None:
    result = await retriever.search_public("production rollout")

    assert result.access_scope is None
    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk:public"]


@pytest.mark.asyncio
async def test_same_resource_id_in_two_tenants_does_not_collide() -> None:
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:rollout"),
            ),
            ("user:ana", "tenant:beta"): (
                ScopedResource(tenant_id="tenant:beta", resource_id="document:rollout"),
            ),
        }
    )

    assert await authorization.check_access(
        Principal(id="user:ana"), "viewer", "document:rollout", "tenant:acme"
    )
    assert await authorization.check_access(
        Principal(id="user:ana"), "viewer", "document:rollout", "tenant:beta"
    )


@pytest.mark.asyncio
async def test_identifier_guess_cannot_add_another_tenants_resource(
    retriever: AccessGatedBm25Retriever,
) -> None:
    result = await retriever.search_tenant(
        Principal(id="user:ana"), "tenant:acme", "document beta rollout"
    )

    assert {candidate.chunk_id for candidate in result.candidates} == {"chunk:public", "chunk:acme"}


@pytest.mark.asyncio
async def test_static_adapter_does_not_treat_unknown_relations_as_viewer() -> None:
    authorization = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )

    assert not await authorization.check_access(
        Principal(id="user:ana"), "editor", "document:acme-rollout", "tenant:acme"
    )


@pytest.mark.asyncio
async def test_tenant_trace_rejects_a_public_search_result(
    retriever: AccessGatedBm25Retriever,
) -> None:
    result = await retriever.search_public("rollout")

    with pytest.raises(ValueError, match="access scope"):
        trace_tenant_retrieval("public", Principal(id="user:ana"), result)
