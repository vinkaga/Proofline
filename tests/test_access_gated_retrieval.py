# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Prove that lexical retrieval excludes unauthorized evidence before ranking."""

import pytest

from proofline.authorization import StaticAuthorizationAdapter
from proofline.domain import Principal
from proofline.retrieval import AccessGatedBm25Retriever, DocumentChunk


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
        {("user:ana", "tenant:acme"): frozenset({"document:acme-rollout"})}
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
