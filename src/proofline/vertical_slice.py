# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provide the deterministic six-document fixture used by the Phase 1.5 demo."""

from proofline.authorization import AuthorizationAdapter, StaticAuthorizationAdapter
from proofline.domain import ScopedResource
from proofline.retrieval import AccessGatedBm25Retriever, DocumentChunk


def vertical_slice_chunks() -> tuple[DocumentChunk, ...]:
    """Build public, allowed, cross-tenant, and child-resource fixture chunks."""

    chunks = (
        DocumentChunk(
            "chunk:public-policy",
            "document:public-policy",
            None,
            "Public release approval policy.",
            is_public=True,
        ),
        DocumentChunk(
            "chunk:public-fga",
            "document:public-fga",
            None,
            "ListObjects returns resources a user may view.",
            is_public=True,
        ),
        DocumentChunk(
            "chunk:acme-rollout",
            "document:acme-rollout",
            "tenant:acme",
            "Acme rollout requires release approval.",
        ),
        DocumentChunk(
            "chunk:acme-secret",
            "document:acme-secret",
            "tenant:acme",
            "Acme child incident notes are private.",
        ),
        DocumentChunk(
            "chunk:beta-rollout",
            "document:beta-rollout",
            "tenant:beta",
            "Beta rollout requires release approval.",
        ),
        DocumentChunk(
            "chunk:beta-secret",
            "document:beta-secret",
            "tenant:beta",
            "Beta incident notes are private.",
        ),
    )
    return chunks


def build_vertical_slice(
    authorization: AuthorizationAdapter | None = None,
) -> AccessGatedBm25Retriever:
    """Build the fixture with a supplied policy adapter or its static test policy."""

    resolved_authorization = authorization or StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            ),
            ("user:ben", "tenant:beta"): (
                ScopedResource(tenant_id="tenant:beta", resource_id="document:beta-rollout"),
            ),
        }
    )
    return AccessGatedBm25Retriever(vertical_slice_chunks(), resolved_authorization)
