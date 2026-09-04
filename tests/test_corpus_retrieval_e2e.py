# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Exercise corpus construction through an ACL-filtered tenant search."""

import pytest

from proofline.authorization import StaticAuthorizationAdapter
from proofline.corpus import AccessAssignments, CorpusManifest, build_corpus
from proofline.domain import Principal, ScopedResource
from proofline.retrieval import AccessGatedBm25Retriever
from proofline.tracing import trace_tenant_retrieval


@pytest.mark.asyncio
async def test_built_assigned_chunks_are_filtered_by_tenant_and_acl(tmp_path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "policy.mdx").write_text("Public policy overview.")
    (tmp_path / "docs" / "rollout.mdx").write_text("Acme deployment runbook details.")
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "unused.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "a" * 40,
                "license": "MIT",
            },
            "documents": [
                {
                    "id": "policy",
                    "path": "docs/policy.mdx",
                    "url": "https://example.test/policy",
                    "visibility": "public",
                },
                {
                    "id": "rollout",
                    "path": "docs/rollout.mdx",
                    "url": "https://example.test/rollout",
                    "visibility": "protected",
                },
            ],
        }
    )
    assignments = AccessAssignments.model_validate(
        {
            "version": "test",
            "assignments": [
                {
                    "source_document": "rollout",
                    "tenant_id": "tenant:acme",
                    "resource_id": "document:rollout",
                }
            ],
        }
    )
    chunks = build_corpus(manifest, tmp_path, assignments)
    retriever = AccessGatedBm25Retriever(
        chunks,
        StaticAuthorizationAdapter(
            {
                ("user:ana", "tenant:acme"): (
                    ScopedResource(tenant_id="tenant:acme", resource_id="document:rollout"),
                )
            }
        ),
    )

    result = await retriever.search_tenant(
        Principal(id="user:ana"), "tenant:acme", "deployment runbook"
    )

    assert [candidate.chunk_id for candidate in result.candidates] == ["chunk:document:rollout:1"]
    assert result.candidates[0].resource_id == "document:rollout"
    trace = trace_tenant_retrieval("tenant-rollout", Principal(id="user:ana"), result)
    assert trace.access_scope is not None
    assert trace.access_scope.resource_ids == ("document:rollout",)
    assert trace.context_chunk_ids == ("chunk:document:rollout:1",)
    assert trace.citations == ()


def test_protected_documents_create_only_assigned_chunks(tmp_path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "secret.mdx").write_text("Private detail.")
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "unused.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "a" * 40,
                "license": "MIT",
            },
            "documents": [
                {
                    "id": "secret",
                    "path": "docs/secret.mdx",
                    "url": "https://example.test/secret",
                    "visibility": "protected",
                }
            ],
        }
    )
    assignments = AccessAssignments.model_validate(
        {
            "version": "test",
            "assignments": [
                {
                    "source_document": "secret",
                    "tenant_id": "tenant:acme",
                    "resource_id": "document:secret",
                }
            ],
        }
    )

    chunks = build_corpus(manifest, tmp_path, assignments)

    assert [chunk.id for chunk in chunks] == ["chunk:document:secret:1"]
