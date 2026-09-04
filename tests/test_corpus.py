# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify deterministic corpus construction and provenance preservation."""

import json

import pytest
from pydantic import ValidationError

from proofline.corpus import (
    AccessAssignments,
    CorpusManifest,
    build_corpus,
    validate_corpus_configuration,
    write_corpus,
)


def test_build_corpus_strips_front_matter_and_preserves_provenance(tmp_path) -> None:
    source = tmp_path / "docs" / "content"
    source.mkdir(parents=True)
    (source / "example.mdx").write_text(
        "---\ntitle: Example\n---\n\nFirst paragraph.\n\nSecond paragraph."
    )
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "data/access/resource-assignments.yaml",
            "source": {
                "repository": "https://github.com/openfga/openfga.dev.git",
                "revision": "a" * 40,
                "license": "Apache-2.0",
            },
            "documents": [
                {
                    "id": "example",
                    "path": "docs/content/example.mdx",
                    "url": "https://openfga.dev/docs/example",
                    "visibility": "public",
                }
            ],
        }
    )

    chunks = build_corpus(manifest, tmp_path, AccessAssignments(version="test", assignments=()))

    assert [chunk.id for chunk in chunks] == ["chunk:example:1", "chunk:example:2"]
    assert chunks[0].content == "First paragraph."
    assert chunks[0].source_revision == "a" * 40


def test_write_corpus_emits_json_lines(tmp_path) -> None:
    output = tmp_path / "corpus.jsonl"
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "data/access/resource-assignments.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "b" * 40,
                "license": "MIT",
            },
            "documents": [],
        }
    )

    write_corpus(
        build_corpus(manifest, tmp_path, AccessAssignments(version="test", assignments=())),
        output,
    )

    assert list(map(json.loads, output.read_text().splitlines())) == []


def test_manifest_rejects_unknown_visibility() -> None:
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(
            {
                "version": "test-v0",
                "retrieved_at": "2026-09-03",
                "access_assignments": "data/access/resource-assignments.yaml",
                "source": {
                    "repository": "https://example.test/repo",
                    "revision": "a" * 40,
                    "license": "MIT",
                },
                "documents": [
                    {
                        "id": "example",
                        "path": "example.mdx",
                        "url": "https://example.test/example",
                        "visibility": "internal",
                    }
                ],
            }
        )


def test_protected_document_requires_assignment(tmp_path) -> None:
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "data/access/resource-assignments.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "a" * 40,
                "license": "MIT",
            },
            "documents": [
                {
                    "id": "secret",
                    "path": "secret.mdx",
                    "url": "https://example.test/secret",
                    "visibility": "protected",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="protected documents require"):
        build_corpus(manifest, tmp_path, AccessAssignments(version="test", assignments=()))


def test_assignments_must_reference_manifest_documents() -> None:
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "data/access/resource-assignments.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "a" * 40,
                "license": "MIT",
            },
            "documents": [],
        }
    )
    assignments = AccessAssignments.model_validate(
        {
            "version": "test",
            "assignments": [
                {
                    "source_document": "missing",
                    "tenant_id": "tenant:acme",
                    "resource_id": "document:secret",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="unknown documents"):
        validate_corpus_configuration(manifest, assignments)


def test_manifest_document_ids_must_be_unique() -> None:
    manifest = CorpusManifest.model_validate(
        {
            "version": "test-v0",
            "retrieved_at": "2026-09-03",
            "access_assignments": "data/access/resource-assignments.yaml",
            "source": {
                "repository": "https://example.test/repo",
                "revision": "a" * 40,
                "license": "MIT",
            },
            "documents": [
                {
                    "id": "duplicate",
                    "path": "first.mdx",
                    "url": "https://example.test/first",
                    "visibility": "public",
                },
                {
                    "id": "duplicate",
                    "path": "second.mdx",
                    "url": "https://example.test/second",
                    "visibility": "public",
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="document IDs must be unique"):
        validate_corpus_configuration(manifest, AccessAssignments(version="test", assignments=()))


def test_manifest_resolves_assignments_relative_to_its_own_directory(tmp_path) -> None:
    manifest_path = tmp_path / "corpus" / "manifest.yaml"
    manifest_path.parent.mkdir()
    assignments = manifest_path.parent / "access.yaml"
    assignments.write_text("version: test\nassignments: []\n")
    manifest_path.write_text(
        "\n".join(
            [
                "version: test-v0",
                "retrieved_at: 2026-09-03",
                "access_assignments: access.yaml",
                "source:",
                "  repository: https://example.test/repo",
                f"  revision: {'a' * 40}",
                "  license: MIT",
                "documents: []",
            ]
        )
    )

    from proofline.corpus import load_manifest

    assert load_manifest(manifest_path).access_assignments == assignments
