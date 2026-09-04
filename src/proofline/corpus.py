# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Build a reviewable chunk corpus from the version-pinned source manifest.

The manifest records where source material came from. This module turns a local
checkout of that exact revision into deterministic, provenance-carrying chunks
without copying a changing external website into application code.
"""

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl

from proofline.retrieval import DocumentChunk

_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_PARAGRAPH = re.compile(r"\n\s*\n")


class ManifestSource(BaseModel):
    """The repository identity and revision from which ingestion is allowed."""

    repository: HttpUrl
    revision: str = Field(min_length=40, max_length=40)
    license: str


class ManifestDocument(BaseModel):
    """One public source document selected for the initial corpus."""

    id: str
    path: Path
    url: HttpUrl
    visibility: Literal["public", "protected"]


class CorpusManifest(BaseModel):
    """The complete, versioned declaration of an ingestible source corpus."""

    version: str
    retrieved_at: date
    access_assignments: Path
    source: ManifestSource
    documents: tuple[ManifestDocument, ...]


class ResourceAssignment(BaseModel):
    """One synthetic protected resource backed by a selected real source document."""

    source_document: str
    tenant_id: str
    resource_id: str


class AccessAssignments(BaseModel):
    """Versioned synthetic resource mapping used by access-controlled retrieval."""

    version: str
    assignments: tuple[ResourceAssignment, ...]


def load_manifest(path: Path) -> CorpusManifest:
    """Parse and validate the corpus manifest before any source file is read."""

    manifest = CorpusManifest.model_validate(yaml.safe_load(path.read_text()))
    if manifest.access_assignments.is_absolute():
        return manifest
    return manifest.model_copy(
        update={"access_assignments": path.parent / manifest.access_assignments}
    )


def load_access_assignments(path: Path) -> AccessAssignments:
    """Load the synthetic resource mapping that augments public source chunks."""

    return AccessAssignments.model_validate(yaml.safe_load(path.read_text()))


def validate_corpus_configuration(manifest: CorpusManifest, assignments: AccessAssignments) -> None:
    """Reject ambiguous documents and protected sources without an ACL mapping."""

    document_ids = [document.id for document in manifest.documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("manifest document IDs must be unique")
    assigned_documents = {assignment.source_document for assignment in assignments.assignments}
    unknown_documents = assigned_documents - set(document_ids)
    if unknown_documents:
        raise ValueError(f"assignments reference unknown documents: {sorted(unknown_documents)}")
    protected_without_assignment = [
        document.id
        for document in manifest.documents
        if document.visibility == "protected" and document.id not in assigned_documents
    ]
    if protected_without_assignment:
        raise ValueError(
            "protected documents require an access assignment: "
            f"{sorted(protected_without_assignment)}"
        )


def build_corpus(
    manifest: CorpusManifest,
    source_root: Path,
    assignments: AccessAssignments,
) -> tuple[DocumentChunk, ...]:
    """Create deterministic public chunks with source revision and URL provenance."""

    validate_corpus_configuration(manifest, assignments)
    chunks: list[DocumentChunk] = []
    for document in manifest.documents:
        source_path = source_root / document.path
        content = _FRONT_MATTER.sub("", source_path.read_text()).strip()
        for index, paragraph in enumerate(_PARAGRAPH.split(content), start=1):
            normalized = " ".join(paragraph.split())
            if normalized:
                if document.visibility == "public":
                    chunks.append(
                        DocumentChunk(
                            id=f"chunk:{document.id}:{index}",
                            resource_id=f"document:{document.id}",
                            tenant_id=None,
                            content=normalized,
                            is_public=True,
                            source_revision=manifest.source.revision,
                            source_url=str(document.url),
                        )
                    )
                for assignment in assignments.assignments:
                    if assignment.source_document == document.id:
                        chunks.append(
                            DocumentChunk(
                                id=f"chunk:{assignment.resource_id}:{index}",
                                resource_id=assignment.resource_id,
                                tenant_id=assignment.tenant_id,
                                content=normalized,
                                source_revision=manifest.source.revision,
                                source_url=str(document.url),
                            )
                        )
    return tuple(chunks)


def write_corpus(chunks: tuple[DocumentChunk, ...], output: Path) -> None:
    """Write generated corpus artifacts outside version control as JSON Lines."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{json.dumps(asdict(chunk), sort_keys=True)}\n" for chunk in chunks))
