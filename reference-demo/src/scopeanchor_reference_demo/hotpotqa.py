# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Load a verified HotpotQA subset and derive a separate security overlay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class HotpotSource(BaseModel):
    """The immutable, externally published benchmark artifact."""

    name: str
    url: str
    license: str
    citation: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mirrors: tuple[str, ...] = ()


class HotpotSubset(BaseModel):
    """Deterministic subset selection parameters."""

    size: int = Field(gt=0)
    selection: str


class HotpotRetrievalQualityGate(BaseModel):
    """Reviewed lower bounds for this pinned benchmark version and subset."""

    supporting_title_recall_at_k: float = Field(ge=0.0, le=1.0)
    answer_evidence_coverage_at_k: float = Field(ge=0.0, le=1.0)


class HotpotOverlaySpec(BaseModel):
    """Versioned synthetic authorization policy applied after source verification."""

    version: Literal["hotpotqa-overlay-v1"]
    caller_tenant: Literal["tenant:acme"]
    protected_tenant: Literal["tenant:beta"]
    allowed_resource_assignment: Literal["supporting-title-resources"]
    protected_resource_assignment: Literal["first-non-supporting-context-resource"]
    poisoned_proposal: Literal["query-plus-resource_id"]
    benign_proposal: Literal["query-only"]


class HotpotManifest(BaseModel):
    """Versioned contract for the downloaded HotpotQA input."""

    version: str
    source: HotpotSource
    subset: HotpotSubset
    retrieval_quality_gate: HotpotRetrievalQualityGate
    overlay: HotpotOverlaySpec


@dataclass(frozen=True, slots=True)
class HotpotCase:
    """The benchmark fields needed for retrieval and supporting-fact scoring."""

    case_id: str
    question: str
    answer: str
    contexts: tuple[tuple[str, tuple[str, ...]], ...]
    supporting_titles: frozenset[str]


@dataclass(frozen=True, slots=True)
class OverlayCase:
    """Synthetic authorization data, deliberately separate from HotpotQA."""

    case_id: str
    policy_version: str
    caller_tenant: str
    protected_tenant: str
    allowed_resource_ids: tuple[str, ...]
    protected_resource_id: str
    poisoned_proposal: dict[str, str]
    benign_proposal: dict[str, str]


@dataclass(frozen=True, slots=True)
class HotpotOverlayReport:
    """Verified structural identity for a separately measured runtime overlay."""

    dataset_sha256: str
    case_count: int


def load_manifest(path: Path) -> HotpotManifest:
    """Read the benchmark contract without downloading data implicitly."""

    return HotpotManifest.model_validate(yaml.safe_load(path.read_text()))


def load_cases(dataset: Path, manifest: HotpotManifest) -> tuple[HotpotCase, ...]:
    """Verify, parse, and select the documented bridge-question subset."""

    payload = dataset.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != manifest.source.sha256:
        raise ValueError(
            f"HotpotQA SHA-256 mismatch: expected {manifest.source.sha256}, got {actual}"
        )
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise ValueError("HotpotQA dataset must be a JSON array")
    bridge_rows = sorted(
        (row for row in rows if isinstance(row, dict) and row.get("type") == "bridge"),
        key=lambda row: str(row.get("_id", "")),
    )
    selected = tuple(_parse_case(row) for row in bridge_rows[: manifest.subset.size])
    if len(selected) != manifest.subset.size:
        raise ValueError("HotpotQA dataset does not contain the requested bridge subset")
    return selected


def build_overlay(
    cases: tuple[HotpotCase, ...], overlay_spec: HotpotOverlaySpec
) -> tuple[OverlayCase, ...]:
    """Assign synthetic authority without changing source questions or passages."""

    if (
        overlay_spec.allowed_resource_assignment != "supporting-title-resources"
        or overlay_spec.protected_resource_assignment != "first-non-supporting-context-resource"
        or overlay_spec.poisoned_proposal != "query-plus-resource_id"
        or overlay_spec.benign_proposal != "query-only"
    ):
        raise ValueError("unsupported HotpotQA overlay policy")

    overlays: list[OverlayCase] = []
    for case in cases:
        resources = tuple(f"hotpot:{case.case_id}:{index}" for index, _ in enumerate(case.contexts))
        title_to_resource = dict(zip((title for title, _ in case.contexts), resources, strict=True))
        allowed = tuple(title_to_resource[title] for title in sorted(case.supporting_titles))
        protected = next((resource for resource in resources if resource not in allowed), None)
        if protected is None:
            raise ValueError(f"HotpotQA case {case.case_id} has no non-supporting context")
        overlays.append(
            OverlayCase(
                case_id=case.case_id,
                policy_version=overlay_spec.version,
                caller_tenant=overlay_spec.caller_tenant,
                protected_tenant=overlay_spec.protected_tenant,
                allowed_resource_ids=allowed,
                protected_resource_id=protected,
                poisoned_proposal={"query": case.question, "resource_id": protected},
                benign_proposal={"query": case.question},
            )
        )
    return tuple(overlays)


def evaluate_overlay(
    cases: tuple[HotpotCase, ...],
    overlays: tuple[OverlayCase, ...],
    sha256: str,
) -> HotpotOverlayReport:
    """Verify overlay structure; runtime control behavior is measured separately."""

    if len(cases) != len(overlays):
        raise ValueError("HotpotQA cases and authorization overlay must have equal length")
    for case, overlay in zip(cases, overlays, strict=True):
        resource_ids = tuple(
            f"hotpot:{case.case_id}:{index}" for index, _ in enumerate(case.contexts)
        )
        titles = dict(zip((title for title, _ in case.contexts), resource_ids, strict=True))
        expected_allowed = tuple(titles[title] for title in sorted(case.supporting_titles))
        if (
            overlay.case_id != case.case_id
            or overlay.allowed_resource_ids != expected_allowed
            or overlay.policy_version != "hotpotqa-overlay-v1"
            or overlay.caller_tenant != "tenant:acme"
            or overlay.protected_tenant != "tenant:beta"
        ):
            raise ValueError(
                f"HotpotQA overlay does not preserve supporting resources for {case.case_id}"
            )
        if (
            overlay.protected_resource_id not in resource_ids
            or overlay.protected_resource_id in expected_allowed
        ):
            raise ValueError(f"HotpotQA overlay protected a supporting resource for {case.case_id}")
        if overlay.poisoned_proposal != {
            "query": case.question,
            "resource_id": overlay.protected_resource_id,
        }:
            raise ValueError(f"HotpotQA overlay poison target is inconsistent for {case.case_id}")
        if overlay.benign_proposal != {"query": case.question}:
            raise ValueError(f"HotpotQA overlay benign proposal is inconsistent for {case.case_id}")
    return HotpotOverlayReport(
        dataset_sha256=sha256,
        case_count=len(cases),
    )


def _parse_case(row: dict[str, object]) -> HotpotCase:
    """Parse only documented HotpotQA fields and reject malformed fixtures."""

    case_id, question, answer = (row.get(key) for key in ("_id", "question", "answer"))
    contexts = row.get("context")
    supporting = row.get("supporting_facts")
    if not (
        isinstance(case_id, str)
        and case_id
        and isinstance(question, str)
        and question
        and isinstance(answer, str)
        and answer
    ):
        raise ValueError("HotpotQA case is missing an ID, question, or answer")
    if not isinstance(contexts, list) or not isinstance(supporting, list):
        raise ValueError(f"HotpotQA case {case_id} has malformed context or supporting facts")
    parsed_contexts = tuple(
        (item[0], tuple(item[1]))
        for item in contexts
        if isinstance(item, list)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], list)
        and all(isinstance(sentence, str) for sentence in item[1])
    )
    titles = frozenset(
        item[0]
        for item in supporting
        if isinstance(item, list) and item and isinstance(item[0], str)
    )
    if len(parsed_contexts) != len(contexts) or not titles:
        raise ValueError(f"HotpotQA case {case_id} has malformed supporting evidence")
    return HotpotCase(case_id, question, answer, parsed_contexts, titles)
