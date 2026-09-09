# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Load a verified HotpotQA subset and derive a separate security overlay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from proofline import ProposedRetrievalStep, ProposedStepError
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


class HotpotManifest(BaseModel):
    """Versioned contract for the downloaded HotpotQA input."""

    version: str
    source: HotpotSource
    subset: HotpotSubset


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
    allowed_resource_ids: tuple[str, ...]
    protected_resource_id: str
    poisoned_proposal: dict[str, str]
    benign_proposal: dict[str, str]


@dataclass(frozen=True, slots=True)
class HotpotOverlayReport:
    """Deterministic structural checks for the benchmark/security composition."""

    dataset_sha256: str
    case_count: int
    clean_supporting_coverage: float
    poisoned_rejection_rate: float
    benign_acceptance_rate: float


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


def build_overlay(cases: tuple[HotpotCase, ...]) -> tuple[OverlayCase, ...]:
    """Assign synthetic authority without changing source questions or passages."""

    overlays: list[OverlayCase] = []
    for case in cases:
        resources = tuple(f"hotpot:{case.case_id}:{index}" for index, _ in enumerate(case.contexts))
        title_to_resource = dict(zip((title for title, _ in case.contexts), resources, strict=True))
        allowed = tuple(title_to_resource[title] for title in sorted(case.supporting_titles))
        protected = next(resource for resource in resources if resource not in allowed)
        overlays.append(
            OverlayCase(
                case_id=case.case_id,
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
    """Verify clean evidence availability and planner-input boundary behavior."""

    if len(cases) != len(overlays):
        raise ValueError("HotpotQA cases and authorization overlay must have equal length")
    clean = sum(bool(overlay.allowed_resource_ids) for overlay in overlays)
    rejected = 0
    accepted = 0
    for overlay in overlays:
        try:
            ProposedRetrievalStep.from_untrusted(overlay.poisoned_proposal)
        except ProposedStepError:
            rejected += 1
        ProposedRetrievalStep.from_untrusted(overlay.benign_proposal)
        accepted += 1
    count = len(overlays)
    return HotpotOverlayReport(
        dataset_sha256=sha256,
        case_count=count,
        clean_supporting_coverage=clean / count,
        poisoned_rejection_rate=rejected / count,
        benign_acceptance_rate=accepted / count,
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
