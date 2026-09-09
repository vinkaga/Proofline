# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministically score retrieval over the pinned HotpotQA distractor contexts."""

from __future__ import annotations

from dataclasses import dataclass

from proofline import ProposedRetrievalStep, ProposedStepError

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import Principal, ScopedResource
from proofline_reference_demo.hotpotqa import HotpotCase, HotpotOverlayReport, OverlayCase
from proofline_reference_demo.retrieval import AccessGatedBm25Retriever, DocumentChunk


@dataclass(frozen=True, slots=True)
class HotpotRetrievalCase:
    """Supporting-evidence retrieval outcome for one unmodified benchmark question."""

    case_id: str
    supporting_title_recall: float
    all_supporting_titles_retrieved: bool


@dataclass(frozen=True, slots=True)
class HotpotEvaluationReport:
    """Measured retrieval utility plus the separate authority-boundary results."""

    case_count: int
    limit: int
    supporting_title_recall_at_k: float
    answer_evidence_coverage_at_k: float
    clean_supporting_coverage: float
    poisoned_rejection_rate: float
    benign_acceptance_rate: float
    cases: tuple[HotpotRetrievalCase, ...]


@dataclass(frozen=True, slots=True)
class HotpotScopeTrace:
    """One actual ACL-overlay retrieval trace, excluding passage contents."""

    case_id: str
    allowed_resource_ids: tuple[str, ...]
    candidate_resource_ids: tuple[str, ...]
    protected_resource_id: str
    poisoned_proposal_rejected: bool


async def evaluate_hotpotqa_scope_overlay(
    cases: tuple[HotpotCase, ...],
    overlays: tuple[OverlayCase, ...],
    *,
    limit: int = 5,
) -> tuple[HotpotScopeTrace, ...]:
    """Run each HotpotQA context set through actual tenant ACL filtering."""

    if len(cases) != len(overlays):
        raise ValueError("HotpotQA cases and overlays must have equal length")
    traces: list[HotpotScopeTrace] = []
    for case, overlay in zip(cases, overlays, strict=True):
        chunks = tuple(
            DocumentChunk(
                id=f"hotpot:{case.case_id}:{index}",
                resource_id=f"hotpot:{case.case_id}:{index}",
                tenant_id=(
                    "tenant:acme"
                    if f"hotpot:{case.case_id}:{index}" in overlay.allowed_resource_ids
                    else "tenant:beta"
                ),
                content=" ".join((title, *sentences)),
                document_id=title,
                source_url="https://hotpotqa.github.io/",
                source_revision="hotpotqa-distractor-dev-v1",
            )
            for index, (title, sentences) in enumerate(case.contexts)
        )
        authorization = StaticAuthorizationAdapter(
            {
                ("user:benchmark", "tenant:acme"): tuple(
                    ScopedResource(tenant_id="tenant:acme", resource_id=resource)
                    for resource in overlay.allowed_resource_ids
                )
            }
        )
        result = await AccessGatedBm25Retriever(chunks, authorization).search_tenant(
            Principal(id="user:benchmark"), "tenant:acme", case.question, limit
        )
        try:
            ProposedRetrievalStep.from_untrusted(overlay.poisoned_proposal)
        except ProposedStepError:
            rejected = True
        else:
            rejected = False
        traces.append(
            HotpotScopeTrace(
                case.case_id,
                overlay.allowed_resource_ids,
                tuple(candidate.resource_id for candidate in result.candidates),
                overlay.protected_resource_id,
                rejected,
            )
        )
    return tuple(traces)


def evaluate_hotpotqa_retrieval(
    cases: tuple[HotpotCase, ...],
    overlay_report: HotpotOverlayReport,
    *,
    limit: int = 5,
) -> HotpotEvaluationReport:
    """Run BM25 over each task's ten official distractor contexts.

    ``answer_evidence_coverage_at_k`` is deliberately not answer accuracy: this
    project has no generator in its benchmark path. It measures whether all
    gold supporting titles needed for an answer were retrieved.
    """

    if limit < 1:
        raise ValueError("limit must be at least one")
    if not cases or overlay_report.case_count != len(cases):
        raise ValueError("HotpotQA retrieval cases and overlay report must agree")
    measurements: list[HotpotRetrievalCase] = []
    for case in cases:
        chunks = tuple(
            DocumentChunk(
                id=f"hotpot:{case.case_id}:{index}",
                resource_id=f"hotpot:{case.case_id}:{index}",
                tenant_id=None,
                content=" ".join((title, *sentences)),
                is_public=True,
                document_id=title,
                source_url="https://hotpotqa.github.io/",
                source_revision=overlay_report.dataset_sha256,
            )
            for index, (title, sentences) in enumerate(case.contexts)
        )
        candidates = AccessGatedBm25Retriever._rank(case.question, chunks, limit)
        retrieved_titles = {candidate.document_id for candidate in candidates}
        found = len(retrieved_titles & case.supporting_titles)
        recall = found / len(case.supporting_titles)
        measurements.append(
            HotpotRetrievalCase(
                case_id=case.case_id,
                supporting_title_recall=recall,
                all_supporting_titles_retrieved=recall == 1,
            )
        )
    return HotpotEvaluationReport(
        case_count=len(cases),
        limit=limit,
        supporting_title_recall_at_k=sum(item.supporting_title_recall for item in measurements)
        / len(measurements),
        answer_evidence_coverage_at_k=sum(
            item.all_supporting_titles_retrieved for item in measurements
        )
        / len(measurements),
        clean_supporting_coverage=overlay_report.clean_supporting_coverage,
        poisoned_rejection_rate=overlay_report.poisoned_rejection_rate,
        benign_acceptance_rate=overlay_report.benign_acceptance_rate,
        cases=tuple(measurements),
    )


def validate_hotpotqa_evaluation(report: HotpotEvaluationReport) -> None:
    """Enforce non-negotiable security properties; utility remains reported, not hidden."""

    if report.clean_supporting_coverage != 1:
        raise ValueError("HotpotQA overlay omitted clean supporting evidence")
    if report.poisoned_rejection_rate != 1:
        raise ValueError("HotpotQA overlay accepted a scope-bearing proposal")
    if report.benign_acceptance_rate != 1:
        raise ValueError("HotpotQA overlay blocked a benign data-only proposal")


def validate_hotpotqa_scope_overlay(traces: tuple[HotpotScopeTrace, ...]) -> None:
    """Block an overlay run that exposes a protected resource or accepts poison."""

    if not traces:
        raise ValueError("HotpotQA scope overlay produced no traces")
    for trace in traces:
        if trace.protected_resource_id in trace.candidate_resource_ids:
            raise ValueError(
                f"HotpotQA scope overlay exposed protected resource in {trace.case_id}"
            )
        if not set(trace.candidate_resource_ids).issubset(trace.allowed_resource_ids):
            raise ValueError(
                f"HotpotQA scope overlay returned out-of-scope evidence in {trace.case_id}"
            )
        if not trace.poisoned_proposal_rejected:
            raise ValueError(f"HotpotQA scope overlay accepted poison in {trace.case_id}")
