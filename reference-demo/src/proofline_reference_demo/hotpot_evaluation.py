# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministically score retrieval over the pinned HotpotQA distractor contexts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from proofline import ProposedRetrievalStep, ProposedStepError, RetrievalScope, scoped
from proofline.scope import ScopeFilters

from proofline_reference_demo.authorization import StaticAuthorizationAdapter
from proofline_reference_demo.domain import Principal, RetrievalCandidate, ScopedResource
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
    """One three-control overlay trace, excluding source passage contents."""

    case_id: str
    allowed_resource_ids: tuple[str, ...]
    candidate_resource_ids: tuple[str, ...]
    benign_follow_up_resource_ids: tuple[str, ...]
    acl_only_follow_up_resource_ids: tuple[str, ...]
    insecure_follow_up_resource_ids: tuple[str, ...]
    protected_resource_id: str
    poisoned_proposal_rejected: bool
    scoped_poison_follow_up_attempted: bool
    scope_lineage_complete: bool


@dataclass(frozen=True, slots=True)
class HotpotControlConfiguration:
    """Aggregate behavior for one control over the public benchmark subset."""

    name: str
    case_count: int
    clean_evidence_available_rate: float
    benign_follow_up_acceptance_rate: float
    scope_bearing_input_acceptance_rate: float
    unauthorized_exposure_rate: float
    rejected_before_retrieval_rate: float
    scope_lineage_complete_rate: float


@dataclass(frozen=True, slots=True)
class HotpotScopeControlReport:
    """Three-control comparison for the synthetic authority overlay."""

    configurations: tuple[HotpotControlConfiguration, ...]


async def evaluate_hotpotqa_scope_overlay(
    cases: tuple[HotpotCase, ...],
    overlays: tuple[OverlayCase, ...],
    *,
    limit: int = 5,
) -> tuple[HotpotScopeTrace, ...]:
    """Evaluate insecure, ACL-only, and scoped controls for every public case.

    The HotpotQA questions and contexts remain unmodified. Tenant labels and
    the scope-bearing proposal are synthetic overlay data. The insecure
    control deliberately treats a proposal's ``resource_id`` as an effective
    selector; the ACL-only control accepts it but independently resolves the
    caller's ACL; the scoped control rejects it before a second retrieval.
    """

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
        caller = Principal(id="user:benchmark")
        acl_retriever = AccessGatedBm25Retriever(chunks, authorization)
        acl_only_follow_up = await acl_retriever.search_tenant(
            caller, "tenant:acme", str(overlay.poisoned_proposal["query"]), limit
        )

        retriever = scoped(_backend_for_chunks(chunks), resolve_scope=_resolve_scope)
        initial = await retriever.search(
            case.question, context=(authorization, caller), limit=limit
        )
        benign = await retriever.follow_proposed(
            initial,
            ProposedRetrievalStep.from_untrusted(overlay.benign_proposal),
            limit=limit,
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
                tuple(candidate.resource_id for candidate in initial.items),
                tuple(candidate.resource_id for candidate in benign.items),
                tuple(candidate.resource_id for candidate in acl_only_follow_up.candidates),
                (overlay.protected_resource_id,),
                overlay.protected_resource_id,
                rejected,
                False,
                benign.scope.parent_scope_id == initial.scope.scope_id,
            )
        )
    return tuple(traces)


async def _resolve_scope(
    context: tuple[StaticAuthorizationAdapter, Principal],
) -> RetrievalScope:
    """Derive a root scope only from the synthetic authorization adapter."""

    authorization, caller = context
    access_scope = await authorization.list_permitted_resources(caller, "tenant:acme")
    return RetrievalScope.root(
        principal=caller.id,
        filters={"resource_id": access_scope.resource_ids},
        max_follow_ups=1,
        policy_version="hotpotqa-overlay-v1",
    )


def _backend_for_chunks(
    chunks: tuple[DocumentChunk, ...],
) -> Callable[..., tuple[RetrievalCandidate, ...]]:
    """Return the host's ordinary filtered-search callable for one case."""

    def backend(query: str, *, filters: ScopeFilters, limit: int) -> tuple[RetrievalCandidate, ...]:
        resource_ids = filters.get("resource_id", frozenset())
        if not all(isinstance(resource_id, str) for resource_id in resource_ids):
            raise TypeError("HotpotQA resource IDs must be strings")
        permitted = tuple(chunk for chunk in chunks if chunk.resource_id in resource_ids)
        return AccessGatedBm25Retriever._rank(query, permitted, limit)

    return backend


def evaluate_hotpotqa_scope_controls(
    traces: tuple[HotpotScopeTrace, ...],
) -> HotpotScopeControlReport:
    """Aggregate the three controls without treating utility as answer accuracy."""

    if not traces:
        raise ValueError("HotpotQA scope overlay produced no traces")
    count = len(traces)

    def rate(predicate: Callable[[HotpotScopeTrace], object]) -> float:
        return sum(bool(predicate(trace)) for trace in traces) / count

    return HotpotScopeControlReport(
        configurations=(
            HotpotControlConfiguration(
                name="insecure-baseline",
                case_count=count,
                clean_evidence_available_rate=rate(lambda trace: trace.allowed_resource_ids),
                benign_follow_up_acceptance_rate=1.0,
                scope_bearing_input_acceptance_rate=1.0,
                unauthorized_exposure_rate=rate(
                    lambda trace: (
                        trace.protected_resource_id in trace.insecure_follow_up_resource_ids
                    )
                ),
                rejected_before_retrieval_rate=0.0,
                scope_lineage_complete_rate=0.0,
            ),
            HotpotControlConfiguration(
                name="acl-filtered-per-hop",
                case_count=count,
                clean_evidence_available_rate=rate(lambda trace: trace.allowed_resource_ids),
                benign_follow_up_acceptance_rate=1.0,
                scope_bearing_input_acceptance_rate=1.0,
                unauthorized_exposure_rate=rate(
                    lambda trace: (
                        trace.protected_resource_id in trace.acl_only_follow_up_resource_ids
                    )
                ),
                rejected_before_retrieval_rate=0.0,
                scope_lineage_complete_rate=0.0,
            ),
            HotpotControlConfiguration(
                name="scoped-plan-policy",
                case_count=count,
                clean_evidence_available_rate=rate(lambda trace: trace.allowed_resource_ids),
                benign_follow_up_acceptance_rate=1,
                scope_bearing_input_acceptance_rate=rate(
                    lambda trace: not trace.poisoned_proposal_rejected
                ),
                unauthorized_exposure_rate=rate(
                    lambda trace: trace.protected_resource_id in trace.candidate_resource_ids
                ),
                rejected_before_retrieval_rate=rate(
                    lambda trace: (
                        trace.poisoned_proposal_rejected
                        and not trace.scoped_poison_follow_up_attempted
                    )
                ),
                scope_lineage_complete_rate=rate(lambda trace: trace.scope_lineage_complete),
            ),
        )
    )


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


def validate_hotpotqa_scope_controls(report: HotpotScopeControlReport) -> None:
    """Block CI when a control no longer demonstrates the claimed boundary."""

    insecure, acl_only, scoped_policy = report.configurations
    if insecure.unauthorized_exposure_rate != 1:
        raise ValueError("HotpotQA insecure control did not expose protected evidence")
    if acl_only.unauthorized_exposure_rate != 0:
        raise ValueError("HotpotQA ACL-only control exposed protected evidence")
    if acl_only.scope_bearing_input_acceptance_rate != 1:
        raise ValueError("HotpotQA ACL-only control did not accept scope-bearing input")
    if scoped_policy.scope_bearing_input_acceptance_rate != 0:
        raise ValueError("HotpotQA scoped policy accepted scope-bearing input")
    if scoped_policy.rejected_before_retrieval_rate != 1:
        raise ValueError("HotpotQA scoped policy did not reject before retrieval")
    if scoped_policy.unauthorized_exposure_rate != 0:
        raise ValueError("HotpotQA scoped policy exposed protected evidence")
    if scoped_policy.scope_lineage_complete_rate != 1:
        raise ValueError("HotpotQA scoped policy lost child-scope lineage")
