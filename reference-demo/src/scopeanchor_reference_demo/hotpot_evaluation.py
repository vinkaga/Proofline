# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministically score retrieval over the pinned HotpotQA distractor contexts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scopeanchor import (
    ProposedRetrievalStep,
    ProposedStepError,
    RetrievalScope,
    matches_scope_filters,
    scoped,
    validate_scope_filter_fields,
)
from scopeanchor.scope import ScopeFilters

from scopeanchor_reference_demo.authorization import StaticAuthorizationAdapter
from scopeanchor_reference_demo.domain import Principal, RetrievalCandidate, ScopedResource
from scopeanchor_reference_demo.hotpotqa import (
    HotpotCase,
    HotpotOverlayReport,
    HotpotRetrievalQualityGate,
    OverlayCase,
)
from scopeanchor_reference_demo.retrieval import AccessGatedBm25Retriever, DocumentChunk


@dataclass(frozen=True, slots=True)
class HotpotRetrievalCase:
    """Supporting-evidence retrieval outcome for one unmodified benchmark question."""

    case_id: str
    supporting_title_recall: float
    all_supporting_titles_retrieved: bool


@dataclass(frozen=True, slots=True)
class HotpotEvaluationReport:
    """Measured retrieval utility over the unmodified public benchmark."""

    case_count: int
    limit: int
    supporting_title_recall_at_k: float
    answer_evidence_coverage_at_k: float
    cases: tuple[HotpotRetrievalCase, ...]


@dataclass(frozen=True, slots=True)
class HotpotScopeTrace:
    """One three-control overlay trace, excluding source passage contents."""

    case_id: str
    allowed_resource_ids: tuple[str, ...]
    supporting_resource_ids: tuple[str, ...]
    insecure_candidate_resource_ids: tuple[str, ...]
    insecure_benign_follow_up_resource_ids: tuple[str, ...]
    acl_only_candidate_resource_ids: tuple[str, ...]
    acl_only_benign_follow_up_resource_ids: tuple[str, ...]
    candidate_resource_ids: tuple[str, ...]
    benign_follow_up_resource_ids: tuple[str, ...]
    acl_only_follow_up_resource_ids: tuple[str, ...]
    insecure_follow_up_resource_ids: tuple[str, ...]
    protected_resource_id: str
    insecure_benign_proposal_accepted: bool
    acl_only_benign_proposal_accepted: bool
    benign_proposal_accepted: bool
    insecure_scope_input_accepted: bool
    acl_only_scope_input_accepted: bool
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
                    overlay.caller_tenant
                    if f"hotpot:{case.case_id}:{index}" in overlay.allowed_resource_ids
                    else overlay.protected_tenant
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
                ("user:benchmark", overlay.caller_tenant): tuple(
                    ScopedResource(tenant_id=overlay.caller_tenant, resource_id=resource)
                    for resource in overlay.allowed_resource_ids
                )
            }
        )
        caller = Principal(id="user:benchmark")
        acl_retriever = AccessGatedBm25Retriever(chunks, authorization)
        _, insecure_initial = _follow_insecure(chunks, {"query": case.question}, limit)
        insecure_benign_accepted, insecure_benign = _follow_insecure(
            chunks, overlay.benign_proposal, limit
        )
        acl_initial = await acl_retriever.search_tenant(
            caller, overlay.caller_tenant, case.question, limit
        )
        acl_benign_accepted, acl_benign = await _follow_acl_only(
            acl_retriever, caller, overlay.caller_tenant, overlay.benign_proposal, limit
        )
        acl_only_accepted, acl_only_follow_up = await _follow_acl_only(
            acl_retriever, caller, overlay.caller_tenant, overlay.poisoned_proposal, limit
        )
        insecure_accepted, insecure_follow_up = _follow_insecure(
            chunks, overlay.poisoned_proposal, limit
        )

        retriever = scoped(_backend_for_chunks(chunks), resolve_scope=_resolve_scope)
        initial = await retriever.search(
            case.question,
            context=(authorization, caller, overlay.caller_tenant, overlay.policy_version),
            limit=limit,
        )
        benign = await retriever.follow_proposed(
            initial,
            ProposedRetrievalStep.from_untrusted(overlay.benign_proposal),
            limit=limit,
        )
        try:
            poisoned_step = ProposedRetrievalStep.from_untrusted(overlay.poisoned_proposal)
        except ProposedStepError:
            rejected = True
            scoped_poison_attempted = False
        else:
            rejected = False
            scoped_poison_attempted = True
            await retriever.follow_proposed(initial, poisoned_step, limit=limit)
        supporting_resource_ids = tuple(
            f"hotpot:{case.case_id}:{index}"
            for index, (title, _) in enumerate(case.contexts)
            if title in case.supporting_titles
        )
        traces.append(
            HotpotScopeTrace(
                case.case_id,
                overlay.allowed_resource_ids,
                supporting_resource_ids,
                tuple(candidate.resource_id for candidate in insecure_initial),
                tuple(candidate.resource_id for candidate in insecure_benign),
                tuple(candidate.resource_id for candidate in acl_initial.candidates),
                tuple(candidate.resource_id for candidate in acl_benign),
                tuple(candidate.resource_id for candidate in initial.items),
                tuple(candidate.resource_id for candidate in benign.items),
                tuple(candidate.resource_id for candidate in acl_only_follow_up),
                tuple(candidate.resource_id for candidate in insecure_follow_up),
                overlay.protected_resource_id,
                insecure_benign_accepted,
                acl_benign_accepted,
                True,
                insecure_accepted,
                acl_only_accepted,
                rejected,
                scoped_poison_attempted,
                benign.scope.parent_scope_id == initial.scope.scope_id,
            )
        )
    return tuple(traces)


def _follow_insecure(
    chunks: tuple[DocumentChunk, ...], proposal: dict[str, str], limit: int
) -> tuple[bool, tuple[RetrievalCandidate, ...]]:
    """Execute the deliberately unsafe selector used by the baseline control."""

    query = proposal.get("query")
    resource_id = proposal.get("resource_id")
    if not isinstance(query, str):
        return False, ()
    if not isinstance(resource_id, str):
        return True, AccessGatedBm25Retriever._rank(query, chunks, limit)
    selected = tuple(chunk for chunk in chunks if chunk.resource_id == resource_id)[:limit]
    return True, tuple(
        RetrievalCandidate(
            chunk_id=chunk.id,
            resource_id=chunk.resource_id,
            tenant_id=chunk.tenant_id,
            rank=index,
            score=1.0,
            document_id=chunk.document_id,
            source_url=chunk.source_url,
            source_revision=chunk.source_revision,
        )
        for index, chunk in enumerate(selected, start=1)
    )


async def _follow_acl_only(
    retriever: AccessGatedBm25Retriever,
    caller: Principal,
    tenant_id: str,
    proposal: dict[str, str],
    limit: int,
) -> tuple[bool, tuple[RetrievalCandidate, ...]]:
    """Execute a control that accepts selectors but reauthorizes each hop."""

    query = proposal.get("query")
    if not isinstance(query, str):
        return False, ()
    result = await retriever.search_tenant(caller, tenant_id, query, limit)
    return True, result.candidates


async def _resolve_scope(
    context: tuple[StaticAuthorizationAdapter, Principal, str, str],
) -> RetrievalScope:
    """Derive a root scope only from the synthetic authorization adapter."""

    authorization, caller, tenant_id, policy_version = context
    access_scope = await authorization.list_permitted_resources(caller, tenant_id)
    return RetrievalScope.root(
        principal=caller.id,
        filters={"tenant_id": [tenant_id], "resource_id": access_scope.resource_ids},
        max_follow_ups=1,
        policy_version=policy_version,
    )


def _backend_for_chunks(
    chunks: tuple[DocumentChunk, ...],
) -> Callable[..., tuple[RetrievalCandidate, ...]]:
    """Return the host's ordinary filtered-search callable for one case."""

    supported_filter_fields = frozenset({"tenant_id", "resource_id"})

    def backend(query: str, *, filters: ScopeFilters, limit: int) -> tuple[RetrievalCandidate, ...]:
        validate_scope_filter_fields(filters, supported_fields=supported_filter_fields)
        permitted = tuple(
            chunk
            for chunk in chunks
            if matches_scope_filters(
                {"tenant_id": chunk.tenant_id, "resource_id": chunk.resource_id},
                filters,
                supported_fields=supported_filter_fields,
            )
        )
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
                clean_evidence_available_rate=rate(
                    lambda trace: _all_supporting_evidence_returned(
                        trace, trace.insecure_candidate_resource_ids
                    )
                ),
                benign_follow_up_acceptance_rate=rate(
                    lambda trace: trace.insecure_benign_proposal_accepted
                ),
                scope_bearing_input_acceptance_rate=rate(
                    lambda trace: trace.insecure_scope_input_accepted
                ),
                unauthorized_exposure_rate=rate(
                    lambda trace: (
                        not set(trace.insecure_follow_up_resource_ids).issubset(
                            trace.allowed_resource_ids
                        )
                    )
                ),
                rejected_before_retrieval_rate=0.0,
                scope_lineage_complete_rate=0.0,
            ),
            HotpotControlConfiguration(
                name="acl-filtered-per-hop",
                case_count=count,
                clean_evidence_available_rate=rate(
                    lambda trace: _all_supporting_evidence_returned(
                        trace, trace.acl_only_candidate_resource_ids
                    )
                ),
                benign_follow_up_acceptance_rate=rate(
                    lambda trace: trace.acl_only_benign_proposal_accepted
                ),
                scope_bearing_input_acceptance_rate=rate(
                    lambda trace: trace.acl_only_scope_input_accepted
                ),
                unauthorized_exposure_rate=rate(
                    lambda trace: (
                        not set(trace.acl_only_follow_up_resource_ids).issubset(
                            trace.allowed_resource_ids
                        )
                    )
                ),
                rejected_before_retrieval_rate=0.0,
                scope_lineage_complete_rate=0.0,
            ),
            HotpotControlConfiguration(
                name="scoped-plan-policy",
                case_count=count,
                clean_evidence_available_rate=rate(
                    lambda trace: _all_supporting_evidence_returned(
                        trace, trace.candidate_resource_ids
                    )
                ),
                benign_follow_up_acceptance_rate=rate(lambda trace: trace.benign_proposal_accepted),
                scope_bearing_input_acceptance_rate=rate(
                    lambda trace: not trace.poisoned_proposal_rejected
                ),
                unauthorized_exposure_rate=rate(
                    lambda trace: _has_out_of_scope_scoped_evidence(trace)
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
        cases=tuple(measurements),
    )


def validate_hotpotqa_evaluation(
    report: HotpotEvaluationReport,
    quality_gate: HotpotRetrievalQualityGate,
) -> None:
    """Enforce recorded retrieval utility baselines."""

    if report.supporting_title_recall_at_k < quality_gate.supporting_title_recall_at_k:
        raise ValueError(
            "HotpotQA supporting-title recall fell below the recorded release baseline"
        )
    if report.answer_evidence_coverage_at_k < quality_gate.answer_evidence_coverage_at_k:
        raise ValueError(
            "HotpotQA complete supporting-evidence coverage fell below "
            "the recorded release baseline"
        )


def validate_hotpotqa_scope_overlay(traces: tuple[HotpotScopeTrace, ...]) -> None:
    """Block a scoped run that exposes out-of-scope evidence or accepts poison."""

    if not traces:
        raise ValueError("HotpotQA scope overlay produced no traces")
    for trace in traces:
        if _has_out_of_scope_scoped_evidence(trace):
            raise ValueError(
                f"HotpotQA scope overlay returned out-of-scope evidence in {trace.case_id}"
            )
        if not trace.poisoned_proposal_rejected:
            raise ValueError(f"HotpotQA scope overlay accepted poison in {trace.case_id}")


def _has_out_of_scope_scoped_evidence(trace: HotpotScopeTrace) -> bool:
    """Check initial and benign scoped follow-up results against the case ACL."""

    returned_resource_ids = (
        *trace.candidate_resource_ids,
        *trace.benign_follow_up_resource_ids,
    )
    return not set(returned_resource_ids).issubset(trace.allowed_resource_ids)


def _all_supporting_evidence_returned(
    trace: HotpotScopeTrace, returned_resource_ids: tuple[str, ...]
) -> bool:
    """Measure whether the executed initial search returned every gold title."""

    return set(trace.supporting_resource_ids).issubset(returned_resource_ids)


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
