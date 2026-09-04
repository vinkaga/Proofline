# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Measure the access-filtered BM25 baseline against reviewed retrieval cases.

The evaluator intentionally measures only evidence retrieval.  Permission
cases are excluded: their correctness belongs to the authorization phase, not
to a ranking metric.  Every search result is also checked independently against
its resolved scope, so an implementation bug cannot be hidden by good ranking.
"""

import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from proofline.domain import InteractionTrace, Principal, RequestMode, RetrievalCandidate
from proofline.evaluation_data import EvaluationCaseSpec, EvaluationSuite, ExpectedOutcome
from proofline.retrieval import AccessGatedRetriever, RetrievalResult


@dataclass(frozen=True, slots=True)
class CaseMeasurement:
    """One retrieval attempt and the evidence labels used to score it."""

    case_id: str
    tags: tuple[str, ...]
    elapsed_ms: float
    candidate_ids: tuple[str, ...]
    relevant_identifiers: frozenset[str]
    ranked_identifiers: tuple[tuple[int, str], ...]
    access_violation: bool
    provenance_violation: bool
    trace: InteractionTrace


@dataclass(frozen=True, slots=True)
class LexicalBaselineMeasurement:
    """Aggregate metrics and trace-safe per-case measurements."""

    corpus_version: str
    suite_version: str
    limit: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    unauthorized_exposure_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    suite_case_count: int
    permission_case_count: int
    cases: tuple[CaseMeasurement, ...]

    @property
    def retrieval_case_count(self) -> int:
        return sum(bool(case.relevant_identifiers) for case in self.cases)

    @property
    def provenance_violation_rate(self) -> float:
        return sum(case.provenance_violation for case in self.cases) / len(self.cases)


async def evaluate_lexical_baseline(
    retriever: AccessGatedRetriever,
    suite: EvaluationSuite,
    corpus_version: str,
    *,
    limit: int = 5,
) -> LexicalBaselineMeasurement:
    """Run reviewed public and tenant retrieval cases with a fixed candidate limit."""

    if limit < 1:
        raise ValueError("limit must be at least one")
    measurements: list[CaseMeasurement] = []
    for case in suite.cases:
        if case.mode is RequestMode.PERMISSION:
            continue
        started = time.perf_counter()
        result = await _search_case(retriever, case, limit)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        relevant = _relevant_identifiers(case)
        ranked = _ranked_identifiers(result.candidates)
        principal = Principal(id=case.principal)
        trace = InteractionTrace(
            case_id=case.id,
            request_mode=case.mode,
            principal=principal,
            access_scope=result.access_scope,
            candidates=result.candidates,
            context_chunk_ids=tuple(candidate.chunk_id for candidate in result.candidates),
        )
        measurements.append(
            CaseMeasurement(
                case_id=case.id,
                tags=case.tags,
                elapsed_ms=elapsed_ms,
                candidate_ids=tuple(candidate.chunk_id for candidate in result.candidates),
                relevant_identifiers=relevant,
                ranked_identifiers=ranked,
                access_violation=_has_access_violation(case, result),
                provenance_violation=_has_provenance_violation(result),
                trace=trace,
            )
        )
    return _aggregate(
        corpus_version,
        suite.version,
        len(suite.cases),
        sum(case.mode is RequestMode.PERMISSION for case in suite.cases),
        limit,
        tuple(measurements),
    )


async def _search_case(
    retriever: AccessGatedRetriever, case: EvaluationCaseSpec, limit: int
) -> RetrievalResult:
    principal = Principal(id=case.principal)
    if case.mode is RequestMode.PUBLIC_DOCUMENTATION:
        return await retriever.search_public(case.query, limit)
    if case.tenant is None:  # Defensive: EvaluationCaseSpec validates this invariant.
        raise ValueError(f"tenant case {case.id} has no tenant")
    return await retriever.search_tenant(principal, case.tenant, case.query, limit)


def _relevant_identifiers(case: EvaluationCaseSpec) -> frozenset[str]:
    if case.expected is not ExpectedOutcome.CITED_ANSWER:
        return frozenset()
    return frozenset(
        [*(f"source:{source}" for source in case.required_sources),
         *(f"resource:{resource}" for resource in case.required_resources)]
    )


def _candidate_identifiers(candidate: RetrievalCandidate) -> str:
    """Choose the reviewed relevance label represented by a candidate."""

    # Protected cases judge the assigned synthetic resource. Public cases judge
    # the source document, which remains stable even if a source URL changes.
    if candidate.tenant_id:
        return f"resource:{candidate.resource_id}"
    return f"source:{candidate.document_id}"


def _ranked_identifiers(candidates: Iterable[RetrievalCandidate]) -> tuple[tuple[int, str], ...]:
    """Keep a source's first *actual* rank without compressing later ranks."""

    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for candidate in candidates:
        identifier = _candidate_identifiers(candidate)
        if identifier not in seen:
            seen.add(identifier)
            unique.append((candidate.rank, identifier))
    return tuple(unique)


def _has_access_violation(case: EvaluationCaseSpec, result: RetrievalResult) -> bool:
    if case.mode is RequestMode.PUBLIC_DOCUMENTATION:
        return any(candidate.tenant_id is not None for candidate in result.candidates)
    if result.access_scope is None or case.tenant is None:
        return True
    allowed = set(result.access_scope.resource_ids)
    return any(
        candidate.tenant_id is not None
        and (candidate.tenant_id != case.tenant or candidate.resource_id not in allowed)
        for candidate in result.candidates
    )


def _has_provenance_violation(result: RetrievalResult) -> bool:
    """Reject evidence that cannot become a checkable citation in a later phase."""

    return any(
        not candidate.document_id or not candidate.source_url or not candidate.source_revision
        for candidate in result.candidates
    )


def _aggregate(
    corpus_version: str,
    suite_version: str,
    suite_case_count: int,
    permission_case_count: int,
    limit: int,
    cases: tuple[CaseMeasurement, ...],
) -> LexicalBaselineMeasurement:
    scored = tuple(case for case in cases if case.relevant_identifiers)
    if not scored:
        raise ValueError("evaluation suite has no retrieval cases with reviewed relevance")
    recalls = [_recall(case) for case in scored]
    reciprocal_ranks = [_reciprocal_rank(case) for case in scored]
    ndcgs = [_ndcg(case, limit) for case in scored]
    latencies = sorted(case.elapsed_ms for case in cases)
    return LexicalBaselineMeasurement(
        corpus_version=corpus_version,
        suite_version=suite_version,
        limit=limit,
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        unauthorized_exposure_rate=sum(case.access_violation for case in cases) / len(cases),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        suite_case_count=suite_case_count,
        permission_case_count=permission_case_count,
        cases=cases,
    )


def _recall(case: CaseMeasurement) -> float:
    returned_identifiers = {identifier for _, identifier in case.ranked_identifiers}
    returned_relevant = case.relevant_identifiers & returned_identifiers
    return len(returned_relevant) / len(case.relevant_identifiers)


def _reciprocal_rank(case: CaseMeasurement) -> float:
    for rank, identifier in case.ranked_identifiers:
        if identifier in case.relevant_identifiers:
            return 1 / rank
    return 0.0


def _ndcg(case: CaseMeasurement, limit: int) -> float:
    gains = [
        1 / _log2(rank + 1)
        for rank, identifier in case.ranked_identifiers
        if identifier in case.relevant_identifiers
    ]
    ideal = sum(
        1 / _log2(rank + 1)
        for rank in range(1, min(len(case.relevant_identifiers), limit) + 1)
    )
    return sum(gains) / ideal if ideal else 0.0


def _log2(value: int) -> float:
    # Avoid importing a second numeric dependency for three conventional metrics.
    import math

    return math.log2(value)


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(quantile * 100) - 1]


def write_lexical_report(measurement: LexicalBaselineMeasurement, output: Path) -> None:
    """Write a human-reviewable report; generated artifacts remain untracked."""

    failures = [
        case.case_id
        for case in measurement.cases
        if case.relevant_identifiers and _recall(case) < 1
    ]
    lines = [
        "# Lexical baseline",
        "",
        (
            f"Corpus: `{measurement.corpus_version}` · "
            f"suite: `{measurement.suite_version}` · k={measurement.limit}"
        ),
        "",
        (
            "| Recall@k | MRR | nDCG@k | Unauthorized exposure | Provenance failures | "
            "p50 latency | p95 latency |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {measurement.recall_at_k:.3f} | {measurement.mrr:.3f} | "
            f"{measurement.ndcg_at_k:.3f} | {measurement.unauthorized_exposure_rate:.3f} | "
            f"{measurement.provenance_violation_rate:.3f} | "
            f"{measurement.p50_latency_ms:.2f} ms | {measurement.p95_latency_ms:.2f} ms |"
        ),
        "",
        (
            f"Executed retrieval for {len(measurement.cases)} of "
            f"{measurement.suite_case_count} suite cases; scored "
            f"{measurement.retrieval_case_count} with reviewed relevance. The remaining "
            f"{measurement.permission_case_count} permission cases belong to Phase 2."
        ),
        "",
        "## Retrieval failures requiring review",
        "",
    ]
    lines.extend(f"- `{case_id}`" for case_id in failures[:10])
    if not failures:
        lines.append("- None")
    lines.extend([
        "",
        "## Access-boundary checks",
        "",
        (
            "All public results must be public. Tenant results may contain public chunks plus only "
            "resources in the resolved tenant scope. Any violation is included in the exposure "
            "metric above."
        ),
        "",
    ])
    reviewed_probes = tuple(
        case for case in measurement.cases if "access_isolation" in case.tags
    )
    lines.extend(["## Access-isolation failure-mode analysis", ""])
    lines.extend(
        (
            f"- `{case.case_id}` — returned `{', '.join(case.candidate_ids) or 'no chunks'}`; "
            f"scope violation: `{case.access_violation}`."
        )
        for case in reviewed_probes
    )
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def write_lexical_traces(measurement: LexicalBaselineMeasurement, output: Path) -> None:
    """Persist one structured, inspectable retrieval trace per executed case."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{case.trace.model_dump_json()}\n" for case in measurement.cases))


def validate_baseline_measurement(measurement: LexicalBaselineMeasurement) -> None:
    """Fail a baseline run that did not preserve the Phase 3 safety contracts."""

    if measurement.unauthorized_exposure_rate:
        raise ValueError("lexical evaluation exposed an unauthorized chunk")
    if measurement.provenance_violation_rate:
        raise ValueError("lexical evaluation returned a chunk without citation provenance")
