# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Run every versioned release case through the bounded host boundary."""

from __future__ import annotations

from dataclasses import dataclass

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.bounded_host import BoundedHostTrace, run_bounded_host
from proofline_reference_demo.domain import Principal
from proofline_reference_demo.evaluation_data import (
    EvaluationCaseSpec,
    EvaluationSuite,
    ExpectedOutcome,
)
from proofline_reference_demo.retrieval import DocumentChunk


@dataclass(frozen=True, slots=True)
class ReleaseCaseResult:
    """The host trace and any contract violations for one reviewed case."""

    case_id: str
    trace: BoundedHostTrace
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseEvaluationReport:
    """Inspectable result of executing the entire versioned release suite."""

    suite_version: str
    cases: tuple[ReleaseCaseResult, ...]

    @property
    def passed(self) -> bool:
        return not any(case.violations for case in self.cases)


async def evaluate_release_suite(
    authorization: AuthorizationAdapter,
    chunks: tuple[DocumentChunk, ...],
    suite: EvaluationSuite,
) -> ReleaseEvaluationReport:
    """Execute every case through public, tenant, or permission host paths."""

    results: list[ReleaseCaseResult] = []
    for case in suite.cases:
        trace = await run_bounded_host(
            authorization,
            principal=Principal(id=case.principal),
            tenant_id=_tenant_for_case(case, chunks),
            query=case.query,
            relation=case.relation or "viewer",
            resource_id=case.resource,
            chunks=chunks,
            request_mode=case.mode,
        )
        results.append(ReleaseCaseResult(case.id, trace, _violations(case, trace, chunks)))
    return ReleaseEvaluationReport(suite.version, tuple(results))


def validate_release_evaluation(report: ReleaseEvaluationReport) -> None:
    """Raise a concise error when any executable release contract failed."""

    failures = [
        f"{case.case_id}: {', '.join(case.violations)}"
        for case in report.cases
        if case.violations
    ]
    if failures:
        raise ValueError("release evaluation failed: " + "; ".join(failures))


def _tenant_for_case(case: EvaluationCaseSpec, chunks: tuple[DocumentChunk, ...]) -> str:
    if case.tenant is not None:
        return case.tenant
    if case.resource is not None:
        for chunk in chunks:
            if chunk.resource_id == case.resource and chunk.tenant_id is not None:
                return chunk.tenant_id
    return "tenant:acme"


def _violations(
    case: EvaluationCaseSpec, trace: BoundedHostTrace, chunks: tuple[DocumentChunk, ...]
) -> tuple[str, ...]:
    violations: list[str] = []
    if trace.request_mode is not case.mode:
        violations.append("wrong request mode")
    if case.expected is ExpectedOutcome.ALLOW and trace.answer != "Access is allowed.":
        violations.append("expected allow")
    if case.expected is ExpectedOutcome.DENY and trace.answer != "Access is denied.":
        violations.append("expected deny")
    if case.expected is ExpectedOutcome.ABSTAIN and not trace.abstained:
        violations.append("expected abstention")
    if case.expected is ExpectedOutcome.CITED_ANSWER and (
        trace.abstained or not trace.citation_chunk_ids
    ):
        violations.append("expected cited evidence")
    if case.required_tool and trace.tool_calls != (case.required_tool,):
        violations.append("required tool was not called")
    if case.required_tool == "check_access":
        expected_arguments = {
            "tenant_id": case.tenant,
            "relation": case.relation,
            "resource_id": case.resource,
        }
        if expected_arguments not in trace.tool_arguments:
            violations.append("access check used incorrect tenant, relation, or resource")
    returned = {chunk.id: chunk for chunk in chunks if chunk.id in trace.candidate_chunk_ids}
    tenant_id = _tenant_for_case(case, chunks)
    if any(
        chunk.tenant_id is not None and chunk.tenant_id != tenant_id
        for chunk in returned.values()
    ):
        violations.append("cross-tenant candidate")
    if not set(case.required_resources).issubset(
        {chunk.resource_id for chunk in returned.values()}
    ):
        violations.append("missing required resource")
    if not set(case.required_sources).issubset({chunk.document_id for chunk in returned.values()}):
        violations.append("missing required source")
    return tuple(violations)
