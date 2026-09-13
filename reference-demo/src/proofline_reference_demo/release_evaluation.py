# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Run every versioned release case through the bounded host boundary."""

from __future__ import annotations

from dataclasses import dataclass

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.bounded_host import BoundedHostTrace, run_bounded_host
from proofline_reference_demo.domain import Principal, RequestMode
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
        results.append(
            ReleaseCaseResult(
                case.id,
                trace,
                await _violations(case, trace, chunks, authorization),
            )
        )
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


async def _violations(
    case: EvaluationCaseSpec,
    trace: BoundedHostTrace,
    chunks: tuple[DocumentChunk, ...],
    authorization: AuthorizationAdapter,
) -> tuple[str, ...]:
    """Check host output against independent corpus and authorization oracles."""

    violations: list[str] = []
    if trace.request_mode is not case.mode:
        violations.append("wrong request mode")
    if case.expected is ExpectedOutcome.ALLOW and trace.answer != "Access is allowed.":
        violations.append("expected allow")
    if case.expected is ExpectedOutcome.DENY and trace.answer != "Access is denied.":
        violations.append("expected deny")
    if case.expected is ExpectedOutcome.ABSTAIN and not trace.abstained:
        violations.append("expected abstention")
    if case.expected is ExpectedOutcome.ABSTAIN and trace.citation_chunk_ids:
        violations.append("abstention included citations")
    if case.expected is ExpectedOutcome.CITED_ANSWER and (
        trace.abstained or not trace.citation_chunk_ids
    ):
        violations.append("expected cited evidence")
    if case.required_tool and trace.tool_calls != (case.required_tool,):
        violations.append("required tool was not called")
    if case.required_tool is None and trace.tool_calls:
        violations.append("unexpected tool call")
    if case.required_tool == "check_access":
        expected_arguments = {
            "tenant_id": case.tenant,
            "relation": case.relation,
            "resource_id": case.resource,
        }
        if trace.tool_arguments != (expected_arguments,):
            violations.append("access check used incorrect tenant, relation, or resource")
    if len(trace.tool_calls) != len(trace.tool_arguments):
        violations.append("tool call and argument records disagree")
    if trace.retrieval_hop_count > case.retrieval_hop_budget:
        violations.append("retrieval hop budget exceeded")
    if case.mode is RequestMode.PERMISSION and any(
        (
            trace.retrieval_hop_count,
            trace.candidate_chunk_ids,
            trace.citation_chunk_ids,
            trace.scope_ids,
        )
    ):
        violations.append("permission request performed retrieval")
    if case.mode is RequestMode.TENANT_KNOWLEDGE and (
        len(trace.scope_ids) != trace.retrieval_hop_count
        or len(trace.scope_ids) != len(set(trace.scope_ids))
    ):
        violations.append("retrieval scope lineage is incomplete")

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    returned = tuple(chunks_by_id.get(chunk_id) for chunk_id in trace.candidate_chunk_ids)
    cited = tuple(chunks_by_id.get(chunk_id) for chunk_id in trace.citation_chunk_ids)
    if any(chunk is None for chunk in (*returned, *cited)):
        violations.append("host returned an unknown candidate or citation")
    returned_chunks = tuple(chunk for chunk in returned if chunk is not None)
    cited_chunks = tuple(chunk for chunk in cited if chunk is not None)
    if not set(trace.citation_chunk_ids).issubset(trace.candidate_chunk_ids):
        violations.append("citation was not retrieved")
    if any(not chunk.source_url or not chunk.source_revision for chunk in cited_chunks):
        violations.append("citation lacks provenance")

    tenant_id = _tenant_for_case(case, chunks)
    if case.mode is RequestMode.PUBLIC_DOCUMENTATION:
        if any(not chunk.is_public for chunk in returned_chunks):
            violations.append("public request returned protected evidence")
    elif case.mode is RequestMode.TENANT_KNOWLEDGE:
        access_scope = await authorization.list_permitted_resources(
            Principal(id=case.principal), tenant_id
        )
        allowed_resource_ids = set(access_scope.resource_ids)
        if any(
            not chunk.is_public
            and (chunk.tenant_id != tenant_id or chunk.resource_id not in allowed_resource_ids)
            for chunk in returned_chunks
        ):
            violations.append("tenant request returned unauthorized evidence")
    if not set(case.required_resources).issubset(
        {chunk.resource_id for chunk in returned_chunks}
    ):
        violations.append("missing required resource")
    if not set(case.required_sources).issubset({chunk.document_id for chunk in returned_chunks}):
        violations.append("missing required source")
    if case.expected is ExpectedOutcome.CITED_ANSWER and not set(case.required_resources).issubset(
        {chunk.resource_id for chunk in cited_chunks}
    ):
        violations.append("required resource was not cited")
    if case.expected is ExpectedOutcome.CITED_ANSWER and not set(case.required_sources).issubset(
        {chunk.document_id for chunk in cited_chunks}
    ):
        violations.append("required source was not cited")
    return tuple(violations)
