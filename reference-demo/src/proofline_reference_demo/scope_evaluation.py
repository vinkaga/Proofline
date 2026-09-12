# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Release-gate evaluation for Proofline's narrow scope-propagation claim.

The intentionally insecure and ACL-per-hop controls are local evaluation code,
not library APIs. They establish that the fixture remains meaningful: ACLs can
prevent exposure after an unsafe plan is accepted, whereas Proofline rejects a
scope-bearing planner input before a second retrieval is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal
from proofline_reference_demo.multi_hop import (
    MultiHopTrace,
    poisoned_proposal_from_initial,
    run_benign_two_hop,
    run_clean_two_hop,
    run_poisoned_two_hop,
)
from proofline_reference_demo.retrieval import (
    AccessGatedBm25Retriever,
    is_permitted_tenant_chunk,
)
from proofline_reference_demo.vertical_slice import vertical_slice_chunks


@dataclass(frozen=True, slots=True)
class ScopeGateConfiguration:
    """One comparable behavior in the deterministic scope-propagation fixture."""

    name: str
    clean_task_succeeds: bool
    benign_follow_up_succeeds: bool
    planner_accepted_scope_input: bool
    unauthorized_exposure: bool
    rejected_scope_fields: tuple[str, ...]
    retrieval_hop_count: int
    scope_lineage_complete: bool


@dataclass(frozen=True, slots=True)
class GateFailure:
    """One release-gate failure with stable references to relevant traces."""

    layer: str
    message: str
    trace_scenarios: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScopeGateReport:
    """A CI-friendly result for the protected scope-propagation cases."""

    version: str
    configurations: tuple[ScopeGateConfiguration, ...]
    passed: bool
    failures: tuple[GateFailure, ...]
    traces: tuple[MultiHopTrace, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable report; random runtime scope IDs are normalized."""

        return {
            "version": self.version,
            "configurations": [asdict(configuration) for configuration in self.configurations],
            "passed": self.passed,
            "failures": [asdict(failure) for failure in self.failures],
            "traces": [_normalized_trace(trace) for trace in self.traces],
        }


class ScopeGateError(RuntimeError):
    """Raised when a protected scope-propagation property regresses."""


async def evaluate_scope_propagation(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal | None = None,
    tenant_id: str = "tenant:acme",
) -> ScopeGateReport:
    """Evaluate clean, benign, and poisoned counterparts across three controls."""

    caller = principal or Principal(id="user:ana")
    clean = await run_clean_two_hop(authorization, principal=caller, tenant_id=tenant_id)
    benign = await run_benign_two_hop(authorization, principal=caller, tenant_id=tenant_id)
    poisoned = await run_poisoned_two_hop(authorization, principal=caller, tenant_id=tenant_id)
    traces = (clean, benign, poisoned)
    unauthorized_exposure_scenarios = await _unauthorized_exposure_scenarios(
        traces,
        authorization=authorization,
        principal=caller,
        tenant_id=tenant_id,
    )
    insecure = await _evaluate_insecure_baseline(
        authorization, principal=caller, tenant_id=tenant_id
    )
    acl_only = await _evaluate_acl_only(authorization, principal=caller, tenant_id=tenant_id)
    scoped = ScopeGateConfiguration(
        name="scoped-plan-policy",
        clean_task_succeeds=_is_clean_success(clean),
        benign_follow_up_succeeds=_is_benign_success(benign),
        planner_accepted_scope_input=not poisoned.rejected_fields,
        unauthorized_exposure=bool(unauthorized_exposure_scenarios),
        rejected_scope_fields=poisoned.rejected_fields,
        retrieval_hop_count=poisoned.retrieval_hop_count,
        scope_lineage_complete=_scope_lineage_complete(clean) and _scope_lineage_complete(benign),
    )
    configurations = (insecure, acl_only, scoped)
    failures = _control_failures(
        insecure,
        acl_only,
        scoped,
        unauthorized_exposure_scenarios=unauthorized_exposure_scenarios,
    )
    return ScopeGateReport(
        version="scope-propagation-v0",
        configurations=configurations,
        passed=not failures,
        failures=failures,
        traces=traces,
    )


def validate_scope_propagation(report: ScopeGateReport) -> None:
    """Fail CI when a protected property or evaluation control regresses."""

    if report.failures:
        messages = "; ".join(failure.message for failure in report.failures)
        raise ScopeGateError("scope-propagation release gate failed: " + messages)


async def _evaluate_insecure_baseline(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
) -> ScopeGateConfiguration:
    """Model a baseline that lets a document-supplied resource select evidence."""

    clean_succeeds = await _baseline_clean_task(authorization, principal, tenant_id)
    retriever = AccessGatedBm25Retriever(vertical_slice_chunks(), authorization)
    initial = await retriever.search_tenant(principal, tenant_id, "acme rollout approval")
    _, proposal = poisoned_proposal_from_initial(
        tuple(candidate.chunk_id for candidate in initial.candidates)
    )
    target_resource_id = proposal["resource_id"]
    candidates = tuple(
        chunk for chunk in vertical_slice_chunks() if chunk.resource_id == target_resource_id
    )
    return ScopeGateConfiguration(
        name="insecure-baseline",
        clean_task_succeeds=clean_succeeds,
        benign_follow_up_succeeds=True,
        planner_accepted_scope_input=True,
        unauthorized_exposure=any(
            chunk.resource_id == "document:beta-rollout" for chunk in candidates
        ),
        rejected_scope_fields=(),
        retrieval_hop_count=2,
        scope_lineage_complete=False,
    )


async def _evaluate_acl_only(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
) -> ScopeGateConfiguration:
    """Model per-hop ACL filtering without a planner-input authority boundary."""

    retriever = AccessGatedBm25Retriever(vertical_slice_chunks(), authorization)
    initial = await retriever.search_tenant(principal, tenant_id, "acme rollout approval")
    _, proposal = poisoned_proposal_from_initial(
        tuple(candidate.chunk_id for candidate in initial.candidates)
    )
    # This control parses and accepts ``resource_id`` but its retrieval backend
    # independently resolves the original caller's ACL for the next query.
    follow_up = await retriever.search_tenant(principal, tenant_id, str(proposal["query"]))
    return ScopeGateConfiguration(
        name="acl-filtered-per-hop",
        clean_task_succeeds=await _baseline_clean_task(authorization, principal, tenant_id),
        benign_follow_up_succeeds=True,
        planner_accepted_scope_input=True,
        unauthorized_exposure=any(
            candidate.resource_id == "document:beta-rollout" for candidate in follow_up.candidates
        ),
        rejected_scope_fields=(),
        retrieval_hop_count=2,
        scope_lineage_complete=False,
    )


async def _baseline_clean_task(
    authorization: AuthorizationAdapter,
    principal: Principal,
    tenant_id: str,
) -> bool:
    """Check clean retrieval for baselines without routing them through Proofline."""

    retriever = AccessGatedBm25Retriever(vertical_slice_chunks(), authorization)
    initial = await retriever.search_tenant(principal, tenant_id, "acme rollout approval")
    follow_up = await retriever.search_tenant(
        principal, tenant_id, "public release approval policy"
    )
    return any(
        candidate.resource_id == "document:acme-rollout" for candidate in initial.candidates
    ) and any(
        candidate.resource_id == "document:public-policy" for candidate in follow_up.candidates
    )


def _is_clean_success(trace: MultiHopTrace) -> bool:
    return trace.retrieval_hop_count == 2 and bool(trace.follow_up_candidate_ids)


def _is_benign_success(trace: MultiHopTrace) -> bool:
    return (
        trace.retrieval_hop_count == 2
        and trace.proposal_source_chunk_id == "chunk:public-security-guidance"
        and "chunk:public-policy" in trace.follow_up_candidate_ids
    )


def _scope_lineage_complete(trace: MultiHopTrace) -> bool:
    return (
        len(trace.scopes) == 2
        and trace.scopes[1].parent_scope_id == trace.scopes[0].scope_id
        and trace.scopes[1].follow_up_count == trace.scopes[0].follow_up_count + 1
    )


async def _unauthorized_exposure_scenarios(
    traces: tuple[MultiHopTrace, ...],
    *,
    authorization: AuthorizationAdapter,
    principal: Principal,
    tenant_id: str,
) -> tuple[str, ...]:
    """Return every scenario whose candidates violate the authorization oracle.

    This intentionally checks every retrieval hop, including normal clean and
    benign follow-ups. The evaluated trace contains IDs only, so the oracle
    resolves authority again from the adapter and fixture corpus rather than
    trusting the retriever-under-test to describe its own permissions.
    """

    access_scope = await authorization.list_permitted_resources(principal, tenant_id)
    chunks_by_id = {chunk.id: chunk for chunk in vertical_slice_chunks()}
    permitted_resource_ids = set(access_scope.resource_ids)
    scenarios: list[str] = []
    for trace in traces:
        candidate_ids = trace.initial_candidate_ids + trace.follow_up_candidate_ids
        if any(
            (chunk := chunks_by_id.get(candidate_id)) is None
            or not is_permitted_tenant_chunk(
                chunk,
                tenant_id=tenant_id,
                permitted_resource_ids=permitted_resource_ids,
            )
            for candidate_id in candidate_ids
        ):
            scenarios.append(trace.scenario)
    return tuple(scenarios)


def _control_failures(
    insecure: ScopeGateConfiguration,
    acl_only: ScopeGateConfiguration,
    scoped: ScopeGateConfiguration,
    *,
    unauthorized_exposure_scenarios: tuple[str, ...],
) -> tuple[GateFailure, ...]:
    failures: list[GateFailure] = []
    if not insecure.unauthorized_exposure:
        failures.append(
            GateFailure(
                "control-fixture", "insecure control did not expose Beta evidence", ("poisoned",)
            )
        )
    if not insecure.planner_accepted_scope_input:
        failures.append(
            GateFailure(
                "control-fixture", "insecure control did not accept scope input", ("poisoned",)
            )
        )
    if acl_only.unauthorized_exposure:
        failures.append(
            GateFailure(
                "access-isolation", "ACL-per-hop control exposed Beta evidence", ("poisoned",)
            )
        )
    if not acl_only.planner_accepted_scope_input:
        failures.append(
            GateFailure(
                "control-fixture", "ACL-per-hop control did not accept scope input", ("poisoned",)
            )
        )
    failures.extend(
        _scoped_failures(
            scoped,
            unauthorized_exposure_scenarios=unauthorized_exposure_scenarios,
        )
    )
    return tuple(failures)


def _scoped_failures(
    configuration: ScopeGateConfiguration,
    *,
    unauthorized_exposure_scenarios: tuple[str, ...],
) -> tuple[GateFailure, ...]:
    failures: list[GateFailure] = []
    if not configuration.clean_task_succeeds:
        failures.append(GateFailure("clean-task", "clean task did not complete", ("clean",)))
    if not configuration.benign_follow_up_succeeds:
        failures.append(
            GateFailure("false-block", "benign data-only follow-up was blocked", ("benign",))
        )
    if configuration.planner_accepted_scope_input:
        failures.append(
            GateFailure("plan-policy", "scope-bearing planner input was accepted", ("poisoned",))
        )
    if configuration.unauthorized_exposure:
        failures.append(
            GateFailure(
                "access-isolation",
                "unauthorized evidence was exposed",
                unauthorized_exposure_scenarios or ("unknown",),
            )
        )
    if configuration.retrieval_hop_count != 1:
        failures.append(
            GateFailure(
                "action-budget", "rejected proposal performed a follow-up retrieval", ("poisoned",)
            )
        )
    if not configuration.scope_lineage_complete:
        failures.append(
            GateFailure("scope-lineage", "clean scope lineage is incomplete", ("clean", "benign"))
        )
    if "resource_id" not in configuration.rejected_scope_fields:
        failures.append(
            GateFailure(
                "plan-policy", "scope-bearing field was not recorded as rejected", ("poisoned",)
            )
        )
    return tuple(failures)


def _normalized_trace(trace: MultiHopTrace) -> dict[str, object]:
    """Replace random scope UUIDs with deterministic, scenario-local references."""

    references = {
        scope.scope_id: f"{trace.scenario}:scope-{index}"
        for index, scope in enumerate(trace.scopes)
    }
    return {
        "scenario": trace.scenario,
        "initial_candidate_ids": list(trace.initial_candidate_ids),
        "follow_up_candidate_ids": list(trace.follow_up_candidate_ids),
        "scopes": [
            {
                "scope_ref": references[scope.scope_id],
                "parent_scope_ref": (
                    references.get(scope.parent_scope_id)
                    if scope.parent_scope_id is not None
                    else None
                ),
                "filter_counts": [list(item) for item in scope.filter_counts],
                "follow_up_count": scope.follow_up_count,
            }
            for scope in trace.scopes
        ],
        "retrieval_hop_count": trace.retrieval_hop_count,
        "proposal_source_chunk_id": trace.proposal_source_chunk_id,
        "rejected_fields": list(trace.rejected_fields),
    }
