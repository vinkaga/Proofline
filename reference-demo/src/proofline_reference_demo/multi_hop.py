# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministic two-hop fixtures for Proofline's scope-propagation contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

from proofline import ProposedRetrievalStep, ProposedStepError, RetrievalScope, ScopeFilters

from proofline_reference_demo.authorization import AuthorizationAdapter
from proofline_reference_demo.domain import Principal
from proofline_reference_demo.scoped_fixture import DemoRequestContext, build_scoped_fixture
from proofline_reference_demo.vertical_slice import vertical_slice_chunks

_POISON_MARKER = "PLANNER_FIXTURE: "


@dataclass(frozen=True, slots=True)
class ScopeTrace:
    """The non-sensitive facts needed to inspect one scope in a retrieval path."""

    scope_id: str
    parent_scope_id: str | None
    filter_counts: tuple[tuple[str, int], ...]
    follow_up_count: int


@dataclass(frozen=True, slots=True)
class MultiHopTrace:
    """An inspectable record of one deterministic two-hop fixture."""

    scenario: str
    initial_candidate_ids: tuple[str, ...]
    follow_up_candidate_ids: tuple[str, ...]
    scopes: tuple[ScopeTrace, ...]
    retrieval_hop_count: int
    proposal_source_chunk_id: str | None = None
    rejected_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe data for the reference-demo CLI."""

        return asdict(self)


def _scope_trace(scope: RetrievalScope) -> ScopeTrace:
    return ScopeTrace(
        scope_id=scope.scope_id,
        parent_scope_id=scope.parent_scope_id,
        filter_counts=tuple(
            sorted((field, len(values)) for field, values in scope.filters.items())
        ),
        follow_up_count=scope.follow_up_count,
    )


def _poisoned_proposal_from_initial(
    initial_candidate_ids: tuple[str, ...],
) -> tuple[str, Mapping[str, object]]:
    """Extract the deliberately untrusted fixture proposal from retrieved evidence.

    This is a deterministic stand-in for a planner that reads the authorized
    Acme chunk and emits structured output. It does not execute document text.
    The output still crosses ``from_untrusted`` before it can request retrieval.
    """

    source_chunk = next(
        (chunk for chunk in vertical_slice_chunks() if chunk.id == "chunk:acme-rollout"),
        None,
    )
    if source_chunk is None or source_chunk.id not in initial_candidate_ids:
        raise ValueError("poisoned fixture requires the authorized Acme rollout chunk")
    _, separator, serialized = source_chunk.content.partition(_POISON_MARKER)
    if not separator:
        raise ValueError("poisoned fixture is missing its planner proposal")
    parsed = json.loads(serialized)
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise ValueError("poisoned fixture proposal must be a string-keyed object")
    return source_chunk.id, cast(Mapping[str, object], parsed)


async def run_clean_two_hop(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
) -> MultiHopTrace:
    """Run a permitted first retrieval and one inherited-scope follow-up."""

    retrieval_hop_count = 0

    def count_retrieval(_: str, __: ScopeFilters) -> None:
        nonlocal retrieval_hop_count
        retrieval_hop_count += 1

    retriever = build_scoped_fixture(
        authorization, max_follow_ups=1, on_retrieval=count_retrieval
    )
    initial = await retriever.search(
        "acme rollout approval",
        context=DemoRequestContext(principal=principal, tenant_id=tenant_id),
    )
    proposed = ProposedRetrievalStep.from_untrusted(
        {
            "query": "public release approval policy",
            "parent_step_id": initial.scope.scope_id,
        }
    )
    follow_up = await retriever.follow_proposed(initial, proposed)
    return MultiHopTrace(
        scenario="clean",
        initial_candidate_ids=tuple(candidate.chunk_id for candidate in initial.items),
        follow_up_candidate_ids=tuple(candidate.chunk_id for candidate in follow_up.items),
        scopes=(_scope_trace(initial.scope), _scope_trace(follow_up.scope)),
        retrieval_hop_count=retrieval_hop_count,
    )


async def run_poisoned_two_hop(
    authorization: AuthorizationAdapter,
    *,
    principal: Principal,
    tenant_id: str,
) -> MultiHopTrace:
    """Reject a retrieved scope-bearing proposal before a follow-up can occur."""

    retrieval_hop_count = 0

    def count_retrieval(_: str, __: ScopeFilters) -> None:
        nonlocal retrieval_hop_count
        retrieval_hop_count += 1

    retriever = build_scoped_fixture(
        authorization, max_follow_ups=1, on_retrieval=count_retrieval
    )
    initial = await retriever.search(
        "acme rollout approval",
        context=DemoRequestContext(principal=principal, tenant_id=tenant_id),
    )
    source_chunk_id, raw_proposal = _poisoned_proposal_from_initial(
        tuple(candidate.chunk_id for candidate in initial.items)
    )
    try:
        ProposedRetrievalStep.from_untrusted(raw_proposal)
    except ProposedStepError as error:
        return MultiHopTrace(
            scenario="poisoned",
            initial_candidate_ids=tuple(candidate.chunk_id for candidate in initial.items),
            follow_up_candidate_ids=(),
            scopes=(_scope_trace(initial.scope),),
            retrieval_hop_count=retrieval_hop_count,
            proposal_source_chunk_id=source_chunk_id,
            rejected_fields=error.fields,
        )
    raise AssertionError("the poisoned fixture must be rejected")
