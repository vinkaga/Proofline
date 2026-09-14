# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Show why a follow-up retrieval needs an authority boundary.

Run after installing ScopeAnchor::

    python hello_scopeanchor.py

The deliberately unsafe path below lets an untrusted proposal select a
document ID. An ACL-filtered path reapplies the caller's filters to its
follow-up search. ScopeAnchor rejects the same scope-bearing proposal before
the backend receives a second search. Both protected paths keep the result
inside the caller's authorization boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from scopeanchor import (
    ProposedRetrievalStep,
    ProposedStepError,
    RetrievalScope,
    matches_scope_filters,
    scoped,
    validate_scope_filter_fields,
)
from scopeanchor.scope import ScopeFilters


@dataclass(frozen=True, slots=True)
class Document:
    resource_id: str
    text: str


@dataclass(frozen=True, slots=True)
class BackendCall:
    """One search and the effective filters the backend received."""

    query: str
    filters: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class FollowUpLineage:
    """The minimum auditable record for an accepted query-only follow-up."""

    parent_retrieval_id: str
    child_retrieval_id: str
    child_parent_retrieval_id: str | None
    parent_principal: str | None
    child_principal: str | None
    parent_filters: tuple[tuple[str, tuple[str, ...]], ...] | None
    child_filters: tuple[tuple[str, tuple[str, ...]], ...] | None
    parent_backend_filters: tuple[tuple[str, tuple[str, ...]], ...] | None
    child_backend_filters: tuple[tuple[str, tuple[str, ...]], ...] | None

    def is_complete(self) -> bool:
        """Require linkage, trusted identity, backend enforcement, and no widening."""

        if (
            self.child_parent_retrieval_id != self.parent_retrieval_id
            or self.parent_principal is None
            or self.child_principal is None
            or self.parent_filters is None
            or self.child_filters is None
            or self.parent_backend_filters is None
            or self.child_backend_filters is None
            or self.parent_principal != self.child_principal
            or self.parent_filters != self.parent_backend_filters
            or self.child_filters != self.child_backend_filters
        ):
            return False
        parent = dict(self.parent_filters)
        child = dict(self.child_filters)
        return bool(parent) and set(parent) == set(child) and all(
            set(child_values).issubset(parent[field]) for field, child_values in child.items()
        )


DOCUMENTS = (
    Document("document:acme-rollout", "Acme rollout plan"),
    Document("document:beta-rollout", "Beta rollout plan"),
)
SUPPORTED_FILTERS = frozenset({"resource_id"})


class SearchBackend(Protocol):
    """The small retriever contract used by this example."""

    def __call__(
        self,
        query: str,
        *,
        filters: ScopeFilters,
        limit: int,
    ) -> tuple[Document, ...]: ...


def format_documents(documents: tuple[Document, ...]) -> str:
    """Return a short, stable terminal representation of retrieved evidence."""

    return ", ".join(document.text for document in documents) or "no documents"


def contains_unauthorized_evidence(
    documents: tuple[Document, ...],
    caller_scope: RetrievalScope,
) -> bool:
    """Return whether any retrieved document falls outside the caller's scope."""

    return any(
        not matches_scope_filters(
            {"resource_id": document.resource_id},
            caller_scope.filters,
            supported_fields=SUPPORTED_FILTERS,
        )
        for document in documents
    )


def canonical_filters(
    filters: Mapping[str, Iterable[object]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Create a stable, displayable snapshot of effective backend filters."""

    return tuple(
        (field, tuple(sorted(str(value) for value in values)))
        for field, values in sorted(filters.items())
    )


def build_backend(*, calls: list[BackendCall]) -> SearchBackend:
    """Return a tiny backend that enforces the filters it receives."""

    def search(query: str, *, filters: ScopeFilters, limit: int) -> tuple[Document, ...]:
        validate_scope_filter_fields(filters, supported_fields=SUPPORTED_FILTERS)
        calls.append(BackendCall(query, canonical_filters(filters)))
        return tuple(
            document
            for document in DOCUMENTS
            if matches_scope_filters(
                {"resource_id": document.resource_id},
                filters,
                supported_fields=SUPPORTED_FILTERS,
            )
        )[:limit]

    return search


def unsafe_follow_up(
    backend: SearchBackend,
    caller_scope: RetrievalScope,
    proposal: Mapping[str, object],
) -> tuple[Document, ...]:
    """Model the unsafe pattern: proposal data chooses retrieval authority."""

    query = proposal["query"]
    assert isinstance(query, str)
    resource_id = proposal.get("resource_id")
    if resource_id is None:
        filters = caller_scope.filters
    else:
        assert isinstance(resource_id, str)
        filters = {"resource_id": (resource_id,)}
    return backend(query, filters=filters, limit=10)


def acl_filtered_follow_up(
    backend: SearchBackend,
    caller_scope: RetrievalScope,
    proposal: Mapping[str, object],
) -> tuple[Document, ...]:
    """Model ACL filtering: trusted code reapplies the caller's filters."""

    query = proposal["query"]
    assert isinstance(query, str)
    return backend(query, filters=caller_scope.filters, limit=10)


def lineage_from_calls(
    *,
    control: str,
    principal: str,
    caller_scope: RetrievalScope,
    parent_call: BackendCall,
    child_call: BackendCall,
    parent_scope: RetrievalScope | None = None,
    child_scope: RetrievalScope | None = None,
) -> FollowUpLineage:
    """Record the effective scope and backend filters used by a trusted follow-up."""

    parent_filters = canonical_filters((parent_scope or caller_scope).filters)
    child_filters = canonical_filters((child_scope or caller_scope).filters)
    return FollowUpLineage(
        parent_retrieval_id=f"{control}:initial",
        child_retrieval_id=f"{control}:query-only-follow-up",
        child_parent_retrieval_id=(
            f"{control}:initial"
            if child_scope is None
            or (parent_scope is not None and child_scope.parent_scope_id == parent_scope.scope_id)
            else None
        ),
        parent_principal=principal,
        child_principal=(child_scope or caller_scope).principal,
        parent_filters=parent_filters,
        child_filters=child_filters,
        parent_backend_filters=parent_call.filters,
        child_backend_filters=child_call.filters,
    )


def print_comparison(
    *,
    unsafe_exposes_unauthorized_evidence: bool,
    acl_exposes_unauthorized_evidence: bool,
    scopeanchor_exposes_unauthorized_evidence: bool,
    unsafe_rejected: bool,
    acl_rejected: bool,
    scopeanchor_rejected: bool,
    unsafe_lineage: FollowUpLineage,
    acl_lineage: FollowUpLineage,
    scopeanchor_lineage: FollowUpLineage,
) -> None:
    """Print the three README measures for this single deterministic trace."""

    print("Comparison for this one trace")
    print("  Measure                                      Unsafe  ACL per hop  ScopeAnchor")
    print(
        "  No unauthorized evidence exposed             "
        f"{'yes' if not unsafe_exposes_unauthorized_evidence else 'no ':>3}"
        f"     {'yes' if not acl_exposes_unauthorized_evidence else 'no ':>3}"
        f"         {'yes' if not scopeanchor_exposes_unauthorized_evidence else 'no ':>3}"
    )
    print(
        "  Scope-bearing proposal rejected before search "
        f"{'yes' if unsafe_rejected else 'no ':>3}"
        f"     {'yes' if acl_rejected else 'no ':>3}"
        f"         {'yes' if scopeanchor_rejected else 'no ':>3}"
    )
    print(
        "  Query-only follow-up lineage complete         "
        f"{'yes' if unsafe_lineage.is_complete() else 'no ':>3}"
        f"     {'yes' if acl_lineage.is_complete() else 'no ':>3}"
        f"         {'yes' if scopeanchor_lineage.is_complete() else 'no ':>3}"
    )


async def main() -> None:
    """Compare an unsafe path, manual ACL filtering, and ScopeAnchor."""

    caller_scope = RetrievalScope.root(
        principal="user:ana",
        filters={"resource_id": ("document:acme-rollout",)},
    )
    scope_changing_proposal = {
        "query": "rollout plan",
        "resource_id": "document:beta-rollout",
    }
    data_only_proposal = {"query": "rollout plan"}

    print("Hello, ScopeAnchor\n")
    print("Caller: user:ana")
    print("Authorized document: Acme rollout plan\n")

    # Mode: pass proposal-provided filters directly to the backend.
    unsafe_calls: list[BackendCall] = []
    unsafe_backend = build_backend(calls=unsafe_calls)
    unsafe_initial = unsafe_backend("rollout plan", filters=caller_scope.filters, limit=10)
    unsafe_follow_up_results = unsafe_follow_up(
        unsafe_backend,
        caller_scope,
        scope_changing_proposal,
    )
    unsafe_exposes_unauthorized_evidence = contains_unauthorized_evidence(
        unsafe_follow_up_results,
        caller_scope,
    )
    print("Unsafe: proposal controls the filter")
    print(f"  Initial search: {format_documents(unsafe_initial)}")
    print("  Scope-bearing follow-up requests: Beta rollout plan")
    print(f"  Result: {format_documents(unsafe_follow_up_results)}")
    print(
        "  Unauthorized evidence was retrieved."
        if unsafe_exposes_unauthorized_evidence
        else "  No unauthorized evidence was retrieved."
    )
    unsafe_data_only_results = unsafe_follow_up(unsafe_backend, caller_scope, data_only_proposal)
    unsafe_lineage = FollowUpLineage(
        parent_retrieval_id="unsafe:initial",
        child_retrieval_id="unsafe:query-only-follow-up",
        child_parent_retrieval_id=None,
        parent_principal=None,
        child_principal=None,
        parent_filters=None,
        child_filters=None,
        parent_backend_filters=None,
        child_backend_filters=None,
    )
    print("  Data-only follow-up")
    print(f"  Result: {format_documents(unsafe_data_only_results)}")
    print()

    # Mode: apply the caller's ACL filters to every backend search.
    acl_calls: list[BackendCall] = []
    acl_backend = build_backend(calls=acl_calls)
    acl_initial = acl_backend("rollout plan", filters=caller_scope.filters, limit=10)
    acl_follow_up_results = acl_filtered_follow_up(
        acl_backend,
        caller_scope,
        scope_changing_proposal,
    )
    acl_exposes_unauthorized_evidence = contains_unauthorized_evidence(
        acl_follow_up_results,
        caller_scope,
    )
    print("Manual ACL filtering on every hop")
    print(f"  Initial search: {format_documents(acl_initial)}")
    print("  Scope-bearing follow-up attempts to request: Beta rollout plan")
    print(f"  Result: {format_documents(acl_follow_up_results)}")
    print(
        "  Unauthorized evidence was retrieved."
        if acl_exposes_unauthorized_evidence
        else "  Acme's filter was reapplied; Beta was excluded."
    )
    acl_data_only_results = acl_filtered_follow_up(acl_backend, caller_scope, data_only_proposal)
    acl_lineage = lineage_from_calls(
        control="acl",
        principal=caller_scope.principal,
        caller_scope=caller_scope,
        parent_call=acl_calls[0],
        child_call=acl_calls[2],
    )
    print("  Data-only follow-up")
    print(f"  Result: {format_documents(acl_data_only_results)}")
    print("  Scope-bearing fields: ignored; trusted code chose the filter\n")

    # Mode: route retrieval through ScopeAnchor with the caller's root scope.
    protected_calls: list[BackendCall] = []
    protected_backend = build_backend(calls=protected_calls)
    retriever = scoped(protected_backend, resolve_scope=lambda _context: caller_scope)
    protected_initial = await retriever.search("rollout plan", context=None)

    backend_calls_before_proposal = len(protected_calls)
    try:
        protected_proposal = ProposedRetrievalStep.from_untrusted(scope_changing_proposal)
    except ProposedStepError as error:
        rejection = f"rejected before retrieval ({', '.join(error.fields)})"
        proposal_rejected = True
    else:  # pragma: no cover - this makes an accidental API regression obvious when run manually.
        await retriever.follow_proposed(protected_initial, protected_proposal)
        rejection = "accepted unexpectedly"
        proposal_rejected = False
    backend_calls_after_rejection = len(protected_calls)
    assert proposal_rejected and backend_calls_after_rejection == backend_calls_before_proposal
    print("ScopeAnchor")
    print(f"  Initial search: {format_documents(protected_initial.items)}")
    print("  Scope-bearing follow-up requests: Beta rollout plan")
    print(f"  Result: {rejection}")

    # Input: a follow-up proposal containing only a query.
    permitted_proposal = ProposedRetrievalStep.from_untrusted(data_only_proposal)
    permitted_follow_up = await retriever.follow_proposed(protected_initial, permitted_proposal)
    lineage_is_complete = (
        permitted_follow_up.scope.parent_scope_id == protected_initial.scope.scope_id
        and permitted_follow_up.scope.is_attenuation_of(protected_initial.scope)
    )
    scopeanchor_lineage = lineage_from_calls(
        control="scopeanchor",
        principal=caller_scope.principal,
        caller_scope=caller_scope,
        parent_call=protected_calls[0],
        child_call=protected_calls[1],
        parent_scope=protected_initial.scope,
        child_scope=permitted_follow_up.scope,
    )
    scopeanchor_exposes_unauthorized_evidence = contains_unauthorized_evidence(
        protected_initial.items + permitted_follow_up.items,
        caller_scope,
    )
    assert unsafe_exposes_unauthorized_evidence
    assert not acl_exposes_unauthorized_evidence
    assert not scopeanchor_exposes_unauthorized_evidence
    assert not unsafe_lineage.is_complete()
    assert acl_lineage.is_complete()
    assert scopeanchor_lineage.is_complete()
    print("  Data-only follow-up")
    print(f"  Result: {format_documents(permitted_follow_up.items)}")
    print(f"  Inherited scope record: {'yes' if lineage_is_complete else 'no'}")
    print(f"  Verified lineage record: {'yes' if scopeanchor_lineage.is_complete() else 'no'}\n")
    print_comparison(
        unsafe_exposes_unauthorized_evidence=unsafe_exposes_unauthorized_evidence,
        acl_exposes_unauthorized_evidence=acl_exposes_unauthorized_evidence,
        scopeanchor_exposes_unauthorized_evidence=scopeanchor_exposes_unauthorized_evidence,
        unsafe_rejected=False,
        acl_rejected=False,
        scopeanchor_rejected=proposal_rejected,
        unsafe_lineage=unsafe_lineage,
        acl_lineage=acl_lineage,
        scopeanchor_lineage=scopeanchor_lineage,
    )
    print()
    print("Fair reading")
    print("  The unsafe path leaks Beta because it lets untrusted data choose filters.")
    print("  Manual ACL filtering and ScopeAnchor both prevent that leak here.")
    print("  ScopeAnchor additionally rejects a scope-bearing proposal before search.")
    print("  The ACL path reaches equivalent lineage only because this example explicitly")
    print("  records and verifies it; each production retrieval path needs the same work.")


if __name__ == "__main__":
    asyncio.run(main())
