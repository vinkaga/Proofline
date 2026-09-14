# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Capability-attenuating retrieval for permission-preserving multi-hop RAG."""

from scopeanchor.proposed_step import ProposedRetrievalStep, ProposedStepError
from scopeanchor.retrieval import (
    BoundScopedRetriever,
    FilteredSearch,
    ScopedResults,
    ScopedRetriever,
    ScopeResolver,
    ScopeValidationError,
    ScopeValidator,
    offload_sync,
    scoped,
)
from scopeanchor.scope import (
    FilterAtom,
    RetrievalScope,
    ScopeCheckpoint,
    ScopeCheckpointError,
    ScopeError,
    ScopeExpiredError,
    ScopeFilters,
    ScopeMetadata,
    matches_scope_filters,
    validate_scope_filter_fields,
)

__all__ = [
    "RetrievalScope",
    "FilterAtom",
    "ScopeFilters",
    "ScopeCheckpoint",
    "ScopeMetadata",
    "matches_scope_filters",
    "validate_scope_filter_fields",
    "ScopeError",
    "ScopeExpiredError",
    "ScopeCheckpointError",
    "ScopeValidationError",
    "BoundScopedRetriever",
    "ScopeResolver",
    "ScopeValidator",
    "offload_sync",
    "FilteredSearch",
    "ScopedResults",
    "ScopedRetriever",
    "ProposedRetrievalStep",
    "ProposedStepError",
    "scoped",
]
