# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Capability-attenuating retrieval for permission-preserving multi-hop RAG."""

from proofline.proposed_step import ProposedRetrievalStep, ProposedStepError
from proofline.retrieval import (
    FilteredSearch,
    ScopedResults,
    ScopedRetriever,
    ScopeResolver,
    ScopeValidationError,
    ScopeValidator,
    scoped,
)
from proofline.scope import (
    FilterAtom,
    RetrievalScope,
    ScopeError,
    ScopeExpiredError,
    ScopeFilters,
    ScopeMetadata,
    matches_scope_filters,
    validate_scope_filter_fields,
)

__version__ = "0.1.0"

__all__ = [
    "RetrievalScope",
    "FilterAtom",
    "ScopeFilters",
    "ScopeMetadata",
    "matches_scope_filters",
    "validate_scope_filter_fields",
    "ScopeError",
    "ScopeExpiredError",
    "ScopeValidationError",
    "ScopeResolver",
    "ScopeValidator",
    "FilteredSearch",
    "ScopedResults",
    "ScopedRetriever",
    "ProposedRetrievalStep",
    "ProposedStepError",
    "scoped",
]
