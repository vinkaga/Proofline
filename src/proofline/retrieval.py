# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Framework-neutral scoped retrieval boundary."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from proofline.proposed_step import ProposedRetrievalStep
from proofline.scope import FilterAtom, RetrievalScope, ScopeFilters

ContextT = TypeVar("ContextT")
ResultT = TypeVar("ResultT")
ContextT_contra = TypeVar("ContextT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)
ScopeResolution = RetrievalScope | Awaitable[RetrievalScope]
ScopeValidation = bool | Awaitable[bool]


class ScopeResolver(Protocol[ContextT_contra]):
    """Trusted host code that resolves authenticated context into a root scope."""

    def __call__(self, context: ContextT_contra) -> ScopeResolution: ...


SearchReturn = Sequence[ResultT] | Awaitable[Sequence[ResultT]]


class ScopeValidator(Protocol):
    """Trusted host check used to reject a revoked scope before retrieval."""

    def __call__(self, scope: RetrievalScope) -> ScopeValidation: ...


class ScopeValidationError(PermissionError):
    """Raised when the host no longer accepts a retrieval scope."""


class FilteredSearch(Protocol[ResultT_co]):
    """The common retrieval shape used by the scoped wrapper."""

    def __call__(
        self,
        query: str,
        *,
        filters: ScopeFilters,
        limit: int,
    ) -> SearchReturn[ResultT_co]: ...


@dataclass(frozen=True, slots=True)
class ScopedResults(Generic[ResultT]):
    """Retrieved items plus the immutable scope for a permitted follow-up."""

    items: tuple[ResultT, ...]
    scope: RetrievalScope


class ScopedRetriever(Generic[ContextT, ResultT]):
    """Apply a trusted scope to every retrieval call.

    The ordinary ``search`` API accepts host context and a query only. It does
    not accept tenant, resource, principal, or raw filter arguments. A
    follow-up starts from a prior ``ScopedResults`` object and inherits the
    parent scope unless trusted application code supplies additional narrowing
    filters.
    """

    def __init__(
        self,
        backend: FilteredSearch[ResultT],
        *,
        resolve_scope: ScopeResolver[ContextT],
        validate_scope: ScopeValidator | None = None,
    ) -> None:
        self._backend = backend
        self._resolve_scope = resolve_scope
        self._validate_scope = validate_scope

    async def search(
        self,
        query: str,
        *,
        context: ContextT,
        limit: int = 10,
    ) -> ScopedResults[ResultT]:
        """Resolve host context once and retrieve using only its root scope."""

        scope = self._resolve_scope(context)
        if inspect.isawaitable(scope):
            scope = await scope
        if not isinstance(scope, RetrievalScope):
            raise TypeError("scope resolver must return RetrievalScope")
        return await self._search_under_scope(query, scope=scope, limit=limit)

    async def follow_up(
        self,
        previous: ScopedResults[ResultT],
        query: str,
        *,
        limit: int = 10,
        narrowing_filters: Mapping[str, Iterable[FilterAtom]] | None = None,
    ) -> ScopedResults[ResultT]:
        """Run one subsequent retrieval under an equal or narrower child scope.

        ``narrowing_filters`` is an advanced host-application API. Never pass
        values produced by a model or retrieved document to it.
        """

        child_scope = previous.scope.attenuate(narrowing_filters)
        return await self._search_under_scope(query, scope=child_scope, limit=limit)

    async def follow_proposed(
        self,
        previous: ScopedResults[ResultT],
        proposed: ProposedRetrievalStep,
        *,
        limit: int = 10,
    ) -> ScopedResults[ResultT]:
        """Execute a data-only planner proposal under the inherited scope.

        Parse model or retrieved-document output with
        :meth:`ProposedRetrievalStep.from_untrusted` before calling this
        method. The proposal has no authority fields, and this method does not
        offer a way to add them.
        """

        return await self.follow_up(previous, proposed.query, limit=limit)

    async def _search_under_scope(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> ScopedResults[ResultT]:
        if limit < 1:
            raise ValueError("limit must be at least one")
        scope.assert_active()
        if self._validate_scope is not None:
            accepted = self._validate_scope(scope)
            if inspect.isawaitable(accepted):
                accepted = await accepted
            if not accepted:
                raise ScopeValidationError("retrieval scope is no longer accepted by host policy")
        result = self._backend(query, filters=scope.filters, limit=limit)
        if inspect.isawaitable(result):
            result = await result
        return ScopedResults(items=tuple(result), scope=scope)


def scoped(
    backend: FilteredSearch[ResultT],
    *,
    resolve_scope: ScopeResolver[ContextT],
    validate_scope: ScopeValidator | None = None,
) -> ScopedRetriever[ContextT, ResultT]:
    """Wrap a sync or async retrieval callable with scope propagation."""

    return ScopedRetriever(backend, resolve_scope=resolve_scope, validate_scope=validate_scope)
