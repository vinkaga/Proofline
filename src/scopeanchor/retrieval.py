# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Framework-neutral scoped retrieval boundary."""

from __future__ import annotations

import inspect
from asyncio import to_thread
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from functools import wraps
from typing import Generic, ParamSpec, Protocol, TypeVar

from scopeanchor.proposed_step import ProposedRetrievalStep
from scopeanchor.scope import (
    FilterAtom,
    RetrievalScope,
    ScopeCheckpoint,
    ScopeCheckpointError,
    ScopeFilters,
)

ContextT = TypeVar("ContextT")
ResultT = TypeVar("ResultT")
ContextT_contra = TypeVar("ContextT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)
Parameters = ParamSpec("Parameters")
ReturnT = TypeVar("ReturnT")
ScopeResolution = RetrievalScope | Awaitable[RetrievalScope]
ScopeValidation = bool | str | Awaitable[bool | str]


class ScopeResolver(Protocol[ContextT_contra]):
    """Trusted host code that resolves authenticated context into a root scope."""

    def __call__(self, context: ContextT_contra) -> ScopeResolution: ...


SearchReturn = Sequence[ResultT] | Awaitable[Sequence[ResultT]]


class ScopeValidator(Protocol):
    """Trusted host check used to reject a scope before retrieval.

    Return ``True`` to accept a scope, ``False`` for a generic rejection, or a
    non-empty string with a safe diagnostic reason. The reason is surfaced to
    the caller; it must not include secrets or protected document content.
    """

    def __call__(self, scope: RetrievalScope) -> ScopeValidation: ...


class ScopeValidationError(PermissionError):
    """Raised when the host no longer accepts a retrieval scope."""


def offload_sync(
    callable_: Callable[Parameters, ReturnT],
) -> Callable[Parameters, Awaitable[ReturnT]]:
    """Adapt a blocking synchronous integration for an async ScopeAnchor host.

    ``ScopedRetriever`` calls ordinary synchronous callbacks on its caller's
    event loop. Wrap only callbacks that are safe to run in a worker thread;
    hosts retain control of client affinity, thread pools, cancellation, and
    timeout policy.
    """

    @wraps(callable_)
    async def offloaded(*args: Parameters.args, **kwargs: Parameters.kwargs) -> ReturnT:
        return await to_thread(callable_, *args, **kwargs)

    return offloaded


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


class _ScopeSearcher(Protocol[ResultT]):
    """Internal operation used by a retriever bound to an existing scope."""

    async def __call__(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> ScopedResults[ResultT]: ...


class ScopedRetriever(Generic[ContextT, ResultT]):
    """Apply a trusted scope to every retrieval call.

    The ordinary ``search`` API accepts host context and a query only. It does
    not accept tenant, resource, principal, or raw filter arguments. A
    follow-up starts from a prior ``ScopedResults`` object and inherits the
    parent scope unless trusted application code supplies additional narrowing
    filters. Synchronous callbacks run inline and may block the event loop;
    use :func:`offload_sync` when the host has approved worker-thread execution.
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

    async def bind(self, context: ContextT) -> BoundScopedRetriever[ResultT]:
        """Bind trusted request context to one reusable retrieval branch.

        Resolve authorization once when an application begins an agent task or
        request, then call :meth:`BoundScopedRetriever.search` for each
        independent query. Those searches share the same immutable authority;
        they are not fabricated into a linear parent/child chain. Trusted host
        code may derive a narrower branch with
        :meth:`BoundScopedRetriever.narrow_trusted`.
        """

        return BoundScopedRetriever(self._search_under_scope, await self._resolve(context))

    async def resume(
        self,
        checkpoint: ScopeCheckpoint,
        *,
        context: ContextT,
        binding: Mapping[str, str],
    ) -> BoundScopedRetriever[ResultT]:
        """Resume a trusted saved branch under the caller's current authority.

        The host must supply the same stable caller/task ``binding`` used when
        it created the checkpoint. ScopeAnchor restores the saved scope, resolves
        the current trusted context, and rejects a saved branch that is broader
        than current authorization. It preserves saved restrictions while
        applying an earlier current expiry or a lower current follow-up limit.
        """

        saved_scope = RetrievalScope.from_checkpoint(checkpoint, binding=binding)
        current_scope = await self._resolve(context)
        current_scope.assert_active()
        if not saved_scope.is_attenuation_of(current_scope):
            raise ScopeCheckpointError(
                "saved scope is broader than the caller's current authorization"
            )
        max_follow_ups = _minimum_optional_int(
            saved_scope.max_follow_ups, current_scope.max_follow_ups
        )
        if max_follow_ups is not None and saved_scope.follow_up_count > max_follow_ups:
            raise ScopeCheckpointError("saved scope exceeds the current follow-up limit")
        resumed_scope = replace(
            saved_scope,
            expires_at=_earlier_expiry(saved_scope.expires_at, current_scope.expires_at),
            max_follow_ups=max_follow_ups,
        )
        resumed_scope.assert_active()
        return BoundScopedRetriever(self._search_under_scope, resumed_scope)

    async def search(
        self,
        query: str,
        *,
        context: ContextT,
        limit: int = 10,
    ) -> ScopedResults[ResultT]:
        """Resolve host context once and retrieve using only its root scope."""

        return await (await self.bind(context)).search(query, limit=limit)

    async def _resolve(self, context: ContextT) -> RetrievalScope:
        scope = self._resolve_scope(context)
        if inspect.isawaitable(scope):
            scope = await scope
        if not isinstance(scope, RetrievalScope):
            raise TypeError("scope resolver must return RetrievalScope")
        return scope

    async def follow_up(
        self,
        previous: ScopedResults[ResultT],
        query: str,
        *,
        limit: int = 10,
    ) -> ScopedResults[ResultT]:
        """Run one subsequent retrieval under the inherited child scope.

        This is safe for a query proposed by a model or retrieved document: it
        has no authority-bearing arguments.
        """

        child_scope = previous.scope.attenuate()
        return await self._search_under_scope(query, scope=child_scope, limit=limit)

    async def follow_up_trusted(
        self,
        previous: ScopedResults[ResultT],
        query: str,
        *,
        limit: int = 10,
        narrowing_filters: Mapping[str, Iterable[FilterAtom]] | None = None,
    ) -> ScopedResults[ResultT]:
        """Run a follow-up with trusted host-supplied narrowing filters.

        Only authentication, authorization, or other trusted application code
        may call this method. Never pass model or retrieved-document values to
        ``narrowing_filters``.
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
            if isinstance(accepted, str):
                if accepted:
                    raise ScopeValidationError(accepted)
                raise ScopeValidationError("retrieval scope is no longer accepted by host policy")
            if not isinstance(accepted, bool):
                raise TypeError("scope validator must return bool or a diagnostic rejection string")
            if not accepted:
                raise ScopeValidationError("retrieval scope is no longer accepted by host policy")
        # Validation may await external policy state. Recheck expiry at the
        # dispatch boundary so a scope that expires while validation runs is
        # never sent to the backend.
        scope.assert_active()
        result = self._backend(query, filters=scope.filters, limit=limit)
        if inspect.isawaitable(result):
            result = await result
        return ScopedResults(items=tuple(result), scope=scope)


@dataclass(frozen=True, slots=True)
class BoundScopedRetriever(Generic[ResultT]):
    """A retriever bound once to trusted authority for one request branch.

    Instances are created by :meth:`ScopedRetriever.bind`. Their ordinary
    ``search`` method accepts only a query and a result limit, so a model tool
    cannot supply a principal, tenant, resource filter, or parent result. A
    bound retriever has no mutable run state, so independent searches use the
    same branch scope instead of creating an artificial retrieval lineage.
    Backend concurrency remains the host's responsibility.
    """

    _search_under_scope: _ScopeSearcher[ResultT]
    _scope: RetrievalScope

    @property
    def scope(self) -> RetrievalScope:
        """Return the immutable authority shared by this branch."""

        return self._scope

    async def search(self, query: str, *, limit: int = 10) -> ScopedResults[ResultT]:
        """Retrieve under this branch's existing trusted authority."""

        return await self._search_under_scope(query, scope=self._scope, limit=limit)

    def narrow_trusted(
        self,
        narrowing_filters: Mapping[str, Iterable[FilterAtom]] | None = None,
    ) -> BoundScopedRetriever[ResultT]:
        """Return a child branch with equal or narrower trusted authority.

        Call this only from authentication, authorization, or other trusted
        application code. Never derive ``narrowing_filters`` from a model or
        retrieved document.
        """

        child_scope = self._scope.attenuate(narrowing_filters)
        return BoundScopedRetriever(self._search_under_scope, child_scope)

    def to_checkpoint(self, *, binding: Mapping[str, str]) -> dict[str, object]:
        """Serialize this branch's authority for trusted host checkpoint storage.

        Retrieved documents are deliberately excluded. Resume with
        :meth:`ScopedRetriever.resume` so current authorization is checked
        before this branch can search again.
        """

        return self._scope.to_checkpoint(binding=binding)


def _minimum_optional_int(left: int | None, right: int | None) -> int | None:
    """Return the stricter of two optional numeric limits."""

    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _earlier_expiry(left: datetime | None, right: datetime | None) -> datetime | None:
    """Return the stricter optional expiry without treating ``None`` as a date."""

    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def scoped(
    backend: FilteredSearch[ResultT],
    *,
    resolve_scope: ScopeResolver[ContextT],
    validate_scope: ScopeValidator | None = None,
) -> ScopedRetriever[ContextT, ResultT]:
    """Wrap a sync or async retrieval callable with scope propagation.

    Synchronous callbacks execute inline. To prevent blocking an async host,
    explicitly wrap thread-safe synchronous callbacks with :func:`offload_sync`.
    """

    return ScopedRetriever(backend, resolve_scope=resolve_scope, validate_scope=validate_scope)
