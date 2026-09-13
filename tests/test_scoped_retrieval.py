# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from scopeanchor import (
    BoundScopedRetriever,
    ProposedRetrievalStep,
    RetrievalScope,
    ScopeCheckpointError,
    ScopedResults,
    ScopeError,
    ScopeExpiredError,
    ScopeValidationError,
    offload_sync,
    scoped,
)


def test_wrapper_derives_filters_from_trusted_context() -> None:
    calls: list[tuple[str, object, int]] = []

    def backend(query: str, *, filters: object, limit: int) -> list[str]:
        calls.append((query, filters, limit))
        return ["permitted-result"]

    def resolve_scope(context: dict[str, str]) -> RetrievalScope:
        return RetrievalScope.root(
            principal=context["principal"],
            filters={"resource_id": ["guide-a"]},
        )

    retriever = scoped(backend, resolve_scope=resolve_scope)
    results = asyncio.run(
        retriever.search("rollout prerequisites", context={"principal": "user:ana"}, limit=3)
    )

    assert results.items == ("permitted-result",)
    assert calls == [("rollout prerequisites", results.scope.filters, 3)]


def test_bound_retriever_resolves_trusted_context_once_for_independent_searches() -> None:
    resolver_calls = 0
    backend_calls: list[str] = []

    def resolve_scope(context: str) -> RetrievalScope:
        nonlocal resolver_calls
        resolver_calls += 1
        return RetrievalScope.root(
            principal=context,
            filters={"resource_id": ["guide-a"]},
            max_follow_ups=0,
        )

    def backend(query: str, *, filters: object, limit: int) -> list[str]:  # noqa: ARG001
        backend_calls.append(query)
        return [query]

    retriever = scoped(backend, resolve_scope=resolve_scope)

    async def run() -> tuple[BoundScopedRetriever[str], ScopedResults[str], ScopedResults[str]]:
        bound = await retriever.bind("user:ana")
        first, second = await asyncio.gather(bound.search("first"), bound.search("second"))
        return bound, first, second

    bound, first, second = asyncio.run(run())

    assert resolver_calls == 1
    assert first.scope is bound.scope
    assert second.scope is bound.scope
    assert first.scope.follow_up_count == 0
    assert backend_calls == ["first", "second"]

    with pytest.raises(TypeError):
        asyncio.run(bound.search("third", filters={"resource_id": ["guide-b"]}))  # type: ignore[call-arg]


def test_bound_retriever_can_create_a_trusted_narrowed_branch() -> None:
    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=lambda context: RetrievalScope.root(  # noqa: ARG005
            principal="user:ana",
            filters={"resource_id": ["guide-a", "guide-b"]},
        ),
    )

    async def run() -> tuple[BoundScopedRetriever[str], ScopedResults[str]]:
        bound = await retriever.bind(None)
        narrowed = bound.narrow_trusted({"resource_id": ["guide-a"]})
        return bound, await narrowed.search("second")

    bound, result = asyncio.run(run())

    assert result.scope.parent_scope_id == bound.scope.scope_id
    assert result.scope.filters["resource_id"] == frozenset({"guide-a"})
    assert bound.scope.filters["resource_id"] == frozenset({"guide-a", "guide-b"})


def test_resume_restores_a_bound_branch_after_rechecking_current_authorization() -> None:
    def resolve_scope(context: frozenset[str]) -> RetrievalScope:
        return RetrievalScope.root(principal="user:ana", filters={"resource_id": context})

    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=resolve_scope,
    )

    async def run() -> tuple[BoundScopedRetriever[str], ScopedResults[str]]:
        initial = await retriever.bind(frozenset({"guide-a", "guide-b"}))
        narrowed = initial.narrow_trusted({"resource_id": ["guide-a"]})
        checkpoint = narrowed.to_checkpoint(
            binding={"principal": "user:ana", "task_id": "task-123"}
        )
        resumed = await retriever.resume(
            checkpoint,
            context=frozenset({"guide-a", "guide-b"}),
            binding={"principal": "user:ana", "task_id": "task-123"},
        )
        return resumed, await resumed.search("continued")

    resumed, result = asyncio.run(run())

    assert result.scope == resumed.scope
    assert result.scope.filters["resource_id"] == frozenset({"guide-a"})
    assert result.scope.follow_up_count == 1


def test_resume_rejects_a_saved_scope_broader_than_current_authorization() -> None:
    def resolve_scope(context: frozenset[str]) -> RetrievalScope:
        return RetrievalScope.root(principal="user:ana", filters={"resource_id": context})

    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=resolve_scope,
    )

    async def run() -> None:
        initial = await retriever.bind(frozenset({"guide-a", "guide-b"}))
        checkpoint = initial.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})
        await retriever.resume(
            checkpoint,
            context=frozenset({"guide-a"}),
            binding={"principal": "user:ana", "task_id": "task-123"},
        )

    with pytest.raises(ScopeCheckpointError, match="broader"):
        asyncio.run(run())


def test_resume_applies_a_stricter_current_follow_up_limit() -> None:
    def resolve_scope(context: int) -> RetrievalScope:
        return RetrievalScope.root(
            principal="user:ana",
            filters={"resource_id": ["guide-a"]},
            max_follow_ups=context,
        )

    retriever = scoped(lambda query, *, filters, limit: [query], resolve_scope=resolve_scope)

    async def run() -> BoundScopedRetriever[str]:
        initial = await retriever.bind(3)
        checkpoint = initial.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})
        return await retriever.resume(
            checkpoint,
            context=1,
            binding={"principal": "user:ana", "task_id": "task-123"},
        )

    resumed = asyncio.run(run())

    assert resumed.scope.max_follow_ups == 1


def test_resume_applies_an_earlier_current_expiry() -> None:
    def resolve_scope(context: datetime) -> RetrievalScope:
        return RetrievalScope.root(
            principal="user:ana",
            filters={"resource_id": ["guide-a"]},
            expires_at=context,
        )

    retriever = scoped(lambda query, *, filters, limit: [query], resolve_scope=resolve_scope)
    saved_expiry = datetime(2031, 1, 1, tzinfo=timezone.utc)
    current_expiry = datetime(2030, 1, 1, tzinfo=timezone.utc)

    async def run() -> BoundScopedRetriever[str]:
        initial = await retriever.bind(saved_expiry)
        checkpoint = initial.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})
        return await retriever.resume(
            checkpoint,
            context=current_expiry,
            binding={"principal": "user:ana", "task_id": "task-123"},
        )

    resumed = asyncio.run(run())

    assert resumed.scope.expires_at == current_expiry


def test_follow_up_inherits_scope_and_rejects_widening() -> None:
    async def backend(query: str, *, filters: object, limit: int) -> list[str]:
        return [query]

    retriever = scoped(
        backend,
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana",
            filters={"resource_id": ["guide-a", "guide-b"]},
        ),
    )
    initial = asyncio.run(retriever.search("first", context=None))
    follow_up = asyncio.run(
        retriever.follow_up_trusted(
            initial, "second", narrowing_filters={"resource_id": ["guide-a"]}
        )
    )

    assert follow_up.scope.parent_scope_id == initial.scope.scope_id
    assert follow_up.scope.filters["resource_id"] == frozenset({"guide-a"})

    with pytest.raises(ScopeError, match="widens"):
        asyncio.run(
            retriever.follow_up_trusted(
                initial, "second", narrowing_filters={"resource_id": ["guide-c"]}
            )
        )


def test_normal_search_api_does_not_accept_model_supplied_filters() -> None:
    retriever = scoped(
        lambda query, *, filters, limit: [],
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana", filters={"resource_id": ["guide-a"]}
        ),
    )

    with pytest.raises(TypeError):
        asyncio.run(retriever.search("query", context=None, filters={"resource_id": ["guide-b"]}))  # type: ignore[call-arg]


def test_wrapper_accepts_an_async_trusted_scope_resolver() -> None:
    async def resolve_scope(context: str) -> RetrievalScope:
        return RetrievalScope.root(principal=context, filters={"resource_id": ["guide-a"]})

    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=resolve_scope,
    )

    results = asyncio.run(retriever.search("query", context="user:ana"))

    assert results.scope.principal == "user:ana"


def test_offload_sync_keeps_a_blocking_backend_off_the_event_loop() -> None:
    def blocking_backend(query: str, *, filters: object, limit: int) -> list[str]:  # noqa: ARG001
        time.sleep(0.05)
        return [query]

    retriever = scoped(
        offload_sync(blocking_backend),
        resolve_scope=lambda context: RetrievalScope.root(  # noqa: ARG005
            principal="user:ana", filters={"resource_id": ["guide-a"]}
        ),
    )

    async def run() -> None:
        search = asyncio.create_task(retriever.search("query", context=None))
        await asyncio.sleep(0.01)
        assert not search.done()
        assert (await search).items == ("query",)

    asyncio.run(run())


def test_proposed_follow_up_cannot_supply_authority() -> None:
    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana", filters={"resource_id": ["guide-a"]}
        ),
    )
    initial = asyncio.run(retriever.search("first", context=None))
    proposed = ProposedRetrievalStep.from_untrusted(
        {"query": "find the prerequisite", "parent_step_id": "step-1"}
    )

    result = asyncio.run(retriever.follow_proposed(initial, proposed))

    assert result.items == ("find the prerequisite",)
    assert result.scope.parent_scope_id == initial.scope.scope_id
    assert result.scope.filters == initial.scope.filters


def test_scope_validator_runs_before_each_retrieval_hop() -> None:
    accepted_scopes: list[str] = []

    def validate_scope(scope: RetrievalScope) -> bool:
        accepted_scopes.append(scope.scope_id)
        return len(accepted_scopes) == 1

    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana", filters={"resource_id": ["guide-a"]}
        ),
        validate_scope=validate_scope,
    )
    initial = asyncio.run(retriever.search("first", context=None))

    with pytest.raises(ScopeValidationError, match="no longer"):
        asyncio.run(retriever.follow_up(initial, "second"))


def test_scope_validator_can_return_a_safe_diagnostic_reason() -> None:
    retriever = scoped(
        lambda query, *, filters, limit: [query],
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana", filters={"resource_id": ["guide-a"]}
        ),
        validate_scope=lambda scope: "authorization policy version was revoked",  # noqa: ARG005
    )

    with pytest.raises(ScopeValidationError, match="policy version was revoked"):
        asyncio.run(retriever.search("first", context=None))


def test_async_scope_validation_cannot_dispatch_an_expired_scope() -> None:
    backend_calls: list[str] = []

    async def validate_scope(scope: RetrievalScope) -> bool:  # noqa: ARG001
        await asyncio.sleep(0.05)
        return True

    def backend(query: str, *, filters: object, limit: int) -> list[str]:  # noqa: ARG001
        backend_calls.append(query)
        return [query]

    retriever = scoped(
        backend,
        resolve_scope=lambda context: RetrievalScope.root(
            principal="user:ana",
            filters={"resource_id": ["guide-a"]},
            expires_at=datetime.now(timezone.utc) + timedelta(milliseconds=10),
        ),
        validate_scope=validate_scope,
    )

    with pytest.raises(ScopeExpiredError):
        asyncio.run(retriever.search("first", context=None))

    assert backend_calls == []
