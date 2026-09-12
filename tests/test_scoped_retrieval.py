# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from proofline import (
    ProposedRetrievalStep,
    RetrievalScope,
    ScopeError,
    ScopeExpiredError,
    ScopeValidationError,
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
