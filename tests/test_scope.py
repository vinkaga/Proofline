# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

from datetime import datetime, timedelta, timezone

import pytest

from proofline import (
    RetrievalScope,
    ScopeError,
    ScopeExpiredError,
    matches_scope_filters,
    validate_scope_filter_fields,
)


def test_scope_filters_are_immutable() -> None:
    metadata = {"trace_id": "trace-123"}
    scope = RetrievalScope.root(
        principal="user:ana",
        filters={"tenant_id": ["acme"], "resource_id": ["guide-a", "guide-b"]},
        metadata=metadata,
    )
    metadata["trace_id"] = "changed"

    with pytest.raises(TypeError):
        scope.filters["tenant_id"] = frozenset({"beta"})  # type: ignore[index]
    with pytest.raises(TypeError):
        scope.metadata["trace_id"] = "changed"  # type: ignore[index]
    assert scope.metadata == {"trace_id": "trace-123"}


def test_child_scope_can_only_narrow_existing_allowlists() -> None:
    root = RetrievalScope.root(
        principal="user:ana",
        filters={"resource_id": ["guide-a", "guide-b"]},
    )

    child = root.attenuate({"resource_id": ["guide-a"], "visibility": ["published"]})

    assert child.principal == root.principal
    assert child.parent_scope_id == root.scope_id
    assert child.filters["resource_id"] == frozenset({"guide-a"})
    assert child.filters["visibility"] == frozenset({"published"})
    assert child.metadata == root.metadata

    with pytest.raises(ScopeError, match="widens"):
        root.attenuate({"resource_id": ["guide-a", "guide-c"]})


@pytest.mark.parametrize(
    ("parent_value", "child_value"),
    [
        (1, True),
        (True, 1),
        (1, 1.0),
        (1.0, 1),
    ],
)
def test_scope_attenuation_distinguishes_equal_values_of_different_types(
    parent_value: int | float | bool, child_value: int | float | bool
) -> None:
    root = RetrievalScope.root(principal="user:ana", filters={"id": [parent_value]})

    with pytest.raises(ScopeError, match="widens"):
        root.attenuate({"id": [child_value]})


def test_scope_rejects_equal_filter_values_with_different_types() -> None:
    with pytest.raises(ScopeError, match="same type"):
        RetrievalScope.root(principal="user:ana", filters={"id": [1, True]})


def test_adapter_filter_contract_rejects_unknown_fields_and_matches_conjunctively() -> None:
    scope = RetrievalScope.root(
        principal="user:ana",
        filters={"tenant_id": ["tenant:acme"], "resource_id": ["document:shared"]},
    )
    supported_fields = {"tenant_id", "resource_id"}

    assert matches_scope_filters(
        {"tenant_id": "tenant:acme", "resource_id": "document:shared"},
        scope.filters,
        supported_fields=supported_fields,
    )
    assert not matches_scope_filters(
        {"tenant_id": "tenant:beta", "resource_id": "document:shared"},
        scope.filters,
        supported_fields=supported_fields,
    )
    assert not matches_scope_filters(
        {"tenant_id": "tenant:acme", "resource_id": "document:other"},
        scope.filters,
        supported_fields=supported_fields,
    )

    empty_allowlist = RetrievalScope.root(principal="user:ana", filters={"resource_id": []})
    assert not matches_scope_filters(
        {"resource_id": "document:shared"},
        empty_allowlist.filters,
        supported_fields={"resource_id"},
    )

    with pytest.raises(ScopeError, match="does not support"):
        validate_scope_filter_fields(
            RetrievalScope.root(
                principal="user:ana", filters={"resource_id": ["document:shared"]}
            ).attenuate({"visibility": ["published"]}).filters,
            supported_fields={"resource_id"},
        )


@pytest.mark.parametrize(
    "filters",
    [
        {1: ["guide-a"]},
        {"id": [float("nan")]},
        {"id": [float("inf")]},
        {"id": [object()]},
    ],
)
def test_scope_rejects_invalid_filter_atoms_or_names(filters: dict[object, list[object]]) -> None:
    with pytest.raises(ScopeError, match="filter"):
        RetrievalScope.root(
            principal="user:ana",
            filters=filters,  # type: ignore[arg-type]
        )


def test_scope_rejects_expiry_before_use() -> None:
    scope = RetrievalScope(
        principal="user:ana",
        filters={"resource_id": frozenset({"guide-a"})},
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    with pytest.raises(ScopeExpiredError):
        scope.assert_active()


def test_scope_can_limit_follow_up_hops() -> None:
    root = RetrievalScope.root(
        principal="user:ana",
        filters={"resource_id": ["guide-a"]},
        max_follow_ups=1,
    )

    child = root.attenuate()

    assert child.follow_up_count == 1
    with pytest.raises(ScopeError, match="budget"):
        child.attenuate()


def test_scope_repr_redacts_filter_values() -> None:
    scope = RetrievalScope.root(
        principal="user:ana",
        filters={"resource_id": ["guide-a", "guide-b"]},
        metadata={"trace_id": "trace-123"},
    )

    representation = repr(scope)

    assert "'resource_id': 2" in representation
    assert "guide-a" not in representation
    assert "metadata_keys=('trace_id',)" in representation


@pytest.mark.parametrize(
    "metadata",
    [
        {"": "trace-123"},
        {"trace_id": 1},
    ],
)
def test_scope_rejects_invalid_metadata(metadata: dict[str, object]) -> None:
    with pytest.raises(ScopeError, match="metadata"):
        RetrievalScope.root(
            principal="user:ana",
            filters={"resource_id": ["guide-a"]},
            metadata=metadata,  # type: ignore[arg-type]
        )
