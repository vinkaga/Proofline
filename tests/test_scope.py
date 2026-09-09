# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

from datetime import datetime, timedelta, timezone

import pytest

from proofline import RetrievalScope, ScopeError, ScopeExpiredError


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
