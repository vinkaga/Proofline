# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

import pytest

import proofline.scope as scope_module
from proofline import (
    FilterAtom,
    RetrievalScope,
    ScopeCheckpointError,
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


def test_scope_rejects_missing_authorization_constraints() -> None:
    with pytest.raises(ScopeError, match="require filters"):
        RetrievalScope.root(principal="user:ana", filters={})

    with pytest.raises(ScopeError, match="require filters"):
        RetrievalScope(principal="user:ana", filters={})


def test_explicit_unrestricted_scope_and_named_empty_allowlist_have_distinct_meanings() -> None:
    unrestricted = RetrievalScope.unrestricted(principal="service:public-search")

    assert unrestricted.is_unrestricted
    assert matches_scope_filters(
        {"resource_id": "any-document"},
        unrestricted.filters,
        supported_fields={"resource_id"},
    )

    no_access = unrestricted.attenuate({"resource_id": []})

    assert not no_access.is_unrestricted
    assert not matches_scope_filters(
        {"resource_id": "any-document"},
        no_access.filters,
        supported_fields={"resource_id"},
    )


def test_unchanged_child_reuses_validated_authority() -> None:
    root = RetrievalScope.root(
        principal="user:ana",
        filters={"tenant_id": ["acme"], "resource_id": ["guide-a", "guide-b"]},
        metadata={"trace_id": "trace-123"},
    )

    child = root.attenuate()

    assert child.filters is root.filters
    assert child.metadata is root.metadata


def test_narrowed_child_reuses_unchanged_filter_values() -> None:
    root = RetrievalScope.root(
        principal="user:ana",
        filters={"tenant_id": ["acme"], "resource_id": ["guide-a", "guide-b"]},
    )

    child = root.attenuate({"resource_id": ["guide-a"]})

    assert child.filters is not root.filters
    assert child.filters["tenant_id"] is root.filters["tenant_id"]
    assert child.filters["resource_id"] == frozenset({"guide-a"})


def test_descendants_do_not_revalidate_inherited_filter_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_freeze = scope_module._freeze_filter_values
    freeze_calls = 0

    def count_freezes(values: Iterable[FilterAtom]) -> frozenset[FilterAtom]:
        nonlocal freeze_calls
        freeze_calls += 1
        return original_freeze(values)

    monkeypatch.setattr(scope_module, "_freeze_filter_values", count_freezes)

    root = RetrievalScope.root(principal="user:ana", filters={"resource_id": ["guide-a"]})
    root.attenuate()
    root.attenuate({"resource_id": ["guide-a"]})

    assert freeze_calls == 2


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


@pytest.mark.parametrize(
    "values",
    [
        [1, True],
        [True, 1],
        [1, 1.0],
        [1.0, 1],
        [False, 0],
        [0.0, False],
    ],
)
def test_scope_rejects_equal_filter_values_with_different_types(
    values: list[int | float | bool],
) -> None:
    with pytest.raises(ScopeError, match="same type"):
        RetrievalScope.root(principal="user:ana", filters={"id": values})


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


def test_scope_checkpoint_round_trips_json_safe_authority_and_lineage() -> None:
    root = RetrievalScope.root(
        principal="user:ana",
        filters={"resource_id": ["guide-a", 3, 2.5, False, None]},
        metadata={"trace_id": "trace-123"},
        policy_version="policy-v7",
        expires_at=datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        max_follow_ups=2,
    )
    child = root.attenuate()

    checkpoint = child.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})
    restored = RetrievalScope.from_checkpoint(
        json.loads(json.dumps(checkpoint)),
        binding={"principal": "user:ana", "task_id": "task-123"},
    )

    assert restored == child
    assert restored.filters["resource_id"] == frozenset({"guide-a", 3, 2.5, False, None})
    assert restored.parent_scope_id == root.scope_id
    assert restored.follow_up_count == 1


def test_scope_checkpoint_rejects_wrong_binding_and_malformed_payload() -> None:
    scope = RetrievalScope.root(principal="user:ana", filters={"resource_id": ["guide-a"]})
    checkpoint = scope.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})

    with pytest.raises(ScopeCheckpointError, match="binding"):
        RetrievalScope.from_checkpoint(
            checkpoint,
            binding={"principal": "user:ana", "task_id": "task-456"},
        )

    checkpoint["version"] = 999
    with pytest.raises(ScopeCheckpointError, match="version"):
        RetrievalScope.from_checkpoint(
            checkpoint,
            binding={"principal": "user:ana", "task_id": "task-123"},
        )


def test_scope_checkpoint_rejects_invalid_scope_fields() -> None:
    scope = RetrievalScope.root(principal="user:ana", filters={"resource_id": ["guide-a"]})
    checkpoint = scope.to_checkpoint(binding={"principal": "user:ana", "task_id": "task-123"})
    checkpoint_scope = checkpoint["scope"]
    assert isinstance(checkpoint_scope, dict)
    checkpoint_scope["filters"] = {"": []}

    with pytest.raises(ScopeCheckpointError, match="invalid scope"):
        RetrievalScope.from_checkpoint(
            checkpoint,
            binding={"principal": "user:ana", "task_id": "task-123"},
        )


def test_scope_checkpoint_preserves_explicit_unrestricted_state() -> None:
    scope = RetrievalScope.unrestricted(principal="service:public-search")
    checkpoint = scope.to_checkpoint(binding={"principal": "service:public-search"})

    restored = RetrievalScope.from_checkpoint(
        json.loads(json.dumps(checkpoint)),
        binding={"principal": "service:public-search"},
    )

    assert restored.is_unrestricted
    assert restored.filters == {}


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
