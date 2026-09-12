# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Immutable, capability-attenuating retrieval scopes."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias
from uuid import uuid4

FilterAtom: TypeAlias = str | int | float | bool | None
ScopeFilters: TypeAlias = Mapping[str, frozenset[FilterAtom]]
# Metadata is intentionally distinct from FilterAtom-based authorization
# filters: it carries only trusted string identifiers such as trace IDs and
# request IDs and is never interpreted as authority.
ScopeMetadata: TypeAlias = Mapping[str, str]


def _freeze_filter_values(values: Iterable[FilterAtom]) -> frozenset[FilterAtom]:
    """Validate and freeze filter atoms without conflating scalar types."""

    validated: list[FilterAtom] = []
    for value in values:
        if type(value) not in (str, int, float, bool, type(None)):
            raise ScopeError("filter values must be strings, integers, floats, booleans, or null")
        if isinstance(value, float) and not isfinite(value):
            raise ScopeError("filter float values must be finite")
        if any(type(value) is not type(existing) and value == existing for existing in validated):
            raise ScopeError("filter values that compare equal must have the same type")
        validated.append(value)
    return frozenset(validated)


def _typed_filter_values(
    values: Iterable[FilterAtom],
) -> frozenset[tuple[type[object], FilterAtom]]:
    """Return filter values keyed by their exact runtime scalar type."""

    return frozenset((type(value), value) for value in values)


class ScopeError(ValueError):
    """Raised when a scope is invalid or attempts to gain authority."""


class ScopeExpiredError(ScopeError):
    """Raised when a scope is used after its trusted expiry time."""


_MISSING_FILTER_VALUE = object()


def validate_scope_filter_fields(
    filters: ScopeFilters, *, supported_fields: Collection[str]
) -> None:
    """Reject scope filters that a backend has not explicitly implemented.

    Call this before ranking or querying a backend. A backend must either
    support every supplied field or reject the request; silently ignoring a
    narrowing field can widen retrieval authority.
    """

    unsupported_fields = sorted(set(filters).difference(supported_fields))
    if unsupported_fields:
        raise ScopeError(f"backend does not support scope filters: {unsupported_fields!r}")


def matches_scope_filters(
    candidate: Mapping[str, FilterAtom],
    filters: ScopeFilters,
    *,
    supported_fields: Collection[str],
) -> bool:
    """Return whether a candidate satisfies every supported scope filter.

    Each field is an allowlist and fields are conjunctive. Empty allowlists and
    missing candidate fields match nothing. Scalar comparisons preserve exact
    runtime type, so ``True``, ``1``, and ``1.0`` remain distinct.
    """

    validate_scope_filter_fields(filters, supported_fields=supported_fields)
    for field_name, permitted_values in filters.items():
        candidate_value = candidate.get(field_name, _MISSING_FILTER_VALUE)
        if candidate_value is _MISSING_FILTER_VALUE or not permitted_values:
            return False
        if not any(
            type(candidate_value) is type(permitted_value) and candidate_value == permitted_value
            for permitted_value in permitted_values
        ):
            return False
    return True


def _freeze_filters(filters: Mapping[str, Iterable[FilterAtom]]) -> ScopeFilters:
    frozen: dict[str, frozenset[FilterAtom]] = {}
    for field_name, values in filters.items():
        if not isinstance(field_name, str) or not field_name:
            raise ScopeError("filter names must be non-empty strings")
        if isinstance(values, (str, bytes)):
            raise ScopeError("filter values must be an iterable of scalar values, not a string")
        frozen[field_name] = _freeze_filter_values(values)
    return MappingProxyType(frozen)


def _freeze_metadata(metadata: Mapping[str, str]) -> ScopeMetadata:
    frozen: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ScopeError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise ScopeError("metadata values must be strings")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """A trusted caller's immutable authority for one retrieval branch.

    The application creates a root scope after authentication and authorization.
    A child scope may keep or narrow each existing allowlist and may add a new
    allowlist constraint; it cannot change a principal or broaden an existing
    allowlist.
    """

    principal: str
    filters: ScopeFilters
    metadata: ScopeMetadata = field(default_factory=lambda: _freeze_metadata({}))
    policy_version: str = ""
    expires_at: datetime | None = None
    max_follow_ups: int | None = None
    follow_up_count: int = 0
    scope_id: str = field(default_factory=lambda: str(uuid4()))
    parent_scope_id: str | None = None

    def __post_init__(self) -> None:
        if not self.principal:
            raise ScopeError("a scope requires a principal")
        if self.max_follow_ups is not None and self.max_follow_ups < 0:
            raise ScopeError("max_follow_ups must be non-negative")
        if self.follow_up_count < 0:
            raise ScopeError("follow_up_count must be non-negative")
        if self.max_follow_ups is not None and self.follow_up_count > self.max_follow_ups:
            raise ScopeError("follow_up_count exceeds max_follow_ups")
        object.__setattr__(self, "filters", _freeze_filters(self.filters))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @classmethod
    def root(
        cls,
        *,
        principal: str,
        filters: Mapping[str, Iterable[FilterAtom]],
        metadata: Mapping[str, str] | None = None,
        policy_version: str = "",
        expires_at: datetime | None = None,
        max_follow_ups: int | None = None,
    ) -> RetrievalScope:
        """Create a root scope from trusted application authorization output."""

        scope = cls(
            principal=principal,
            filters=_freeze_filters(filters),
            metadata=_freeze_metadata(metadata or {}),
            policy_version=policy_version,
            expires_at=expires_at,
            max_follow_ups=max_follow_ups,
        )
        scope.assert_active()
        return scope

    def assert_active(self, *, now: datetime | None = None) -> None:
        """Reject expired scopes before a backend retrieval is attempted."""

        if self.expires_at is None:
            return
        current_time = now or datetime.now(timezone.utc)
        expiry = self.expires_at
        if expiry.tzinfo is None:
            raise ScopeError("scope expiry must be timezone-aware")
        if current_time >= expiry:
            raise ScopeExpiredError("retrieval scope has expired")

    def attenuate(
        self, narrowing_filters: Mapping[str, Iterable[FilterAtom]] | None = None
    ) -> RetrievalScope:
        """Create a child scope with equal or narrower authority.

        New filter fields add a constraint. Values for an existing field must be
        a subset of the parent allowlist. This method deliberately has no
        counterpart that expands a scope.
        """

        self.assert_active()
        if self.max_follow_ups is not None and self.follow_up_count >= self.max_follow_ups:
            raise ScopeError("retrieval scope has exhausted its follow-up budget")
        requested = _freeze_filters(narrowing_filters or {})
        child_filters = dict(self.filters)
        for field_name, child_values in requested.items():
            parent_values = self.filters.get(field_name)
            if parent_values is not None and not _typed_filter_values(child_values).issubset(
                _typed_filter_values(parent_values)
            ):
                raise ScopeError(f"child scope widens filter {field_name!r}")
            child_filters[field_name] = child_values
        return RetrievalScope(
            principal=self.principal,
            filters=_freeze_filters(child_filters),
            metadata=self.metadata,
            policy_version=self.policy_version,
            expires_at=self.expires_at,
            max_follow_ups=self.max_follow_ups,
            follow_up_count=self.follow_up_count + 1,
            parent_scope_id=self.scope_id,
        )

    def __repr__(self) -> str:
        """Return a useful trace-friendly representation without filter values."""

        filter_counts = {field_name: len(values) for field_name, values in self.filters.items()}
        scope_id = f"{self.scope_id[:8]}…"
        parent_scope_id = None
        if self.parent_scope_id is not None:
            parent_scope_id = f"{self.parent_scope_id[:8]}…"
        return (
            "RetrievalScope("
            f"principal={self.principal!r}, filters={filter_counts!r}, "
            f"metadata_keys={tuple(self.metadata)!r}, scope_id={scope_id!r}, "
            f"parent_scope_id={parent_scope_id!r})"
        )
