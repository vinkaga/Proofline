# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Immutable, capability-attenuating retrieval scopes."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
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
ScopeCheckpoint: TypeAlias = Mapping[str, object]
_SCOPE_CHECKPOINT_VERSION = 1


class _FrozenFilterValues(frozenset[FilterAtom]):
    """Validated filter values with cached exact-type membership keys."""

    _typed_values: frozenset[tuple[type[object], FilterAtom]]

    def __new__(
        cls,
        values: Iterable[FilterAtom],
        typed_values: frozenset[tuple[type[object], FilterAtom]],
    ) -> _FrozenFilterValues:
        frozen = super().__new__(cls, values)
        frozen._typed_values = typed_values
        return frozen


class _FrozenFilters(Mapping[str, frozenset[FilterAtom]]):
    """An immutable mapping that marks already validated scope filters."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, frozenset[FilterAtom]]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> frozenset[FilterAtom]:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class _FrozenMetadata(Mapping[str, str]):
    """An immutable mapping that marks already validated scope metadata."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _values_compare_equal_across_types(
    value: FilterAtom,
    *,
    integers: set[int],
    integral_floats: set[int],
    booleans: set[bool],
) -> bool:
    """Return whether ``value`` equals a previously seen value of another type.

    Filter atoms have only five supported types.  The only cross-type equality
    possible between them is the numeric relationship among ``bool``, ``int``,
    and finite integral ``float`` values.  Tracking that relationship directly
    avoids pairwise comparisons as an allowlist grows.
    """

    if isinstance(value, bool):
        return int(value) in integers or int(value) in integral_floats
    if isinstance(value, int):
        return (value in (0, 1) and bool(value) in booleans) or value in integral_floats
    if isinstance(value, float) and value.is_integer():
        integer_value = int(value)
        return integer_value in integers or (
            integer_value in (0, 1) and bool(integer_value) in booleans
        )
    return False


def _freeze_filter_values(values: Iterable[FilterAtom]) -> frozenset[FilterAtom]:
    """Validate and freeze filter atoms without conflating scalar types."""

    validated: list[FilterAtom] = []
    typed_values: set[tuple[type[object], FilterAtom]] = set()
    integers: set[int] = set()
    integral_floats: set[int] = set()
    booleans: set[bool] = set()
    for value in values:
        if type(value) not in (str, int, float, bool, type(None)):
            raise ScopeError("filter values must be strings, integers, floats, booleans, or null")
        if isinstance(value, float) and not isfinite(value):
            raise ScopeError("filter float values must be finite")
        typed_value = (type(value), value)
        if typed_value in typed_values:
            continue
        if _values_compare_equal_across_types(
            value,
            integers=integers,
            integral_floats=integral_floats,
            booleans=booleans,
        ):
            raise ScopeError("filter values that compare equal must have the same type")
        validated.append(value)
        typed_values.add(typed_value)
        if isinstance(value, bool):
            booleans.add(value)
        elif isinstance(value, int):
            integers.add(value)
        elif isinstance(value, float) and value.is_integer():
            integral_floats.add(int(value))
    return _FrozenFilterValues(validated, frozenset(typed_values))


def _typed_filter_values(
    values: Iterable[FilterAtom],
) -> frozenset[tuple[type[object], FilterAtom]]:
    """Return filter values keyed by their exact runtime scalar type."""

    if isinstance(values, _FrozenFilterValues):
        return values._typed_values
    return frozenset((type(value), value) for value in values)


class ScopeError(ValueError):
    """Raised when a scope is invalid or attempts to gain authority."""


class ScopeExpiredError(ScopeError):
    """Raised when a scope is used after its trusted expiry time."""


class ScopeCheckpointError(ScopeError):
    """Raised when a persisted scope checkpoint is malformed or misbound."""


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
        if (type(candidate_value), candidate_value) not in _typed_filter_values(permitted_values):
            return False
    return True


def _freeze_filters(filters: Mapping[str, Iterable[FilterAtom]]) -> ScopeFilters:
    if isinstance(filters, _FrozenFilters):
        return filters
    frozen: dict[str, frozenset[FilterAtom]] = {}
    for field_name, values in filters.items():
        if not isinstance(field_name, str) or not field_name:
            raise ScopeError("filter names must be non-empty strings")
        if isinstance(values, (str, bytes)):
            raise ScopeError("filter values must be an iterable of scalar values, not a string")
        frozen[field_name] = _freeze_filter_values(values)
    return _FrozenFilters(frozen)


def _freeze_metadata(metadata: Mapping[str, str]) -> ScopeMetadata:
    if isinstance(metadata, _FrozenMetadata):
        return metadata
    frozen: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ScopeError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise ScopeError("metadata values must be strings")
        frozen[key] = value
    return _FrozenMetadata(frozen)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ScopeCheckpointError(f"scope checkpoint {name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ScopeCheckpointError(f"scope checkpoint {name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object], *, name: str, keys: frozenset[str]
) -> None:
    if set(value) != keys:
        raise ScopeCheckpointError(f"scope checkpoint {name} has unsupported or missing fields")


def _checkpoint_binding(binding: Mapping[str, str]) -> dict[str, str]:
    try:
        return dict(_freeze_metadata(binding))
    except ScopeError as error:
        raise ScopeCheckpointError(str(error)) from error


def _encode_filter_atom(value: FilterAtom) -> dict[str, FilterAtom | str]:
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "value": value}
    return {"type": "str", "value": value}


def _decode_filter_atom(value: object) -> FilterAtom:
    encoded = _require_mapping(value, name="filter value")
    _require_exact_keys(encoded, name="filter value", keys=frozenset({"type", "value"}))
    atom_type = encoded["type"]
    atom_value = encoded["value"]
    if atom_type == "null" and atom_value is None:
        return None
    if atom_type == "bool" and type(atom_value) is bool:
        return atom_value
    if atom_type == "int" and type(atom_value) is int:
        return atom_value
    if atom_type == "float" and type(atom_value) is float and isfinite(atom_value):
        return atom_value
    if atom_type == "str" and type(atom_value) is str:
        return atom_value
    raise ScopeCheckpointError("scope checkpoint filter value has an invalid type or value")


def _decode_filters(value: object) -> dict[str, list[FilterAtom]]:
    encoded_filters = _require_mapping(value, name="filters")
    filters: dict[str, list[FilterAtom]] = {}
    for field_name, encoded_values in encoded_filters.items():
        if not isinstance(encoded_values, list):
            raise ScopeCheckpointError("scope checkpoint filter values must be lists")
        filters[field_name] = [
            _decode_filter_atom(encoded_value) for encoded_value in encoded_values
        ]
    return filters


def _decode_string_mapping(value: object, *, name: str) -> dict[str, str]:
    encoded = _require_mapping(value, name=name)
    if not all(type(item) is str for item in encoded.values()):
        raise ScopeCheckpointError(f"scope checkpoint {name} values must be strings")
    return {key: item for key, item in encoded.items() if type(item) is str}


def _decode_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ScopeCheckpointError("scope checkpoint expiry must be an ISO 8601 string or null")
    try:
        expiry = datetime.fromisoformat(value)
    except ValueError as error:
        raise ScopeCheckpointError("scope checkpoint expiry must be ISO 8601") from error
    if expiry.tzinfo is None:
        raise ScopeCheckpointError("scope checkpoint expiry must be timezone-aware")
    return expiry


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

    def is_attenuation_of(self, parent: RetrievalScope) -> bool:
        """Return whether this scope grants no more filter authority than ``parent``.

        A child may add constraints. For every parent filter field, it must
        retain that field and use an exact-type subset of the parent's values.
        Principal equality is required. Lineage IDs, metadata, and policy
        labels are not authority; callers that need to compare them should do
        so separately.
        """

        if self.principal != parent.principal:
            return False
        for field_name, parent_values in parent.filters.items():
            child_values = self.filters.get(field_name)
            if child_values is None or not _typed_filter_values(child_values).issubset(
                _typed_filter_values(parent_values)
            ):
                return False
        return True

    def to_checkpoint(self, *, binding: Mapping[str, str]) -> dict[str, object]:
        """Return a versioned, JSON-safe checkpoint for trusted host storage.

        ``binding`` must be built by trusted host code from stable caller and
        task identifiers. It is checked during restoration to prevent a
        checkpoint for one authenticated task from being resumed in another.
        This payload is not a bearer credential: store it where callers cannot
        alter it, and never construct it from model or client input.
        """

        self.assert_active()
        return {
            "version": _SCOPE_CHECKPOINT_VERSION,
            "binding": _checkpoint_binding(binding),
            "scope": {
                "principal": self.principal,
                "filters": {
                    field_name: [_encode_filter_atom(value) for value in values]
                    for field_name, values in self.filters.items()
                },
                "metadata": dict(self.metadata),
                "policy_version": self.policy_version,
                "expires_at": self.expires_at.isoformat() if self.expires_at is not None else None,
                "max_follow_ups": self.max_follow_ups,
                "follow_up_count": self.follow_up_count,
                "scope_id": self.scope_id,
                "parent_scope_id": self.parent_scope_id,
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: ScopeCheckpoint,
        *,
        binding: Mapping[str, str],
    ) -> RetrievalScope:
        """Restore a trusted checkpoint after validating its format and binding.

        Do not call this on client, model, or document data. Applications
        should normally use :meth:`proofline.ScopedRetriever.resume`, which
        additionally re-resolves current authorization before it returns a
        retriever capable of dispatching searches.
        """

        encoded = _require_mapping(checkpoint, name="payload")
        _require_exact_keys(
            encoded,
            name="payload",
            keys=frozenset({"version", "binding", "scope"}),
        )
        if type(encoded["version"]) is not int or encoded["version"] != _SCOPE_CHECKPOINT_VERSION:
            raise ScopeCheckpointError("scope checkpoint has an unsupported version")
        if _decode_string_mapping(
            encoded["binding"], name="binding"
        ) != _checkpoint_binding(binding):
            raise ScopeCheckpointError(
                "scope checkpoint does not match the trusted caller/task binding"
            )

        scope_data = _require_mapping(encoded["scope"], name="scope")
        _require_exact_keys(
            scope_data,
            name="scope",
            keys=frozenset(
                {
                    "principal",
                    "filters",
                    "metadata",
                    "policy_version",
                    "expires_at",
                    "max_follow_ups",
                    "follow_up_count",
                    "scope_id",
                    "parent_scope_id",
                }
            ),
        )
        principal = scope_data["principal"]
        policy_version = scope_data["policy_version"]
        scope_id = scope_data["scope_id"]
        parent_scope_id = scope_data["parent_scope_id"]
        max_follow_ups = scope_data["max_follow_ups"]
        follow_up_count = scope_data["follow_up_count"]
        if type(principal) is not str or not principal:
            raise ScopeCheckpointError("scope checkpoint principal must be a non-empty string")
        if type(policy_version) is not str:
            raise ScopeCheckpointError("scope checkpoint policy_version must be a string")
        if type(scope_id) is not str or not scope_id:
            raise ScopeCheckpointError("scope checkpoint scope_id must be a non-empty string")
        if parent_scope_id is not None and (
            type(parent_scope_id) is not str or not parent_scope_id
        ):
            raise ScopeCheckpointError(
                "scope checkpoint parent_scope_id must be a non-empty string or null"
            )
        if max_follow_ups is not None and type(max_follow_ups) is not int:
            raise ScopeCheckpointError("scope checkpoint max_follow_ups must be an integer or null")
        if type(follow_up_count) is not int:
            raise ScopeCheckpointError("scope checkpoint follow_up_count must be an integer")

        try:
            scope = cls(
                principal=principal,
                filters=_freeze_filters(_decode_filters(scope_data["filters"])),
                metadata=_decode_string_mapping(scope_data["metadata"], name="metadata"),
                policy_version=policy_version,
                expires_at=_decode_optional_datetime(scope_data["expires_at"]),
                max_follow_ups=max_follow_ups,
                follow_up_count=follow_up_count,
                scope_id=scope_id,
                parent_scope_id=parent_scope_id,
            )
        except ScopeError as error:
            raise ScopeCheckpointError(
                f"scope checkpoint contains an invalid scope: {error}"
            ) from error
        scope.assert_active()
        return scope

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
        updated_filters: dict[str, frozenset[FilterAtom]] | None = None
        for field_name, child_values in requested.items():
            parent_values = self.filters.get(field_name)
            if parent_values is not None and not _typed_filter_values(child_values).issubset(
                _typed_filter_values(parent_values)
            ):
                raise ScopeError(f"child scope widens filter {field_name!r}")
            if updated_filters is None:
                updated_filters = dict(self.filters)
            updated_filters[field_name] = child_values
        return RetrievalScope(
            principal=self.principal,
            filters=(
                _FrozenFilters(updated_filters)
                if updated_filters is not None
                else self.filters
            ),
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
