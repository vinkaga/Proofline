# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Data-only retrieval steps proposed by an untrusted planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ProposedStepError(ValueError):
    """Raised when an untrusted proposed step is not data-only."""

    def __init__(self, message: str, *, fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.fields = fields


@dataclass(frozen=True, slots=True)
class ProposedRetrievalStep:
    """A planner proposal that cannot carry retrieval authority.

    A valid proposal identifies only the query and, optionally, its logical
    parent. The scoped retriever obtains authority exclusively from the
    preceding trusted scope; it never reads authority from this object.
    """

    query: str
    parent_step_id: str | None = None

    @classmethod
    def from_untrusted(cls, value: Mapping[str, object]) -> ProposedRetrievalStep:
        """Parse an external proposal, rejecting every non-data field.

        Call this at the boundary where model or retrieved-document output is
        converted into a retrieval request. In particular, fields such as
        ``principal``, ``tenant_id``, ``filters``, and ``resource_id`` are not
        silently ignored: they reject the entire proposal.
        """

        allowed_fields = {"query", "parent_step_id"}
        unexpected = [
            field for field in value if not isinstance(field, str) or field not in allowed_fields
        ]
        if unexpected:
            rejected_fields = tuple(
                sorted(field if isinstance(field, str) else repr(field) for field in unexpected)
            )
            fields = ", ".join(rejected_fields)
            raise ProposedStepError(
                f"proposed step contains forbidden fields: {fields}",
                fields=rejected_fields,
            )

        query = value.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ProposedStepError("proposed step requires a non-empty query")

        parent_step_id = value.get("parent_step_id")
        if parent_step_id is not None and not isinstance(parent_step_id, str):
            raise ProposedStepError("parent_step_id must be a string or null")

        return cls(query=query, parent_step_id=parent_step_id)
