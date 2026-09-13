# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministic citation and abstention composition for the reference host."""

from __future__ import annotations

from dataclasses import dataclass

from scopeanchor_reference_demo.domain import Citation, RetrievalCandidate


@dataclass(frozen=True, slots=True)
class ComposedResponse:
    """A bounded host response and the exact evidence it cites."""

    text: str
    citations: tuple[Citation, ...]
    abstained: bool


def compose_response(query: str, candidates: tuple[RetrievalCandidate, ...]) -> ComposedResponse:
    """Return a citation-backed fixture response or an explicit abstention."""

    if not candidates:
        return ComposedResponse(
            text="I do not have enough permitted evidence to answer that question.",
            citations=(),
            abstained=True,
        )
    if any(not candidate.source_url or not candidate.source_revision for candidate in candidates):
        raise ValueError("citation-ready candidates require source URL and revision")
    return ComposedResponse(
        text=f"Retrieved permitted evidence for: {query}",
        # Retain the complete evidence set considered by this deterministic
        # host.  A release case can therefore check that each required source
        # was actually cited instead of merely present somewhere in retrieval.
        citations=tuple(
            Citation.model_validate(
                {
                    "chunk_id": candidate.chunk_id,
                    "source_url": candidate.source_url,
                    "source_revision": candidate.source_revision,
                }
            )
            for candidate in candidates
        ),
        abstained=False,
    )
