# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministic citation and abstention composition for the reference host."""

from __future__ import annotations

from dataclasses import dataclass

from proofline_reference_demo.domain import Citation, RetrievalCandidate


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
    primary = candidates[0]
    if not primary.source_url or not primary.source_revision:
        raise ValueError("citation-ready candidates require source URL and revision")
    return ComposedResponse(
        text=f"Retrieved permitted evidence for: {query}",
        citations=(
            Citation.model_validate(
                {
                    "chunk_id": primary.chunk_id,
                    "source_url": primary.source_url,
                    "source_revision": primary.source_revision,
                }
            ),
        ),
        abstained=False,
    )
