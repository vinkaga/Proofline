# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Deterministic request routing for the deliberately bounded reference host."""

from __future__ import annotations

from proofline_reference_demo.domain import RequestMode


def classify_request(query: str) -> RequestMode:
    """Classify one fixture query without asking a model to choose authority.

    This intentionally small classifier is an evaluated demo policy, not a
    general natural-language router. Permission-shaped wording takes priority;
    documentation terms route to public retrieval; all other queries use the
    caller's tenant-scoped knowledge path.
    """

    normalized = query.lower()
    if any(term in normalized for term in ("can ", "allowed", "permission", "view ", "access ")):
        return RequestMode.PERMISSION
    if any(
        term in normalized
        for term in ("what is", "how does", "listobjects", "documentation", "public policy")
    ):
        return RequestMode.PUBLIC_DOCUMENTATION
    return RequestMode.TENANT_KNOWLEDGE
