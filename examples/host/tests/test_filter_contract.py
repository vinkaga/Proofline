# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio

import pytest
from proofline_example_host import TrustedRequest, retriever

from proofline import ScopeError


def test_host_backend_rejects_unknown_filters_and_empty_allowlists_match_nothing() -> None:
    boundary = retriever()
    request = TrustedRequest(
        principal="user:ana", resource_ids=frozenset({"document:acme-rollout"})
    )
    initial = asyncio.run(boundary.search("rollout", context=request))

    empty = asyncio.run(
        boundary.follow_up_trusted(
            initial,
            "rollout",
            narrowing_filters={"resource_id": []},
        )
    )

    assert empty.items == ()
    initial = asyncio.run(boundary.search("rollout", context=request))
    with pytest.raises(ScopeError, match="does not support"):
        asyncio.run(
            boundary.follow_up_trusted(
                initial,
                "rollout",
                narrowing_filters={"visibility": ["public"]},
            )
        )
