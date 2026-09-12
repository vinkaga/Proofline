import asyncio

import pytest
from proofline import ProposedStepError, ScopeError

from app import RequestContext, build_retriever, retrieve


def test_authorized_retrieval_and_data_only_follow_up_preserve_scope() -> None:
    context = RequestContext(
        principal="user:ana",
        authorized_resource_ids=frozenset(
            {"document:public-policy", "document:acme-rollout"}
        ),
    )

    documents = asyncio.run(
        retrieve(
            "rollout approval",
            context=context,
            planner_output={"query": "release-manager approval"},
        )
    )

    assert {document.resource_id for document in documents} == {
        "document:public-policy",
        "document:acme-rollout",
    }


def test_scope_bearing_planner_output_is_rejected_not_ignored() -> None:
    context = RequestContext(
        principal="user:ana",
        authorized_resource_ids=frozenset({"document:acme-rollout"}),
    )

    with pytest.raises(ProposedStepError, match="resource_id"):
        asyncio.run(
            retrieve(
                "rollout approval",
                context=context,
                planner_output={
                    "query": "beta rollout",
                    "resource_id": "document:beta-rollout",
                },
            )
        )


def test_backend_rejects_unknown_filters_and_empty_allowlists_match_nothing() -> None:
    boundary = build_retriever()
    context = RequestContext(
        principal="user:ana", authorized_resource_ids=frozenset({"document:acme-rollout"})
    )
    initial = asyncio.run(boundary.search("rollout", context=context))

    empty = asyncio.run(
        boundary.follow_up_trusted(
            initial,
            "rollout",
            narrowing_filters={"resource_id": []},
        )
    )

    assert empty.items == ()
    initial = asyncio.run(boundary.search("rollout", context=context))
    with pytest.raises(ScopeError, match="does not support"):
        asyncio.run(
            boundary.follow_up_trusted(
                initial,
                "rollout",
                narrowing_filters={"visibility": ["public"]},
            )
        )
