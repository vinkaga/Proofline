import asyncio

import pytest
from app import ProoflineRetriever
from proofline_example_host import TrustedRequest

from proofline import ProposedStepError


def test_llamaindex_retriever_preserves_the_host_scope() -> None:
    retriever = ProoflineRetriever(TrustedRequest("user:ana", frozenset({"document:acme-rollout"})))

    nodes = retriever.retrieve("rollout approval")

    assert [node.node.node_id for node in nodes] == ["document:acme-rollout"]
    assert retriever.retrieve("beta rollout") == []


def test_llamaindex_async_retriever_works_inside_an_event_loop() -> None:
    retriever = ProoflineRetriever(TrustedRequest("user:ana", frozenset({"document:acme-rollout"})))

    nodes = asyncio.run(retriever.aretrieve("rollout approval"))

    assert [node.node.node_id for node in nodes] == ["document:acme-rollout"]


def test_llamaindex_planned_follow_up_preserves_scope() -> None:
    retriever = ProoflineRetriever(TrustedRequest("user:ana", frozenset({"document:acme-rollout"})))

    async def run_follow_up() -> None:
        initial = await retriever.search_scoped("rollout approval")
        follow_up = await retriever.follow_proposed(initial, {"query": "release-manager approval"})
        assert [node.node.node_id for node in follow_up.nodes] == ["document:acme-rollout"]
        assert follow_up.parent_scope_id == initial.scope.scope_id

        with pytest.raises(ProposedStepError, match="resource_id"):
            await retriever.follow_proposed(
                initial,
                {"query": "beta rollout", "resource_id": "document:beta-rollout"},
            )

    asyncio.run(run_follow_up())
