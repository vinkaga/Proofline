# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""LlamaIndex retriever integration backed by the shared Proofline host boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from proofline_example_host import Evidence, TrustedRequest, retriever

from proofline import ProposedRetrievalStep, ScopedResults


@dataclass(frozen=True, slots=True)
class FollowUpResult:
    """LlamaIndex nodes plus the Proofline scope lineage for a planned hop."""

    nodes: tuple[NodeWithScore, ...]
    scope_id: str
    parent_scope_id: str | None


class ProoflineRetriever(BaseRetriever):
    """A LlamaIndex retriever whose authority comes from trusted construction context."""

    def __init__(self, request: TrustedRequest) -> None:
        super().__init__()
        self._request = request
        self._boundary = retriever()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = asyncio.run(self.search_scoped(query_bundle.query_str))
        return self._nodes(results.items)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Use the host's async path when LlamaIndex is already in an event loop."""

        results = await self.search_scoped(query_bundle.query_str)
        return self._nodes(results.items)

    async def search_scoped(self, query: str) -> ScopedResults[Evidence]:
        """Run a normal LlamaIndex retrieval while retaining its trusted scope."""

        return await self._boundary.search(query, context=self._request)

    async def follow_proposed(
        self,
        previous: ScopedResults[Evidence],
        planner_output: dict[str, object],
    ) -> FollowUpResult:
        """Run a model proposal under the prior LlamaIndex retrieval scope."""

        step = ProposedRetrievalStep.from_untrusted(planner_output)
        results = await self._boundary.follow_proposed(previous, step)
        return FollowUpResult(
            nodes=tuple(self._nodes(results.items)),
            scope_id=results.scope.scope_id,
            parent_scope_id=results.scope.parent_scope_id,
        )

    @staticmethod
    def _nodes(evidence: tuple[Evidence, ...]) -> list[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(text=item.text, id_=item.resource_id), score=1.0)
            for item in evidence
        ]
