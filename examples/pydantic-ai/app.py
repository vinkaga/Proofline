# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Pydantic AI tool integration: dependencies carry trusted host context."""

from __future__ import annotations

from proofline_example_host import TrustedRequest, retrieve
from pydantic_ai import Agent, RunContext

# The host supplies its production model at ``agent.run`` time.  Tests supply
# Pydantic AI's deterministic ``TestModel`` instead; this example never embeds
# a testing model in application configuration.
agent = Agent(deps_type=TrustedRequest)


async def retrieve_for_host(request: TrustedRequest, query: str) -> list[str]:
    """The host-tested retrieval implementation used by the Pydantic AI tool."""

    return [item.resource_id for item in await retrieve(request, query)]


@agent.tool
async def retrieve_evidence(ctx: RunContext[TrustedRequest], query: str) -> list[str]:
    """Retrieve evidence using trusted agent dependencies, never tool arguments for scope."""

    return await retrieve_for_host(ctx.deps, query)


async def follow_up(request: TrustedRequest, planner_output: dict[str, object]) -> list[str]:
    """Parse planner output at the same strict host boundary as any custom loop."""

    return [
        item.resource_id
        for item in await retrieve(request, "rollout approval", proposal=planner_output)
    ]
