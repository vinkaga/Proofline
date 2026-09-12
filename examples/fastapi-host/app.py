# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Application-style HTTP search host with trusted authentication dependency."""

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from proofline_example_host import TrustedRequest, retrieve, retrieve_results

from proofline import ProposedStepError

app = FastAPI(title="Proofline document search")

_CALLER_SCOPES = {
    "ana": frozenset({"document:acme-rollout"}),
    "ben": frozenset(),
}


def authenticated_request(x_principal: str = Header()) -> TrustedRequest:
    """Replace this demo lookup with the application's authentication layer."""

    resource_ids = _CALLER_SCOPES.get(x_principal)
    if resource_ids is None:
        raise HTTPException(status_code=401, detail="unknown principal")
    return TrustedRequest(principal=f"user:{x_principal}", resource_ids=resource_ids)


AuthenticatedRequest = Annotated[TrustedRequest, Depends(authenticated_request)]


@app.get("/search")
async def search(query: str, request: AuthenticatedRequest) -> dict[str, list[str]]:
    evidence = await retrieve(request, query)
    return {"resource_ids": [item.resource_id for item in evidence]}


@app.post("/search/follow-up")
async def follow_up(
    planner_output: dict[str, object], request: AuthenticatedRequest
) -> dict[str, object]:
    """Apply one untrusted planner proposal under the authenticated scope only."""

    try:
        results = await retrieve_results(request, "rollout approval", proposal=planner_output)
    except ProposedStepError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "resource_ids": [item.resource_id for item in results.items],
        "scope_id": results.scope.scope_id,
        "parent_scope_id": results.scope.parent_scope_id,
    }
