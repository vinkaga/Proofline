# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Create inspectable traces for the implemented retrieval boundary."""

from proofline.domain import InteractionTrace, Principal, RequestMode
from proofline.retrieval import RetrievalResult


def trace_tenant_retrieval(
    case_id: str,
    principal: Principal,
    result: RetrievalResult,
) -> InteractionTrace:
    """Record the resolved scope and only the candidates that survived it."""

    if result.access_scope is None:
        raise ValueError("tenant retrieval traces require an access scope")
    return InteractionTrace(
        case_id=case_id,
        request_mode=RequestMode.TENANT_KNOWLEDGE,
        principal=principal,
        access_scope=result.access_scope,
        candidates=result.candidates,
        context_chunk_ids=tuple(candidate.chunk_id for candidate in result.candidates),
    )
