# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Define the typed contracts that preserve Proofline's access boundary.

These models give authorization, retrieval, agent, trace, and evaluation code a
shared vocabulary. In particular, they make a principal, resolved access scope,
candidate, citation, and tool call explicit and inspectable rather than passing
unstructured dictionaries between system layers.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RequestMode(StrEnum):
    """The three request paths that Proofline treats differently."""

    PUBLIC_DOCUMENTATION = "public_documentation"
    TENANT_KNOWLEDGE = "tenant_knowledge"
    PERMISSION = "permission"


class Principal(BaseModel):
    """The caller whose access scope is resolved before protected retrieval."""

    id: str = Field(pattern=r"^user:[a-z0-9_-]+$")


class AccessScope(BaseModel):
    """Resources a principal may use for one tenant-scoped retrieval request."""

    tenant_id: str = Field(pattern=r"^tenant:[a-z0-9_-]+$")
    resource_ids: tuple[str, ...] = ()


class ScopedResource(BaseModel):
    """A resource identifier whose meaning is confined to one tenant."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(pattern=r"^tenant:[a-z0-9_-]+$")
    resource_id: str = Field(min_length=1)


class Citation(BaseModel):
    """A source reference that can be checked against retrieved evidence."""

    chunk_id: str
    source_url: HttpUrl
    source_revision: str


class ToolCall(BaseModel):
    """A redacted record of a typed tool interaction."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None


class RetrievalCandidate(BaseModel):
    """One permitted candidate returned by a retrieval implementation."""

    chunk_id: str
    resource_id: str
    rank: int = Field(ge=1)
    score: float


class InteractionTrace(BaseModel):
    """The minimum inspectable record for an evaluated interaction."""

    case_id: str
    request_mode: RequestMode
    principal: Principal
    access_scope: AccessScope | None = None
    candidates: tuple[RetrievalCandidate, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    citations: tuple[Citation, ...] = ()
    abstained: bool = False


class EvaluationCase(BaseModel):
    """A versioned, hand-authored quality case."""

    id: str
    request_mode: RequestMode
    principal: Principal
    query: str = Field(min_length=1)
    expected_abstention: bool = False
    required_tool: str | None = None
    model_config = ConfigDict(frozen=True)
