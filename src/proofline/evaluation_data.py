# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Validate reviewed evaluation data before it becomes a release contract."""

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, model_validator

from proofline.domain import RequestMode


class ExpectedOutcome(StrEnum):
    """The observable behavior expected for one evaluation case."""

    CITED_ANSWER = "cited_answer"
    ABSTAIN = "abstain"
    ALLOW = "allow"
    DENY = "deny"


class EvaluationCaseSpec(BaseModel):
    """One reviewed case with the required contract for its request mode."""

    id: str
    mode: RequestMode
    principal: str
    query: str
    expected: ExpectedOutcome
    tenant: str | None = None
    relation: str | None = None
    resource: str | None = None
    required_tool: str | None = None
    required_sources: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    retry_eligible: bool = False

    @model_validator(mode="after")
    def validate_mode_contract(self) -> Self:
        if self.mode is RequestMode.PERMISSION:
            if not (self.relation and self.resource and self.required_tool):
                raise ValueError("permission cases require relation, resource, and required_tool")
            if self.expected not in {ExpectedOutcome.ALLOW, ExpectedOutcome.DENY}:
                raise ValueError("permission cases must expect allow or deny")
        elif self.mode is RequestMode.TENANT_KNOWLEDGE and not self.tenant:
            raise ValueError("tenant knowledge cases require tenant")
        elif self.expected in {ExpectedOutcome.ALLOW, ExpectedOutcome.DENY}:
            raise ValueError("only permission cases may expect allow or deny")
        if self.mode is RequestMode.PUBLIC_DOCUMENTATION and self.required_resources:
            raise ValueError("public documentation cases cannot require protected resources")
        if self.expected is ExpectedOutcome.CITED_ANSWER and not (
            self.required_sources or self.required_resources
        ):
            raise ValueError("cited answers require expected source or resource evidence")
        return self


class EvaluationSuite(BaseModel):
    """A versioned set of reviewed cases used by later release gates."""

    version: str
    cases: tuple[EvaluationCaseSpec, ...]

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case IDs must be unique")
        return self


def load_evaluation_suite(path: Path) -> EvaluationSuite:
    """Load and validate reviewed evaluation YAML from a versioned path."""

    return EvaluationSuite.model_validate(yaml.safe_load(path.read_text()))
