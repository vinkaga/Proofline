# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify release-suite contracts and reject incomplete evaluation cases."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proofline.evaluation_data import EvaluationSuite, load_evaluation_suite


def test_release_suite_is_typed_and_unique() -> None:
    suite = load_evaluation_suite(Path("data/eval/release-v0.yaml"))

    assert len(suite.cases) == 25
    assert len({case.id for case in suite.cases}) == 25
    assert suite.cases[0].required_sources == ("perform-check",)


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "p",
            "mode": "permission",
            "principal": "user:ana",
            "query": "q",
            "expected": "allow",
        },
        {
            "id": "t",
            "mode": "tenant_knowledge",
            "principal": "user:ana",
            "query": "q",
            "expected": "cited_answer",
        },
        {
            "id": "a",
            "mode": "public_documentation",
            "principal": "user:ana",
            "query": "q",
            "expected": "allow",
        },
        {
            "id": "c",
            "mode": "public_documentation",
            "principal": "user:ana",
            "query": "q",
            "expected": "cited_answer",
        },
    ],
)
def test_invalid_case_contracts_are_rejected(case: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate({"version": "test", "cases": [case]})
