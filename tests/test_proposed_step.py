# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

from collections.abc import Mapping
from typing import cast

import pytest

from proofline import ProposedRetrievalStep, ProposedStepError


@pytest.mark.parametrize("field", ["principal", "tenant_id", "resource_id", "filters", "scope"])
def test_untrusted_step_rejects_authority_bearing_fields(field: str) -> None:
    with pytest.raises(ProposedStepError, match="forbidden"):
        ProposedRetrievalStep.from_untrusted({"query": "follow up", field: "attacker-value"})


def test_untrusted_step_requires_data_only_valid_shape() -> None:
    step = ProposedRetrievalStep.from_untrusted(
        {"query": "compare the two policies", "parent_step_id": "hop-1"}
    )

    assert step.query == "compare the two policies"
    assert step.parent_step_id == "hop-1"


def test_untrusted_step_rejects_non_string_field_names() -> None:
    with pytest.raises(ProposedStepError, match="forbidden"):
        ProposedRetrievalStep.from_untrusted(
            cast(Mapping[str, object], {"query": "follow up", 1: "unexpected"})
        )
