# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio
import hashlib
import json

import pytest

from proofline_reference_demo.hotpot_evaluation import (
    evaluate_hotpotqa_retrieval,
    evaluate_hotpotqa_scope_overlay,
    validate_hotpotqa_evaluation,
    validate_hotpotqa_scope_overlay,
)
from proofline_reference_demo.hotpotqa import (
    HotpotManifest,
    build_overlay,
    evaluate_overlay,
    load_cases,
)


def _dataset() -> bytes:
    return json.dumps(
        [
            {
                "_id": "b",
                "type": "bridge",
                "question": "What evidence gives the answer?",
                "answer": "Example",
                "context": [["Evidence", ["An answer."]], ["Distractor", ["Noise."]]],
                "supporting_facts": [["Evidence", 0]],
            },
            {
                "_id": "a",
                "type": "comparison",
                "question": "Ignored?",
                "answer": "No",
                "context": [["Ignored", ["No."]]],
                "supporting_facts": [["Ignored", 0]],
            },
        ]
    ).encode()


def _manifest(payload: bytes) -> HotpotManifest:
    return HotpotManifest.model_validate(
        {
            "version": "test",
            "source": {
                "name": "test",
                "url": "https://example.test/hotpot.json",
                "license": "CC-BY-SA-4.0",
                "citation": "test",
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "subset": {"size": 1, "selection": "bridge"},
        }
    )


def test_verified_hotpotqa_subset_and_overlay_preserve_the_source_data(tmp_path) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload)
    cases = load_cases(path, _manifest(payload))
    overlays = build_overlay(cases)
    report = evaluate_overlay(cases, overlays, hashlib.sha256(payload).hexdigest())
    retrieval = evaluate_hotpotqa_retrieval(cases, report, limit=2)

    assert cases[0].case_id == "b"
    assert cases[0].supporting_titles == {"Evidence"}
    assert overlays[0].poisoned_proposal["resource_id"] == "hotpot:b:1"
    assert report.clean_supporting_coverage == 1
    assert report.poisoned_rejection_rate == 1
    assert report.benign_acceptance_rate == 1
    assert retrieval.supporting_title_recall_at_k == 1
    assert retrieval.answer_evidence_coverage_at_k == 1
    validate_hotpotqa_evaluation(retrieval)
    scope_traces = asyncio.run(evaluate_hotpotqa_scope_overlay(cases, overlays, limit=2))
    assert scope_traces[0].candidate_resource_ids == ("hotpot:b:0",)
    assert scope_traces[0].poisoned_proposal_rejected
    validate_hotpotqa_scope_overlay(scope_traces)


def test_hotpotqa_loader_rejects_a_hash_mismatch(tmp_path) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_cases(path, _manifest(payload))
