# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal

import asyncio
import hashlib
import json
from dataclasses import replace

import pytest
from scopeanchor import RetrievalScope
from typer.testing import CliRunner

import scopeanchor_reference_demo.cli as cli
import scopeanchor_reference_demo.hotpot_evaluation as hotpot_evaluation
from scopeanchor_reference_demo.domain import RetrievalCandidate
from scopeanchor_reference_demo.hotpot_evaluation import (
    evaluate_hotpotqa_retrieval,
    evaluate_hotpotqa_scope_controls,
    evaluate_hotpotqa_scope_overlay,
    validate_hotpotqa_evaluation,
    validate_hotpotqa_scope_controls,
    validate_hotpotqa_scope_overlay,
)
from scopeanchor_reference_demo.hotpotqa import (
    HotpotManifest,
    build_overlay,
    evaluate_overlay,
    load_cases,
)
from scopeanchor_reference_demo.retrieval import DocumentChunk

runner = CliRunner()


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
            "retrieval_quality_gate": {
                "supporting_title_recall_at_k": 1.0,
                "answer_evidence_coverage_at_k": 1.0,
            },
            "overlay": {
                "version": "hotpotqa-overlay-v1",
                "caller_tenant": "tenant:acme",
                "protected_tenant": "tenant:beta",
                "allowed_resource_assignment": "supporting-title-resources",
                "protected_resource_assignment": "first-non-supporting-context-resource",
                "poisoned_proposal": "query-plus-resource_id",
                "benign_proposal": "query-only",
            },
        }
    )


def test_verified_hotpotqa_subset_and_overlay_preserve_the_source_data(tmp_path) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload)
    cases = load_cases(path, _manifest(payload))
    overlays = build_overlay(cases, _manifest(payload).overlay)
    report = evaluate_overlay(cases, overlays, hashlib.sha256(payload).hexdigest())
    retrieval = evaluate_hotpotqa_retrieval(cases, report, limit=2)

    assert cases[0].case_id == "b"
    assert cases[0].supporting_titles == {"Evidence"}
    assert overlays[0].poisoned_proposal["resource_id"] == "hotpot:b:1"
    assert report.case_count == 1
    assert retrieval.supporting_title_recall_at_k == 1
    assert retrieval.answer_evidence_coverage_at_k == 1
    validate_hotpotqa_evaluation(retrieval, _manifest(payload).retrieval_quality_gate)
    scope_traces = asyncio.run(evaluate_hotpotqa_scope_overlay(cases, overlays, limit=2))
    assert scope_traces[0].candidate_resource_ids == ("hotpot:b:0",)
    assert scope_traces[0].insecure_follow_up_resource_ids == ("hotpot:b:1",)
    assert scope_traces[0].insecure_scope_input_accepted
    assert scope_traces[0].acl_only_scope_input_accepted
    assert scope_traces[0].poisoned_proposal_rejected
    validate_hotpotqa_scope_overlay(scope_traces)
    controls = evaluate_hotpotqa_scope_controls(scope_traces)
    insecure, acl_only, scoped_policy = controls.configurations
    assert insecure.unauthorized_exposure_rate == 1
    assert acl_only.unauthorized_exposure_rate == 0
    assert scoped_policy.scope_bearing_input_acceptance_rate == 0
    assert scoped_policy.rejected_before_retrieval_rate == 1
    assert scoped_policy.scope_lineage_complete_rate == 1
    validate_hotpotqa_scope_controls(controls)

    regressed = replace(
        controls,
        configurations=(
            controls.configurations[0],
            controls.configurations[1],
            replace(controls.configurations[2], rejected_before_retrieval_rate=0.0),
        ),
    )
    with pytest.raises(ValueError, match="did not reject before retrieval"):
        validate_hotpotqa_scope_controls(regressed)

    with pytest.raises(ValueError, match="supporting-title recall"):
        validate_hotpotqa_evaluation(
            replace(retrieval, supporting_title_recall_at_k=0.0),
            _manifest(payload).retrieval_quality_gate,
        )
    with pytest.raises(ValueError, match="complete supporting-evidence coverage"):
        validate_hotpotqa_evaluation(
            replace(retrieval, answer_evidence_coverage_at_k=0.0),
            _manifest(payload).retrieval_quality_gate,
        )


def test_hotpotqa_loader_rejects_a_hash_mismatch(tmp_path) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload + b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_cases(path, _manifest(payload))


def test_hotpot_backend_applies_tenant_and_resource_filters_conjunctively() -> None:
    backend = hotpot_evaluation._backend_for_chunks(
        (
            DocumentChunk("acme", "document:shared", "tenant:acme", "shared evidence"),
            DocumentChunk("beta", "document:shared", "tenant:beta", "shared evidence"),
        )
    )
    filters = RetrievalScope.root(
        principal="user:benchmark",
        filters={"tenant_id": ["tenant:acme"], "resource_id": ["document:shared"]},
    ).filters

    candidates = backend("shared", filters=filters, limit=10)

    assert [candidate.chunk_id for candidate in candidates] == ["acme"]


def test_hotpotqa_utility_gate_rejects_an_actual_ranking_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload)
    cases = load_cases(path, _manifest(payload))
    overlay = evaluate_overlay(
        cases, build_overlay(cases, _manifest(payload).overlay), hashlib.sha256(payload).hexdigest()
    )

    monkeypatch.setattr(
        hotpot_evaluation.AccessGatedBm25Retriever,
        "_rank",
        staticmethod(lambda query, chunks, limit: ()),
    )
    report = evaluate_hotpotqa_retrieval(cases, overlay, limit=2)

    assert report.supporting_title_recall_at_k == 0
    with pytest.raises(ValueError, match="supporting-title recall"):
        validate_hotpotqa_evaluation(report, _manifest(payload).retrieval_quality_gate)


def test_hotpotqa_scope_gate_detects_an_actual_benign_follow_up_exposure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload)
    cases = load_cases(path, _manifest(payload))
    overlays = build_overlay(cases, _manifest(payload).overlay)
    original_backend_for_chunks = hotpot_evaluation._backend_for_chunks

    def leaking_backend_for_chunks(chunks):  # noqa: ANN001
        backend = original_backend_for_chunks(chunks)
        call_count = 0

        def leaking_backend(query, *, filters, limit):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            candidates = backend(query, filters=filters, limit=limit)
            if call_count != 2:
                return candidates
            protected = next(
                chunk for chunk in chunks if chunk.resource_id not in filters["resource_id"]
            )
            return (
                *candidates,
                RetrievalCandidate(
                    chunk_id=protected.id,
                    resource_id=protected.resource_id,
                    tenant_id=protected.tenant_id,
                    rank=len(candidates) + 1,
                    score=0.0,
                ),
            )

        return leaking_backend

    monkeypatch.setattr(hotpot_evaluation, "_backend_for_chunks", leaking_backend_for_chunks)
    traces = asyncio.run(evaluate_hotpotqa_scope_overlay(cases, overlays, limit=2))

    assert traces[0].protected_resource_id in traces[0].benign_follow_up_resource_ids
    with pytest.raises(ValueError, match="out-of-scope evidence"):
        validate_hotpotqa_scope_overlay(traces)

    controls = evaluate_hotpotqa_scope_controls(traces)
    scoped_policy = controls.configurations[-1]
    assert scoped_policy.unauthorized_exposure_rate == 1
    with pytest.raises(ValueError, match="scoped policy exposed protected evidence"):
        validate_hotpotqa_scope_controls(controls)


def test_hotpotqa_control_metrics_follow_executed_insecure_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    payload = _dataset()
    path = tmp_path / "hotpot.json"
    path.write_bytes(payload)
    cases = load_cases(path, _manifest(payload))
    overlays = build_overlay(cases, _manifest(payload).overlay)

    monkeypatch.setattr(
        hotpot_evaluation,
        "_follow_insecure",
        lambda chunks, proposal, limit: (False, ()),
    )
    traces = asyncio.run(evaluate_hotpotqa_scope_overlay(cases, overlays, limit=2))
    controls = evaluate_hotpotqa_scope_controls(traces)
    insecure = controls.configurations[0]

    assert insecure.scope_bearing_input_acceptance_rate == 0
    assert insecure.unauthorized_exposure_rate == 0
    with pytest.raises(ValueError, match="did not expose protected evidence"):
        validate_hotpotqa_scope_controls(controls)


def test_hotpotqa_cli_reports_all_three_controls(tmp_path) -> None:
    payload = _dataset()
    dataset = tmp_path / "hotpot.json"
    dataset.write_bytes(payload)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            (
                "version: test",
                "source:",
                "  name: test",
                "  url: https://example.test/hotpot.json",
                "  license: CC-BY-SA-4.0",
                "  citation: test",
                f"  sha256: {hashlib.sha256(payload).hexdigest()}",
                "subset:",
                "  size: 1",
                "  selection: bridge",
                "retrieval_quality_gate:",
                "  supporting_title_recall_at_k: 1.0",
                "  answer_evidence_coverage_at_k: 1.0",
                "overlay:",
                "  version: hotpotqa-overlay-v1",
                "  caller_tenant: tenant:acme",
                "  protected_tenant: tenant:beta",
                "  allowed_resource_assignment: supporting-title-resources",
                "  protected_resource_assignment: first-non-supporting-context-resource",
                "  poisoned_proposal: query-plus-resource_id",
                "  benign_proposal: query-only",
            )
        )
    )

    report_output = tmp_path / "hotpot-report.json"
    traces_output = tmp_path / "hotpot-traces.jsonl"
    result = runner.invoke(
        cli.app,
        [
            "evaluate-hotpotqa",
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--output",
            str(report_output),
            "--traces-output",
            str(traces_output),
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert [item["name"] for item in output["scope_overlay"]["configurations"]] == [
        "insecure-baseline",
        "acl-filtered-per-hop",
        "scoped-plan-policy",
    ]
    assert json.loads(report_output.read_text())["version"] == "test"
    trace = json.loads(traces_output.read_text())
    assert trace["case_id"] == "b"
    assert "query" not in trace
