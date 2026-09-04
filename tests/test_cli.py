# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify that the public CLI accurately communicates implemented capabilities."""

import json
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

from qdrant_client import QdrantClient
from typer.testing import CliRunner

import proofline.cli as cli
from proofline.authorization import StaticAuthorizationAdapter
from proofline.domain import ScopedResource

runner = CliRunner()


def test_future_commands_are_explicitly_unavailable() -> None:
    result = runner.invoke(cli.app, ["query"])

    assert result.exit_code == 2
    assert "planned for Phase 6" in result.stdout


def test_ingest_writes_a_corpus_from_a_pinned_manifest(tmp_path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "example.mdx").write_text("One searchable paragraph.")
    assignments = tmp_path / "assignments.yaml"
    assignments.write_text("version: test\nassignments: []\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: test-v0",
                "retrieved_at: 2026-09-03",
                f"access_assignments: {assignments}",
                "source:",
                "  repository: https://example.test/repo",
                f"  revision: {'a' * 40}",
                "  license: MIT",
                "documents:",
                "  - id: example",
                "    path: docs/example.mdx",
                "    url: https://example.test/example",
                "    visibility: public",
            ]
        )
    )
    output = tmp_path / "corpus.jsonl"

    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "--source-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote 1 chunks" in result.stdout
    assert output.exists()


def test_validate_data_accepts_the_versioned_fixtures() -> None:
    result = runner.invoke(cli.app, ["validate-data"])

    assert result.exit_code == 0
    assert "Validated corpus-v0" in result.stdout


def test_evaluate_lexical_writes_report_and_retrieval_traces(tmp_path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "check.mdx").write_text("Check decides whether a user may view a document.")
    assignments = tmp_path / "assignments.yaml"
    assignments.write_text("version: test\nassignments: []\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: test-v0",
                "retrieved_at: 2026-09-03",
                f"access_assignments: {assignments}",
                "source:",
                "  repository: https://example.test/repo",
                f"  revision: {'a' * 40}",
                "  license: MIT",
                "documents:",
                "  - id: check",
                "    path: docs/check.mdx",
                "    url: https://example.test/check",
                "    visibility: public",
            ]
        )
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """version: test-suite
cases:
  - id: check
    mode: public_documentation
    principal: user:ana
    query: What does Check decide?
    expected: cited_answer
    required_sources: [check]
"""
    )
    report = tmp_path / "report.md"
    traces = tmp_path / "traces.jsonl"

    result = runner.invoke(
        cli.app,
        [
            "evaluate-lexical",
            "--source-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--suite",
            str(suite),
            "--output",
            str(report),
            "--traces-output",
            str(traces),
        ],
    )

    assert result.exit_code == 0
    assert "1 retrieval cases" in result.stdout
    assert "Recall@k" in report.read_text()
    assert len(traces.read_text().splitlines()) == 1


def test_evaluate_dense_writes_comparison_report(tmp_path, monkeypatch) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "check.mdx").write_text("Check decides whether a user may view a document.")
    assignments = tmp_path / "assignments.yaml"
    assignments.write_text("version: test\nassignments: []\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: test-v0",
                "retrieved_at: 2026-09-03",
                f"access_assignments: {assignments}",
                "source:",
                "  repository: https://example.test/repo",
                f"  revision: {'a' * 40}",
                "  license: MIT",
                "documents:",
                "  - id: check",
                "    path: docs/check.mdx",
                "    url: https://example.test/check",
                "    visibility: public",
            ]
        )
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """version: test-suite
cases:
  - id: check
    mode: public_documentation
    principal: user:ana
    query: What does Check decide?
    expected: cited_answer
    required_sources: [check]
"""
    )
    report = tmp_path / "dense.md"
    monkeypatch.setattr(cli, "QdrantClient", lambda **kwargs: QdrantClient(":memory:"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        result = runner.invoke(
            cli.app,
            [
                "evaluate-dense",
                "--source-root",
                str(tmp_path),
                "--manifest",
                str(manifest),
                "--suite",
                str(suite),
                "--collection",
                "dense-test",
                "--output",
                str(report),
            ],
        )

    assert result.exit_code == 0
    assert "dense retrieval comparison" in result.stdout
    assert "Estimated vector index size" in report.read_text()


def test_ingest_writes_only_assigned_chunks_for_protected_documents(tmp_path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "secret.mdx").write_text("Protected rollout detail.")
    assignments = tmp_path / "assignments.yaml"
    assignments.write_text(
        "\n".join(
            [
                "version: test",
                "assignments:",
                "  - source_document: secret",
                "    tenant_id: tenant:acme",
                "    resource_id: document:secret",
            ]
        )
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "version: test-v0",
                "retrieved_at: 2026-09-03",
                f"access_assignments: {assignments}",
                "source:",
                "  repository: https://example.test/repo",
                f"  revision: {'a' * 40}",
                "  license: MIT",
                "documents:",
                "  - id: secret",
                "    path: docs/secret.mdx",
                "    url: https://example.test/secret",
                "    visibility: protected",
            ]
        )
    )
    output = tmp_path / "corpus.jsonl"

    result = runner.invoke(
        cli.app,
        [
            "ingest",
            "--source-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert [json.loads(line)["resource_id"] for line in output.read_text().splitlines()] == [
        "document:secret"
    ]


def test_demo_tenant_search_prints_only_allowed_trace_candidates() -> None:
    result = runner.invoke(cli.app, ["demo-tenant-search", "--authorization", "static"])

    trace = json.loads(result.stdout)
    assert result.exit_code == 0
    assert trace["access_scope"]["resources"] == [
        {"tenant_id": "tenant:acme", "resource_id": "document:acme-rollout"}
    ]
    assert {candidate["chunk_id"] for candidate in trace["candidates"]} == {
        "chunk:public-policy",
        "chunk:acme-rollout",
    }


def test_demo_tenant_search_uses_provisioned_openfga_adapter(monkeypatch) -> None:
    adapter = StaticAuthorizationAdapter(
        {
            ("user:ana", "tenant:acme"): (
                ScopedResource(tenant_id="tenant:acme", resource_id="document:acme-rollout"),
            )
        }
    )
    provisioned = SimpleNamespace(adapter=adapter, delete=AsyncMock())
    provision = AsyncMock(return_value=provisioned)
    monkeypatch.setattr(cli, "provision_openfga", provision)

    result = runner.invoke(
        cli.app,
        ["demo-tenant-search", "--authorization", "openfga"],
        env={"OPENFGA_URL": "http://openfga.test"},
    )

    assert result.exit_code == 0
    assert {candidate["chunk_id"] for candidate in json.loads(result.stdout)["candidates"]} == {
        "chunk:public-policy",
        "chunk:acme-rollout",
    }
    provision.assert_awaited_once_with("http://openfga.test")
    provisioned.delete.assert_awaited_once()


def test_demo_check_access_uses_provisioned_openfga_adapter(monkeypatch) -> None:
    adapter = SimpleNamespace(check_access=AsyncMock(return_value=True))
    provisioned = SimpleNamespace(adapter=adapter, delete=AsyncMock())
    provision = AsyncMock(return_value=provisioned)
    monkeypatch.setattr(cli, "provision_openfga", provision)

    result = runner.invoke(
        cli.app,
        ["demo-check-access", "--resource", "document:acme-rollout"],
        env={"OPENFGA_URL": "http://openfga.test"},
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"allowed": True}
    adapter.check_access.assert_awaited_once()
    provisioned.delete.assert_awaited_once()
