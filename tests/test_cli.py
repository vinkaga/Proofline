# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify that the public CLI accurately communicates implemented capabilities."""

from typer.testing import CliRunner

from proofline.cli import app

runner = CliRunner()


def test_future_commands_are_explicitly_unavailable() -> None:
    result = runner.invoke(app, ["query"])

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
        app,
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
    result = runner.invoke(app, ["validate-data"])

    assert result.exit_code == 0
    assert "Validated corpus-v0" in result.stdout
