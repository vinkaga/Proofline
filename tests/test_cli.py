# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify that the public CLI accurately communicates implemented capabilities."""

from typer.testing import CliRunner

from proofline.cli import app

runner = CliRunner()


def test_status_reports_phase_zero_foundation() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Phase 0 foundation ready" in result.stdout


def test_future_commands_are_explicitly_unavailable() -> None:
    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 2
    assert "planned for Phase 3" in result.stdout
