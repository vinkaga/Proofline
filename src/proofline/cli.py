# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provide the stable command-line boundary for reproducible project workflows.

The CLI names the eventual ingest, query, evaluation, and reporting workflows
before their implementations arrive. Explicit unavailable commands prevent a
reviewer from mistaking scaffolding for a completed capability.
"""

from __future__ import annotations

import typer

from proofline import __version__

app = typer.Typer(
    name="proofline",
    help="Access-gated retrieval and bounded agent evaluation.",
    no_args_is_help=True,
)


@app.command()
def status() -> None:
    """Report the currently implemented project foundation."""

    typer.echo(f"Proofline {__version__}: Phase 0 foundation ready")


def _not_available(command: str, phase: int) -> None:
    typer.echo(f"`proofline {command}` is planned for Phase {phase}.")
    raise typer.Exit(code=2)


@app.command()
def ingest() -> None:
    """Ingest the pinned corpus. Available in Phase 3."""

    _not_available("ingest", phase=3)


@app.command()
def query() -> None:
    """Run one request through the bounded agent. Available in Phase 6."""

    _not_available("query", phase=6)


@app.command()
def evaluate() -> None:
    """Evaluate a configuration. Available in Phase 7."""

    _not_available("evaluate", phase=7)


@app.command()
def report() -> None:
    """Generate a static evaluation report. Available in Phase 8."""

    _not_available("report", phase=8)
