# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provide the stable command-line boundary for reproducible project workflows.

The CLI names the eventual ingest, query, evaluation, and reporting workflows
before their implementations arrive. Explicit unavailable commands prevent a
reviewer from mistaking scaffolding for a completed capability.
"""

from pathlib import Path
from typing import Annotated

import typer

from proofline.corpus import (
    build_corpus,
    load_access_assignments,
    load_manifest,
    validate_corpus_configuration,
    write_corpus,
)
from proofline.evaluation_data import load_evaluation_suite

app = typer.Typer(
    name="proofline",
    help="Access-gated retrieval and bounded agent evaluation.",
    no_args_is_help=True,
)


def _not_available(command: str, phase: int) -> None:
    typer.echo(f"`proofline {command}` is planned for Phase {phase}.")
    raise typer.Exit(code=2)


@app.command()
def ingest(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/corpus.jsonl"),
) -> None:
    """Build a provenance-carrying corpus from a pinned source checkout."""

    corpus_manifest = load_manifest(manifest)
    assignments = load_access_assignments(corpus_manifest.access_assignments)
    chunks = build_corpus(corpus_manifest, source_root, assignments)
    write_corpus(chunks, output)
    typer.echo(f"Wrote {len(chunks)} chunks to {output}")


@app.command("validate-data")
def validate_data(
    manifest: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    suite: Annotated[Path, typer.Option(exists=True)] = Path("data/eval/release-v0.yaml"),
) -> None:
    """Validate the pinned corpus manifest and reviewed release suite."""

    corpus = load_manifest(manifest)
    assignments = load_access_assignments(corpus.access_assignments)
    validate_corpus_configuration(corpus, assignments)
    release_suite = load_evaluation_suite(suite)
    typer.echo(
        f"Validated {corpus.version} and {len(release_suite.cases)} {release_suite.version} cases"
    )


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
