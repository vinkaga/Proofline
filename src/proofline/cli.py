# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provide the stable command-line boundary for reproducible project workflows.

The CLI names the eventual ingest, query, evaluation, and reporting workflows
before their implementations arrive. Explicit unavailable commands prevent a
reviewer from mistaking scaffolding for a completed capability.
"""

import asyncio
import json
import os
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
from proofline.domain import Principal
from proofline.evaluation_data import load_evaluation_suite
from proofline.openfga_fixture import provision_openfga
from proofline.retrieval import RetrievalResult
from proofline.tracing import trace_tenant_retrieval
from proofline.vertical_slice import build_vertical_slice

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


@app.command("demo-tenant-search")
def demo_tenant_search(
    principal: Annotated[str, typer.Option()] = "user:ana",
    tenant: Annotated[str, typer.Option()] = "tenant:acme",
    query: Annotated[str, typer.Option()] = "release approval",
    authorization: Annotated[str, typer.Option()] = "openfga",
) -> None:
    """Run the ACL-filtered fixture and print its trace as JSON."""

    caller = Principal(id=principal)
    if authorization == "static":
        retriever = build_vertical_slice()
        result = asyncio.run(retriever.search_tenant(caller, tenant, query))
    elif authorization == "openfga":
        server_url = os.environ.get("OPENFGA_URL")
        if not server_url:
            raise typer.BadParameter("set OPENFGA_URL or pass --authorization static")
        result = asyncio.run(_search_with_openfga(server_url, caller, tenant, query))
    else:
        raise typer.BadParameter("authorization must be openfga or static")
    trace = trace_tenant_retrieval("phase-1.5-demo", caller, result)
    typer.echo(trace.model_dump_json(indent=2))


@app.command("demo-check-access")
def demo_check_access(
    principal: Annotated[str, typer.Option()] = "user:ana",
    tenant: Annotated[str, typer.Option()] = "tenant:acme",
    resource: Annotated[str, typer.Option()] = "document:acme-rollout",
    relation: Annotated[str, typer.Option()] = "viewer",
) -> None:
    """Run one OpenFGA permission decision against the synthetic fixture."""

    server_url = os.environ.get("OPENFGA_URL")
    if not server_url:
        raise typer.BadParameter("set OPENFGA_URL")
    caller = Principal(id=principal)
    allowed = asyncio.run(
        _check_access_with_openfga(server_url, caller, relation, resource, tenant)
    )
    typer.echo(json.dumps({"allowed": allowed}))


async def _search_with_openfga(
    server_url: str, caller: Principal, tenant: str, query: str
) -> RetrievalResult:
    """Run the tenant-search demo with an isolated OpenFGA fixture."""

    provisioned = await provision_openfga(server_url)
    try:
        return await build_vertical_slice(provisioned.adapter).search_tenant(caller, tenant, query)
    finally:
        await provisioned.delete()


async def _check_access_with_openfga(
    server_url: str,
    caller: Principal,
    relation: str,
    resource: str,
    tenant: str,
) -> bool:
    """Run one permission check with an isolated OpenFGA fixture."""

    provisioned = await provision_openfga(server_url)
    try:
        return await provisioned.adapter.check_access(caller, relation, resource, tenant)
    finally:
        await provisioned.delete()


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
