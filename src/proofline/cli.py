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
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from qdrant_client import QdrantClient

from proofline.authorization import StaticAuthorizationAdapter
from proofline.corpus import (
    build_corpus,
    load_access_assignments,
    load_manifest,
    validate_corpus_configuration,
    write_corpus,
)
from proofline.dense_retrieval import (
    FastEmbedEmbeddingProvider,
    OpenAiEmbeddingProvider,
    QdrantDenseRetriever,
    TokenHashEmbeddingProvider,
    write_dense_comparison_report,
)
from proofline.domain import Principal
from proofline.evaluation_data import load_evaluation_suite
from proofline.hybrid_retrieval import HybridRrfRetriever
from proofline.lexical_evaluation import (
    evaluate_lexical_baseline,
    validate_baseline_measurement,
    write_lexical_report,
    write_lexical_traces,
)
from proofline.openfga_fixture import load_static_permissions, provision_openfga
from proofline.reranking import RerankingRetriever, TokenCoverageReranker
from proofline.retrieval import AccessGatedBm25Retriever, RetrievalResult
from proofline.retrieval_comparison import write_method_comparison_report
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


@app.command("evaluate-lexical")
def evaluate_lexical(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    suite: Annotated[Path, typer.Option(exists=True)] = Path("data/eval/release-v0.yaml"),
    output: Annotated[Path, typer.Option()] = Path("artifacts/lexical-baseline.md"),
    traces_output: Annotated[Path, typer.Option()] = Path(
        "artifacts/lexical-baseline-traces.jsonl"
    ),
    limit: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Evaluate ACL-filtered BM25 and write a reproducible Phase 3 report."""

    corpus_manifest = load_manifest(manifest)
    assignments = load_access_assignments(corpus_manifest.access_assignments)
    chunks = build_corpus(corpus_manifest, source_root, assignments)
    measurement = asyncio.run(
        evaluate_lexical_baseline(
            AccessGatedBm25Retriever(chunks, StaticAuthorizationAdapter(load_static_permissions())),
            load_evaluation_suite(suite),
            corpus_manifest.version,
            limit=limit,
        )
    )
    validate_baseline_measurement(measurement)
    write_lexical_report(measurement, output)
    write_lexical_traces(measurement, traces_output)
    typer.echo(
        "Wrote lexical baseline report for "
        f"{measurement.retrieval_case_count} retrieval cases to {output}"
    )


@app.command("evaluate-dense")
def evaluate_dense(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    suite: Annotated[Path, typer.Option(exists=True)] = Path("data/eval/release-v0.yaml"),
    qdrant_url: Annotated[str, typer.Option()] = "http://localhost:6333",
    collection: Annotated[str, typer.Option()] = "proofline-dense-evaluation",
    recreate: Annotated[bool, typer.Option()] = False,
    embedding_model: Annotated[str, typer.Option()] = "token-hash",
    output: Annotated[Path, typer.Option()] = Path("artifacts/dense-baseline.md"),
    limit: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Compare filtered Qdrant dense retrieval with the lexical baseline."""

    corpus_manifest = load_manifest(manifest)
    assignments = load_access_assignments(corpus_manifest.access_assignments)
    chunks = build_corpus(corpus_manifest, source_root, assignments)
    suite_data = load_evaluation_suite(suite)
    authorization = StaticAuthorizationAdapter(load_static_permissions())
    lexical = AccessGatedBm25Retriever(chunks, authorization)
    provider = _embedding_provider(embedding_model)
    dense = QdrantDenseRetriever(
        QdrantClient(url=qdrant_url),
        collection,
        provider,
        authorization,
    )
    index = dense.index(chunks, recreate=recreate)
    lexical_measurement = asyncio.run(
        evaluate_lexical_baseline(lexical, suite_data, corpus_manifest.version, limit=limit)
    )
    dense_measurement = asyncio.run(
        evaluate_lexical_baseline(dense, suite_data, corpus_manifest.version, limit=limit)
    )
    validate_baseline_measurement(dense_measurement)
    index = replace(index, estimated_query_cost_usd=provider.estimated_query_cost_usd)
    write_dense_comparison_report(dense_measurement, lexical_measurement, index, output)
    typer.echo(f"Wrote dense retrieval comparison to {output}")


def _embedding_provider(
    model: str,
) -> TokenHashEmbeddingProvider | FastEmbedEmbeddingProvider | OpenAiEmbeddingProvider:
    """Select the reproducible vector control or a local learned embedding model."""

    if model == "token-hash":
        return TokenHashEmbeddingProvider()
    if model.startswith("openai:"):
        return OpenAiEmbeddingProvider(model.removeprefix("openai:"))
    return FastEmbedEmbeddingProvider(model)


@app.command("evaluate-hybrid")
def evaluate_hybrid(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest_path: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    suite_path: Annotated[Path, typer.Option(exists=True)] = Path("data/eval/release-v0.yaml"),
    qdrant_url: Annotated[str, typer.Option()] = "http://localhost:6333",
    collection: Annotated[str, typer.Option()] = "proofline-hybrid-evaluation",
    recreate: Annotated[bool, typer.Option()] = False,
    embedding_model: Annotated[str, typer.Option()] = "token-hash",
    output: Annotated[Path, typer.Option()] = Path("artifacts/retrieval-comparison.md"),
    limit: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    """Compare lexical, dense, RRF hybrid, and fixed-candidate reranking."""

    manifest = load_manifest(manifest_path)
    assignments = load_access_assignments(manifest.access_assignments)
    chunks = build_corpus(manifest, source_root, assignments)
    suite = load_evaluation_suite(suite_path)
    authorization = StaticAuthorizationAdapter(load_static_permissions())
    lexical = AccessGatedBm25Retriever(chunks, authorization)
    provider = _embedding_provider(embedding_model)
    dense = QdrantDenseRetriever(
        QdrantClient(url=qdrant_url),
        collection,
        provider,
        authorization,
    )
    index = dense.index(chunks, recreate=recreate)
    hybrid = HybridRrfRetriever(lexical, dense)
    reranked = RerankingRetriever(hybrid, TokenCoverageReranker(chunks))
    measurements = {
        name: asyncio.run(
            evaluate_lexical_baseline(retriever, suite, manifest.version, limit=limit)
        )
        for name, retriever in {
            "lexical": lexical,
            "dense": dense,
            "hybrid-rrf": hybrid,
            "hybrid-rrf-token-coverage": reranked,
        }.items()
    }
    for measurement in measurements.values():
        validate_baseline_measurement(measurement)
    index = replace(index, estimated_query_cost_usd=provider.estimated_query_cost_usd)
    write_method_comparison_report(measurements, index, output)
    typer.echo(f"Wrote retrieval comparison to {output}")


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
