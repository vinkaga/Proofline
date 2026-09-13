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
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, cast

import typer
from qdrant_client import QdrantClient

from scopeanchor_reference_demo.authorization import (
    AuthorizationAdapter,
    StaticAuthorizationAdapter,
)
from scopeanchor_reference_demo.bounded_host import run_bounded_host
from scopeanchor_reference_demo.corpus import (
    build_corpus,
    load_access_assignments,
    load_manifest,
    validate_corpus_configuration,
    write_corpus,
)
from scopeanchor_reference_demo.dense_retrieval import (
    FastEmbedEmbeddingProvider,
    OpenAiEmbeddingProvider,
    QdrantDenseRetriever,
    TokenHashEmbeddingProvider,
    write_dense_comparison_report,
)
from scopeanchor_reference_demo.domain import Principal
from scopeanchor_reference_demo.evaluation_data import load_evaluation_suite
from scopeanchor_reference_demo.hotpot_evaluation import (
    evaluate_hotpotqa_retrieval,
    evaluate_hotpotqa_scope_controls,
    evaluate_hotpotqa_scope_overlay,
    validate_hotpotqa_evaluation,
    validate_hotpotqa_scope_controls,
    validate_hotpotqa_scope_overlay,
)
from scopeanchor_reference_demo.hotpotqa import (
    build_overlay,
    evaluate_overlay,
)
from scopeanchor_reference_demo.hotpotqa import (
    load_cases as load_hotpotqa_cases,
)
from scopeanchor_reference_demo.hotpotqa import (
    load_manifest as load_hotpotqa_manifest,
)
from scopeanchor_reference_demo.hybrid_retrieval import HybridRrfRetriever
from scopeanchor_reference_demo.lexical_evaluation import (
    evaluate_lexical_baseline,
    validate_baseline_measurement,
    write_lexical_report,
    write_lexical_traces,
)
from scopeanchor_reference_demo.multi_hop import run_clean_two_hop, run_poisoned_two_hop
from scopeanchor_reference_demo.openfga_fixture import load_static_permissions, provision_openfga
from scopeanchor_reference_demo.permission_mcp import build_permission_server
from scopeanchor_reference_demo.release_evaluation import (
    evaluate_release_suite,
    validate_release_evaluation,
)
from scopeanchor_reference_demo.reranking import RerankingRetriever, TokenCoverageReranker
from scopeanchor_reference_demo.retrieval import AccessGatedBm25Retriever, RetrievalResult
from scopeanchor_reference_demo.retrieval_comparison import write_method_comparison_report
from scopeanchor_reference_demo.scope_evaluation import (
    ScopeGateError,
    evaluate_scope_propagation,
    validate_scope_propagation,
)
from scopeanchor_reference_demo.scoped_fixture import DemoRequestContext, build_scoped_fixture
from scopeanchor_reference_demo.tracing import configure_otlp_tracing, trace_tenant_retrieval

app = typer.Typer(
    name="scopeanchor-reference-demo",
    help="Public-data evaluation and demonstration for ScopeAnchor.",
    no_args_is_help=True,
)


@app.callback()
def configure_observability(
    otlp_endpoint: Annotated[
        str | None,
        typer.Option(
            "--otlp-endpoint",
            help="Explicit OTLP/HTTP trace endpoint; no telemetry is configured by default.",
        ),
    ] = None,
) -> None:
    """Configure opt-in telemetry before the selected demonstration command."""

    if otlp_endpoint is not None:
        configure_otlp_tracing(otlp_endpoint)


def _not_available(command: str, phase: int) -> None:
    typer.echo(f"`scopeanchor-reference-demo {command}` is planned for Phase {phase}.")
    raise typer.Exit(code=2)


def _openfga_url(value: str | None) -> str:
    """Resolve the explicit option first, then the local-demo environment."""

    server_url = value or os.environ.get("OPENFGA_URL")
    if not server_url:
        raise typer.BadParameter(
            "set --openfga-url or OPENFGA_URL, or explicitly select --authorization static"
        )
    return server_url


@asynccontextmanager
async def _selected_authorization(
    authorization: str,
    openfga_url: str | None,
) -> AsyncIterator[AuthorizationAdapter]:
    """Yield the selected policy adapter and clean up temporary OpenFGA stores."""

    if authorization == "static":
        yield StaticAuthorizationAdapter(load_static_permissions())
        return
    if authorization != "openfga":
        raise typer.BadParameter("authorization must be openfga or static")
    provisioned = await provision_openfga(_openfga_url(openfga_url))
    try:
        yield provisioned.adapter
    finally:
        await provisioned.delete()


def _warn_static_authorization() -> None:
    """Keep the offline fixture from being mistaken for relationship evaluation."""

    typer.echo(
        "Using fixture-only static direct-viewer grants; inherited OpenFGA relationships are not "
        "evaluated.",
        err=True,
    )


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
        result = asyncio.run(
            _search_through_scopeanchor(
                StaticAuthorizationAdapter(load_static_permissions()), caller, tenant, query
            )
        )
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


@app.command("serve-mcp")
def serve_mcp(
    principal: Annotated[str, typer.Option()] = "user:ana",
    tenant: Annotated[str, typer.Option()] = "tenant:acme",
    authorization: Annotated[
        str,
        typer.Option(help="openfga evaluates the checked-in policy; static is a limited fixture."),
    ] = "openfga",
    openfga_url: Annotated[
        str | None,
        typer.Option(help="OpenFGA server URL; defaults to OPENFGA_URL."),
    ] = None,
) -> None:
    """Serve the context-bound ``check_access`` MCP tool over stdio."""

    if authorization == "static":
        _warn_static_authorization()
    asyncio.run(
        _serve_permission_mcp(
            principal=Principal(id=principal),
            tenant_id=tenant,
            authorization=authorization,
            openfga_url=openfga_url,
        )
    )


@app.command("demo-multi-hop")
def demo_multi_hop(
    scenario: Annotated[str, typer.Option()] = "clean",
    principal: Annotated[str, typer.Option()] = "user:ana",
    tenant: Annotated[str, typer.Option()] = "tenant:acme",
) -> None:
    """Run the clean or poisoned deterministic two-hop fixture."""

    authorization = StaticAuthorizationAdapter(load_static_permissions())
    caller = Principal(id=principal)
    if scenario == "clean":
        trace = asyncio.run(run_clean_two_hop(authorization, principal=caller, tenant_id=tenant))
    elif scenario == "poisoned":
        trace = asyncio.run(run_poisoned_two_hop(authorization, principal=caller, tenant_id=tenant))
    else:
        raise typer.BadParameter("scenario must be clean or poisoned")
    typer.echo(json.dumps(trace.as_dict(), indent=2))


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
    suite_data = load_evaluation_suite(suite)
    measurement = asyncio.run(
        evaluate_lexical_baseline(
            AccessGatedBm25Retriever(chunks, StaticAuthorizationAdapter(load_static_permissions())),
            suite_data,
            corpus_manifest.version,
            limit=limit,
        )
    )
    validate_baseline_measurement(measurement, suite_data.lexical_quality_gate)
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
    collection: Annotated[str, typer.Option()] = "scopeanchor-dense-evaluation",
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
    collection: Annotated[str, typer.Option()] = "scopeanchor-hybrid-evaluation",
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
        return await _search_through_scopeanchor(provisioned.adapter, caller, tenant, query)
    finally:
        await provisioned.delete()


async def _search_through_scopeanchor(
    authorization: AuthorizationAdapter,
    caller: Principal,
    tenant: str,
    query: str,
) -> RetrievalResult:
    """Run the fixture through the published scoped-retrieval boundary."""

    results = await build_scoped_fixture(authorization).search(
        query,
        context=DemoRequestContext(principal=caller, tenant_id=tenant),
    )
    # The retrieval scope also contains public-resource allowlist entries so
    # the backend can apply all filters conjunctively. Report authorization
    # output separately rather than mislabeling those public entries as grants.
    access_scope = await authorization.list_permitted_resources(caller, tenant)
    return RetrievalResult(
        access_scope=access_scope,
        candidates=results.items,
    )


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


async def _serve_permission_mcp(
    *,
    principal: Principal,
    tenant_id: str,
    authorization: str,
    openfga_url: str | None,
) -> None:
    """Run MCP and OpenFGA in one event-loop lifecycle for safe SDK cleanup."""

    async with _selected_authorization(authorization, openfga_url) as adapter:
        await build_permission_server(
            adapter,
            principal=principal,
            tenant_id=tenant_id,
        ).run_stdio_async()


@app.command()
def query(
    query_text: Annotated[str, typer.Option("--query")] = (
        "What approval does Acme need for rollout?"
    ),
    principal: Annotated[str, typer.Option()] = "user:ana",
    tenant: Annotated[str, typer.Option()] = "tenant:acme",
    relation: Annotated[str, typer.Option()] = "viewer",
    resource: Annotated[str | None, typer.Option()] = None,
    authorization: Annotated[
        str,
        typer.Option(help="openfga evaluates the checked-in policy; static is a limited fixture."),
    ] = "openfga",
    openfga_url: Annotated[
        str | None,
        typer.Option(help="OpenFGA server URL; defaults to OPENFGA_URL."),
    ] = None,
) -> None:
    """Run one bounded host request through the selected authorization backend."""

    if authorization == "static":
        _warn_static_authorization()
    trace = asyncio.run(
        _run_query(
            authorization=authorization,
            openfga_url=openfga_url,
            principal=Principal(id=principal),
            tenant_id=tenant,
            query=query_text,
            relation=relation,
            resource_id=resource,
        )
    )
    typer.echo(json.dumps(trace.as_dict(), indent=2))


async def _run_query(
    *,
    authorization: str,
    openfga_url: str | None,
    principal: Principal,
    tenant_id: str,
    query: str,
    relation: str,
    resource_id: str | None,
):
    """Execute the normal host path with the requested policy adapter."""

    async with _selected_authorization(authorization, openfga_url) as adapter:
        return await run_bounded_host(
            adapter,
            principal=principal,
            tenant_id=tenant_id,
            query=query,
            relation=relation,
            resource_id=resource_id,
        )


@app.command()
def evaluate() -> None:
    """Run the deterministic scope-propagation release gate."""

    report = asyncio.run(
        evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))
    )
    typer.echo(json.dumps(report.as_dict(), indent=2))
    try:
        validate_scope_propagation(report)
    except ScopeGateError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command("evaluate-release")
def evaluate_release(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True)] = Path("data/corpus/manifest.yaml"),
    suite: Annotated[Path, typer.Option(exists=True)] = Path("data/eval/release-v0.yaml"),
    openfga_url: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Execute every versioned release case through the corpus-backed host."""

    corpus_manifest = load_manifest(manifest)
    chunks = build_corpus(
        corpus_manifest, source_root, load_access_assignments(corpus_manifest.access_assignments)
    )
    release_suite = load_evaluation_suite(suite)
    report = asyncio.run(_evaluate_release(chunks, release_suite, openfga_url))
    typer.echo(json.dumps(asdict(report), indent=2))
    try:
        validate_release_evaluation(report)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


async def _evaluate_release(
    chunks: tuple,
    suite,
    openfga_url: str | None,
):
    """Use the real policy adapter for release execution when configured."""

    if openfga_url is None:
        return await evaluate_release_suite(
            StaticAuthorizationAdapter(load_static_permissions()), chunks, suite
        )
    provisioned = await provision_openfga(openfga_url)
    try:
        return await evaluate_release_suite(provisioned.adapter, chunks, suite)
    finally:
        await provisioned.delete()


@app.command("evaluate-hotpotqa")
def evaluate_hotpotqa(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "../hotpot_dev_distractor_v1.json"
    ),
    manifest: Annotated[Path, typer.Option(exists=True)] = Path(
        "data/benchmarks/hotpotqa-distractor-dev.yaml"
    ),
    output: Annotated[Path | None, typer.Option()] = None,
    traces_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Verify the pinned HotpotQA subset and its separate security overlay."""

    benchmark = load_hotpotqa_manifest(manifest)
    cases = load_hotpotqa_cases(dataset, benchmark)
    overlays = build_overlay(cases, benchmark.overlay)
    overlay = evaluate_overlay(cases, overlays, benchmark.source.sha256)
    report = evaluate_hotpotqa_retrieval(cases, overlay)
    scope_traces = asyncio.run(evaluate_hotpotqa_scope_overlay(cases, overlays))
    controls = evaluate_hotpotqa_scope_controls(scope_traces)
    payload = {
        "version": benchmark.version,
        "retrieval": asdict(report),
        "scope_overlay": {
            "trace_count": len(scope_traces),
            "configurations": [asdict(configuration) for configuration in controls.configurations],
        },
    }
    typer.echo(json.dumps(payload, indent=2))
    try:
        validate_hotpotqa_evaluation(report, benchmark.retrieval_quality_gate)
        validate_hotpotqa_scope_overlay(scope_traces)
        validate_hotpotqa_scope_controls(controls)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    if output is not None:
        _write_json(payload, output)
    if traces_output is not None:
        _write_jsonl((asdict(trace) for trace in scope_traces), traces_output)


@app.command()
def report(
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/scope-propagation-v0"),
) -> None:
    """Write a versioned scope-gate report and redacted representative traces."""

    scope_report = asyncio.run(
        evaluate_scope_propagation(StaticAuthorizationAdapter(load_static_permissions()))
    )
    try:
        validate_scope_propagation(scope_report)
    except ScopeGateError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    payload = scope_report.as_dict()
    report_path = output_dir / f"{scope_report.version}-report.json"
    traces_path = output_dir / f"{scope_report.version}-traces.jsonl"
    _write_json(payload, report_path)
    _write_jsonl(cast(list[object], payload["traces"]), traces_path)
    typer.echo(f"Wrote {report_path} and {traces_path}")


def _write_json(payload: object, output: Path) -> None:
    """Persist a stable, pretty-printed public evidence artifact."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(rows: Iterable[object], output: Path) -> None:
    """Persist redacted per-case evidence with one stable JSON object per line."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
