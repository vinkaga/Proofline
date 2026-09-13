# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify the opt-in OTLP path without requiring Phoenix or Docker in CI."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Collector(BaseHTTPRequestHandler):
    payloads: list[bytes] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        self.__class__.payloads.append(self.rfile.read(length))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def test_configured_otlp_exporter_posts_a_span_to_a_local_collector() -> None:
    _Collector.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
    script = """
from opentelemetry import trace
from scopeanchor_reference_demo.tracing import configure_otlp_tracing

configure_otlp_tracing(__import__('sys').argv[1])
with trace.get_tracer('scopeanchor.reference_demo').start_as_current_span('scopeanchor.test'):
    pass
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, endpoint],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        worker.join()
        server.server_close()

    assert completed.returncode == 0, completed.stderr
    assert _Collector.payloads


def test_bounded_host_exports_one_parented_request_trace() -> None:
    """The host path must make authorization a child of its retrieval operation."""

    script = """
import asyncio
import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from scopeanchor_reference_demo.authorization import StaticAuthorizationAdapter
from scopeanchor_reference_demo.bounded_host import run_bounded_host
from scopeanchor_reference_demo.domain import Principal, ScopedResource

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
authorization = StaticAuthorizationAdapter({
    (\"user:ana\", \"tenant:acme\"): (
        ScopedResource(tenant_id=\"tenant:acme\", resource_id=\"document:acme-rollout\"),
    ),
})
asyncio.run(run_bounded_host(
    authorization,
    principal=Principal(id=\"user:ana\"),
    tenant_id=\"tenant:acme\",
    query=\"What approval does Acme need for rollout?\",
))
print(json.dumps([
    {
        \"name\": span.name,
        \"trace_id\": span.context.trace_id,
        \"span_id\": span.context.span_id,
        \"parent_id\": span.parent.span_id if span.parent else 0,
        \"attributes\": dict(span.attributes),
    }
    for span in exporter.get_finished_spans()
]))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    spans = json.loads(completed.stdout)
    assert {span["trace_id"] for span in spans}
    assert len({span["trace_id"] for span in spans}) == 1
    request = next(span for span in spans if span["name"] == "scopeanchor.request")
    initial_retrieval = next(
        span
        for span in spans
        if span["name"] == "scopeanchor.retrieval" and span["parent_id"] == request["span_id"]
    )
    authorization = next(
        span for span in spans if span["name"] == "scopeanchor.authorization.resolve_scope"
    )
    assert request["parent_id"] == 0
    assert request["attributes"]["enduser.id"] == "user:ana"
    assert request["attributes"]["scopeanchor.scope.count"] == 2
    assert initial_retrieval["attributes"]["scopeanchor.scope.id"]
    assert authorization["parent_id"] == initial_retrieval["span_id"]
    assert all(span["parent_id"] for span in spans if span is not request)
