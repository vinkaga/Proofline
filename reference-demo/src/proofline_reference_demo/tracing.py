# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Create inspectable traces for the implemented retrieval boundary."""

from opentelemetry import trace

from proofline_reference_demo.domain import InteractionTrace, Principal, RequestMode
from proofline_reference_demo.retrieval import RetrievalResult

_TRACER_NAME = "proofline.reference_demo"
_configured_endpoint: str | None = None


def trace_operation(name: str, attributes: dict[str, str | int | bool]) -> trace.Span:
    """Start an opt-in span without recording document text or filter values."""

    span = trace.get_tracer(_TRACER_NAME).start_span(name)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    return span


def configure_otlp_tracing(endpoint: str) -> None:
    """Export reference-demo spans to an explicit OTLP/HTTP endpoint.

    For a local Phoenix server, pass its OTLP endpoint (commonly
    ``http://localhost:6006/v1/traces``). This is opt-in: the library itself
    never reads environment variables or configures application telemetry.
    """

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    global _configured_endpoint
    if _configured_endpoint is not None:
        if endpoint != _configured_endpoint:
            raise RuntimeError("OTLP tracing is already configured for a different endpoint")
        return

    provider = TracerProvider()
    # The demo's commands are short-lived. Export synchronously so a process
    # does not exit before a batch processor has sent its final span.
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured_endpoint = endpoint


def trace_tenant_retrieval(
    case_id: str,
    principal: Principal,
    result: RetrievalResult,
) -> InteractionTrace:
    """Record the resolved scope and only the candidates that survived it."""

    with trace_operation(
        "proofline.retrieval",
        {"proofline.case_id": case_id, "enduser.id": principal.id},
    ) as span:
        if result.access_scope is None:
            span.set_attribute("proofline.scope.resolved", False)
            raise ValueError("tenant retrieval traces require an access scope")
        span.set_attribute("proofline.scope.resolved", True)
        span.set_attribute("proofline.scope.resource_count", len(result.access_scope.resource_ids))
        span.set_attribute("proofline.candidate_count", len(result.candidates))
        return InteractionTrace(
            case_id=case_id,
            request_mode=RequestMode.TENANT_KNOWLEDGE,
            principal=principal,
            access_scope=result.access_scope,
            candidates=result.candidates,
            context_chunk_ids=tuple(candidate.chunk_id for candidate in result.candidates),
        )
