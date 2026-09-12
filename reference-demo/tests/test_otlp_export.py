# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify the opt-in OTLP path without requiring Phoenix or Docker in CI."""

from __future__ import annotations

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
from proofline_reference_demo.tracing import configure_otlp_tracing

configure_otlp_tracing(__import__('sys').argv[1])
with trace.get_tracer('proofline.reference_demo').start_as_current_span('proofline.test'):
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
