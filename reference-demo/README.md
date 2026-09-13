# Proofline reference demonstration

This package contains the public OpenFGA/Qdrant corpus, evaluation harness,
CLI, and optional integrations used to demonstrate Proofline. It is not a
dependency of the `proofline` library.

Run it from this directory with `uv sync --all-groups` and
`uv run proofline-reference-demo --help`.

## Proofline Qdrant adapter contract

`QdrantDenseRetriever.search(query, *, filters, limit)` implements the core
Proofline backend contract for the indexed string fields `tenant_id` and
globally qualified `resource_id`. It validates every supplied field, rejects
non-string values rather than silently translating them, and turns each
allowlist into a conjunctive Qdrant payload condition. A named empty allowlist
returns no results without querying Qdrant.

`build_scoped_qdrant_retriever()` is the reference host boundary: it resolves
the caller's trusted authorization once through Proofline, includes public
documents as ordinary allowed resource IDs, and passes the resulting filters to
Qdrant. The older `search_tenant()` and hybrid evaluation paths remain controls
for the reference evaluation; they are not the library adapter contract.
