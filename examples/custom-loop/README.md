# Custom-loop example

This is the smallest useful Proofline integration. Authentication and
authorization create `RequestContext`; the host converts it to a root
`RetrievalScope` inside `build_retriever()`. The planner can propose a query,
but `ProposedRetrievalStep.from_untrusted()` rejects any attempt to provide a
resource, tenant, principal, or raw filter.

```bash
uv sync --all-groups
uv run pytest
```

The in-memory documents make the boundary inspectable. Replace only `search`
with the application's existing retriever; keep scope construction in trusted
host code.
