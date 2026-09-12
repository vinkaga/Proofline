# LlamaIndex example

`ProoflineRetriever` is constructed with trusted host context, then implements
LlamaIndex's normal `BaseRetriever` interface. The model-facing query is passed
to Proofline; resource filters are not part of the LlamaIndex query interface.
Both synchronous and asynchronous retrieval paths are implemented.
For an orchestrated second hop, retain `search_scoped()` results and call
`follow_proposed()`; it rejects scope-bearing planner fields and returns child
scope lineage alongside LlamaIndex nodes.

```bash
uv run pytest
```
