# Pydantic AI example

`retrieve_evidence` is a real Pydantic AI tool. Its `RunContext.deps` carries
trusted host-authentication output; the tool schema exposes only `query`.
Do not add tenant, principal, resource, or filter parameters to the tool.
Create fresh `PydanticRunDependencies` for every `agent.run` call. They own one
root result and the most recent scoped result: the first tool call resolves the
trusted root scope and later calls are inherited follow-ups. The registered
tool is sequential, and the shared host session also serializes direct
concurrent calls; a run cannot reset its follow-up budget by invoking the tool
again.

```python
request = TrustedRequest("user:ana", frozenset({"document:acme-rollout"}))
result = await agent.run(
    "Find rollout evidence",
    deps=PydanticRunDependencies.from_request(request),
    model=production_model,
)
```

The application supplies its chosen model when calling `agent.run`; tests use
Pydantic AI's deterministic models only to exercise the registered tool.

```bash
uv run pytest
```
