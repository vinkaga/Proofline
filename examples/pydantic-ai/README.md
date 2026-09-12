# Pydantic AI example

`retrieve_evidence` is a real Pydantic AI tool. Its `RunContext.deps` carries
trusted host-authentication output; the tool schema exposes only `query`.
Do not add tenant, principal, resource, or filter parameters to the tool.
The application supplies its chosen model when calling `agent.run`; tests use
Pydantic AI's `TestModel` only to exercise the registered tool deterministically.

```bash
uv run pytest
```
