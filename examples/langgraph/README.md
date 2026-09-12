# LangGraph example

Trusted request context is captured when the host constructs the graph. The
graph's optional second node parses planner output through Proofline, then uses
the first node's `ScopedResults` to create an inherited child scope. Graph state
never contains or constructs resource filters.

```bash
uv run pytest
```
