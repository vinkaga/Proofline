# Proofline examples

Each example is an independent host application with its own `pyproject.toml`.
It depends on `proofline` and its selected framework only, then routes its
existing retriever through the same scoped-retrieval boundary.

Planned examples cover a custom Python loop, LangChain/LangGraph, Pydantic AI,
and LlamaIndex. They are not framework-specific adapter packages.
