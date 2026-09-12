# Proofline examples

Each example is runnable from its own directory in this repository and has its
own `pyproject.toml`. It depends on its selected framework plus the local
`proofline-example-host` fixture package, which in turn depends on `proofline`.
The fixture keeps the documents and trusted caller context constant so the
examples differ only in framework wiring.

Available examples cover a custom Python loop, an application-style FastAPI
host, LangGraph, Pydantic AI, and LlamaIndex. They are not framework-specific
adapter packages: each keeps its normal framework integration and calls
Proofline only at the retrieval boundary.

`host/` is a normal Python package, not a parent-directory import or test
helper. Each example can therefore run with its own dependency manager and
working directory. The custom-loop example is deliberately self-contained.
