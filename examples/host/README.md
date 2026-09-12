# Shared example host fixture

`proofline-example-host` is an internal package used only by the framework
examples in this repository. It keeps a tiny retriever, trusted request
context, and fixed Acme/Beta evidence constant, so each example demonstrates
framework wiring rather than a different security model.

It is not a production integration package and is not published independently.
Applications should replace its trusted request resolver and in-memory backend
with their own authentication, authorization, and retriever implementations.
