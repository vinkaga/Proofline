<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Vinay Agarwal. This document explains Proofline's purpose and contracts. -->

# Proofline

Access-gated retrieval and bounded agent evaluation for reliable AI assistants.

## Implemented now

- Version-pinned public corpus manifests, deterministic paragraph chunking, and
  source revision/URL provenance.
- Synthetic tenant/resource assignments and typed validation of the corpus and
  25-case release suite.
- A static test adapter plus an OpenFGA-shaped authorization adapter.
- BM25 retrieval that applies tenant ACL filtering before scoring candidates.
- A reproducible lexical-baseline command that measures ACL-filtered Recall@k,
  MRR, nDCG@k, latency, and unauthorized exposure, plus a local OpenFGA-backed
  `demo-tenant-search` command.
- Qdrant-backed dense retrieval with payload filters applied before candidates
  are returned, and a reproducible dense-versus-lexical comparison command.

## Planned architecture

The remaining architecture is intentional future work: MCP tool serving, a
bounded agent, hybrid/reranked retrieval, semantic embedding-model comparison,
Phoenix tracing, and a release-gated end-to-end evaluation runner. The sections
below describe that target system; they do not claim those capabilities already
exist.

## The problem

Most retrieval demos answer questions from a document collection. An
enterprise assistant has a harder contract: it must find the right evidence,
enforce what the caller is allowed to see or do, use authoritative systems for
decisions, and make regressions visible before release.

Proofline is a small, inspectable reference system for that contract. It is
not a general-purpose chatbot or a benchmark for model intelligence.

Its defining invariant is:

> Retrieval provides evidence. An authorization service decides what a
> principal may retrieve or do. The assistant cannot override either boundary.

## A concrete interaction

```text
User: Can Ana see the production rollout guide?

1. The agent identifies a tenant-scoped permission question.
2. The authorization service evaluates
   check_access(user:ana, viewer, document:production-rollout-guide).
3. The decision is deny. Proofline does not retrieve the guide or pass any of
   its chunks to the model.
4. The assistant explains that access is unavailable. It may cite permitted
   public policy documentation, but it cannot explain protected content that it
   did not retrieve.
```

A relevant answer is still a failure if it exposes the wrong evidence.

## What Proofline demonstrates

- **Access-gated retrieval.** Before a tenant-scoped search returns content,
  the retrieval layer receives the requester's authorized tenant and resource
  scope. Results outside that scope cannot enter the candidate set, trace,
  prompt context, or response.
- **Bounded agentic retrieval.** The agent classifies the question, obtains
  authorization when necessary, retrieves evidence, may reformulate a query
  once under defined conditions, then answers with citations or abstains.
- **Measured retrieval choices.** A keyword baseline, vector retrieval, hybrid
  retrieval, and optional reranking are compared on the same versioned corpus
  and relevance set. Embedding models are compared on quality, latency, and
  estimated per-query cost.
- **Evidence-backed answers.** The assistant returns citations and abstains
  when its retrieved evidence is insufficient.
- **Inspectable quality.** The evaluation suite tests retrieval, grounding,
  abstention, agent traces, authorization boundaries, and regressions. Every
  run records candidates, scores, tool calls, citations, outcomes, latency, and
  configuration.

## How Proofline differs from retrieval frameworks

LangChain, LlamaIndex, Haystack, and similar frameworks can assemble retrieval,
tool use, and evaluation components. Proofline is not a replacement for them.
It makes a narrower contract explicit and testable: authorization constrains the
retrieval candidate universe before model context is assembled, the agent has a
small action budget, and protected outcomes are release-gated in CI. Evaluation
libraries can score parts of that system, but they do not by themselves enforce
this architecture or its security invariants.

## What it contains

The corpus uses version-pinned, public OpenFGA documentation, examples, and
selected issue discussions. The source material is real. A synthetic
organization, tenant, project, resource, and relationship model creates
controlled allow and deny cases without using private information.

Every chunk carries source revision, tenant and resource identifiers, visibility,
and a source URL. Public chunks are available to every principal. Tenant-scoped
chunks are available only when the authorization adapter includes their resource
in the requester's permitted scope.

Questions intentionally mix three kinds of work:

1. Documentation questions, such as how a relation or model behaves. These
   require grounded retrieval and cited answers.
2. Tenant-scoped knowledge questions, such as what a team's rollout guide says.
   These resolve the permitted scope, search only within it, and answer with
   citations or abstain.
3. Permission questions, such as whether a user can view a document. These
   require the authorization tool for the outcome. Documentation may explain the
   result, but cannot be treated as the source of truth.

This is deliberately narrower than an enterprise knowledge assistant. The
model never receives tenant-scoped content before its authorized scope is
resolved. The goal is to make retrieval and authorization failures easy to
reproduce and evaluate.

### Threat model and limits

Proofline verifies that unauthorized chunks, chunk metadata, citations, and
prompt context do not leave the access-filtered retrieval path. Its adversarial
cases include cross-tenant queries, identifier guessing, partial-access
hierarchies, and instruction-like text embedded in documents.

It does not claim to eliminate every information side channel. Response timing,
result-count differences, and broader identity or transport security require
separate production controls. Those boundaries are documented rather than
silently treated as solved.

## Deliberate boundaries

- Building a polished chat application or a general agent framework.
- Claiming production-grade identity, multi-tenancy, or security from a local
  demonstration.
- Building a vector database, ANN algorithm, embedding model, or authorization
  engine from scratch.
- Optimizing a leaderboard score without analyzing why a system succeeds or
  fails.
- Adding multimodal ingestion, a fleet of agents, or integrations unrelated to
  the core access-gated retrieval question.

## Design principles

### Enforce access before evidence

The authorization service evaluates the caller and requested resource before
retrieval returns protected content. The retrieval system supplies evidence for
an explanation. The orchestrator keeps those responsibilities separate and
records the decision path.

### Start with baselines

Each retrieval improvement must beat or clarify a measurable baseline. The
first comparison is lexical retrieval versus vector retrieval. Hybrid retrieval
and reranking are added only when the evaluation set shows a specific weakness
they address.

### Let the agent take only bounded, testable actions

The agent has a small action budget: classify the question, resolve
authorization when needed, retrieve only permitted evidence, optionally
reformulate once or decompose one retry-eligible compound question into at most
two subqueries when evidence is insufficient, then answer or abstain. It does
not freely plan, call arbitrary tools, retrieve after a denial, or iterate until
it appears confident. Every step is part of the trace and evaluation contract.

### Make failure a first-class result

"I do not have enough evidence" and "I cannot determine access without the
authorization tool" are valid outcomes. The system should abstain rather than
invent a citation, a policy interpretation, or a permission result.

### Prefer deterministic checks where possible

Tool selection, authorization results, response schemas, citation IDs, and
required refusals should be checked deterministically. Model-based graders are
reserved for open-ended properties such as explanation quality and whether an
answer is well grounded. They will be calibrated against a small human-reviewed
set.

### Keep every result reproducible

The corpus snapshot, chunking configuration, embedding model, retriever
configuration, reranker, prompt version, model version, and evaluation set are
recorded with each run.

## How it works

```text
version-pinned corpus
        |
        v
ingest -> chunk + attach access metadata -> tenant/public indexes

user question
        |
        v
bounded agent ---- permission question? ----> check_access
        |                         |                 |
        |                         |                 +--> allow or deny
        |                         v
        |                 resolve authorized scope
        |                         |
        v                         v
public retrieval --------> ACL-filtered retrieval ---> ranked evidence
        |                         |                         |
        +-------------------------+-------------------------+
                                  |
           retry eligible and evidence insufficient?
                                  |
             one query reformulation or decomposition
                                  |
                                  v
                         one additional retrieval
                                  |
                                  v
                    cited answer, denial, or abstention
                                  |
                                  v
                         trace + evaluation runner
```

The retrieval layer applies scope before returning candidates. Public and
tenant-scoped indexes are separate namespaces. A policy-derived resource
allowlist further constrains tenant-scoped retrieval. This makes access control
a property of retrieval itself, not a filter applied after an LLM has seen the
results.

Question classification can be deterministic. An LLM is used only for bounded
query reformulation and response composition. That separation makes the access
boundary and the agent's additional value easy to inspect and test.

## Retrieval experiments

| Stage | Method | Purpose |
| --- | --- | --- |
| 0 | BM25 lexical search | Establish a transparent baseline for identifiers and exact terminology. |
| 1 | Dense vector search | Test semantic matching where wording differs, including embedding-model trade-offs. |
| 2 | Hybrid search with reciprocal-rank fusion | Combine complementary lexical and semantic recall. |
| 3 | Reranking of a fixed candidate set | Improve the ordering of already-retrieved candidates. |

Each experiment uses the same corpus revision, chunking policy, access policy,
and test cases. Embedding model comparisons report ranking quality, latency,
index size, and estimated cost. Retrieval is evaluated within the authorized
candidate universe, not against documents the principal is not permitted to
see. Changing more than one variable at a time makes results difficult to
interpret.

## Evaluation

The evaluation set has three tiers. The release suite contains roughly 50 to 75
hand-authored, versioned cases that gate changes. An adversarial suite adds
cross-tenant, partial-access, identifier-guessing, prompt-injection, and
existence-inference probes. A later expansion set adds held-out, realistically
authored questions. Generated cases may add stress coverage but are not the
quality source of truth.

| Evaluation layer | What it checks | Primary measure |
| --- | --- | --- |
| Access isolation | Protected content stays out of unauthorized retrieval and prompts | unauthorized-chunk exposure rate, cross-tenant leakage pass rate |
| Retrieval | Whether permitted relevant evidence appears in the candidate set and near the top | ACL-filtered Recall@k, MRR, nDCG |
| Grounding | Whether answer claims are supported by returned sources | deterministic citation validation plus calibrated rubric |
| Abstention | Whether unsupported or ambiguous questions avoid invented answers | exact expected outcome |
| Tool behavior | Whether access questions call the tool with correct arguments | tool-call and result assertions |
| Authorization | Whether allowed and denied cases match the policy model | exact expected decision |
| Agent trace | Whether query reformulation and retrieval follow the permitted action budget | trace assertions and pass rate |
| End to end | Whether the final answer, citations, tool behavior, and refusal behavior work together | pass rate by scenario type |
| Regression | Whether a proposed change degrades a protected metric or case | CI comparison to baseline |

Initial case categories:

- Exact-name and identifier queries.
- Paraphrased documentation queries.
- Multi-document questions.
- Near-miss questions designed to retrieve plausible but wrong material.
- Unsupported questions that require abstention.
- Allowed and denied access checks.
- Tool-required questions where retrieved text alone would be insufficient.
- Ambiguous queries where one reformulation is useful, and cases where it must
  not be attempted.
- Cross-tenant, partial-access, and identifier-guessing probes.
- Prompt-injection-like text embedded in the corpus, to test that documents
  are evidence rather than instructions.

## Quality gates

A change should fail CI when it:

- changes a known authorization decision;
- omits a required tool call or calls the tool with incorrect arguments;
- produces a citation that is absent or does not support the answer;
- fails a mandatory abstention case;
- exposes an unauthorized chunk;
- exceeds the allowed agent action budget; or
- regresses a protected retrieval or end-to-end metric beyond the agreed
  tolerance.

The project should report trade-offs, not hide them. For example, a reranker
may improve nDCG while increasing latency. A result is useful even when the
new method loses, provided the evaluation explains why.

## Trace record

Each evaluated interaction should write a structured record with:

- case and corpus version;
- request mode and resolved access scope;
- retrieval method, candidates, ranks, and scores;
- the permitted chunk IDs supplied to response context;
- router decision and tool calls with redacted inputs and outputs;
- final answer, citations, and abstention state;
- latency and token or cost metadata when applicable; and
- deterministic checks and model-grader results.

No secrets, private documents, or personal data belong in the repository.

## Operational view

The evaluation runner produces a compact quality view for each corpus and
configuration version: pass rate by request mode, ACL-filtered retrieval
metrics, unauthorized exposure, access denials, abstentions, reformulation
rate, tool failures, latency, and change from baseline. This is intentionally a
small operational surface, not a full observability platform. Its purpose is to
make a regression or unexpected trade-off visible quickly.

Every retrieval comparison also reports ingestion time and cost per 1,000
chunks, estimated query, reranking, and response cost per 1,000 queries, and
p50 and p95 latency. Extrapolations are labeled as such and derived from the
measured corpus and documented model pricing.

## Technology choices

Proofline uses a small, current Python stack. Each dependency supports a
specific part of the access-gated retrieval contract.

| Concern | Choice | Role in Proofline |
| --- | --- | --- |
| Runtime and packaging | Python 3.13+ and `uv` | Modern type syntax, reproducible environments, and fast dependency management. CI tests the supported baseline. |
| Contracts | Pydantic | Typed request modes, tool arguments, traces, citations, and evaluation-case schemas. |
| Authorization | OpenFGA and `openfga-sdk` | `ListObjects` resolves permitted scope. `Check` makes authoritative access decisions. |
| Retrieval | Qdrant and `qdrant-client` | Local dense, hybrid, and payload-filtered retrieval. Qdrant payload filters enforce tenant, resource, and visibility constraints during search. |
| Lexical baseline | BM25 | A transparent baseline for exact terminology and identifiers, evaluated under the same access filter. |
| Tool boundary | Official MCP Python SDK | Exposes `check_access` as a typed tool without turning the project into an MCP platform. |
| Evaluation | `pytest` and `ir_measures` | Deterministic authorization and tool assertions, plus standard retrieval metrics in CI. |
| Traces | OpenTelemetry and Phoenix | Trace visualization and evaluation inspection. Proofline's structured trace remains the source artifact. |

Embedding and reranker providers sit behind small internal interfaces. The
evaluation results, rather than a provider name in the implementation, decide
which model is useful for a given corpus.

The index can be rebuilt from the pinned corpus for the initial project.
Incremental indexing, caching, remote index operations, and more complex policy
models are extensions to justify with measured need, not prerequisites.

LangGraph is deliberately not required. The initial bounded agent is a direct,
typed state machine so that its decision path remains obvious and testable.

## Local setup

Prerequisites: Python 3.13+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync --all-groups
docker compose up -d
uv run proofline --help
```

Run the foundation checks with:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

Run the deterministic access-gated fixture and inspect its JSON trace with:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline demo-tenant-search
```

The same fixture exposes one authoritative permission decision with:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline demo-check-access
```

Build and evaluate the pinned documentation corpus with a checkout at the
revision named in `data/corpus/manifest.yaml`:

```bash
uv run proofline evaluate-lexical --source-root /path/to/openfga.dev
```

This writes `artifacts/lexical-baseline.md` and one inspectable trace per
retrieval-path case to `artifacts/lexical-baseline-traces.jsonl`. It scores the
reviewed evidence-retrieval cases at a fixed `k` and records latency plus
independent access-scope and citation-provenance checks. Permission cases remain
outside these ranking metrics; their release-gated evaluation arrives in Phase 7.
The offline baseline derives direct-viewer grants from the checked-in OpenFGA
tuples; inherited relationship behavior remains covered by the OpenFGA adapter
and integration tests.

With local Qdrant running, compare its access-filtered dense-vector control to
BM25 on the same corpus and cases:

```bash
uv run proofline evaluate-dense --source-root /path/to/openfga.dev
```

Pass `--recreate` to replace that command's named local collection on a repeat
run. To use the optional learned FastEmbed provider, install it in an ONNX
Runtime-compatible environment with `uv sync --extra fastembed`, then pass its
model ID through `--embedding-model`.

The built-in token-hash embedding is deterministic and zero-cost; it validates
the Qdrant boundary and provides a reproducible control, not a claim of
semantic-model quality. A learned embedding-model comparison is future work.

To run the real OpenFGA policy integration test, start the local service and
set its URL for pytest:

```bash
docker compose up -d openfga
OPENFGA_URL=http://localhost:8080 uv run pytest -m integration
```

The test suite enforces more than 85% branch coverage. Ingestion, querying,
evaluation, and reporting commands become available as the corresponding
capabilities are implemented.

## How to extend it

Proofline is designed to make an extension falsifiable. Add one capability,
add or revise the relevant evaluation cases, and compare it with the existing
baseline. A feature is only an improvement if the evaluation results support
it.

Useful extensions include:

- **A different corpus.** Replace the OpenFGA corpus with a versioned set of
  product, API, or support documents. Preserve source provenance, resource
  boundaries, and relevance cases drawn from real information needs rather
  than synthetic prompts.
- **A stronger retrieval stack.** Try a different embedding model, fusion
  method, or reranker. Report both ranking quality and latency, including the
  queries that changed most.
- **More authorization complexity.** Add resource hierarchies, delegated
  agents, time-bounded grants, or multi-tenant isolation. Extend the policy
  cases before changing the tool.
- **A more capable agent.** Add another carefully scoped tool or a second
  reasoning step. Preserve a strict action budget and evaluate the complete
  trace, not only the final wording.
- **Production-facing operations.** Add dataset approval workflows, scheduled
  evaluation runs, dashboards, alerts, or a CI release gate.

The project should not grow into a collection of integrations. Each extension
should clarify one engineering question, establish a baseline, and leave behind
a reproducible result.

## Why this is useful

The value is not a chat screenshot. It is an evidence-backed engineering
artifact:

1. A baseline made retrieval limitations visible.
2. Each retrieval and embedding choice was tested against a fixed,
   access-aware evaluation set.
3. The system enforced authorization before an LLM could see protected evidence
   and kept explanation separate from authorization decisions.
4. A bounded agent improved retrieval without becoming an opaque planner.
5. Traces and quality gates made failures and trade-offs inspectable.

## License

Proofline is available under the [MIT License](LICENSE).

That is the proof line: a traceable path from authorized source evidence and
authoritative tools to a response that can be inspected, tested, and improved.
