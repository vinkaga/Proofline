<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Vinay Agarwal. This document explains Proofline's purpose and contracts. -->

# Proofline

Keep your RAG app's searches within the data each user is allowed to access.

Your RAG app may search a shared document collection, but each user should only
receive information from documents they're allowed to read. If an agent runs
additional searches to answer a question, those searches need the same access
restrictions.

Proofline automatically passes your application's permission filters to every
search through its wrapper, including agent follow-ups. You keep your existing
retriever.

It is a small, MIT-licensed Python library. The permitted tenants, projects, or
document IDs, together with the caller's identity, form a **retrieval scope**.

## What you get

- Permission filters supplied to every search through the wrapper, including
  follow-ups, so you do not have to pass them along manually at each step.
- Follow-ups that keep the previous search's access limits. Your application
  can narrow those limits, but a model or retrieved document cannot widen them.
- A wrapper around your existing retriever, with your framework and document
  types unchanged.

Your application still identifies the user and decides what they may access.
Your retrieval backend must enforce the filters Proofline supplies. The model
can suggest a search query; it cannot choose whose permissions to use.

Proofline is useful when your RAG app or agent searches data with different
access rules for different users or tasks. It protects calls made through its
wrapper; it does not secure direct backend calls, detect prompt injection, or
decide whether retrieved content is trustworthy.

### Comparison with access filtering alone

The following results come from the
[50-case benchmark with synthetic permissions and search proposals](reference-demo/benchmark-results/hotpotqa-distractor-dev-v1/bm25-k5-scope-overlay-v1.md).
Higher is better in every row. These are fixture results, not general attack
success rates. ACL filtering means applying access-control rules to each search.

| Measure | Intentionally insecure control | ACL filtering on every hop | Proofline |
| --- | ---: | ---: | ---: |
| Cases where no unauthorized evidence was exposed | 0% | 100% | 100% |
| Proposals containing permission-setting fields rejected before search | 0% | 0% | 100% |
| Follow-ups with a complete record of inherited access restrictions | 0% | 0% | 100% |

Permission-setting fields include a caller identity, tenant, or resource filter.
These must come from trusted application code. The rejection measure applies
only to proposals containing those forbidden fields; ordinary query-only
proposals remain accepted in the Proofline fixture.

Both ACL filtering on every hop and Proofline prevent unauthorized evidence
exposure in this fixture. Proofline also rejects invalid proposals before
search and records how each follow-up inherits its access restrictions. These
are additional enforcement and traceability properties, not a measured reduction
in exposure compared with the ACL-filtered control.

## Quickstart

From a repository checkout, install the core development environment with
`uv sync`.

In this example, your application has already authenticated the user and resolved
the document IDs they may access. `request.authorized_resource_ids` comes from
that trusted authorization step, not from user input or a model response.
`resolve_scope` packages those permissions for Proofline:

```python
from proofline import RetrievalScope, scoped


def resolve_scope(request) -> RetrievalScope:
    return RetrievalScope.root(
        principal=request.user_id,
        filters={"resource_id": request.authorized_resource_ids},
    )


retriever = scoped(existing_retriever.search, resolve_scope=resolve_scope)
request_retriever = await retriever.bind(request)
results = await request_retriever.search("rollout prerequisites")
```

The wrapped search function must accept `query`, keyword-only `filters`, and
`limit`, and enforce the filters before returning results. Framework retrievers
often need a small adapter because their parameter names and filter formats
differ. See the [backend filter contract](#backend-filter-contract) below.

Pass `request_retriever.search` to an agent tool or application service. Every
call uses the same trusted permissions resolved at `bind()` time; independent
searches are not artificially recorded as a parent/child chain. Trusted
application code can create a narrower branch with
`request_retriever.narrow_trusted(...)`.

For an explicit, audited retrieval tree, parse a model proposal and call
`follow_proposed(previous_results, proposal)`. Proofline uses the previous
results' scope for that child search. Only trusted application code may use
`follow_up_trusted(..., narrowing_filters=...)` to restrict it further.

## Backend filter contract

Proofline supplies the filters; the backend is responsible for applying them.
It must reject filter fields it does not support, require every supplied field
to match, and return no matches for an empty list of permitted values.
`validate_scope_filter_fields` and
`matches_scope_filters` provide this contract for adapters whose candidate
metadata can be represented as scalar fields:

```python
from proofline import matches_scope_filters, validate_scope_filter_fields

SUPPORTED_FILTERS = frozenset({"tenant_id", "resource_id"})


def search(query, *, filters, limit):
    validate_scope_filter_fields(filters, supported_fields=SUPPORTED_FILTERS)
    candidates = (
        document
        for document in documents
        if matches_scope_filters(
            {"tenant_id": document.tenant_id, "resource_id": document.resource_id},
            filters,
            supported_fields=SUPPORTED_FILTERS,
        )
    )
    return rank(query, candidates, limit)
```

The matcher treats values with different Python scalar types as distinct. If a
backend needs a public-content exception, represent that content in the trusted
root allowlists or use a separate public retrieval path; do not bypass supplied
filters while selecting candidates.

## The problem

A document can be relevant to a question without being available to the user
asking it. Access restrictions must apply whenever an assistant searches,
including when it follows a reference in a document or asks a second question.

Proofline centralizes the work of carrying those restrictions between retrieval
steps. A follow-up receives the same or narrower scope as its parent. Switching
to a broader scope requires a separate authorization operation in trusted
application code.

The repository's reference demonstration also explores evidence quality,
authoritative permission decisions, and regression testing. Those experiments
support the library; they do not make it a chatbot or a benchmark for model
intelligence.

## Reference demonstration

The repository includes a public-data reference demonstration and evaluation
harness for the library. One permission scenario is:

```text
User: Can Ana see the production rollout guide?

1. The host application identifies a tenant-scoped permission question.
2. The authorization service evaluates
   check_access(user:ana, viewer, document:production-rollout-guide).
3. The decision is deny. Proofline does not retrieve the guide or pass any of
   its chunks to the model.
4. The assistant explains that access is unavailable. It may cite permitted
   public policy documentation, but it cannot explain protected content that it
   did not retrieve.
```

A relevant answer is still a failure if it exposes the wrong evidence.

## What the reference demonstration evaluates

- **Access-gated retrieval.** Before a tenant-scoped search returns content,
  the retrieval layer receives the requester's authorized tenant and resource
  scope. Results outside that scope cannot enter the candidate set, trace,
  prompt context, or response.
- **Permission-preserving multi-hop retrieval.** A trusted request scope is
  propagated to every retrieval step. A later step may narrow that scope, but
  retrieved text cannot supply a principal, tenant, resource filter, or
  authorization decision that widens it.
- **Bounded host-controlled retrieval.** The host may perform one tested,
  scope-preserving follow-up step before answering with citations or abstaining.
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

## What Proofline is and is not

Proofline's target product is a small, framework-neutral retrieval wrapper. It
is not a RAG framework, agent runtime, planner, vector store, prompt-injection
classifier, policy-language platform, chat application, or authorization engine.

LangChain, LangGraph, LlamaIndex, Haystack, custom Python loops, and other
systems can keep their own orchestration and document models. Proofline sits at
their retrieval boundary. Every retrieval routed through it receives the
authenticated caller's authorized scope or a stricter descendant scope.

It does not replace an application's retriever, vector database, planner, or
document schema; build an ANN algorithm, embedding model, or a fleet of agents;
or optimize a leaderboard score without explaining the observed failures and
trade-offs.

It does not claim to solve prompt injection, factual poisoning, or calls that
bypass the wrapper. Its narrow guarantee is architectural and testable:
retrieval performed through Proofline cannot implicitly receive broader
authority from model or retrieved-text output. It is a reference implementation
and evaluation harness, not a claim of production-grade identity, multi-tenancy,
or security.

## What it contains today

1. **The core library.** A framework-neutral scoped-retrieval wrapper with
   immutable scopes, monotonic attenuation, optional follow-up budgets, and a
   parser that rejects scope-bearing planner input.
2. **Core contract tests.** Tests cover immutable filters, scope widening,
   expiry, optional revalidation, follow-up budgets, and untrusted-step
   rejection.
3. **A reproducible public-data demonstration.** A version-pinned corpus and
   synthetic access relationships. Its `demo-tenant-search` command resolves
   authorization and routes the fixture through the core wrapper.
4. **A deterministic multi-hop fixture.** A clean two-hop path preserves its
   inherited scope; a synthetic scope-bearing planner proposal is rejected
   before it can trigger a follow-up retrieval.
5. **A public-data scope-propagation benchmark.** A pinned 50-case HotpotQA
   distractor-dev subset measures lexical evidence retrieval and applies a
   separate synthetic ACL/poison overlay across insecure, ACL-per-hop, and
   scoped-plan-policy controls. The checked-in
   [result report](reference-demo/benchmark-results/hotpotqa-distractor-dev-v1/bm25-k5-scope-overlay-v1.md)
   records the measured outcome.
6. **A bounded reference host.** Deterministic request routing uses a
   context-bound MCP permission tool or scoped retrieval, then returns a cited
   evidence response or an explicit abstention.

The repository also includes runnable custom-loop, FastAPI host, LangGraph,
Pydantic AI, and LlamaIndex examples. Each uses the same host-controlled
retrieval boundary.
The reference demo emits OpenTelemetry spans for scope resolution, retrieval,
MCP permission decisions, deterministic planner proposals, and response
composition when configured with an explicit OTLP endpoint, such as a local
Phoenix collector. It has no runtime model call to trace.

For local trace viewing, start Phoenix with
`docker compose --profile observability up phoenix` from `reference-demo`, then
run a reference-demo command with
`--otlp-endpoint http://localhost:6006/v1/traces`, for example
`uv run proofline-reference-demo --otlp-endpoint http://localhost:6006/v1/traces
demo-tenant-search --authorization static`. Phoenix’s UI and OTLP/HTTP collector
share port 6006; Proofline records IDs and counts, not passage text or raw
allowlists.

## Repository layout

The repository separates the library, reference demonstration, and examples:

```text
src/proofline/                  published library only
tests/                          library contract tests, Python 3.10–3.14

reference-demo/                 reproducible OpenFGA/Qdrant public-data demo
  src/proofline_reference_demo/
  data/
  tests/
  .env.example

examples/README.md              design constraints for future integrations
examples/host/                  installable shared fixture for framework examples
```

`proofline` has only core dependencies and never reads `.env`. The reference
demo owns OpenFGA, Qdrant, MCP, Pydantic Settings, tracing, public data, and
evaluation. Each framework example depends on its selected host framework and
the installable shared fixture package; the fixture depends on Proofline.

### Evaluation fixture

The demonstration corpus uses version-pinned, public OpenFGA documentation,
examples, and selected issue discussions. The source material is real. A
synthetic organization, tenant, project, resource, and relationship model
creates controlled allow and deny cases without using private information.

Every chunk carries source revision, tenant and resource identifiers, visibility,
and a source URL. Public chunks are available to every principal. Tenant-scoped
chunks are available only when the authorization backend includes their resource
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

Proofline passes an immutable, trusted filter set to its wrapped backend. The
backend is part of the security boundary: it must apply every supplied filter
as a conjunction, match an empty allowlist to no protected records, and reject
unknown filters rather than ignoring them. Proofline cannot secure a backend
that bypasses or misimplements that contract.

Its target adversarial evaluation additionally tests whether an authorized
shared document can steer a multi-hop planner toward an out-of-scope resource.
A valid proposed step carries a query and parent-step reference, not
scope-bearing fields. The wrapper rejects attempts to supply a principal,
tenant, resource filter, or ACL field, and executes every valid query only with
inherited, server-derived authorization filters. Query text is not treated as
reliable evidence of an intended authority change.

It does not claim to eliminate every information side channel or identify every
poisoned document. Response timing, result-count differences, factual
misinformation, broader identity/transport security, and misuse of resources
the caller is already allowed to access require separate controls. Those
boundaries are documented rather than silently treated as solved.

## Design principles

### Enforce access before evidence

The authorization service evaluates the caller and requested resource before
retrieval returns protected content. The retrieval system supplies evidence for
an explanation. The orchestrator keeps those responsibilities separate and
records the decision path.

### Attenuate authority across hops

A trusted application creates a request scope after authentication and
authorization. The model may propose a query, but not an effective principal,
tenant, resource filter, or permission. Every child retrieval scope is equal to
or narrower than its parent. A real workspace change is a separate trusted
authorization operation, never a side effect of retrieved text.

The recommended integration is a wrapped existing retriever:

```python
from proofline import RetrievalScope, scoped


def resolve_scope(request_context: RequestContext) -> RetrievalScope:
    # Trusted application/authentication code produces this scope.
    return RetrievalScope.root(
        principal=request_context.principal_id,
        filters={"resource_id": request_context.authorized_resource_ids},
    )


retriever = scoped(existing_retriever.search, resolve_scope=resolve_scope)
request_retriever = await retriever.bind(request_context_from_authenticated_user)
results = await request_retriever.search("rollout prerequisites", limit=5)
```

For an ordinary agent or RAG request, bind once and pass the bound retriever's
query-only `search` method to the tool or service. Independent searches share
that immutable authorized branch, including when they run concurrently. The
bound handle adds no mutable run state; your backend client remains responsible
for its own concurrency guarantees. The
lower-level explicit-parent API remains available for trees, parallel workers,
and custom policy flows; ordinary users should not manage scope algebra. A host
may provide `validate_scope` to recheck revocation before every retrieval call,
and may set `max_follow_ups` on a root scope when it needs a bounded branch.
Scope expiry is checked again immediately before backend dispatch, including
after an asynchronous validator returns; it does not cancel an already-dispatched
backend call. Both controls are optional and disabled by default. A root may
also carry immutable string `metadata` such as a trace or request ID; metadata
is propagated unchanged and is never used to grant retrieval authority.
Underneath, the wrapper targets the ordinary Python retrieval shape: a sync or
async callable/protocol that accepts `query`, enforced `filters`, and `limit`.
Synchronous callbacks run on the calling event loop. For a thread-safe blocking
SDK, explicitly opt into worker-thread execution with `offload_sync`; the host
remains responsible for client thread affinity, timeout, and cancellation policy:

```python
from proofline import offload_sync, scoped

retriever = scoped(offload_sync(existing_retriever.search), resolve_scope=resolve_scope)
```
It preserves the application's document/result model rather than imposing a
new one.

### Reference-demo principles

The following principles govern the public demonstration and evaluation harness;
they do not make Proofline an agent or retrieval framework.

#### Start with baselines

Each retrieval improvement must beat or clarify a measurable baseline. The
first comparison is lexical retrieval versus vector retrieval. Hybrid retrieval
and reranking are added only when the evaluation set shows a specific weakness
they address.

#### Keep host-controlled actions bounded and testable

The reference host has a small action budget: classify the question, resolve
authorization when needed, retrieve only permitted evidence, optionally perform
one scope-preserving follow-up retrieval, then answer or abstain. It does not
freely plan, call arbitrary tools, mutate authority, retrieve after a denial,
or iterate until it appears confident. Every step is part of the trace and
evaluation contract.

#### Make failure a first-class result

"I do not have enough evidence" and "I cannot determine access without the
authorization tool" are valid outcomes. The system should abstain rather than
invent a citation, a policy interpretation, or a permission result.

#### Prefer deterministic checks where possible

Tool selection, authorization results, response schemas, citation IDs, and
required refusals should be checked deterministically. Model-based graders are
reserved for open-ended properties such as explanation quality and whether an
answer is well grounded. They will be calibrated against a small human-reviewed
set.

#### Keep every result reproducible

Corpus, evaluation-suite, retrieval-method, and embedding-provider versions are
recorded by the evaluation artifacts that use them. The demo has no prompt or
chat-model version because it makes no runtime model call.

## Reference-demo flow

```text
version-pinned corpus
        |
        v
ingest -> chunk + attach access metadata -> tenant/public indexes

user question
        |
        v
trusted request scope ---- permission question? ----> check_access
        |                         |                 |
        |                         |                 +--> allow or deny
        |                         v
        |                 resolve authorized scope
        |                         |
        v                         v
host proposes step ------> scoped retrieval wrapper ---> ranked evidence
        |                         |                         |
        |                         +--> reject scope-bearing input
        +-------------------------+-------------------------+
                                  |
                  bounded follow-up under equal-or-narrower scope
                                  |
                                  v
                         one additional filtered retrieval
                                  |
                                  v
                    cited answer, denial, or abstention
                                  |
                                  v
                         trace + evaluation runner
```

The scoped wrapper applies scope before returning candidates. Public and
tenant-scoped indexes are separate namespaces. A policy-derived resource
allowlist further constrains tenant-scoped retrieval. This makes access control
a property of every retrieval hop, not a filter applied after an LLM has seen
the results.

Question classification, planner fixtures, and response composition are
deterministic. The reference demonstration makes no runtime LLM call. Its
responses acknowledge cited evidence or abstain; they are not answer-quality
or generation evaluations.

## Reference-demo retrieval experiments

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

## Reference-demo evaluation

The checked-in release suite contains 50 hand-authored, versioned retrieval and
permission cases. A separate deterministic scope-propagation gate compares
clean, benign, and poisoned multi-hop counterparts across three controls: an
intentionally insecure baseline, ACL filtering on every hop, and Proofline's
scoped plan policy.

The broader public-data gate uses the pinned 50-case HotpotQA distractor-dev
bridge-question subset. It measures BM25 retrieval against HotpotQA's supplied
supporting-title labels, then applies a synthetic ACL and scope-bearing-plan
overlay without modifying the questions or passages. It is not an answer
accuracy or universal poisoning claim. The
[versioned result report](reference-demo/benchmark-results/hotpotqa-distractor-dev-v1/bm25-k5-scope-overlay-v1.md)
records its exact data hash, configuration, and control comparison.

See the [comparison with access filtering alone](#comparison-with-access-filtering-alone)
near the top of this README for the reported results.

To reproduce the public-data gate locally, download the artifact named by its
manifest, then run the evaluator; it verifies the recorded SHA-256 before
parsing the file:

```bash
curl --fail --location --output hotpot_dev_distractor_v1.json \
  https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json
cd reference-demo
uv run proofline-reference-demo evaluate-hotpotqa \
  --dataset ../hotpot_dev_distractor_v1.json
```

| Evaluation layer | What it checks | Primary measure |
| --- | --- | --- |
| Access isolation | Protected content stays out of unauthorized retrieval and prompts | unauthorized-chunk exposure rate, cross-tenant leakage pass rate |
| Retrieval | Whether permitted relevant evidence appears in the candidate set and near the top | ACL-filtered Recall@k, MRR, nDCG |
| Evidence provenance | Whether returned chunks have reviewed source metadata | deterministic citation-provenance checks |
| Abstention | Whether unsupported or ambiguous questions avoid invented answers | exact expected outcome |
| Tool behavior | Whether access questions call the tool with correct arguments | tool-call and result assertions |
| Authorization | Whether allowed and denied cases match the policy model | exact expected decision |
| Scope propagation | Whether every child retrieval is equal to or narrower than its parent and poison cannot supply authority | rejected scope-bearing inputs, unauthorized exposure, scope-lineage checks |
| Host trace | Whether follow-up retrieval follows the permitted action budget and inherited scope | trace assertions and pass rate |
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
- Cross-tenant, partial-access, and identifier-guessing probes.
- Authorized shared documents containing realistic instruction-like
  cross-references, to test that documents are evidence rather than authority.
- Clean and poisoned multi-hop counterparts, plus benign documents that discuss
  security, to measure both attack handling and false blocks.

## Reference-demo quality gates

A change should fail CI when it:

- changes a known authorization decision;
- omits a required tool call or calls the tool with incorrect arguments;
- produces a citation that is absent or does not support the answer;
- fails a mandatory abstention case;
- exposes an unauthorized chunk;
- permits an authority-expanding retrieval step without a separate trusted
  authorization operation;
- exceeds the allowed agent action budget; or
- regresses a protected retrieval or end-to-end metric beyond the agreed
  tolerance.

The project should report trade-offs, not hide them. For example, a reranker
may improve nDCG while increasing latency. A result is useful even when the
new method loses, provided the evaluation explains why.

## Reference-demo trace record

The current generated evidence artifacts contain the following safe fields when
their workflow produces them:

- case and corpus version;
- request mode and resolved access scope;
- scope lineage references and rejected proposal fields for scope-gate traces;
- retrieval method, candidates, ranks, and scores;
- the permitted chunk IDs supplied to response context;
- evidence provenance, including source revision and parent retrieval step;
- host decision and tool calls with redacted inputs and outputs;
- cited-evidence response or abstention state; and
- deterministic gate results.

No secrets, private documents, or personal data belong in the repository.

## Reference-demo operational view

The evaluation commands report the metrics their writers actually compute:
retrieval quality, unauthorized exposure, provenance checks, and latency where
the relevant retrieval evaluator measures it. They do not currently produce a
complete per-run cost, token, answer-quality, or regression-delta report.

## Technology choices

### Core library

| Concern | Choice | Role in Proofline |
| --- | --- | --- |
| Compatibility | Python 3.10–3.14 | Target public-library support; Python 3.11+ is recommended. The reference demonstration remains on Python 3.13. |
| Integration contract | Python protocol/callable | Wraps a host retriever using `query`, enforced `filters`, and `limit`, without imposing a document model. |
| Contracts | Standard-library dataclasses and protocols | Typed, 3.10-compatible public contracts without imposing Pydantic on the host application. |

### Reference demonstration

| Concern | Choice | Role in the demonstration |
| --- | --- | --- |
| Authorization | OpenFGA and `openfga-sdk` | `ListObjects` resolves permitted scope; `Check` makes authoritative permission decisions. |
| Retrieval | BM25, Qdrant, and `qdrant-client` | Compares lexical, dense, hybrid, and reranked retrieval under the same filters. |
| Tool boundary | Official MCP Python SDK | Demonstrates an authoritative `check_access` tool. |
| Configuration | Pydantic Settings and local `.env` | Configures the demonstration only; the core library never reads host environment or secret files. |
| Evaluation | `pytest` and local metric implementations | Provides deterministic assertions and Recall@k, MRR, and nDCG calculations. |
| Traces | OpenTelemetry and Phoenix | Supports trace visualization; the structured trace remains the source artifact. |

Embedding and reranker providers are demonstration components selected by
evaluation results, not dependencies of the core library.

The demonstration index can be rebuilt from the pinned corpus. Optional library
policy layers—provenance export, revocation, cache partitioning, approval,
budgets, and tool governance—must be disabled by default and document their
latency, storage, or operational cost when enabled.

## Run the reference demonstration locally

Prerequisites for the current reference demonstration: Python 3.13+,
[uv](https://docs.astral.sh/uv/), and Docker. The core library supports Python
3.10+ independently of this demonstration runtime.

```bash
cd reference-demo
uv sync --all-groups
docker compose up -d
uv run proofline-reference-demo --help
```

Run the foundation checks with:

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

Run the deterministic access-gated fixture and inspect its JSON trace with:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline-reference-demo demo-tenant-search
```

Run the deterministic clean and poisoned two-hop fixtures with the static test
authorization adapter:

```bash
uv run proofline-reference-demo demo-multi-hop --scenario clean
uv run proofline-reference-demo demo-multi-hop --scenario poisoned
```

The poisoned fixture is a deterministic test of the planner-input boundary; it
does not claim to detect arbitrary prompt injection or document poisoning.

Run the CI-friendly scope-propagation release gate:

```bash
uv run proofline-reference-demo evaluate
```

It exits nonzero if the scoped configuration accepts a scope-bearing planner
input, exposes unauthorized evidence, loses clean or benign follow-up behavior,
exceeds the rejected-step budget, or records incomplete scope lineage.

Run the bounded reference host with a tenant-knowledge or permission request:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline-reference-demo query \
  --query "What approval does Acme need for rollout?"
OPENFGA_URL=http://localhost:8080 uv run proofline-reference-demo query \
  --query "Can Ana view the Beta rollout?" \
  --tenant tenant:beta --resource document:beta-rollout
```

The fixture's MCP server exposes only `check_access` over stdio. Its caller
context is bound when the server starts, so the tool accepts `relation` and
`resource_id`, never a model-supplied principal or tenant:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline-reference-demo serve-mcp \
  --principal user:ana --tenant tenant:acme
```

Both commands default to a temporary store containing the checked-in OpenFGA
model and tuples. `--authorization static` is an explicit offline
direct-viewer fixture only: it does not evaluate inherited relationships and
must not be used as evidence of policy-engine behavior.

The same fixture exposes one authoritative permission decision with:

```bash
OPENFGA_URL=http://localhost:8080 uv run proofline-reference-demo demo-check-access
```

Build and evaluate the pinned documentation corpus with a checkout at the
revision named in `data/corpus/manifest.yaml`:

```bash
uv run proofline-reference-demo evaluate-lexical --source-root /path/to/openfga.dev
```

This writes `artifacts/lexical-baseline.md` and one inspectable trace per
retrieval-path case to `artifacts/lexical-baseline-traces.jsonl`. It scores the
reviewed evidence-retrieval cases at a fixed `k` and records latency plus
independent access-scope and citation-provenance checks. Permission cases remain
outside these ranking metrics; the scope-propagation release gate is evaluated
separately.
The offline baseline derives direct-viewer grants from the checked-in OpenFGA
tuples; inherited relationship behavior remains covered by the OpenFGA backend
and integration tests.

Write the reviewed scope-propagation gate and its redacted clean, benign, and
rejected-proposal traces to stable versioned paths:

```bash
uv run proofline-reference-demo report
```

This creates `artifacts/scope-propagation-v0/scope-propagation-v0-report.json`
and `artifacts/scope-propagation-v0/scope-propagation-v0-traces.jsonl`. To
persist the HotpotQA runtime overlay evidence as well, pass `--output` and
`--traces-output` to `evaluate-hotpotqa`.

With local Qdrant running, compare its access-filtered dense-vector control to
BM25 on the same corpus and cases:

```bash
uv run proofline-reference-demo evaluate-dense --source-root /path/to/openfga.dev
```

Pass `--recreate` to replace that command's named local collection on a repeat
run. To use the optional learned FastEmbed provider, install it in an ONNX
Runtime-compatible environment with `uv sync --extra fastembed`, then pass its
model ID through `--embedding-model`.

For OpenAI embeddings, set `OPENAI_API_KEY` in the project's ignored local
`.env`; the typed Pydantic settings boundary loads it automatically, while a
shell environment variable takes precedence. Then select
`--embedding-model openai:text-embedding-3-small`. This calls the embeddings
endpoint only; it does not use a chat or reasoning model.

The built-in token-hash embedding is deterministic and zero-cost; it validates
the Qdrant boundary and provides a reproducible control, not a claim of
semantic-model quality. Compare learned models with the same ACL-filtered suite
through `evaluate-hybrid`; reports record quality, latency, index size, and
embedding/query cost for each run.

To run the real OpenFGA policy integration test, start the local service and
set its URL for pytest:

```bash
docker compose up -d openfga
OPENFGA_URL=http://localhost:8080 uv run pytest -m integration --no-cov
```

The focused integration command disables the repository-wide coverage gate;
run `uv run pytest` without test selection to enforce the 85% combined coverage
threshold computed by coverage.py across statements and branches.

Branch collection is enabled to expose untested decision paths, but CI does not
currently enforce a separate branch-only percentage threshold.

## How to extend it

Proofline is designed to make an extension falsifiable. Add one capability,
add or revise the relevant evaluation cases, and compare it with the existing
baseline. A feature is only an improvement if the evaluation results support
it.

Useful extensions include:

- **Retriever backends.** Implement the standard retrieval protocol for a
  different vector store, search engine, or application-specific retriever.
  Proofline supplies authorization-derived filters; the host keeps its document
  and result model.
- **Policy layers.** Add opt-in provenance export, revocation checks, cache
  partitioning, approval requirements, retrieval budgets, or tool governance.
  Document the layer's latency, storage, and operational cost.
- **Execution shapes.** Add parallel branches, a deeper bounded retrieval tree,
  or a framework integration. The library remains at the retrieval boundary;
  the host still owns planning and orchestration.
- **Evaluation fixtures.** Add a versioned public corpus and synthetic access
  relationships for a new domain. Preserve source provenance, resource
  boundaries, clean tasks, adversarial counterparts, and deterministic checks.

Every extension may narrow, approve, or reject authority; it must never silently
widen it. The project should not grow into a collection of integrations. Each
extension should clarify one engineering question, establish a baseline, and
leave behind a reproducible result.

## License

Proofline is available under the [MIT License](LICENSE).

That is the proof line: a traceable path from authorized source evidence and
authoritative tools to a response that can be inspected, tested, and improved.
