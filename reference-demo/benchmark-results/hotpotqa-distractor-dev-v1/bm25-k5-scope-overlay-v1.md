# HotpotQA distractor-dev v1: BM25 k=5 with scope overlay v1

## Run identity

| Field | Value |
| --- | --- |
| Benchmark manifest | `data/benchmarks/hotpotqa-distractor-dev.yaml` (`hotpotqa-distractor-dev-v1`) |
| Dataset | HotpotQA distractor development set |
| Dataset SHA-256 | `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` |
| License | CC BY-SA 4.0 |
| Subset | First 50 `bridge` cases after lexical `_id` ordering |
| Retrieval | BM25 over each question's 10 official distractor contexts, `k=5` |
| Overlay | `scope-overlay-v1`: synthetic ACLs and proposals; source records unchanged |

## Results

| Measure | Result |
| --- | ---: |
| Supporting-title Recall@5 | 0.830 |
| Complete gold-supporting-evidence coverage@5 | 0.680 |
| ACL-overlay traces | 50 |
| ACL clean supporting-evidence availability | 1.000 |
| ACL unauthorized-evidence exposure | 0.000 |
| Scope-bearing poisoned-proposal rejection | 1.000 |
| Benign data-only proposal acceptance | 1.000 |

## Control comparison

The source questions and contexts are identical across controls. The ACL labels
and the scope-bearing proposal are synthetic overlay data. The intentionally
insecure control is a test control: it treats the proposal's protected
`resource_id` as an effective retrieval selector. It establishes that the
fixture can expose protected evidence; it is not a deployable baseline.

| Control | Accepts scope-bearing input | Protected-evidence exposure | Rejects before retrieval | Complete child-scope lineage |
| --- | ---: | ---: | ---: | ---: |
| Insecure baseline | 1.000 | 1.000 | 0.000 | 0.000 |
| ACL-filtered per hop | 1.000 | 0.000 | 0.000 | 0.000 |
| Proofline scoped-plan policy | 0.000 | 0.000 | 1.000 | 1.000 |

## Interpretation

The retrieval result is a transparent lexical baseline, not an answer-quality
claim: 34 of 50 questions retrieved all gold supporting titles in the top five;
the other 16 missed at least one title. The ACL/poison overlay is evaluated
separately from the benchmark's source data. ACL filtering alone prevents
exposure after an unsafe planner input has been accepted. Proofline adds the
narrower property that the scope-bearing proposal is rejected before it can
produce a second retrieval; ordinary data-only follow-up retains its
child-scope lineage.

Run the evaluation with:

```bash
curl --fail --location --output hotpot_dev_distractor_v1.json \
  https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json
cd reference-demo
uv run proofline-reference-demo evaluate-hotpotqa \
  --dataset ../hotpot_dev_distractor_v1.json
```

This report deliberately does not claim answer accuracy or LLM grounding:
Proofline does not supply a generator, and no calibrated external judge was
used for this run.
