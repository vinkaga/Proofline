# HotpotQA distractor-dev v1: BM25 k=5 with scope overlay v1

## Run identity

| Field | Value |
| --- | --- |
| Code base | `767ca8e592ba0ce06c7a16e4d36dc8203e2c1d31` plus uncommitted Phase 7 benchmark changes |
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

## Interpretation

The retrieval result is a transparent lexical baseline, not an answer-quality
claim: 34 of 50 questions retrieved all gold supporting titles in the top five;
the other 16 missed at least one title. The ACL/poison overlay is evaluated
separately from the benchmark's source data. Each case runs through actual
ACL-filtered retrieval: all returned resources must be in the caller's allowed
set, the protected resource must not be returned, a scope-bearing proposal must
be rejected, and a data-only proposal remains allowed.

Run the evaluation with:

```bash
cd reference-demo
uv run proofline-reference-demo evaluate-hotpotqa \
  --dataset ../hotpot_dev_distractor_v1.json
```

This report deliberately does not claim answer accuracy or LLM grounding:
Proofline does not supply a generator, and no calibrated external judge was
used for this run.
