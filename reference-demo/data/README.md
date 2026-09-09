<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Vinay Agarwal. This directory defines reproducible corpus and evaluation inputs. -->

# Proofline data contracts

This directory contains versioned specifications, not downloaded source content
or generated artifacts. `corpus/manifest.yaml` identifies the exact public
source revision that ingestion must fetch. `eval/release-v0.yaml` contains the
50-case `release-v1` hand-authored suite used to evaluate retrieval,
authorization, and abstention. `benchmarks/hotpotqa-distractor-dev.yaml` pins a
separate public HotpotQA artifact and its synthetic access-control overlay.
`openfga/model.json` and `openfga/tuples.yaml` define the synthetic policy used
by the access-gated fixture.

Keeping these inputs small, reviewable, and versioned makes a result
reproducible and prevents the corpus or quality definition from changing
silently.
