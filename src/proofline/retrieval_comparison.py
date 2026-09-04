# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Write measured comparisons for access-filtered retrieval configurations."""

from pathlib import Path

from proofline.dense_retrieval import DenseIndexMetadata
from proofline.lexical_evaluation import LexicalBaselineMeasurement


def write_method_comparison_report(
    measurements: dict[str, LexicalBaselineMeasurement],
    index: DenseIndexMetadata,
    output: Path,
) -> None:
    """Report comparable quality, latency, and changed evidence cases by method."""

    if "lexical" not in measurements:
        raise ValueError("comparison requires a lexical baseline")
    baseline = measurements["lexical"]
    lines = [
        "# Retrieval comparison",
        "",
        f"Embedding model: `{index.embedding_model}` · dimensions: {index.dimensions}",
        (
            f"Corpus revision: `{index.corpus_revision}` · index size: "
            f"{index.estimated_vector_index_bytes:,} bytes"
        ),
        (
            f"Estimated embedding cost: ${index.estimated_embedding_cost_usd:.6f} · "
            f"query cost: ${index.estimated_query_cost_usd:.6f}"
        ),
        "",
        "| Method | Recall@k | MRR | nDCG@k | Exposure | p50 latency | p95 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, measurement in measurements.items():
        lines.append(
            f"| {name} | {measurement.recall_at_k:.3f} | {measurement.mrr:.3f} | "
            f"{measurement.ndcg_at_k:.3f} | {measurement.unauthorized_exposure_rate:.3f} | "
            f"{measurement.p50_latency_ms:.2f} ms | {measurement.p95_latency_ms:.2f} ms |"
        )
    lines.extend(["", "## Changed evidence-retrieval cases", ""])
    for name, measurement in measurements.items():
        if name == "lexical":
            continue
        changed = _changes(baseline, measurement)
        lines.append(f"### {name}")
        lines.extend(f"- `{case_id}` — {outcome}" for case_id, outcome in changed)
        if not changed:
            lines.append("- None")
        lines.append("")
    # Prefer evidence ordering first, then retrieval coverage, then tail latency.
    preferred = max(
        measurements,
        key=lambda name: (
            measurements[name].ndcg_at_k,
            measurements[name].mrr,
            measurements[name].recall_at_k,
            -measurements[name].p95_latency_ms,
        ),
    )
    lines.extend(["## Selection", "", f"Preferred configuration: `{preferred}`.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def _hits(measurement: LexicalBaselineMeasurement) -> dict[str, bool]:
    return {
        case.case_id: bool(
            case.relevant_identifiers
            & {identifier for _, identifier in case.ranked_identifiers}
        )
        for case in measurement.cases
        if case.relevant_identifiers
    }


def _changes(
    baseline: LexicalBaselineMeasurement, candidate: LexicalBaselineMeasurement
) -> list[tuple[str, str]]:
    baseline_ranks = _first_ranks(baseline)
    candidate_ranks = _first_ranks(candidate)
    changes: list[tuple[str, str]] = []
    for case_id, baseline_rank in baseline_ranks.items():
        rank = candidate_ranks[case_id]
        if rank is None and baseline_rank is not None:
            changes.append((case_id, "regressed: relevant evidence dropped"))
        elif rank is not None and baseline_rank is None:
            changes.append((case_id, "improved: relevant evidence recovered"))
        elif rank is not None and baseline_rank is not None and rank != baseline_rank:
            direction = "improved" if rank < baseline_rank else "regressed"
            changes.append((case_id, f"{direction}: first relevant rank {baseline_rank} → {rank}"))
    return changes


def _first_ranks(measurement: LexicalBaselineMeasurement) -> dict[str, int | None]:
    return {
        case.case_id: next(
            (
                rank
                for rank, identifier in case.ranked_identifiers
                if identifier in case.relevant_identifiers
            ),
            None,
        )
        for case in measurement.cases
        if case.relevant_identifiers
    }
