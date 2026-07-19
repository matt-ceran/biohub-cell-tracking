#!/usr/bin/env python3
"""Compare a frozen Phase 10 holdout run with the exact Phase 8 baseline rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance_evaluation import (  # noqa: E402
    file_sha256,
    read_frozen_policy,
)
from scripts.validate_appearance import read_result_rows as read_appearance_rows  # noqa: E402
from scripts.validate_pruning import read_result_rows as read_baseline_rows  # noqa: E402

SUMMARY_SCHEMA_VERSION = 1
EXPECTED_HOLDOUT_MOVIES = 71
BASELINE_DETECTOR_K = 3.0
BASELINE_MIN_TRACK_LENGTH = 8


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the exact headline metrics shared by baseline and learned rows."""
    if not rows:
        raise ValueError("cannot aggregate an empty result set")
    edge_scores = np.array([row["edge_jaccard"] for row in rows], dtype=float)
    return {
        "mean_edge_jaccard": float(edge_scores.mean()),
        "median_edge_jaccard": float(np.median(edge_scores)),
        "aggregate_node_ratio": float(
            sum(row["predicted_nodes"] for row in rows)
            / sum(row["estimated_nodes"] for row in rows)
        ),
        "mean_sparse_label_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
    }


def compare_rows(
    learned_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require identical movies and report exact per-movie wins, ties, and losses."""
    learned_by_dataset = {row["dataset"]: row for row in learned_rows}
    baseline_by_dataset = {row["dataset"]: row for row in baseline_rows}
    if len(learned_by_dataset) != len(learned_rows):
        raise ValueError("learned holdout results contain duplicate movies")
    if len(baseline_by_dataset) != len(baseline_rows):
        raise ValueError("baseline results contain duplicate movies")
    if set(learned_by_dataset) != set(baseline_by_dataset):
        raise ValueError("learned and baseline results do not contain identical movies")

    wins = ties = losses = 0
    per_movie_deltas: dict[str, float] = {}
    for dataset in sorted(learned_by_dataset):
        delta = (
            learned_by_dataset[dataset]["edge_jaccard"]
            - baseline_by_dataset[dataset]["edge_jaccard"]
        )
        per_movie_deltas[dataset] = float(delta)
        if delta > 0.0:
            wins += 1
        elif delta < 0.0:
            losses += 1
        else:
            ties += 1

    learned = aggregate_rows(learned_rows)
    baseline = aggregate_rows(baseline_rows)
    mean_gain = learned["mean_edge_jaccard"] - baseline["mean_edge_jaccard"]
    density_within_guardrail = learned["aggregate_node_ratio"] <= baseline["aggregate_node_ratio"]
    return {
        "learned": learned,
        "baseline": baseline,
        "mean_edge_jaccard_gain": float(mean_gain),
        "aggregate_node_ratio_change": float(
            learned["aggregate_node_ratio"] - baseline["aggregate_node_ratio"]
        ),
        "per_movie_wins": wins,
        "per_movie_ties": ties,
        "per_movie_losses": losses,
        "per_movie_edge_jaccard_deltas": per_movie_deltas,
        "edge_improved": mean_gain > 0.0,
        "density_within_guardrail": density_within_guardrail,
        "phase10_success": mean_gain > 0.0 and density_within_guardrail,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Atomically write the final comparison record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learned-results",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_locked_holdout.csv",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "data" / "working" / "phase8_full71_validation.csv",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_frozen_candidate_policy.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "data"
        / "working"
        / "appearance_checkpoints"
        / "cellness_cnn_phase10_full.pt",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_locked_comparison.json",
    )
    args = parser.parse_args()

    for name, path in (
        ("learned results", args.learned_results),
        ("baseline results", args.baseline_results),
        ("frozen policy", args.policy),
        ("checkpoint", args.checkpoint),
    ):
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
    if args.output_json.exists():
        parser.error(f"summary output already exists: {args.output_json}")

    policy = read_frozen_policy(args.policy)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if policy["checkpoint_sha256"] != checkpoint_sha256:
        parser.error("frozen policy belongs to a different checkpoint")

    learned_rows = read_appearance_rows(args.learned_results)
    if len(learned_rows) != EXPECTED_HOLDOUT_MOVIES:
        parser.error(
            f"learned results require {EXPECTED_HOLDOUT_MOVIES} rows, found {len(learned_rows)}"
        )
    if any(row["cohort"] != "locked-holdout" for row in learned_rows):
        parser.error("learned results contain rows outside the locked holdout")
    if any(row["checkpoint_sha256"] != checkpoint_sha256 for row in learned_rows):
        parser.error("learned results contain a different checkpoint identity")
    if any(row["evaluation_sha256"] != policy["evaluation_sha256"] for row in learned_rows):
        parser.error("learned results use different evaluation source from the policy")
    if any(
        not np.isclose(row["score_threshold"], policy["score_threshold"])
        or row["min_track_length"] != policy["min_track_length"]
        for row in learned_rows
    ):
        parser.error("learned results do not use the exact frozen candidate policy")

    baseline_rows = [
        row
        for row in read_baseline_rows(args.baseline_results)
        if np.isclose(row["threshold_k"], BASELINE_DETECTOR_K)
        and row["min_track_length"] == BASELINE_MIN_TRACK_LENGTH
    ]
    if len(baseline_rows) != EXPECTED_HOLDOUT_MOVIES:
        parser.error(
            f"baseline requires {EXPECTED_HOLDOUT_MOVIES} rows, found {len(baseline_rows)}"
        )
    try:
        comparison = compare_rows(learned_rows, baseline_rows)
    except ValueError as exc:
        parser.error(str(exc))

    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_sha256": policy["evaluation_sha256"],
        "frozen_policy_sha256": file_sha256(args.policy),
        "learned_results_sha256": file_sha256(args.learned_results),
        "baseline_results_sha256": file_sha256(args.baseline_results),
        "score_threshold": policy["score_threshold"],
        "min_track_length": policy["min_track_length"],
        **comparison,
    }
    write_summary(args.output_json, summary)

    learned = comparison["learned"]
    baseline = comparison["baseline"]
    print(
        f"Mean edge Jaccard: {baseline['mean_edge_jaccard']:.6f} -> "
        f"{learned['mean_edge_jaccard']:.6f} "
        f"({comparison['mean_edge_jaccard_gain']:+.6f})"
    )
    print(
        f"Aggregate node ratio: {baseline['aggregate_node_ratio']:.6f} -> "
        f"{learned['aggregate_node_ratio']:.6f}"
    )
    print(
        "Per-movie wins/ties/losses: "
        f"{comparison['per_movie_wins']}/"
        f"{comparison['per_movie_ties']}/"
        f"{comparison['per_movie_losses']}"
    )
    print(f"Phase 10 success: {comparison['phase10_success']}")
    print(f"Saved comparison: {args.output_json}")


if __name__ == "__main__":
    main()
