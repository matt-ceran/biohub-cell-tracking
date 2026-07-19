#!/usr/bin/env python3
"""Freeze one CNN candidate policy selected only from complete internal-dev results.

The selected policy must use a positive cellness threshold, stay within the locked node
density guardrail, and improve mean internal-dev edge Jaccard over the unfiltered DoG
baseline at prune length 8. Only such a policy can unlock the one-time 44b6 evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance_evaluation import (  # noqa: E402
    BASELINE_MIN_TRACK_LENGTH,
    BASELINE_THRESHOLD,
    CANDIDATE_POLICY_SCHEMA_VERSION,
    CANDIDATE_SELECTION_RULE,
    LOCKED_NODE_DENSITY_LIMIT,
    aggregate_policy_rows,
    file_sha256,
    require_complete_checkpoint_dev,
    select_policy,
    write_frozen_policy,
)
from biohub.appearance_model import load_cellness_checkpoint  # noqa: E402
from scripts.validate_appearance import read_result_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_internal_dev.csv",
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
        "--output",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_frozen_candidate_policy.json",
    )
    parser.add_argument("--density-limit", type=float, default=LOCKED_NODE_DENSITY_LIMIT)
    args = parser.parse_args()

    if not args.results.is_file():
        parser.error(f"internal-dev results do not exist: {args.results}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output.exists():
        parser.error(f"frozen policy already exists: {args.output}")
    if args.density_limit <= 0.0:
        parser.error("--density-limit must be positive")
    if args.density_limit > LOCKED_NODE_DENSITY_LIMIT:
        parser.error(
            f"--density-limit cannot exceed the locked limit {LOCKED_NODE_DENSITY_LIMIT:.12f}"
        )

    rows = read_result_rows(args.results)
    if any(row["cohort"] != "internal-dev" for row in rows):
        parser.error("policy selection accepts only internal-dev rows")
    checkpoint_sha256 = file_sha256(args.checkpoint)
    try:
        _, checkpoint = load_cellness_checkpoint(args.checkpoint, device="cpu")
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid checkpoint: {exc}")
    row_checkpoint_ids = {row["checkpoint_sha256"] for row in rows}
    if row_checkpoint_ids != {checkpoint_sha256}:
        parser.error("result rows do not belong exclusively to the requested checkpoint")
    evaluation_ids = {row["evaluation_sha256"] for row in rows}
    if len(evaluation_ids) != 1:
        parser.error("result rows mix different evaluation source fingerprints")
    evaluation_sha256 = next(iter(evaluation_ids))
    try:
        aggregates = aggregate_policy_rows(rows)
        require_complete_checkpoint_dev(aggregates, checkpoint)
        selected, baseline = select_policy(aggregates, density_limit=args.density_limit)
    except ValueError as exc:
        parser.error(str(exc))

    by_dataset = {
        (row["dataset"], row["score_threshold"], row["min_track_length"]): row for row in rows
    }
    wins = ties = losses = 0
    for dataset in selected["datasets"]:
        learned = by_dataset[(dataset, selected["score_threshold"], selected["min_track_length"])][
            "edge_jaccard"
        ]
        classical = by_dataset[(dataset, BASELINE_THRESHOLD, BASELINE_MIN_TRACK_LENGTH)][
            "edge_jaccard"
        ]
        if learned > classical:
            wins += 1
        elif learned < classical:
            losses += 1
        else:
            ties += 1

    policy = {
        "candidate_policy_schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "selection_cohort": "internal-dev",
        "selection_datasets": selected["datasets"],
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_sha256": evaluation_sha256,
        "source_results_sha256": file_sha256(args.results),
        "score_threshold": selected["score_threshold"],
        "min_track_length": selected["min_track_length"],
        "density_limit": args.density_limit,
        "selection_rule": CANDIDATE_SELECTION_RULE,
        "internal_improvement_supported": True,
        "selected_metrics": {
            key: selected[key]
            for key in (
                "mean_edge_jaccard",
                "median_edge_jaccard",
                "aggregate_node_ratio",
                "mean_recall",
                "mean_runtime_seconds",
            )
        },
        "baseline_metrics": {
            key: baseline[key]
            for key in (
                "mean_edge_jaccard",
                "median_edge_jaccard",
                "aggregate_node_ratio",
                "mean_recall",
                "mean_runtime_seconds",
            )
        },
        "mean_edge_jaccard_gain": (selected["mean_edge_jaccard"] - baseline["mean_edge_jaccard"]),
        "per_movie_wins": wins,
        "per_movie_ties": ties,
        "per_movie_losses": losses,
    }
    write_frozen_policy(args.output, policy)
    print(
        f"Frozen threshold={selected['score_threshold']:.3f}, "
        f"min_track_length={selected['min_track_length']}"
    )
    print(
        f"Internal mean edge Jaccard: {baseline['mean_edge_jaccard']:.6f} -> "
        f"{selected['mean_edge_jaccard']:.6f}"
    )
    print(f"Aggregate node ratio: {selected['aggregate_node_ratio']:.6f}")
    print(f"Per-movie wins/ties/losses: {wins}/{ties}/{losses}")
    print(f"Frozen policy: {args.output}")


if __name__ == "__main__":
    main()
