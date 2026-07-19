#!/usr/bin/env python3
"""Evaluate CNN-filtered DoG candidates through the complete tracking pipeline.

Internal development is the default and uses only the whole-movie ``6bba`` shards
recorded in the checkpoint. The locked ``44b6`` cohort cannot be opened unless a
separately frozen internal-development policy is supplied and explicitly confirmed.

Candidate scores are cached once per checkpoint. Each requested score threshold then
reruns whole-movie min-cost flow before applying the pruning frontier, because filtering
can change which links the global solver selects.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import (  # noqa: E402
    APPEARANCE_HOLDOUT_EMBRYO,
    APPEARANCE_TRAIN_EMBRYO,
    filter_candidates,
)
from biohub.appearance_evaluation import (  # noqa: E402
    file_sha256,
    files_sha256,
    read_frozen_policy,
    read_score_cache,
    verify_frozen_policy_results,
    write_score_cache,
)
from biohub.appearance_model import (  # noqa: E402
    load_cellness_checkpoint,
    resolve_device,
    score_candidates,
)
from biohub.detect import DetectorConfig, detect_movie  # noqa: E402
from biohub.io import SamplePaths, discover_split, open_image_array, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402

EXPECTED_HOLDOUT_MOVIES = 71
DEFAULT_THRESHOLDS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_MIN_LENGTHS = (1, 2, 3, 8)
SCORING_SOURCE_FILES = [
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "appearance_model.py",
    SRC / "biohub" / "detect.py",
    SRC / "biohub" / "io.py",
    SRC / "biohub" / "constants.py",
]
EVALUATION_SOURCE_FILES = [
    *SCORING_SOURCE_FILES,
    SRC / "biohub" / "appearance_evaluation.py",
    SRC / "biohub" / "link.py",
    SRC / "biohub" / "metric.py",
    Path(__file__).with_name("freeze_appearance_policy.py"),
    Path(__file__).resolve(),
]

RESULT_FIELDS = [
    "dataset",
    "cohort",
    "checkpoint_sha256",
    "evaluation_sha256",
    "score_threshold",
    "min_track_length",
    "raw_candidates",
    "kept_candidates",
    "kept_fraction",
    "detect_seconds",
    "score_seconds",
    "link_seconds",
    "runtime_seconds",
    "edge_jaccard",
    "score",
    "division_jaccard",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "edge_ignored",
    "division_tp",
    "division_fp",
    "division_fn",
    "matched_nodes",
    "predicted_nodes",
    "truth_nodes",
    "estimated_nodes",
    "predicted_to_estimated",
    "recall",
]

FLOAT_FIELDS = {
    "score_threshold",
    "kept_fraction",
    "detect_seconds",
    "score_seconds",
    "link_seconds",
    "runtime_seconds",
    "edge_jaccard",
    "score",
    "division_jaccard",
    "predicted_to_estimated",
    "recall",
}
INT_FIELDS = {
    "min_track_length",
    "raw_candidates",
    "kept_candidates",
    "edge_tp",
    "edge_fp",
    "edge_fn",
    "edge_ignored",
    "division_tp",
    "division_fp",
    "division_fn",
    "matched_nodes",
    "predicted_nodes",
    "truth_nodes",
    "estimated_nodes",
}


def parse_float_list(value: str) -> list[float]:
    """Parse a unique comma-separated list of finite score thresholds."""
    try:
        values = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not values or any(not np.isfinite(item) or not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("thresholds must all be between 0 and 1")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return values


def parse_int_list(value: str) -> list[int]:
    """Parse a unique comma-separated list of positive pruning lengths."""
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "minimum lengths must be comma-separated integers"
        ) from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("minimum lengths must all be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("minimum lengths must be unique")
    return values


def estimated_node_count(labels: Path) -> int:
    """Read the competition-provided estimated true node count for one movie."""
    metadata = json.loads((labels / "zarr.json").read_text())
    return int(metadata["attributes"]["geff"]["extra"]["estimated_number_of_nodes"])


def internal_dev_samples(data_root: Path, checkpoint: dict[str, Any]) -> list[SamplePaths]:
    """Resolve only the whole 6bba movies recorded as development shards."""
    if checkpoint.get("train_embryo") != APPEARANCE_TRAIN_EMBRYO:
        raise ValueError("checkpoint does not use the configured training embryo")
    requested = list(checkpoint.get("dev_shards", ()))
    if not requested:
        raise ValueError("checkpoint does not record an internal movie-level dev split")
    train = discover_split(data_root, "train")
    by_name = {sample.dataset: sample for sample in train}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"checkpoint development movies are missing from disk: {missing}")
    samples = [by_name[name] for name in requested]
    invalid = [sample.dataset for sample in samples if sample.embryo != APPEARANCE_TRAIN_EMBRYO]
    if invalid:
        raise ValueError(f"internal development contains non-6bba movies: {invalid}")
    return samples


def locked_holdout_samples(data_root: Path, checkpoint: dict[str, Any]) -> list[SamplePaths]:
    """Resolve the exact labeled 44b6 cohort after the policy guard has passed."""
    if checkpoint.get("holdout_embryo") != APPEARANCE_HOLDOUT_EMBRYO:
        raise ValueError("checkpoint does not preserve the configured holdout embryo")
    samples = [
        sample
        for sample in discover_split(data_root, "train")
        if sample.embryo == APPEARANCE_HOLDOUT_EMBRYO and sample.labels is not None
    ]
    if len(samples) != EXPECTED_HOLDOUT_MOVIES:
        raise ValueError(
            f"locked protocol requires {EXPECTED_HOLDOUT_MOVIES} holdout movies, "
            f"found {len(samples)}"
        )
    return samples


def _parse_result_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for field in RESULT_FIELDS:
        if field in FLOAT_FIELDS:
            parsed[field] = float(row[field])
        elif field in INT_FIELDS:
            parsed[field] = int(row[field])
        else:
            parsed[field] = row[field]
    return parsed


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    """Read an existing atomic checkpoint for safe long-run resume."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(f"unexpected result columns in {path}: {reader.fieldnames}")
        return [_parse_result_row(row) for row in reader]


def write_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically checkpoint every completed movie, threshold, and prune result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    row["dataset"],
                    row["score_threshold"],
                    row["min_track_length"],
                ),
            )
        )
    temporary.replace(path)


def _score_cache_path(
    directory: Path,
    dataset: str,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
) -> Path:
    return directory / (f"{dataset}.{checkpoint_sha256[:12]}.{scoring_source_sha256[:12]}.npz")


def load_or_build_scores(
    sample: SamplePaths,
    *,
    data_split: str,
    cache_dir: Path,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
    checkpoint_k: float,
    patch_shape: tuple[int, int, int],
    model,
    device,
    batch_size: int,
    overwrite: bool,
) -> tuple[TrackingGraph, np.ndarray, dict[str, Any], Path]:
    """Reuse an exact score cache or run label-free DoG and CNN inference once."""
    cache = _score_cache_path(
        cache_dir,
        sample.dataset,
        checkpoint_sha256,
        scoring_source_sha256,
    )
    if cache.exists() and not overwrite:
        graph, scores, metadata = read_score_cache(
            cache,
            expected_dataset=sample.dataset,
            expected_data_split=data_split,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_scoring_source_sha256=scoring_source_sha256,
            expected_detector_threshold_k=checkpoint_k,
            expected_patch_shape=patch_shape,
        )
        return graph, scores, metadata, cache

    image = open_image_array(sample.image)
    start = time.perf_counter()
    candidates = detect_movie(
        image,
        DetectorConfig(method="dog", threshold_k=checkpoint_k),
    )
    detect_seconds = time.perf_counter() - start
    score_start = time.perf_counter()
    scores = score_candidates(
        model,
        image,
        candidates,
        patch_shape=patch_shape,
        batch_size=batch_size,
        device=device,
    )
    score_seconds = time.perf_counter() - score_start
    write_score_cache(
        cache,
        candidates,
        scores,
        dataset=sample.dataset,
        data_split=data_split,
        checkpoint_sha256=checkpoint_sha256,
        scoring_source_sha256=scoring_source_sha256,
        detector_threshold_k=checkpoint_k,
        patch_shape=patch_shape,
        detect_seconds=detect_seconds,
        score_seconds=score_seconds,
    )
    metadata = {
        "detect_seconds": detect_seconds,
        "score_seconds": score_seconds,
    }
    return candidates, scores, metadata, cache


def print_summary(
    rows: list[dict[str, Any]],
    *,
    datasets: list[str],
    thresholds: list[float],
    min_lengths: list[int],
) -> None:
    """Print complete aggregate policies while clearly marking partial checkpoints."""
    print("\n=== Candidate policy summary ===", flush=True)
    print(
        f"{'threshold':>10}{'min_len':>9}{'movies':>8}{'edgeJ':>9}"
        f"{'median':>9}{'node ratio':>12}{'recall':>9}",
        flush=True,
    )
    expected = set(datasets)
    for threshold in thresholds:
        for min_length in min_lengths:
            current = [
                row
                for row in rows
                if np.isclose(row["score_threshold"], threshold)
                and row["min_track_length"] == min_length
                and row["dataset"] in expected
            ]
            observed = {row["dataset"] for row in current}
            if observed != expected:
                print(
                    f"{threshold:>10.3f}{min_length:>9}{len(observed):>8}{'partial':>30}",
                    flush=True,
                )
                continue
            edge_scores = np.array([row["edge_jaccard"] for row in current])
            aggregate_ratio = sum(row["predicted_nodes"] for row in current) / sum(
                row["estimated_nodes"] for row in current
            )
            mean_recall = float(np.mean([row["recall"] for row in current]))
            print(
                f"{threshold:>10.3f}{min_length:>9}{len(current):>8}"
                f"{edge_scores.mean():>9.4f}{np.median(edge_scores):>9.4f}"
                f"{aggregate_ratio:>12.3f}{mean_recall:>9.4f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--cohort",
        choices=("internal-dev", "locked-holdout"),
        default="internal-dev",
    )
    parser.add_argument("--score-thresholds", type=parse_float_list)
    parser.add_argument("--min-lengths", type=parse_int_list)
    parser.add_argument("--policy", type=Path, help="frozen policy required for locked holdout")
    parser.add_argument(
        "--selection-results",
        type=Path,
        help="internal-dev results that produced the frozen holdout policy",
    )
    parser.add_argument(
        "--confirm-locked-holdout",
        action="store_true",
        help="confirm the one-time frozen-policy evaluation",
    )
    parser.add_argument("--max-movies", type=int, help="internal-dev smoke limit only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--score-cache-dir",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_scores" / "phase10",
    )
    parser.add_argument("--overwrite-score-cache", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_movies is not None and args.max_movies < 1:
        parser.error("--max-movies must be positive")
    if args.overwrite_score_cache and args.resume:
        parser.error("--overwrite-score-cache cannot be combined with --resume")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    checkpoint_sha256 = file_sha256(args.checkpoint)
    scoring_source_sha256 = files_sha256(SCORING_SOURCE_FILES)
    evaluation_sha256 = files_sha256(EVALUATION_SOURCE_FILES)
    device = resolve_device(args.device)
    model, checkpoint = load_cellness_checkpoint(args.checkpoint, device=device)
    checkpoint_k = float(checkpoint["detector_threshold_k"])
    patch_shape = tuple(int(value) for value in checkpoint["patch_shape"])

    if args.cohort == "locked-holdout":
        if args.policy is None or not args.confirm_locked_holdout:
            parser.error(
                "locked-holdout requires --policy and --confirm-locked-holdout before any 44b6 read"
            )
        if args.score_thresholds is not None or args.min_lengths is not None:
            parser.error("locked-holdout settings come only from the frozen policy")
        if args.max_movies is not None:
            parser.error("locked-holdout must run the exact complete cohort")
        policy = read_frozen_policy(args.policy)
        if policy["checkpoint_sha256"] != checkpoint_sha256:
            parser.error("frozen policy belongs to a different checkpoint")
        if policy["evaluation_sha256"] != evaluation_sha256:
            parser.error("frozen policy belongs to different evaluation source")
        selection_results = args.selection_results or (
            ROOT / "data" / "working" / "phase10_internal_dev.csv"
        )
        if not selection_results.is_file():
            parser.error(f"frozen policy source results do not exist: {selection_results}")
        if policy["source_results_sha256"] != file_sha256(selection_results):
            parser.error("frozen policy source-results hash does not match the result file")
        selection_rows = read_result_rows(selection_results)
        if any(row["cohort"] != "internal-dev" for row in selection_rows):
            parser.error("frozen policy source results contain rows outside internal-dev")
        if {row["checkpoint_sha256"] for row in selection_rows} != {checkpoint_sha256}:
            parser.error("frozen policy source results use a different checkpoint")
        if {row["evaluation_sha256"] for row in selection_rows} != {evaluation_sha256}:
            parser.error("frozen policy source results use different evaluation source")
        try:
            verify_frozen_policy_results(policy, selection_rows, checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
        thresholds = [float(policy["score_threshold"])]
        min_lengths = [int(policy["min_track_length"])]
        try:
            samples = locked_holdout_samples(args.data_root, checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
        default_output = ROOT / "data" / "working" / "phase10_locked_holdout.csv"
    else:
        if (
            args.policy is not None
            or args.selection_results is not None
            or args.confirm_locked_holdout
        ):
            parser.error("holdout policy flags are invalid for internal-dev")
        thresholds = args.score_thresholds or list(DEFAULT_THRESHOLDS)
        min_lengths = args.min_lengths or list(DEFAULT_MIN_LENGTHS)
        try:
            samples = internal_dev_samples(args.data_root, checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
        if args.max_movies is not None:
            samples = samples[: args.max_movies]
        default_output = ROOT / "data" / "working" / "phase10_internal_dev.csv"

    output_csv = args.output_csv or default_output
    if args.resume and not output_csv.exists():
        parser.error(f"resume output does not exist: {output_csv}")
    if output_csv.exists() and not args.resume:
        parser.error(f"output already exists: {output_csv}; pass --resume to continue")

    rows = read_result_rows(output_csv) if args.resume else []
    incompatible = [
        row
        for row in rows
        if row["cohort"] != args.cohort
        or row["checkpoint_sha256"] != checkpoint_sha256
        or row["evaluation_sha256"] != evaluation_sha256
    ]
    if incompatible:
        parser.error(f"result checkpoint is incompatible with this run: {output_csv}")
    selected_datasets = {sample.dataset for sample in samples}
    rows = [row for row in rows if row["dataset"] in selected_datasets]
    rows_by_key = {
        (row["dataset"], row["score_threshold"], row["min_track_length"]): row for row in rows
    }

    print(f"Cohort: {args.cohort}   movies={len(samples)}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Checkpoint SHA-256: {checkpoint_sha256}", flush=True)
    print(f"Evaluation SHA-256: {evaluation_sha256}", flush=True)
    print(f"Device: {device}   DoG k={checkpoint_k}   patch={patch_shape}", flush=True)
    print(f"Score thresholds: {thresholds}", flush=True)
    print(f"Prune min-lengths: {min_lengths}", flush=True)

    for sample_index, sample in enumerate(samples, start=1):
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        estimated_nodes = estimated_node_count(sample.labels)
        candidates, scores, score_metadata, cache = load_or_build_scores(
            sample,
            data_split="train",
            cache_dir=args.score_cache_dir,
            checkpoint_sha256=checkpoint_sha256,
            scoring_source_sha256=scoring_source_sha256,
            checkpoint_k=checkpoint_k,
            patch_shape=patch_shape,
            model=model,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite_score_cache,
        )
        detect_seconds = float(score_metadata["detect_seconds"])
        score_seconds = float(score_metadata["score_seconds"])
        print(
            f"[{sample_index}/{len(samples)}] {sample.dataset} "
            f"candidates={len(scores):,} score_cache={cache.name} "
            f"detect={detect_seconds:.1f}s score={score_seconds:.1f}s",
            flush=True,
        )

        for threshold in thresholds:
            pending_lengths = [
                min_length
                for min_length in min_lengths
                if (sample.dataset, threshold, min_length) not in rows_by_key
            ]
            if not pending_lengths:
                print(f"  threshold={threshold:.3f} cached", flush=True)
                continue
            selected = filter_candidates(candidates, scores, min_score=threshold)
            link_start = time.perf_counter()
            linked = link_graph_flow(selected)
            link_seconds = time.perf_counter() - link_start
            line = [
                f"  threshold={threshold:.3f}",
                f"keep={len(selected.node_ids):,}/{len(scores):,}",
                f"link={link_seconds:.1f}s",
            ]
            for min_length in pending_lengths:
                pruned = prune_to_tracks(linked, min_track_length=min_length)
                result = evaluate(pruned, truth)
                recall = result.matched_nodes / result.truth_nodes if result.truth_nodes else 0.0
                row = {
                    "dataset": sample.dataset,
                    "cohort": args.cohort,
                    "checkpoint_sha256": checkpoint_sha256,
                    "evaluation_sha256": evaluation_sha256,
                    "score_threshold": threshold,
                    "min_track_length": min_length,
                    "raw_candidates": len(scores),
                    "kept_candidates": len(selected.node_ids),
                    "kept_fraction": len(selected.node_ids) / max(1, len(scores)),
                    "detect_seconds": detect_seconds,
                    "score_seconds": score_seconds,
                    "link_seconds": link_seconds,
                    "runtime_seconds": detect_seconds + score_seconds + link_seconds,
                    **result.as_dict(),
                    "estimated_nodes": estimated_nodes,
                    "predicted_to_estimated": result.predicted_nodes / estimated_nodes,
                    "recall": recall,
                }
                rows_by_key[(sample.dataset, threshold, min_length)] = row
                line.append(
                    f"m>={min_length}:J={result.edge_jaccard:.3f} nodes={result.predicted_nodes:,}"
                )
            print(" | ".join(line), flush=True)
            write_result_rows(output_csv, list(rows_by_key.values()))

    rows = list(rows_by_key.values())
    print_summary(
        rows,
        datasets=[sample.dataset for sample in samples],
        thresholds=thresholds,
        min_lengths=min_lengths,
    )
    print(f"\nSaved results: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
