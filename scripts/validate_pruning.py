#!/usr/bin/env python3
"""Measure linker-aware pruning: detect at a low bar, link, then keep only detections
that participate in a track (drop the isolated junk).

Phase 5 showed that lowering the DoG threshold k recovers faint cells and lifts edge
Jaccard, but at the cost of many more detections that a strength cap cannot trim without
discarding the same faint cells. This harness tests the alternative: rank detections by
*track participation* instead of strength. For each movie it detects once at ``--k``,
links once with the whole-movie flow linker, then evaluates several ``prune_to_tracks``
thresholds:

* min_track_length=1  -> no pruning (the raw low-bar baseline)
* min_track_length=2  -> drop only edgeless nodes (no edge removed; edge score unchanged)
* min_track_length>=3 -> also drop short stubs (more nodes cut, may cost true short tracks)

Reports edge Jaccard (the score), detections/movie (the over-prediction cost), and node
recall for each threshold. Usage:

    python scripts/validate_pruning.py --n 6 --k 3.0
    python scripts/validate_pruning.py --k 3.0 --min-lengths 1,2,3
    python scripts/validate_pruning.py --n 71 --min-lengths 1,2,3,8 \
        --output-csv data/working/full_validation.csv
    python scripts/validate_pruning.py --n 71 --min-lengths 1,2,3,8 \
        --output-csv data/working/full_validation.csv --resume

Use ``--start-index`` with ``--n`` to run deterministic non-overlapping shards.
The checkpoint stores one row per movie and pruning threshold and is replaced atomically
after every completed movie.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.detect import DetectorConfig, detect_movie  # noqa: E402
from biohub.io import discover_split, open_image_array, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402

RESULT_FIELDS = [
    "dataset",
    "threshold_k",
    "min_track_length",
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


def validation_samples(data_root: Path, n: int, start_index: int = 0):
    train = discover_split(data_root, "train")
    validation = set(make_embryo_holdout([s.dataset for s in train], seed=0).validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    return labeled[start_index : start_index + n]


def estimated_node_count(labels: Path) -> int:
    """Read the competition-provided estimated true node count for one movie."""
    metadata = json.loads((labels / "zarr.json").read_text())
    return int(metadata["attributes"]["geff"]["extra"]["estimated_number_of_nodes"])


def _parse_result_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {"dataset": row["dataset"]}
    parsed["threshold_k"] = float(row["threshold_k"])
    parsed["min_track_length"] = int(row["min_track_length"])
    for field in (
        "runtime_seconds",
        "edge_jaccard",
        "score",
        "division_jaccard",
        "predicted_to_estimated",
        "recall",
    ):
        parsed[field] = float(row[field])
    for field in (
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
    ):
        parsed[field] = int(row[field])
    return parsed


def read_result_rows(path: Path) -> list[dict[str, Any]]:
    """Read a prior checkpoint so a long validation run can resume safely."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(f"Unexpected result columns in {path}: {reader.fieldnames}")
        return [_parse_result_row(row) for row in reader]


def write_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically checkpoint all completed per-movie measurements."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["dataset"], row["min_track_length"]))
        )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="zero-based offset into the sorted validation movies",
    )
    parser.add_argument("--k", type=float, default=3.0, help="DoG threshold k (low bar)")
    parser.add_argument("--min-lengths", default="1,2,3", help="prune thresholds to compare")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="checkpoint one result row per movie and pruning threshold",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed movies from --output-csv",
    )
    args = parser.parse_args()
    min_lengths = [int(x) for x in args.min_lengths.split(",")]

    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.n < 1:
        parser.error("--n must be positive")
    if args.resume and args.output_csv is None:
        parser.error("--resume requires --output-csv")
    if args.output_csv and args.output_csv.exists() and not args.resume:
        parser.error(f"output already exists: {args.output_csv}; pass --resume to reuse it")

    samples = validation_samples(ROOT / "data" / "raw", args.n, args.start_index)
    if not samples:
        parser.error("no validation samples selected")
    print(f"Samples: {len(samples)}   detector k={args.k}", flush=True)
    print(f"prune min-lengths: {min_lengths}\n", flush=True)

    rows: list[dict[str, Any]] = []
    if args.resume and args.output_csv and args.output_csv.exists():
        rows = read_result_rows(args.output_csv)
        incompatible = [row for row in rows if row["threshold_k"] != args.k]
        if incompatible:
            parser.error(f"checkpoint uses a different detector threshold: {args.output_csv}")

    selected = {sample.dataset for sample in samples}
    requested = set(min_lengths)
    rows = [
        row
        for row in rows
        if row["dataset"] in selected and row["min_track_length"] in requested
    ]
    rows_by_key = {(row["dataset"], row["min_track_length"]): row for row in rows}

    for sample in samples:
        if all((sample.dataset, m) in rows_by_key for m in min_lengths):
            cached_runtime = rows_by_key[(sample.dataset, min_lengths[0])]["runtime_seconds"]
            print(f"{sample.dataset:<16} ({cached_runtime:.0f}s cached)", flush=True)
            continue

        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        estimated_nodes = estimated_node_count(sample.labels)
        image = open_image_array(sample.image)
        start = time.time()
        nodes = detect_movie(image, DetectorConfig(method="dog", threshold_k=args.k))
        linked = link_graph_flow(nodes)
        runtime_seconds = time.time() - start
        line = [f"{sample.dataset:<16} ({runtime_seconds:.0f}s)"]
        for m in min_lengths:
            pruned = prune_to_tracks(linked, min_track_length=m)
            res = evaluate(pruned, truth)
            recall = res.matched_nodes / res.truth_nodes if res.truth_nodes else 0.0
            row = {
                "dataset": sample.dataset,
                "threshold_k": args.k,
                "min_track_length": m,
                "runtime_seconds": runtime_seconds,
                **res.as_dict(),
                "estimated_nodes": estimated_nodes,
                "predicted_to_estimated": res.predicted_nodes / estimated_nodes,
                "recall": recall,
            }
            rows_by_key[(sample.dataset, m)] = row
            line.append(f"| m>={m}: J={res.edge_jaccard:.3f} det={res.predicted_nodes:>6,}")
        print("  ".join(line), flush=True)

        rows = list(rows_by_key.values())
        if args.output_csv:
            write_result_rows(args.output_csv, rows)

    print("\n=== Means ===", flush=True)
    print(
        f"{'min_len':>8}{'edgeJ':>9}{'det/movie':>12}{'est ratio':>11}{'recall':>9}",
        flush=True,
    )
    for m in min_lengths:
        current = [row for row in rows_by_key.values() if row["min_track_length"] == m]
        j = sum(row["edge_jaccard"] for row in current) / len(current)
        det = sum(row["predicted_nodes"] for row in current) / len(current)
        ratio = sum(row["predicted_nodes"] for row in current) / sum(
            row["estimated_nodes"] for row in current
        )
        rec = sum(row["recall"] for row in current) / len(current)
        label = "none" if m == 1 else str(m)
        print(f"{label:>8}{j:>9.3f}{det:>12,.0f}{ratio:>11.2f}{rec:>9.3f}", flush=True)

    if args.output_csv:
        print(f"\nSaved per-movie results: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
