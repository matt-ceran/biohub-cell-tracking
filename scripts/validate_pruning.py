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
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.detect import DetectorConfig, detect_movie  # noqa: E402
from biohub.io import discover_split, open_image_array, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402


def validation_samples(data_root: Path, n: int):
    train = discover_split(data_root, "train")
    validation = set(make_embryo_holdout([s.dataset for s in train], seed=0).validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    return labeled[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--k", type=float, default=3.0, help="DoG threshold k (low bar)")
    parser.add_argument("--min-lengths", default="1,2,3", help="prune thresholds to compare")
    args = parser.parse_args()
    min_lengths = [int(x) for x in args.min_lengths.split(",")]

    samples = validation_samples(ROOT / "data" / "raw", args.n)
    print(f"Samples: {len(samples)}   detector k={args.k}", flush=True)
    print(f"prune min-lengths: {min_lengths}\n", flush=True)

    results: dict[int, list[tuple[float, int, float]]] = {m: [] for m in min_lengths}
    for sample in samples:
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        image = open_image_array(sample.image)
        start = time.time()
        nodes = detect_movie(image, DetectorConfig(method="dog", threshold_k=args.k))
        linked = link_graph_flow(nodes)
        line = [f"{sample.dataset:<16} ({time.time() - start:.0f}s)"]
        for m in min_lengths:
            pruned = prune_to_tracks(linked, min_track_length=m)
            res = evaluate(pruned, truth)
            recall = res.matched_nodes / res.truth_nodes if res.truth_nodes else 0.0
            results[m].append((res.edge_jaccard, res.predicted_nodes, recall))
            line.append(f"| m>={m}: J={res.edge_jaccard:.3f} det={res.predicted_nodes:>6,}")
        print("  ".join(line), flush=True)

    print("\n=== Means ===", flush=True)
    print(f"{'min_len':>8}{'edgeJ':>9}{'det/movie':>12}{'recall':>9}", flush=True)
    for m in min_lengths:
        rows = results[m]
        j = sum(r[0] for r in rows) / len(rows)
        det = sum(r[1] for r in rows) / len(rows)
        rec = sum(r[2] for r in rows) / len(rows)
        label = "none" if m == 1 else str(m)
        print(f"{label:>8}{j:>9.3f}{det:>12,.0f}{rec:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
