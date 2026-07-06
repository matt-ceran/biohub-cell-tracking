#!/usr/bin/env python3
"""Demonstrate the local metric on one real GEFF sample.

This builds reference predictions from a real ground-truth graph (a perfect copy,
a copy with half the edges dropped, and a copy jittered past the match radius) and
scores each one. It is a sanity demonstration that the metric responds sensibly on
real data, and a usage example for the local validation loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.io import discover_split, read_geff_graph  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402


def pick_validation_sample(data_root: Path):
    """Choose the first labeled sample from the seed-0 validation embryo."""
    train = discover_split(data_root, "train")
    holdout = make_embryo_holdout([s.dataset for s in train], seed=0)
    validation = set(holdout.validation)
    for sample in train:
        if sample.dataset in validation and sample.labels is not None:
            return sample
    raise SystemExit("No labeled validation sample found under data/raw/train.")


def drop_half_edges(truth: TrackingGraph) -> TrackingGraph:
    """Return a copy that keeps only every other edge (a recall-starved linker)."""
    kept = truth.edges[::2]
    return TrackingGraph(truth.node_ids, truth.t, truth.z, truth.y, truth.x, kept)


def jitter_beyond_gate(truth: TrackingGraph, shift: float = 40.0) -> TrackingGraph:
    """Return a copy whose nodes are moved far enough that none match (bad detector)."""
    return TrackingGraph(
        truth.node_ids,
        truth.t,
        truth.z,
        truth.y + shift,
        truth.x + shift,
        truth.edges,
    )


def main() -> None:
    data_root = ROOT / "data" / "raw"
    sample = pick_validation_sample(data_root)
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))

    perfect = truth
    half = drop_half_edges(truth)
    jittered = jitter_beyond_gate(truth)

    print(f"Sample: {sample.dataset} (embryo {sample.embryo})")
    print(f"Truth nodes: {len(truth.node_ids):,}   truth edges: {len(truth.edges):,}")
    print()
    cols = f"{'score':>8}{'edgeJ':>8}{'divJ':>8}{'TP':>7}{'FP':>7}{'FN':>7}{'ign':>7}"
    header = f"{'prediction':<22}{cols}"
    print(header)
    print("-" * len(header))
    predictions = [
        ("perfect copy", perfect),
        ("half edges dropped", half),
        ("jittered off-target", jittered),
    ]
    for name, pred in predictions:
        r = evaluate(pred, truth)
        print(
            f"{name:<22}{r.score:>8.3f}{r.edge_jaccard:>8.3f}{r.division_jaccard:>8.3f}"
            f"{r.edge_tp:>7}{r.edge_fp:>7}{r.edge_fn:>7}{r.edge_ignored:>7}"
        )


if __name__ == "__main__":
    main()
