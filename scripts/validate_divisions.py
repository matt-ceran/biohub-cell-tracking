#!/usr/bin/env python3
"""Measure the conservative fork classifier: after detect -> flow-link -> prune, re-attach
orphaned daughters with ``add_divisions`` and check that it recovers true divisions without
denting the dominant edge score.

The flow linker gives every detection one successor, so its tracks can never contain a
division. ``add_divisions`` is the post-linking step that puts forks back: at a real
mitosis it finds the mother's kept child plus an orphaned second daughter on the opposite
side and adds the missing edge (see ``biohub.link.add_divisions``). Divisions are rare
(~0.2 per movie) and weighted 0.1, and a false fork lands as an edge false-positive on the
*dominant* score, so the classifier is tuned for precision.

For each validation movie this detects once at ``--k``, flow-links, prunes, then evaluates
the graph twice -- before and after ``add_divisions`` -- reporting division TP/FP/FN and the
edge Jaccard on each side (the edge score must not regress). By default it runs only the
movies that actually contain a division in the ground truth, so recall is observable
without paying for detection on movies with nothing to find; pass ``--all`` to include the
rest as a false-positive control. Usage:

    python scripts/validate_divisions.py --n 8
    python scripts/validate_divisions.py --all --n 12
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
from biohub.link import add_divisions, link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import (  # noqa: E402
    TrackingGraph,
    _directed_edges,
    _out_degree,
    evaluate,
)
from biohub.split import make_embryo_holdout  # noqa: E402


def truth_division_count(labels: Path) -> int:
    graph = TrackingGraph.from_geff(read_geff_graph(labels))
    return sum(1 for _, deg in _out_degree(_directed_edges(graph)).items() if deg >= 2)


def validation_samples(data_root: Path, n: int, only_divisions: bool):
    train = discover_split(data_root, "train")
    validation = set(make_embryo_holdout([s.dataset for s in train], seed=0).validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    if only_divisions:
        labeled = [s for s in labeled if truth_division_count(s.labels) > 0]
    return labeled[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=float, default=3.0, help="DoG threshold k (low bar)")
    parser.add_argument("--min-length", type=int, default=3, help="prune_to_tracks threshold")
    parser.add_argument("--all", action="store_true", help="include movies with no true division")
    args = parser.parse_args()

    samples = validation_samples(ROOT / "data" / "raw", args.n, only_divisions=not args.all)
    print(f"Samples: {len(samples)}   detector k={args.k}   prune m={args.min_length}", flush=True)
    print(f"{'dataset':<16}{'edgeJ_before':>13}{'edgeJ_after':>12}"
          f"{'div tp/fp/fn':>14}{'true':>6}", flush=True)

    tot = {"tp": 0, "fp": 0, "fn": 0}
    ej_before: list[float] = []
    ej_after: list[float] = []
    for sample in samples:
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        image = open_image_array(sample.image)
        start = time.time()
        nodes = detect_movie(image, DetectorConfig(method="dog", threshold_k=args.k))
        pruned = prune_to_tracks(link_graph_flow(nodes), min_track_length=args.min_length)
        forked = add_divisions(pruned)

        before = evaluate(pruned, truth)
        after = evaluate(forked, truth)
        ej_before.append(before.edge_jaccard)
        ej_after.append(after.edge_jaccard)
        tot["tp"] += after.division_tp
        tot["fp"] += after.division_fp
        tot["fn"] += after.division_fn
        n_true = truth_division_count(sample.labels)
        print(f"{sample.dataset:<16}{before.edge_jaccard:>13.3f}{after.edge_jaccard:>12.3f}"
              f"{after.division_tp:>6}/{after.division_fp}/{after.division_fn}"
              f"{n_true:>6}   ({time.time() - start:.0f}s)", flush=True)

    n = len(samples)
    denom = tot["tp"] + tot["fp"] + tot["fn"]
    micro_div_jaccard = tot["tp"] / denom if denom else 1.0
    print("\n=== Means ===", flush=True)
    print(f"mean edge Jaccard: before={sum(ej_before) / n:.3f}  after={sum(ej_after) / n:.3f}",
          flush=True)
    print(f"division totals: tp={tot['tp']} fp={tot['fp']} fn={tot['fn']}  "
          f"(micro Jaccard {micro_div_jaccard:.3f})", flush=True)


if __name__ == "__main__":
    main()
