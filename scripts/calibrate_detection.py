#!/usr/bin/env python3
"""Calibrate the detection operating point: a recall x over-prediction sweep.

The DoG detector's threshold ``k`` trades detection recall against the number of
detections: a lower bar recovers more true (often faint) cells but also admits more
false ones. The local edge metric ignores false positives, so it always prefers a
lower bar -- but the real competition penalises node over-prediction, so the raw
density is not free. This harness makes that trade-off explicit.

For each ``(k, cap)`` it runs the single-scale DoG detector (optionally capping to the
``cap`` strongest detections per frame via ``max_peaks``), links with the requested
linker, and reports:

* node recall    = matched truth nodes / truth nodes
* edge Jaccard   = the local competition edge score
* detections/movie = the over-prediction cost to weigh against the real node penalty

Usage:
    python scripts/calibrate_detection.py --n 6
    python scripts/calibrate_detection.py --ks 4,3.5,3 --caps none,400,250
    python scripts/calibrate_detection.py --ks 3.5 --caps 400 --linker flow
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
from biohub.link import link_graph, link_graph_flow  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402


def validation_samples(data_root: Path, n: int):
    """Return the first ``n`` labeled seed-0 validation samples."""
    train = discover_split(data_root, "train")
    validation = set(make_embryo_holdout([s.dataset for s in train], seed=0).validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    return labeled[:n]


def parse_caps(text: str) -> list[int | None]:
    """Parse a comma list of per-frame caps; ``none`` means uncapped."""
    caps: list[int | None] = []
    for token in text.split(","):
        token = token.strip()
        caps.append(None if token.lower() == "none" else int(token))
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="number of validation samples")
    parser.add_argument("--ks", default="4.0,3.5,3.0,2.5", help="comma list of DoG thresholds k")
    parser.add_argument("--caps", default="none,600,400,250", help="comma list of per-frame caps")
    parser.add_argument("--linker", choices=["greedy", "flow"], default="greedy")
    parser.add_argument("--end-cost", type=float, default=4.0, help="flow linker end-cost (um)")
    args = parser.parse_args()

    ks = [float(x) for x in args.ks.split(",")]
    caps = parse_caps(args.caps)
    samples = validation_samples(ROOT / "data" / "raw", args.n)
    print(f"Samples: {len(samples)}   linker: {args.linker}", flush=True)
    print(f"ks: {ks}   caps: {caps}\n", flush=True)

    def link(nodes):
        if args.linker == "flow":
            return link_graph_flow(nodes, end_cost_um=args.end_cost)
        return link_graph(nodes)

    # results[(k, cap)] = list of (recall, edge_jaccard, detections)
    results: dict[tuple[float, int | None], list[tuple[float, float, int]]] = {}
    for sample in samples:
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        image = open_image_array(sample.image)
        for k in ks:
            for cap in caps:
                cfg = DetectorConfig(method="dog", threshold_k=k, max_peaks=cap)
                start = time.time()
                nodes = detect_movie(image, cfg)
                m = evaluate(link(nodes), truth)
                recall = m.matched_nodes / m.truth_nodes if m.truth_nodes else 0.0
                results.setdefault((k, cap), []).append((recall, m.edge_jaccard, m.predicted_nodes))
                print(
                    f"{sample.dataset:<16} k={k:<4} cap={str(cap):<5} "
                    f"rec={recall:.2f} J={m.edge_jaccard:.3f} det={m.predicted_nodes:>7,} "
                    f"({time.time() - start:.0f}s)",
                    flush=True,
                )

    print("\n=== Means ===", flush=True)
    print(f"{'k':>5}{'cap':>7}{'recall':>9}{'edgeJ':>9}{'det/movie':>12}", flush=True)
    for k in ks:
        for cap in caps:
            rows = results[(k, cap)]
            n = len(rows)
            rec = sum(r[0] for r in rows) / n
            j = sum(r[1] for r in rows) / n
            det = sum(r[2] for r in rows) / n
            print(f"{k:>5}{str(cap):>7}{rec:>9.3f}{j:>9.3f}{det:>12,.0f}", flush=True)


if __name__ == "__main__":
    main()
