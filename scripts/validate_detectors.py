#!/usr/bin/env python3
"""Compare detector configs across a batch of validation samples.

Runs one or more named detector configs (e.g. the Phase-2 ``peak`` baseline and the
Phase-3 ``dog`` adaptive detector) over the first N labeled seed-0 validation samples,
and reports two numbers per config:

* node recall  = matched truth nodes / truth nodes   (did the detector find the cell?)
* edge score   = the local competition metric        (did tracking improve?)

The image and ground truth for each sample are loaded once and shared across configs,
so adding a config only costs its own detection time. Usage:

    python scripts/validate_detectors.py                       # baseline vs dog, 8 samples
    python scripts/validate_detectors.py --n 12
    python scripts/validate_detectors.py --configs dog --dog-k 4 --dog-large 2.5
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
from biohub.link import link_graph  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402


def validation_samples(data_root: Path, n: int):
    """Return the first ``n`` labeled seed-0 validation samples."""
    train = discover_split(data_root, "train")
    holdout = make_embryo_holdout([s.dataset for s in train], seed=0)
    validation = set(holdout.validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    return labeled[:n]


def build_configs(args) -> dict[str, DetectorConfig]:
    """Assemble the named detector configs requested on the command line."""
    available = {
        "baseline": DetectorConfig(method="peak", threshold_percentile=args.percentile),
        "dog": DetectorConfig(
            method="dog",
            dog_sigma_small=args.dog_small,
            dog_sigma_large=args.dog_large,
            threshold_k=args.dog_k,
            min_distance=args.min_distance,
        ),
    }
    return {name: available[name] for name in args.configs.split(",") if name in available}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="number of validation samples")
    parser.add_argument("--configs", default="baseline,dog", help="comma list: baseline,dog")
    parser.add_argument("--percentile", type=float, default=99.5, help="baseline peak percentile")
    parser.add_argument("--dog-small", type=float, default=1.0, help="dog inner sigma (voxels)")
    parser.add_argument("--dog-large", type=float, default=2.0, help="dog outer sigma (voxels)")
    parser.add_argument("--dog-k", type=float, default=4.0, help="dog adaptive threshold k")
    parser.add_argument("--min-distance", type=int, default=3, help="dog peak separation (voxels)")
    args = parser.parse_args()

    data_root = ROOT / "data" / "raw"
    samples = validation_samples(data_root, args.n)
    configs = build_configs(args)
    print(f"Samples: {len(samples)}   Configs: {', '.join(configs)}\n", flush=True)

    # results[config][dataset] = dict of metrics
    results: dict[str, dict[str, dict]] = {name: {} for name in configs}
    for sample in samples:
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        image = open_image_array(sample.image)
        line = [f"{sample.dataset:<16}", f"truth={len(truth.node_ids):>3}n/{len(truth.edges):>3}e"]
        for name, config in configs.items():
            start = time.time()
            nodes = detect_movie(image, config)
            linked = link_graph(nodes)
            m = evaluate(linked, truth)
            elapsed = time.time() - start
            recall = m.matched_nodes / m.truth_nodes if m.truth_nodes else 0.0
            results[name][sample.dataset] = {
                "recall": recall,
                "score": m.score,
                "edge_jaccard": m.edge_jaccard,
                "detections": m.predicted_nodes,
                "tp": m.edge_tp,
                "fn": m.edge_fn,
                "elapsed": elapsed,
            }
            line.append(
                f"| {name}: rec={recall:.2f} J={m.edge_jaccard:.3f} "
                f"det={m.predicted_nodes:>6,} ({elapsed:.0f}s)"
            )
        print("  ".join(line), flush=True)

    print("\n=== Means ===", flush=True)
    for name in configs:
        rows = results[name].values()
        n = len(rows) or 1
        mean_recall = sum(r["recall"] for r in rows) / n
        mean_j = sum(r["edge_jaccard"] for r in rows) / n
        mean_score = sum(r["score"] for r in rows) / n
        mean_det = sum(r["detections"] for r in rows) / n
        print(
            f"{name:<10} node_recall={mean_recall:.3f}  edge_jaccard={mean_j:.3f}  "
            f"score={mean_score:.3f}  mean_detections={mean_det:,.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
