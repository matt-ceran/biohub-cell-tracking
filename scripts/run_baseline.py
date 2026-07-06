#!/usr/bin/env python3
"""Run the classical detector + linker baseline on a validation sample and score it.

This is the first end-to-end modeling path: detect cell centers per frame, link them
across frames, and grade the result with the local metric. Usage:

    python scripts/run_baseline.py                  # first seed-0 validation sample
    python scripts/run_baseline.py 44b6_0113de3b    # a specific dataset
    python scripts/run_baseline.py 44b6_0113de3b 99.8   # with a threshold percentile
"""

from __future__ import annotations

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


def resolve_sample(data_root: Path, dataset: str | None):
    """Return the requested labeled sample, or the first seed-0 validation sample."""
    train = discover_split(data_root, "train")
    if dataset is not None:
        for sample in train:
            if sample.dataset == dataset and sample.labels is not None:
                return sample
        raise SystemExit(f"No labeled train sample named {dataset}.")
    holdout = make_embryo_holdout([s.dataset for s in train], seed=0)
    validation = set(holdout.validation)
    for sample in train:
        if sample.dataset in validation and sample.labels is not None:
            return sample
    raise SystemExit("No labeled validation sample found under data/raw/train.")


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 99.5

    data_root = ROOT / "data" / "raw"
    sample = resolve_sample(data_root, dataset)
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    image = open_image_array(sample.image)

    config = DetectorConfig(threshold_percentile=percentile)
    print(f"Sample: {sample.dataset} (embryo {sample.embryo})", flush=True)
    print(f"Detector threshold percentile: {percentile}", flush=True)

    start = time.time()
    nodes = detect_movie(image, config)
    linked = link_graph(nodes)
    result = evaluate(linked, truth)
    elapsed = time.time() - start

    print(f"Detections: {len(nodes.node_ids):,}   linked edges: {len(linked.edges):,}", flush=True)
    print(f"Truth: {len(truth.node_ids):,} nodes, {len(truth.edges):,} edges", flush=True)
    print(f"Runtime: {elapsed:.1f}s", flush=True)
    print(flush=True)
    print(f"Score:          {result.score:.3f}", flush=True)
    print(f"Edge Jaccard:   {result.edge_jaccard:.3f}", flush=True)
    print(
        f"Edges  TP={result.edge_tp}  FP={result.edge_fp}  "
        f"FN={result.edge_fn}  ignored={result.edge_ignored}",
        flush=True,
    )
    print(f"Matched nodes:  {result.matched_nodes} of {result.truth_nodes} truth nodes", flush=True)


if __name__ == "__main__":
    main()
