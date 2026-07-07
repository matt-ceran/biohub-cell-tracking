#!/usr/bin/env python3
"""Compare the greedy and min-cost-flow linkers across a batch of validation samples.

Both linkers consume the same detections (the Phase-3 adaptive DoG detector), so this
isolates the linking step: given identical cell centers, does solving the whole movie as
one min-cost flow reproduce more true edges than solving each frame pair greedily?

For each seed-0 validation movie it reports, per linker:

* edge Jaccard  = the local competition metric (here effectively edge recall, because a
  predicted edge whose endpoints do not both match a truth node is ignored, not
  penalised, so false positives stay ~0).
* TP / FN       = true edges reproduced / missed.

Detections for each movie are computed once and shared across linkers. Usage:

    python scripts/validate_linkers.py                 # greedy vs flow, 6 samples
    python scripts/validate_linkers.py --n 12
    python scripts/validate_linkers.py --end-cost 4 --appear-reward 8 --max-gap 1
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

from biohub.detect import DOG_DETECTOR, detect_movie  # noqa: E402
from biohub.io import discover_split, open_image_array, read_geff_graph  # noqa: E402
from biohub.link import link_graph, link_graph_flow  # noqa: E402
from biohub.metric import TrackingGraph, evaluate  # noqa: E402
from biohub.split import make_embryo_holdout  # noqa: E402


def validation_samples(data_root: Path, n: int):
    """Return the first ``n`` labeled seed-0 validation samples."""
    train = discover_split(data_root, "train")
    holdout = make_embryo_holdout([s.dataset for s in train], seed=0)
    validation = set(holdout.validation)
    labeled = [s for s in train if s.dataset in validation and s.labels is not None]
    return labeled[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6, help="number of validation samples")
    parser.add_argument("--end-cost", type=float, default=4.0, help="flow birth/death fee (um)")
    parser.add_argument("--appear-reward", type=float, default=8.0, help="flow use reward (um)")
    parser.add_argument("--max-gap", type=int, default=1, help="flow max frame skip (1 = none)")
    args = parser.parse_args()

    data_root = ROOT / "data" / "raw"
    samples = validation_samples(data_root, args.n)
    print(f"Samples: {len(samples)}   Detector: DoG k=4\n", flush=True)

    scores: dict[str, list[float]] = {"greedy": [], "flow": []}
    for sample in samples:
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        image = open_image_array(sample.image)
        nodes = detect_movie(image, DOG_DETECTOR)
        line = [
            f"{sample.dataset:<16}",
            f"det={len(nodes.node_ids):>6,}",
            f"truth={len(truth.edges):>3}e",
        ]

        linkers = {
            "greedy": lambda n: link_graph(n),
            "flow": lambda n: link_graph_flow(
                n,
                end_cost_um=args.end_cost,
                appear_reward_um=args.appear_reward,
                max_gap=args.max_gap,
            ),
        }
        for name, run in linkers.items():
            start = time.time()
            linked = run(nodes)
            elapsed = time.time() - start
            m = evaluate(linked, truth)
            scores[name].append(m.edge_jaccard)
            line.append(
                f"| {name}: J={m.edge_jaccard:.3f} "
                f"TP={m.edge_tp:>4} FN={m.edge_fn:>4} ({elapsed:.0f}s)"
            )
        print("  ".join(line), flush=True)

    print("\n=== Mean edge Jaccard ===", flush=True)
    for name in ("greedy", "flow"):
        vals = scores[name]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"{name:<8} {mean:.3f}", flush=True)
    g = sum(scores["greedy"]) / len(scores["greedy"])
    f = sum(scores["flow"]) / len(scores["flow"])
    print(f"\nflow - greedy = {f - g:+.3f}", flush=True)


if __name__ == "__main__":
    main()
