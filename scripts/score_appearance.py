#!/usr/bin/env python3
"""Run DoG k=3, score each 3D candidate patch, and emit cellness decisions.

This is the first learned inference boundary.  It does not use GEFF labels and can run
unchanged on Kaggle test movies.  The output keeps every candidate and its score so a
later flow-linking experiment can sweep thresholds without rerunning the CNN.  Usage:

    python scripts/score_appearance.py 6bba_0e7c0d07 --split train
    python scripts/score_appearance.py 44b6_0113de3b --split test --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import filter_candidates  # noqa: E402
from biohub.appearance_model import (  # noqa: E402
    load_cellness_checkpoint,
    resolve_device,
    score_candidates,
)
from biohub.detect import DetectorConfig, detect_movie  # noqa: E402
from biohub.io import discover_split, open_image_array  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-timepoints",
        type=int,
        help="score only the first N frames for an integration smoke run",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_timepoints is not None and args.max_timepoints < 1:
        parser.error("--max-timepoints must be positive")

    samples = discover_split(args.data_root, args.split)
    sample = next((item for item in samples if item.dataset == args.dataset), None)
    if sample is None:
        parser.error(f"dataset {args.dataset!r} was not found in the {args.split} split")
    image = open_image_array(sample.image)
    n_timepoints = image.shape[0]
    if args.max_timepoints is not None:
        n_timepoints = min(n_timepoints, args.max_timepoints)
    t_range = range(n_timepoints)

    device = resolve_device(args.device)
    model, checkpoint = load_cellness_checkpoint(args.checkpoint, device=device)
    patch_shape = tuple(int(value) for value in checkpoint["patch_shape"])
    checkpoint_k = float(checkpoint["detector_threshold_k"])
    if not np.isclose(args.k, checkpoint_k):
        parser.error(
            f"checkpoint was trained on DoG k={checkpoint_k}, but inference requested k={args.k}"
        )
    print(f"Dataset: {sample.dataset}   split={args.split}   frames={n_timepoints}")
    print(f"Device: {device}   patch={patch_shape}   DoG k={args.k}")
    start = time.time()
    candidates = detect_movie(
        image,
        DetectorConfig(method="dog", threshold_k=args.k),
        t_range=t_range,
    )
    detect_seconds = time.time() - start
    scores = score_candidates(
        model,
        image,
        candidates,
        patch_shape=patch_shape,
        batch_size=args.batch_size,
        device=device,
    )
    score_seconds = time.time() - start - detect_seconds
    selected = filter_candidates(candidates, scores, min_score=args.threshold)

    output = args.output or (
        ROOT / "data" / "working" / "appearance_scores" / f"{sample.dataset}.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            node_ids=np.asarray(candidates.node_ids),
            t=np.asarray(candidates.t),
            z=np.asarray(candidates.z),
            y=np.asarray(candidates.y),
            x=np.asarray(candidates.x),
            scores=scores,
            is_cell=scores >= args.threshold,
            threshold=np.array(args.threshold, dtype=np.float32),
        )
    temporary.replace(output)

    quantiles = np.quantile(scores, [0.1, 0.5, 0.9]) if len(scores) else np.zeros(3)
    print(f"DoG candidates: {len(candidates.node_ids):,} in {detect_seconds:.1f}s")
    print(f"CNN scoring: {score_seconds:.1f}s")
    print(f"Cellness p10={quantiles[0]:.3f} p50={quantiles[1]:.3f} p90={quantiles[2]:.3f}")
    print(
        f"Classified as cells at threshold {args.threshold:.3f}: "
        f"{len(selected.node_ids):,} ({len(selected.node_ids) / max(1, len(scores)):.1%})"
    )
    print(f"Saved candidate decisions: {output}")


if __name__ == "__main__":
    main()
