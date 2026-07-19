#!/usr/bin/env python3
"""Build deterministic 3D patch shards for positive-unlabeled cellness training.

Only embryo 6bba is eligible.  Each movie is detected at DoG k=3, GEFF-matched
candidates become trusted positives, and a uniform sample of all candidates remains
explicitly unlabeled.  Usage:

    python scripts/build_appearance_dataset.py
    python scripts/build_appearance_dataset.py --n 8 --start-index 0
    python scripts/build_appearance_dataset.py --all
    python scripts/build_appearance_dataset.py --dataset 6bba_0e7c0d07
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

from biohub.appearance import (  # noqa: E402
    APPEARANCE_TRAIN_EMBRYO,
    DEFAULT_PATCH_SHAPE,
    build_appearance_shard,
    validate_patch_shape,
)
from biohub.io import discover_split  # noqa: E402


def parse_patch_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
        return validate_patch_shape(shape)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_shards",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", action="append", help="specific 6bba dataset name")
    selection.add_argument("--all", action="store_true", help="build all 6bba movies")
    parser.add_argument("--n", type=int, default=1, help="movies to select when not using --all")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--k", type=float, default=3.0, help="DoG candidate threshold")
    parser.add_argument(
        "--patch-shape",
        type=parse_patch_shape,
        default=DEFAULT_PATCH_SHAPE,
        metavar="Z,Y,X",
    )
    parser.add_argument(
        "--unlabeled-ratio",
        type=float,
        default=4.0,
        help="unlabeled population samples per known positive",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.unlabeled_ratio <= 0:
        parser.error("--unlabeled-ratio must be positive")

    train = [
        sample
        for sample in discover_split(args.data_root, "train")
        if sample.embryo == APPEARANCE_TRAIN_EMBRYO and sample.labels is not None
    ]
    by_name = {sample.dataset: sample for sample in train}
    if args.dataset:
        missing = sorted(set(args.dataset) - set(by_name))
        if missing:
            parser.error(
                f"requested datasets are absent or outside embryo {APPEARANCE_TRAIN_EMBRYO}: "
                f"{', '.join(missing)}"
            )
        samples = [by_name[name] for name in args.dataset]
    elif args.all:
        samples = train
    else:
        samples = train[args.start_index : args.start_index + args.n]
    if not samples:
        parser.error("no training movies selected")

    print(f"Training embryo: {APPEARANCE_TRAIN_EMBRYO}")
    print("Held-out embryo is excluded by code: 44b6")
    print(f"Movies: {len(samples)}   DoG k={args.k}   patch={args.patch_shape}")
    print(f"Output: {args.output_dir}")
    for index, sample in enumerate(samples, start=1):
        start = time.time()
        summary = build_appearance_shard(
            sample,
            args.output_dir,
            detector_threshold_k=args.k,
            patch_shape=args.patch_shape,
            unlabeled_ratio=args.unlabeled_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(
            f"[{index}/{len(samples)}] {sample.dataset} "
            f"candidates={summary.raw_candidate_count:,} "
            f"positives={summary.known_positive_samples:,} "
            f"unlabeled={summary.unlabeled_samples:,} "
            f"recall={summary.sparse_label_recall:.3f} "
            f"prior={summary.estimated_candidate_prior:.3f} "
            f"elapsed={time.time() - start:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
