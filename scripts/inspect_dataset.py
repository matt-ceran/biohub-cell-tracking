#!/usr/bin/env python3
"""Print a small, read-only summary of a downloaded Biohub dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.io import discover_split, read_image_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, nargs="?", default=Path("data/raw"))
    args = parser.parse_args()

    for split in ("train", "test"):
        split_dir = args.data_root / split
        if not split_dir.exists():
            print(f"{split}: missing {split_dir}")
            continue
        samples = discover_split(args.data_root, split)
        embryos = sorted({sample.embryo for sample in samples})
        print(f"{split}: {len(samples)} samples from {len(embryos)} embryos")
        if samples:
            first = samples[0]
            print(f"  first dataset: {first.dataset}")
            print(f"  labels present: {first.labels is not None}")
            try:
                meta = read_image_metadata(first.image)
            except FileNotFoundError as exc:
                print(f"  metadata error: {exc}")
            else:
                print(f"  shape: {meta.get('shape')}")
                print(f"  dtype: {meta.get('data_type') or meta.get('dtype')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
