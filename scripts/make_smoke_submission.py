#!/usr/bin/env python3
"""Create a tiny format-only submission from discovered test samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.io import discover_split, read_image_metadata  # noqa: E402
from biohub.submission import (  # noqa: E402
    assemble_submission,
    edges_frame,
    nodes_frame,
    write_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, nargs="?", default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/smoke_submission.csv"),
    )
    args = parser.parse_args()

    try:
        samples = discover_split(args.data_root, "test")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if not samples:
        raise SystemExit("No test samples found.")

    parts = []
    for sample in samples:
        shape = read_image_metadata(sample.image)["shape"]
        t_count, z_count, y_count, x_count = map(int, shape)
        n_nodes = min(3, t_count)
        node_ids = list(range(1, n_nodes + 1))
        parts.append(
            nodes_frame(
                sample.dataset,
                node_ids=node_ids,
                t=list(range(n_nodes)),
                z=[z_count // 2] * n_nodes,
                y=[y_count // 2] * n_nodes,
                x=[x_count // 2] * n_nodes,
            )
        )
        if n_nodes > 1:
            parts.append(
                edges_frame(
                    sample.dataset,
                    source_ids=node_ids[:-1],
                    target_ids=node_ids[1:],
                )
            )

    submission = assemble_submission(parts)
    write_submission(
        submission,
        args.output,
        expected_datasets=[sample.dataset for sample in samples],
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
