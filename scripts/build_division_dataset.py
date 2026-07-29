#!/usr/bin/env python3
"""Build strict sampled fit records or print the read-only development funnel."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true", help="build the complete frozen fit role")
    mode.add_argument(
        "--smoke-dataset",
        help="build one explicit fit movie as a non-production smoke artifact",
    )
    mode.add_argument(
        "--development-funnel",
        action="store_true",
        help="inspect development labels read-only; writes no records or fit statistics",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "working" / "phase11_division_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="required for smoke; fit defaults to the canonical production directory",
    )
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--upstream-manifest",
        type=Path,
        default=ROOT / "data" / "working" / "phase11_division_manifest.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data" / "working" / "phase11_division_manifest_summary.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "data"
        / "working"
        / "appearance_checkpoints"
        / "cellness_cnn_phase10_full.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Imports stay below argument parsing so ``--help`` works in a clean base install.
    from biohub.division_data import canonical_json_bytes
    from biohub.division_dataset import (
        build_fit_dataset,
        build_smoke_dataset,
        development_proposal_funnel,
    )

    common = {
        "project_root": ROOT,
        "data_root": args.data_root,
        "config_dir": args.config_dir,
        "upstream_manifest_path": args.upstream_manifest,
        "summary_path": args.summary,
        "cache_dir": args.cache_dir,
        "checkpoint_path": args.checkpoint,
    }
    try:
        if args.development_funnel:
            if args.overwrite or args.output_dir is not None:
                parser.error(
                    "read-only development funnel mode does not accept output or overwrite"
                )
            report = development_proposal_funnel(**common)
            print(canonical_json_bytes(report).decode("ascii"), end="", flush=True)
            return
        if args.smoke_dataset is not None:
            if args.output_dir is None:
                parser.error("--smoke-dataset requires an explicit --output-dir")
            result = build_smoke_dataset(
                args.output_dir,
                dataset=args.smoke_dataset,
                overwrite=args.overwrite,
                **common,
            )
        else:
            output = args.output_dir or (
                ROOT / "data" / "working" / "phase11_division_dataset" / "fit_complete"
            )
            result = build_fit_dataset(
                output,
                overwrite=args.overwrite,
                **common,
            )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    status = "validated" if result.skipped else "built"
    print(
        f"{status} {result.build_mode} division dataset: records={result.record_count} "
        f"feature_schema_sha256={result.feature_schema_sha256} "
        f"sample_tuple_sha256={result.sample_tuple_sha256} "
        f"model_inputs_sha256={result.model_inputs_sha256} "
        f"dataset_manifest_sha256={result.dataset_manifest_sha256} path={result.path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
