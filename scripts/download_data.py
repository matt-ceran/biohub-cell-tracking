#!/usr/bin/env python3
"""Download and unzip the Kaggle competition data."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    kaggle_bin = Path(sys.executable).with_name("kaggle")
    cmd = [
        str(kaggle_bin),
        "competitions",
        "download",
        "-c",
        args.competition,
        "-p",
        str(args.data_root),
    ]
    if args.force:
        cmd.append("--force")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    unzip_downloads(args.data_root, keep_zip=args.keep_zip)
    print(args.data_root)
    return 0


def unzip_downloads(data_root: Path, keep_zip: bool) -> None:
    archives = sorted(data_root.glob("*.zip"))
    if not archives:
        return
    for archive in archives:
        print(f"Extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_root)
        if not keep_zip:
            archive.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
