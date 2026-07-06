"""Dataset discovery and lightweight Zarr or GEFF readers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamplePaths:
    dataset: str
    image: Path
    labels: Path | None
    embryo: str


@dataclass(frozen=True)
class GeffGraph:
    node_ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray


def embryo_id(dataset: str) -> str:
    """Return the embryo prefix from a competition dataset name."""
    return dataset.split("_", 1)[0]


def dataset_name(path: str | Path) -> str:
    """Return the dataset name without `.zarr` or `.geff`."""
    path = Path(path)
    if path.suffix in {".zarr", ".geff"}:
        return path.name[: -len(path.suffix)]
    return path.name


def discover_split(data_root: str | Path, split: str) -> list[SamplePaths]:
    """Discover image samples for `train` or `test` under a Kaggle data root."""
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    samples: list[SamplePaths] = []
    for image in sorted(split_dir.glob("*.zarr")):
        name = dataset_name(image)
        labels = split_dir / f"{name}.geff"
        samples.append(
            SamplePaths(
                dataset=name,
                image=image,
                labels=labels if labels.exists() else None,
                embryo=embryo_id(name),
            )
        )
    return samples


def read_image_metadata(zarr_path: str | Path) -> dict[str, Any]:
    """Read array metadata from a sample `.zarr` without loading image data."""
    meta_path = Path(zarr_path) / "0" / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Zarr array metadata: {meta_path}")
    return json.loads(meta_path.read_text())


def open_image_array(zarr_path: str | Path):
    """Open the image array at path `0/` using zarr."""
    zarr = _require_zarr()
    return zarr.open(str(Path(zarr_path) / "0"), mode="r")


def read_geff_graph(geff_path: str | Path) -> GeffGraph:
    """Load sparse training graph arrays from a `.geff` directory."""
    zarr = _require_zarr()
    root = zarr.open(str(geff_path), mode="r")
    return GeffGraph(
        node_ids=np.asarray(root["nodes/ids"][:]),
        t=np.asarray(root["nodes/props/t/values"][:]),
        z=np.asarray(root["nodes/props/z/values"][:]),
        y=np.asarray(root["nodes/props/y/values"][:]),
        x=np.asarray(root["nodes/props/x/values"][:]),
        edges=np.asarray(root["edges/ids"][:]),
    )


def _require_zarr():
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install project dependencies with `python -m pip install -e .` "
            "before reading Zarr or GEFF data."
        ) from exc
    return zarr
