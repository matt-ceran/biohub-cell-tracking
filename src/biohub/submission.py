"""Submission assembly and validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from biohub.constants import SUBMISSION_COLUMNS

INT_COLUMNS = ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id")


def assemble_submission(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate node and edge rows and assign the required throwaway id."""
    df = pd.concat(list(parts), ignore_index=True)
    df = df.loc[:, list(SUBMISSION_COLUMNS)].copy()
    df["id"] = np.arange(len(df), dtype=np.int64)
    return df


def nodes_frame(dataset: str, node_ids, t, z, y, x) -> pd.DataFrame:
    """Create submission rows for detected nodes."""
    return pd.DataFrame(
        {
            "id": -1,
            "dataset": dataset,
            "row_type": "node",
            "node_id": node_ids,
            "t": t,
            "z": z,
            "y": y,
            "x": x,
            "source_id": -1,
            "target_id": -1,
        }
    )


def edges_frame(dataset: str, source_ids, target_ids) -> pd.DataFrame:
    """Create submission rows for tracking edges."""
    n = len(source_ids)
    return pd.DataFrame(
        {
            "id": -1,
            "dataset": dataset,
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": source_ids,
            "target_id": target_ids,
        },
        index=range(n),
    )


def validate_submission(df: pd.DataFrame, expected_datasets: Iterable[str] | None = None) -> None:
    """Raise ValueError if a submission dataframe violates the competition format."""
    if tuple(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}.")
    if len(df) and df["id"].tolist() != list(range(len(df))):
        raise ValueError("The id column must be consecutive integers starting at 0.")
    if not set(df["row_type"]).issubset({"node", "edge"}):
        raise ValueError("row_type must contain only `node` or `edge`.")
    if df[list(INT_COLUMNS)].isna().any().any():
        raise ValueError("Integer submission columns cannot contain missing values.")
    for column in INT_COLUMNS:
        if not _is_integer_like(df[column]):
            raise ValueError(f"Column {column} must contain integer values.")

    expected = set(expected_datasets or [])
    if expected and set(df["dataset"]) != expected:
        missing = sorted(expected - set(df["dataset"]))
        extra = sorted(set(df["dataset"]) - expected)
        raise ValueError(f"Dataset mismatch. Missing={missing}, extra={extra}.")

    for dataset, group in df.groupby("dataset", sort=False):
        _validate_dataset_group(dataset, group)


def write_submission(
    df: pd.DataFrame,
    path: str | Path,
    expected_datasets: Iterable[str] | None = None,
) -> Path:
    """Validate and write a submission CSV."""
    validate_submission(df, expected_datasets=expected_datasets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _validate_dataset_group(dataset: str, group: pd.DataFrame) -> None:
    nodes = group[group["row_type"] == "node"]
    edges = group[group["row_type"] == "edge"]

    if nodes.empty:
        raise ValueError(f"Dataset {dataset} must contain at least one node row.")
    if nodes["node_id"].duplicated().any():
        raise ValueError(f"Dataset {dataset} has duplicate node_id values.")
    if (nodes[["node_id", "t", "z", "y", "x"]] < 0).any().any():
        raise ValueError(f"Dataset {dataset} node rows need non-negative node_id,t,z,y,x.")
    if not (nodes[["source_id", "target_id"]] == -1).all().all():
        raise ValueError(f"Dataset {dataset} node rows must use -1 for source_id,target_id.")

    if not edges.empty:
        if not (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all():
            raise ValueError(f"Dataset {dataset} edge rows must use -1 for node_id,t,z,y,x.")
        node_ids = set(nodes["node_id"].astype(int))
        edge_ids = set(edges["source_id"].astype(int)) | set(edges["target_id"].astype(int))
        missing = sorted(edge_ids - node_ids)
        if missing:
            raise ValueError(f"Dataset {dataset} edges reference missing node IDs: {missing[:5]}.")


def _is_integer_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna().all() and np.equal(values, np.floor(values)).all()
