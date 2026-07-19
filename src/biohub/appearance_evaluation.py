"""Auditable score caches and frozen policies for appearance-model evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from biohub.metric import TrackingGraph

SCORE_CACHE_SCHEMA_VERSION = 2
CANDIDATE_POLICY_SCHEMA_VERSION = 1
LOCKED_NODE_DENSITY_LIMIT = 1.4464667407415892
CANDIDATE_SELECTION_RULE = (
    "highest mean internal-dev edge Jaccard among positive cellness thresholds "
    "within the locked node-density limit"
)
BASELINE_THRESHOLD = 0.0
BASELINE_MIN_TRACK_LENGTH = 8
MIN_INTERNAL_EDGE_GAIN = 1e-6


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 identity for a checkpoint or result artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_sha256(paths: list[str | Path]) -> str:
    """Hash an ordered source bundle without depending on its absolute directory."""
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_dev_datasets(checkpoint: dict[str, Any]) -> list[str]:
    """Return the checkpoint's complete, unique, sorted internal-dev movie list."""
    datasets = checkpoint.get("dev_shards")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("checkpoint does not record a non-empty internal-dev movie split")
    if not all(isinstance(dataset, str) and dataset.startswith("6bba_") for dataset in datasets):
        raise ValueError("checkpoint internal-dev split must contain only 6bba movies")
    if len(set(datasets)) != len(datasets):
        raise ValueError("checkpoint internal-dev split contains duplicate movies")
    return sorted(datasets)


def aggregate_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate a complete rectangular policy grid across identical movie sets."""
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["score_threshold"], row["min_track_length"])].append(row)
    if not grouped:
        raise ValueError("internal-dev result file is empty")
    dataset_sets = [{row["dataset"] for row in current} for current in grouped.values()]
    expected = dataset_sets[0]
    if not expected or any(datasets != expected for datasets in dataset_sets[1:]):
        raise ValueError("every candidate policy must contain the same complete movie set")

    aggregates: list[dict[str, Any]] = []
    for (threshold, min_length), current in sorted(grouped.items()):
        if len(current) != len(expected):
            raise ValueError("candidate policy contains duplicate or missing movie rows")
        edge_scores = np.array([row["edge_jaccard"] for row in current], dtype=float)
        aggregate_ratio = sum(row["predicted_nodes"] for row in current) / sum(
            row["estimated_nodes"] for row in current
        )
        aggregates.append(
            {
                "score_threshold": threshold,
                "min_track_length": min_length,
                "datasets": sorted(expected),
                "mean_edge_jaccard": float(edge_scores.mean()),
                "median_edge_jaccard": float(np.median(edge_scores)),
                "aggregate_node_ratio": float(aggregate_ratio),
                "mean_recall": float(np.mean([row["recall"] for row in current])),
                "mean_runtime_seconds": float(
                    np.mean([row["runtime_seconds"] for row in current])
                ),
            }
        )
    return aggregates


def select_policy(
    aggregates: list[dict[str, Any]],
    *,
    density_limit: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the strongest real CNN policy under the predeclared density contract."""
    baseline = next(
        (
            policy
            for policy in aggregates
            if np.isclose(policy["score_threshold"], BASELINE_THRESHOLD)
            and policy["min_track_length"] == BASELINE_MIN_TRACK_LENGTH
        ),
        None,
    )
    if baseline is None:
        raise ValueError("results must include the threshold 0, min-track-length 8 baseline")
    eligible = [
        policy
        for policy in aggregates
        if policy["score_threshold"] > 0.0
        and policy["aggregate_node_ratio"] <= density_limit
    ]
    if not eligible:
        raise ValueError("no positive-threshold candidate policy meets the node-density limit")
    selected = max(
        eligible,
        key=lambda policy: (
            policy["mean_edge_jaccard"],
            -policy["aggregate_node_ratio"],
            -policy["score_threshold"],
        ),
    )
    gain = selected["mean_edge_jaccard"] - baseline["mean_edge_jaccard"]
    if gain <= MIN_INTERNAL_EDGE_GAIN:
        raise ValueError(
            "no eligible CNN policy improves the internal-dev edge baseline; do not unlock 44b6"
        )
    return selected, baseline


def require_complete_checkpoint_dev(
    aggregates: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> list[str]:
    """Reject a smoke subset before it can be frozen as the selection cohort."""
    expected = checkpoint_dev_datasets(checkpoint)
    observed = aggregates[0]["datasets"] if aggregates else []
    if observed != expected:
        raise ValueError(
            "internal-dev results must contain the checkpoint's exact complete movie split"
        )
    return expected


def verify_frozen_policy_results(
    policy: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> None:
    """Recompute a frozen winner so edited policy settings cannot unlock holdout data."""
    validate_frozen_policy(policy)
    aggregates = aggregate_policy_rows(rows)
    require_complete_checkpoint_dev(aggregates, checkpoint)
    selected, baseline = select_policy(aggregates, density_limit=float(policy["density_limit"]))
    if not np.isclose(selected["score_threshold"], policy["score_threshold"]) or int(
        selected["min_track_length"]
    ) != int(policy["min_track_length"]):
        raise ValueError("frozen policy is not the winning policy in its source results")
    for name, aggregate in (("selected_metrics", selected), ("baseline_metrics", baseline)):
        for metric, recorded in policy[name].items():
            if not np.isclose(float(aggregate[metric]), float(recorded)):
                raise ValueError(f"frozen policy has stale or edited {name}")


def _validate_candidate_scores(graph: TrackingGraph, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    expected = (len(graph.node_ids),)
    if values.shape != expected:
        raise ValueError(f"scores must have shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate scores must all be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("candidate scores must be between 0 and 1")
    node_ids = np.asarray(graph.node_ids)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("candidate node identifiers must be unique")
    for name in ("t", "z", "y", "x"):
        coordinates = np.asarray(getattr(graph, name))
        if coordinates.shape != expected:
            raise ValueError(
                f"candidate {name} must have shape {expected}, got {coordinates.shape}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"candidate {name} values must all be finite")
    return values


def write_score_cache(
    path: str | Path,
    graph: TrackingGraph,
    scores: np.ndarray,
    *,
    dataset: str,
    data_split: str,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
    detector_threshold_k: float,
    patch_shape: tuple[int, int, int],
    detect_seconds: float,
    score_seconds: float,
) -> None:
    """Atomically save raw candidates, scores, runtimes, and their exact contract."""
    values = _validate_candidate_scores(graph, scores)
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a complete SHA-256 hex digest")
    if not _is_sha256(scoring_source_sha256):
        raise ValueError("scoring_source_sha256 must be a complete SHA-256 hex digest")
    if detector_threshold_k <= 0.0:
        raise ValueError("detector_threshold_k must be positive")
    if detect_seconds < 0.0 or score_seconds < 0.0:
        raise ValueError("score-cache runtimes must be non-negative")
    metadata = {
        "score_cache_schema_version": SCORE_CACHE_SCHEMA_VERSION,
        "dataset": dataset,
        "data_split": data_split,
        "checkpoint_sha256": checkpoint_sha256,
        "scoring_source_sha256": scoring_source_sha256,
        "detector_threshold_k": float(detector_threshold_k),
        "patch_shape": [int(value) for value in patch_shape],
        "detect_seconds": float(detect_seconds),
        "score_seconds": float(score_seconds),
        "candidate_count": len(values),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            node_ids=np.asarray(graph.node_ids, dtype=np.int64),
            t=np.asarray(graph.t),
            z=np.asarray(graph.z, dtype=np.float64),
            y=np.asarray(graph.y, dtype=np.float64),
            x=np.asarray(graph.x, dtype=np.float64),
            scores=values,
        )
    temporary.replace(output)


def read_score_cache(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_data_split: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_scoring_source_sha256: str | None = None,
    expected_detector_threshold_k: float | None = None,
    expected_patch_shape: tuple[int, int, int] | None = None,
) -> tuple[TrackingGraph, np.ndarray, dict[str, Any]]:
    """Load a score cache and reject stale or incompatible evaluation inputs."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {"metadata_json", "node_ids", "t", "z", "y", "x", "scores"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"score cache {source} is missing fields: {missing}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        graph = TrackingGraph(
            node_ids=np.asarray(payload["node_ids"], dtype=np.int64),
            t=np.asarray(payload["t"]),
            z=np.asarray(payload["z"], dtype=np.float64),
            y=np.asarray(payload["y"], dtype=np.float64),
            x=np.asarray(payload["x"], dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
        )
        scores = np.asarray(payload["scores"], dtype=np.float32)
    _validate_candidate_scores(graph, scores)
    if metadata.get("score_cache_schema_version") != SCORE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"score cache {source} uses an unsupported schema")
    if int(metadata.get("candidate_count", -1)) != len(scores):
        raise ValueError(f"score cache {source} has an inconsistent candidate count")

    expected = {
        "dataset": expected_dataset,
        "data_split": expected_data_split,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "scoring_source_sha256": expected_scoring_source_sha256,
    }
    for field, value in expected.items():
        if value is not None and metadata.get(field) != value:
            raise ValueError(
                f"score cache {source} has {field}={metadata.get(field)!r}, expected {value!r}"
            )
    if expected_detector_threshold_k is not None and not np.isclose(
        float(metadata.get("detector_threshold_k", -1.0)),
        expected_detector_threshold_k,
    ):
        raise ValueError(f"score cache {source} uses a different DoG threshold")
    if expected_patch_shape is not None and tuple(metadata.get("patch_shape", ())) != tuple(
        expected_patch_shape
    ):
        raise ValueError(f"score cache {source} uses a different patch shape")
    return graph, scores, metadata


def write_frozen_policy(path: str | Path, policy: dict[str, Any]) -> None:
    """Validate and atomically write one candidate policy selected without holdout data."""
    validate_frozen_policy(policy)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def validate_frozen_policy(policy: dict[str, Any]) -> None:
    """Reject an incomplete policy before it can unlock the held-out embryo."""
    if policy.get("candidate_policy_schema_version") != CANDIDATE_POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported frozen candidate policy schema")
    if policy.get("selection_cohort") != "internal-dev":
        raise ValueError("candidate policy must be selected exclusively on internal-dev movies")
    checkpoint_sha256 = policy.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("candidate policy must record a complete checkpoint SHA-256")
    evaluation_sha256 = policy.get("evaluation_sha256")
    if not _is_sha256(evaluation_sha256):
        raise ValueError("candidate policy must record a complete evaluation SHA-256")
    if not _is_sha256(policy.get("source_results_sha256")):
        raise ValueError("candidate policy must record a complete source-results SHA-256")
    threshold = float(policy.get("score_threshold", -1.0))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("candidate policy score_threshold must be greater than 0 and at most 1")
    if int(policy.get("min_track_length", 0)) < 1:
        raise ValueError("candidate policy min_track_length must be positive")
    if policy.get("internal_improvement_supported") is not True:
        raise ValueError("candidate policy must improve the internal-dev tracking baseline")
    datasets = policy.get("selection_datasets")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(dataset, str) and dataset.startswith("6bba_") for dataset in datasets)
        or datasets != sorted(set(datasets))
    ):
        raise ValueError(
            "candidate policy must record unique sorted internal 6bba selection movies"
        )
    density_limit = float(policy.get("density_limit", -1.0))
    if not 0.0 < density_limit <= LOCKED_NODE_DENSITY_LIMIT:
        raise ValueError("candidate policy exceeds the locked node-density limit")
    if policy.get("selection_rule") != CANDIDATE_SELECTION_RULE:
        raise ValueError("candidate policy does not use the locked selection rule")
    selected_metrics = policy.get("selected_metrics")
    baseline_metrics = policy.get("baseline_metrics")
    required_metrics = {
        "mean_edge_jaccard",
        "median_edge_jaccard",
        "aggregate_node_ratio",
        "mean_recall",
        "mean_runtime_seconds",
    }
    if not isinstance(selected_metrics, dict) or set(selected_metrics) != required_metrics:
        raise ValueError("candidate policy has incomplete selected metrics")
    if not isinstance(baseline_metrics, dict) or set(baseline_metrics) != required_metrics:
        raise ValueError("candidate policy has incomplete baseline metrics")
    for name, metrics in (("selected", selected_metrics), ("baseline", baseline_metrics)):
        values = np.array(list(metrics.values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"candidate policy has non-finite {name} metrics")
        if not 0.0 <= float(metrics["mean_edge_jaccard"]) <= 1.0:
            raise ValueError(f"candidate policy has invalid {name} mean edge Jaccard")
        if not 0.0 <= float(metrics["median_edge_jaccard"]) <= 1.0:
            raise ValueError(f"candidate policy has invalid {name} median edge Jaccard")
        if float(metrics["aggregate_node_ratio"]) < 0.0:
            raise ValueError(f"candidate policy has invalid {name} node ratio")
        if not 0.0 <= float(metrics["mean_recall"]) <= 1.0:
            raise ValueError(f"candidate policy has invalid {name} recall")
        if float(metrics["mean_runtime_seconds"]) < 0.0:
            raise ValueError(f"candidate policy has invalid {name} runtime")
    if float(selected_metrics["aggregate_node_ratio"]) > density_limit:
        raise ValueError("candidate policy selected metrics exceed its density limit")
    gain = float(policy.get("mean_edge_jaccard_gain", 0.0))
    expected_gain = float(selected_metrics["mean_edge_jaccard"]) - float(
        baseline_metrics["mean_edge_jaccard"]
    )
    if gain <= 0.0 or not np.isclose(gain, expected_gain):
        raise ValueError("candidate policy does not record a consistent positive internal gain")
    counts = [
        policy.get("per_movie_wins"),
        policy.get("per_movie_ties"),
        policy.get("per_movie_losses"),
    ]
    if any(not isinstance(value, int) or value < 0 for value in counts) or sum(counts) != len(
        datasets
    ):
        raise ValueError("candidate policy has inconsistent per-movie comparison counts")


def read_frozen_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate a candidate policy before locked evaluation."""
    policy = json.loads(Path(path).read_text())
    validate_frozen_policy(policy)
    return policy
