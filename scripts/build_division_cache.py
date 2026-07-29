#!/usr/bin/env python3
"""Build strict frozen-encoder node caches for selected Phase 11 ``6bba`` movies."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import io
import json
import pickle
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import (  # noqa: E402
    normalize_patches,
    validate_patch_shape,
)
from biohub.appearance_evaluation import read_score_cache  # noqa: E402
from biohub.appearance_model import (  # noqa: E402
    CellnessCNN3D,
    load_cellness_checkpoint,
)
from biohub.detect import DetectorConfig, detect_movie  # noqa: E402
from biohub.division_data import (  # noqa: E402
    ROLE_NAMES,
    DivisionManifestRow,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    read_division_contracts,
    read_division_manifest,
    source_bundle_sha256,
    verify_division_contract_bundle,
)
from biohub.division_model import (  # noqa: E402
    CACHE_DEVICE,
    CELLNESS_MATCH_ATOL,
    DIVISION_CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_BASE_CHECKPOINT_SHA256,
    EXPECTED_MODEL_STATE_SHA256,
    EXPECTED_SCORING_SOURCE_SHA256,
    NODE_CACHE_DETECTOR_THRESHOLD_K,
    NODE_CACHE_EMBEDDING_DIM,
    NODE_CACHE_NORMALIZATION,
    NODE_CACHE_PATCH_SHAPE,
    DivisionModel,
    DivisionModelConfig,
    FrozenCellnessEncoder,
    canonical_model_state_sha256,
    current_division_contract_hashes,
    raw_patch_intensity_summaries,
    read_division_node_cache,
    write_division_node_cache,
)
from biohub.io import SamplePaths, discover_split, open_image_array  # noqa: E402
from biohub.metric import TrackingGraph  # noqa: E402

EVALUATION_UNLOCK_POLICY_SCHEMA_VERSION = 3
DEVELOPMENT_RESULT_SCHEMA_VERSION = 3
CACHE_SOURCE_PATHS = (
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "appearance_evaluation.py",
    SRC / "biohub" / "appearance_model.py",
    SRC / "biohub" / "detect.py",
    SRC / "biohub" / "division_data.py",
    SRC / "biohub" / "division_model.py",
    SRC / "biohub" / "io.py",
    Path(__file__).resolve(),
)
IMAGE_HASH_SCHEMA = "biohub-logical-tzyx-image-v1"
_EVALUATION_POLICY_FIELDS = {
    "assignment_sha256",
    "base_checkpoint_sha256",
    "contract_sha256",
    "dataset_manifest_sha256",
    "development_grid",
    "development_guard_results",
    "development_result",
    "division_checkpoint",
    "division_policy_schema_version",
    "evaluation_unlocked",
    "feature_schema_sha256",
    "model_config",
    "model_inputs_sha256",
    "policy_source_sha256",
    "score_threshold",
    "selection_role",
    "trainer_source_sha256",
}
_DEVELOPMENT_RESULT_FIELDS = {
    "assignment_sha256",
    "base_checkpoint_sha256",
    "contract_sha256",
    "dataset_manifest_sha256",
    "development_result_schema_version",
    "division_checkpoint",
    "feature_schema_sha256",
    "model_config",
    "model_inputs_sha256",
    "role",
    "rows",
    "rows_sha256",
    "score_threshold",
    "trainer_source_sha256",
}
_DEVELOPMENT_ROW_FIELDS = {
    "accepted_forks",
    "added_edge_count",
    "baseline_edge_jaccard",
    "baseline_full_score",
    "baseline_predicted_nodes",
    "candidate_edge_jaccard",
    "candidate_full_score",
    "candidate_graph_sha256",
    "candidate_predicted_nodes",
    "dataset",
    "edge_removal_count",
    "estimated_nodes",
    "fallback_graph_sha256",
    "fully_matched_wrong_forks",
    "matched_division_false_positives",
    "restored_node_count",
    "strict_true_recoveries",
    "truth_division_count",
    "truth_graph_sha256",
}
_DIVISION_CHECKPOINT_FIELDS = {
    "base_checkpoint_sha256",
    "contract_sha256",
    "dataset_manifest_sha256",
    "division_checkpoint_schema_version",
    "epoch",
    "feature_schema_sha256",
    "head_state_dict",
    "metrics",
    "model_config",
    "model_inputs_sha256",
    "trainer_source_sha256",
}


@dataclass(frozen=True)
class CacheBuildResult:
    dataset: str
    role: str
    path: Path
    candidate_count: int
    content_sha256: str
    file_sha256: str
    max_cellness_abs_diff: float | None
    skipped: bool


_EVALUATION_AUTHORIZATION_KEY = object()


@dataclass(frozen=True, slots=True, init=False)
class _EvaluationAuthorization:
    evaluation_datasets: tuple[str, ...]
    upstream_manifest_sha256: str
    policy_sha256: str
    cache_source_sha256: str

    def __init__(
        self,
        *,
        evaluation_datasets: Sequence[str],
        upstream_manifest_sha256: str,
        policy_sha256: str,
        cache_source_sha256: str,
        _key: object,
    ) -> None:
        if _key is not _EVALUATION_AUTHORIZATION_KEY:
            raise PermissionError(
                "evaluation authorization can be created only by strict policy verification"
            )
        datasets = tuple(evaluation_datasets)
        if len(datasets) != 20 or datasets != tuple(sorted(set(datasets))):
            raise ValueError("evaluation authorization requires the frozen 20-movie cohort")
        object.__setattr__(self, "evaluation_datasets", datasets)
        object.__setattr__(
            self,
            "upstream_manifest_sha256",
            _require_sha256(
                upstream_manifest_sha256,
                field="authorization upstream manifest SHA-256",
            ),
        )
        object.__setattr__(
            self,
            "policy_sha256",
            _require_sha256(policy_sha256, field="authorization policy SHA-256"),
        )
        object.__setattr__(
            self,
            "cache_source_sha256",
            _require_sha256(
                cache_source_sha256,
                field="authorization cache source SHA-256",
            ),
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_policy_artifact(
    record: Any,
    *,
    field: str,
    project_root: Path,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"evaluation policy {field} must contain exact path and SHA-256 fields")
    logical_path = record["path"]
    expected_hash = _require_sha256(record["sha256"], field=f"evaluation policy {field} SHA-256")
    if not isinstance(logical_path, str) or Path(logical_path).is_absolute():
        raise ValueError(f"evaluation policy {field} path must be a project-relative string")
    path = project_root / logical_path
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"evaluation policy {field} path escapes the project root") from exc
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"evaluation policy {field} artifact is missing or has changed")
    return path


def _require_policy_artifact_snapshot(
    record: Any,
    *,
    field: str,
    project_root: Path,
) -> tuple[Path, bytes]:
    path = _require_policy_artifact(
        record,
        field=field,
        project_root=project_root,
    )
    content = path.read_bytes()
    expected_hash = _require_sha256(
        record["sha256"],
        field=f"evaluation policy {field} SHA-256",
    )
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"evaluation policy {field} artifact changed while reading")
    return path, content


def _canonical_json_snapshot(content: bytes, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise ValueError(f"{field} is not strict canonical JSON")
    return payload


def _resolve_relative_directory(
    logical_path: Any,
    *,
    field: str,
    project_root: Path,
) -> Path:
    if not isinstance(logical_path, str) or Path(logical_path).is_absolute():
        raise ValueError(f"{field} must be a project-relative directory")
    path = (project_root / logical_path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the project root") from exc
    if not path.is_dir():
        raise ValueError(f"{field} is missing")
    return path


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        requirement = (
            "a non-negative integer" if minimum == 0 else f"an integer of at least {minimum}"
        )
        raise ValueError(f"development row {field} must be {requirement}")
    return int(value)


def _require_unit_float(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"development row {field} must be finite and between zero and one")
    return float(value)


def _development_division_counts(split: Mapping[str, Any]) -> dict[str, int]:
    expected_datasets = split["role_datasets"]["development"]
    counts = {
        dataset: int(division_count)
        for division_count, roles in split["strata"].items()
        for dataset in roles["development"]
    }
    if sorted(counts) != expected_datasets or len(counts) != 20:
        raise ValueError("frozen split does not define the exact 20 development datasets")
    return counts


def _manifest_truth_by_dataset(
    manifest_rows: Sequence[DivisionManifestRow],
    *,
    split: Mapping[str, Any],
) -> dict[str, DivisionManifestRow]:
    rows = tuple(manifest_rows)
    if len(rows) != 128 or any(not isinstance(row, DivisionManifestRow) for row in rows):
        raise ValueError("frozen manifest must contain exactly 128 DivisionManifestRow records")
    by_dataset = {row.dataset: row for row in rows}
    if len(by_dataset) != len(rows):
        raise ValueError("frozen manifest contains duplicate datasets")
    expected_all = set().union(*map(set, split["role_datasets"].values()))
    if set(by_dataset) != expected_all:
        raise ValueError("frozen manifest dataset tuple does not match the split contract")
    for role in ROLE_NAMES:
        observed = sorted(row.dataset for row in rows if row.role == role)
        if observed != split["role_datasets"][role]:
            raise ValueError(f"frozen manifest {role} role does not match the split contract")
    split_counts = {
        dataset: int(division_count)
        for division_count, roles in split["strata"].items()
        for role in ROLE_NAMES
        for dataset in roles[role]
    }
    for dataset, row in by_dataset.items():
        if (
            row.embryo != "6bba"
            or row.division_count != split_counts[dataset]
            or row.no_division != (row.division_count == 0)
        ):
            raise ValueError(f"frozen manifest truth is inconsistent for {dataset}")
    return by_dataset


def _validate_development_rows(
    rows: Any,
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    manifest_rows: Sequence[DivisionManifestRow],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("development result rows must be a JSON list")
    split = contracts["split"]
    expected_datasets = split["role_datasets"]["development"]
    observed_datasets = [row.get("dataset") if isinstance(row, dict) else None for row in rows]
    if not all(isinstance(dataset, str) for dataset in observed_datasets):
        raise ValueError("development result dataset values must be strings")
    if len(set(observed_datasets)) != len(observed_datasets):
        raise ValueError("development result contains a duplicate dataset")
    if observed_datasets != expected_datasets:
        raise ValueError("development result does not contain the exact development dataset cohort")
    division_counts = _development_division_counts(split)
    manifest_by_dataset = _manifest_truth_by_dataset(manifest_rows, split=split)
    max_forks = int(contracts["assignment"]["max_forks_per_movie"])
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _DEVELOPMENT_ROW_FIELDS:
            raise ValueError("development result row has incomplete or unexpected fields")
        dataset = row["dataset"]
        truth_graph_hash = _require_sha256(
            row["truth_graph_sha256"],
            field="truth_graph_sha256",
        )
        manifest_truth = manifest_by_dataset[dataset]
        if truth_graph_hash != manifest_truth.truth_graph_sha256:
            raise ValueError(f"development row truth_graph_sha256 is wrong for {dataset}")
        truth_count = _require_int(row["truth_division_count"], field="truth_division_count")
        manifest_truth_count = manifest_truth.division_count
        if truth_count != manifest_truth_count:
            raise ValueError(
                f"development row manifest truth_division_count is wrong for {dataset}"
            )
        if manifest_truth_count != division_counts[dataset]:
            raise ValueError(f"frozen manifest and split truth disagree for {dataset}")
        accepted_forks = _require_int(row["accepted_forks"], field="accepted_forks")
        if accepted_forks > max_forks:
            raise ValueError("development row accepted_forks exceeds the frozen assignment cap")
        added_edges = _require_int(row["added_edge_count"], field="added_edge_count")
        removed_edges = _require_int(row["edge_removal_count"], field="edge_removal_count")
        restored_nodes = _require_int(row["restored_node_count"], field="restored_node_count")
        baseline_nodes = _require_int(
            row["baseline_predicted_nodes"],
            field="baseline_predicted_nodes",
        )
        candidate_nodes = _require_int(
            row["candidate_predicted_nodes"],
            field="candidate_predicted_nodes",
        )
        estimated_nodes = _require_int(row["estimated_nodes"], field="estimated_nodes", minimum=1)
        false_positives = _require_int(
            row["matched_division_false_positives"],
            field="matched_division_false_positives",
        )
        wrong_forks = _require_int(
            row["fully_matched_wrong_forks"],
            field="fully_matched_wrong_forks",
        )
        strict_recoveries = _require_int(
            row["strict_true_recoveries"],
            field="strict_true_recoveries",
        )
        if strict_recoveries > truth_count:
            raise ValueError(
                "development row strict_true_recoveries cannot exceed truth_division_count"
            )
        if wrong_forks > truth_count:
            raise ValueError(
                "development row fully_matched_wrong_forks cannot exceed truth_division_count"
            )
        if accepted_forks < strict_recoveries + wrong_forks + false_positives:
            raise ValueError(
                "development row accepted_forks must cover strict_true_recoveries, "
                "fully_matched_wrong_forks, and matched division false positives"
            )
        fallback_hash = _require_sha256(
            row["fallback_graph_sha256"],
            field="fallback graph SHA-256",
        )
        candidate_hash = _require_sha256(
            row["candidate_graph_sha256"],
            field="candidate graph SHA-256",
        )
        baseline_edge = _require_unit_float(
            row["baseline_edge_jaccard"], field="baseline_edge_jaccard"
        )
        baseline_full = _require_unit_float(row["baseline_full_score"], field="baseline_full_score")
        candidate_edge = _require_unit_float(
            row["candidate_edge_jaccard"], field="candidate_edge_jaccard"
        )
        candidate_full = _require_unit_float(
            row["candidate_full_score"], field="candidate_full_score"
        )
        hashes_equal = candidate_hash == fallback_hash
        if hashes_equal and accepted_forks != 0:
            raise ValueError("same graph hash cannot record accepted forks or candidate mutations")
        if accepted_forks == 0:
            if not hashes_equal:
                raise ValueError("no-fork row violates no_fork_fallback_identity")
            if (
                candidate_full != baseline_full
                or candidate_edge != baseline_edge
                or candidate_nodes != baseline_nodes
            ):
                raise ValueError(
                    "no-fork row metrics must exactly equal fallback metrics "
                    "for the same graph hash"
                )
            if restored_nodes != 0 or removed_edges != 0 or added_edges != 0:
                raise ValueError("no-fork row mutation counts must all be zero")
            if strict_recoveries != 0 or wrong_forks != 0 or false_positives != 0:
                raise ValueError("no-fork row graph-derived fork counts must all be zero")
        else:
            if hashes_equal:
                raise ValueError("accepted forks require a changed candidate graph hash")
            if not accepted_forks <= added_edges <= 2 * accepted_forks:
                raise ValueError("added_edge_count is impossible for the accepted_forks count")
            if restored_nodes > 3 * accepted_forks or removed_edges > 3 * accepted_forks:
                raise ValueError("mutation counts are impossible for the accepted_forks count")
            if candidate_nodes != baseline_nodes + restored_nodes:
                raise ValueError(
                    "candidate_predicted_nodes must equal fallback nodes plus restored_node_count"
                )
        normalized.append(
            {
                "accepted_forks": accepted_forks,
                "added_edge_count": added_edges,
                "baseline_edge_jaccard": baseline_edge,
                "baseline_full_score": baseline_full,
                "baseline_predicted_nodes": baseline_nodes,
                "candidate_edge_jaccard": candidate_edge,
                "candidate_full_score": candidate_full,
                "candidate_graph_sha256": candidate_hash,
                "candidate_predicted_nodes": candidate_nodes,
                "dataset": dataset,
                "edge_removal_count": removed_edges,
                "estimated_nodes": estimated_nodes,
                "fallback_graph_sha256": fallback_hash,
                "fully_matched_wrong_forks": wrong_forks,
                "matched_division_false_positives": false_positives,
                "restored_node_count": restored_nodes,
                "strict_true_recoveries": strict_recoveries,
                "truth_division_count": truth_count,
                "truth_graph_sha256": truth_graph_hash,
            }
        )
    return normalized


def recompute_development_guards(
    rows: Any,
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    manifest_rows: Sequence[DivisionManifestRow],
) -> tuple[dict[str, bool], dict[str, float | int | bool]]:
    """Recompute every development unlock guard from exact row-level values."""
    normalized = _validate_development_rows(
        rows,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    full_deltas = np.array(
        [row["candidate_full_score"] - row["baseline_full_score"] for row in normalized],
        dtype=np.float64,
    )
    edge_deltas = np.array(
        [row["candidate_edge_jaccard"] - row["baseline_edge_jaccard"] for row in normalized],
        dtype=np.float64,
    )
    estimated_nodes = sum(row["estimated_nodes"] for row in normalized)
    baseline_ratio = sum(row["baseline_predicted_nodes"] for row in normalized) / estimated_nodes
    candidate_ratio = sum(row["candidate_predicted_nodes"] for row in normalized) / estimated_nodes
    wins = int(np.sum(full_deltas > 0.0))
    losses = int(np.sum(full_deltas < 0.0))
    ties = len(normalized) - wins - losses
    no_division_datasets = set(contracts["split"]["strata"]["0"]["development"])
    control_false_positives_zero = all(
        row["matched_division_false_positives"] == 0
        for row in normalized
        if row["dataset"] in no_division_datasets
    )
    no_fork_identity = all(
        row["accepted_forks"] != 0 or row["candidate_graph_sha256"] == row["fallback_graph_sha256"]
        for row in normalized
    )
    wrong_forks = sum(row["fully_matched_wrong_forks"] for row in normalized)
    strict_recoveries = sum(row["strict_true_recoveries"] for row in normalized)
    mean_full_delta = float(full_deltas.mean())
    mean_edge_delta = float(edge_deltas.mean())
    density_delta = float(candidate_ratio - baseline_ratio)
    guards = contracts["acceptance"]["guards"]
    results = {
        "aggregate_node_ratio_delta": density_delta
        <= float(guards["aggregate_node_ratio_delta"]["value"]),
        "complete_cohort_movies": len(normalized) == int(guards["complete_cohort_movies"]["value"]),
        "fully_matched_wrong_forks": wrong_forks
        == int(guards["fully_matched_wrong_forks"]["value"]),
        "full_score_wins_vs_losses": wins > losses,
        "matched_division_false_positives_per_no_division_movie": (
            len(no_division_datasets)
            == int(guards["matched_division_false_positives_per_no_division_movie"]["movie_count"])
            and control_false_positives_zero
        ),
        "mean_edge_jaccard_delta": mean_edge_delta
        >= float(guards["mean_edge_jaccard_delta"]["value"]),
        "mean_full_score_delta": mean_full_delta >= float(guards["mean_full_score_delta"]["value"]),
        "no_fork_fallback_identity": no_fork_identity,
        "strict_true_recoveries": strict_recoveries
        >= int(guards["strict_true_recoveries"]["value"]),
    }
    if set(results) != set(guards):
        raise ValueError("development guard implementation disagrees with the acceptance contract")
    metrics: dict[str, float | int | bool] = {
        "aggregate_node_ratio_delta": density_delta,
        "fully_matched_wrong_forks": wrong_forks,
        "full_score_losses": losses,
        "full_score_ties": ties,
        "full_score_wins": wins,
        "mean_edge_jaccard_delta": mean_edge_delta,
        "mean_full_score_delta": mean_full_delta,
        "no_fork_fallback_identity": no_fork_identity,
        "strict_true_recoveries": strict_recoveries,
    }
    return results, metrics


def _validate_division_checkpoint_identity(
    checkpoint_bytes: bytes,
    *,
    policy: Mapping[str, Any],
) -> None:
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        raise ValueError(f"evaluation policy division checkpoint is invalid: {exc}") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != _DIVISION_CHECKPOINT_FIELDS:
        raise ValueError("evaluation policy division checkpoint has an incomplete contract chain")
    if checkpoint["division_checkpoint_schema_version"] != DIVISION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("evaluation policy division checkpoint has an unsupported schema")
    expected = {
        "base_checkpoint_sha256": policy["base_checkpoint_sha256"],
        "contract_sha256": policy["contract_sha256"],
        "dataset_manifest_sha256": policy["dataset_manifest_sha256"],
        "feature_schema_sha256": policy["feature_schema_sha256"],
        "model_config": policy["model_config"],
        "model_inputs_sha256": policy["model_inputs_sha256"],
        "trainer_source_sha256": policy["trainer_source_sha256"],
    }
    if any(checkpoint[field] != value for field, value in expected.items()):
        raise ValueError("evaluation policy checkpoint contract chain does not match the policy")
    config = DivisionModelConfig.from_dict(checkpoint["model_config"])
    state = checkpoint["head_state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str)
        or not isinstance(value, torch.Tensor)
        or not bool(torch.isfinite(value).all())
        for name, value in state.items()
    ):
        raise ValueError("evaluation policy checkpoint head state is invalid")
    try:
        DivisionModel(config).head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("evaluation policy checkpoint head does not match model config") from exc
    if (
        isinstance(checkpoint["epoch"], bool)
        or not isinstance(checkpoint["epoch"], Integral)
        or int(checkpoint["epoch"]) != 100
    ):
        raise ValueError("evaluation policy checkpoint must be the exact production epoch 100")
    metrics = checkpoint["metrics"]
    if not isinstance(metrics, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        for name, value in metrics.items()
    ):
        raise ValueError("evaluation policy checkpoint metrics are invalid")


def _validate_selected_identity(
    payload: Mapping[str, Any],
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    contract_sha256: Mapping[str, str],
    label: str,
) -> DivisionModelConfig:
    if payload["base_checkpoint_sha256"] != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError(f"{label} references the wrong external base checkpoint")
    if payload["contract_sha256"] != dict(contract_sha256):
        raise ValueError(f"{label} contract chain does not match current frozen configs")
    if payload["assignment_sha256"] != contract_sha256["assignment"]:
        raise ValueError(f"{label} assignment identity is not frozen")
    _require_sha256(payload["feature_schema_sha256"], field=f"{label} feature schema SHA-256")
    _require_sha256(
        payload["dataset_manifest_sha256"],
        field=f"{label} dataset manifest SHA-256",
    )
    _require_sha256(payload["model_inputs_sha256"], field=f"{label} model inputs SHA-256")
    _require_sha256(payload["trainer_source_sha256"], field=f"{label} trainer source SHA-256")
    try:
        config = DivisionModelConfig.from_dict(payload["model_config"])
    except ValueError as exc:
        raise ValueError(f"{label} model config is not one of the fixed ladder: {exc}") from exc
    threshold = payload["score_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(float(threshold))
        or float(threshold) not in contracts["assignment"]["threshold_grid"]
    ):
        raise ValueError(f"{label} threshold is outside the frozen assignment grid")
    return config


def verify_evaluation_unlock_policy(
    policy_path: str | Path,
    *,
    expected_policy_sha256: str,
    config_dir: str | Path,
    manifest_rows: Sequence[DivisionManifestRow],
    project_root: str | Path,
) -> _EvaluationAuthorization:
    """Replay the selected development policy before authorizing evaluation access."""
    contracts = read_division_contracts(config_dir)
    contract_sha256 = current_division_contract_hashes(config_dir)
    policy_file = Path(policy_path)
    expected_policy_hash = _require_sha256(
        expected_policy_sha256,
        field="expected evaluation policy SHA-256",
    )
    policy_bytes = policy_file.read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != expected_policy_hash:
        raise ValueError("frozen evaluation policy SHA-256 does not match the supplied identity")
    policy = _canonical_json_snapshot(policy_bytes, field="evaluation policy")
    if set(policy) != _EVALUATION_POLICY_FIELDS:
        raise ValueError("evaluation policy has incomplete or unexpected fields")
    if policy["division_policy_schema_version"] != EVALUATION_UNLOCK_POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation-unlock policy schema")
    if policy["selection_role"] != "development":
        raise ValueError("evaluation policy must be selected only from development")
    _validate_selected_identity(
        policy,
        contracts=contracts,
        contract_sha256=contract_sha256,
        label="evaluation policy",
    )
    root = Path(project_root)
    _, checkpoint_bytes = _require_policy_artifact_snapshot(
        policy["division_checkpoint"],
        field="division checkpoint",
        project_root=root,
    )
    _validate_division_checkpoint_identity(checkpoint_bytes, policy=policy)

    result_record = policy["development_result"]
    if not isinstance(result_record, Mapping) or set(result_record) != {
        "path",
        "rows_sha256",
        "sha256",
    }:
        raise ValueError("evaluation policy development result reference is invalid")
    rows_hash = _require_sha256(
        result_record["rows_sha256"],
        field="development rows SHA-256",
    )
    _, result_bytes = _require_policy_artifact_snapshot(
        {"path": result_record["path"], "sha256": result_record["sha256"]},
        field="development result",
        project_root=root,
    )
    try:
        development_result = _canonical_json_snapshot(
            result_bytes,
            field="development result",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"development result is not strict canonical JSON: {exc}") from exc
    if set(development_result) != _DEVELOPMENT_RESULT_FIELDS:
        raise ValueError("development result has incomplete or unexpected fields")
    if development_result["development_result_schema_version"] != DEVELOPMENT_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported development result schema")
    if development_result["role"] != "development":
        raise ValueError("development result must use only the development role")
    _validate_selected_identity(
        development_result,
        contracts=contracts,
        contract_sha256=contract_sha256,
        label="development result",
    )
    identity_fields = {
        "assignment_sha256",
        "base_checkpoint_sha256",
        "contract_sha256",
        "dataset_manifest_sha256",
        "division_checkpoint",
        "feature_schema_sha256",
        "model_config",
        "model_inputs_sha256",
        "score_threshold",
        "trainer_source_sha256",
    }
    if any(development_result[field] != policy[field] for field in identity_fields):
        raise ValueError("development result selected identities do not match the frozen policy")
    observed_rows_hash = canonical_json_sha256(development_result["rows"])
    if development_result["rows_sha256"] != observed_rows_hash or rows_hash != observed_rows_hash:
        raise ValueError("development result canonical row hash does not match its rows")
    guard_results, _ = recompute_development_guards(
        development_result["rows"],
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    failed = sorted(name for name, passed in guard_results.items() if not passed)
    if failed:
        raise ValueError("development guards failed: " + ", ".join(failed))
    claimed_guards = policy["development_guard_results"]
    if (
        not isinstance(claimed_guards, Mapping)
        or set(claimed_guards) != set(guard_results)
        or any(not isinstance(value, bool) for value in claimed_guards.values())
        or dict(claimed_guards) != guard_results
    ):
        raise ValueError("evaluation policy guard claims disagree with independent recomputation")
    if policy["evaluation_unlocked"] is not True:
        raise ValueError("development policy did not unlock evaluation")

    grid_record = policy["development_grid"]
    if not isinstance(grid_record, Mapping) or set(grid_record) != {
        "path",
        "rows_sha256",
        "sha256",
    }:
        raise ValueError("evaluation policy development grid reference is invalid")
    grid_rows_hash = _require_sha256(
        grid_record["rows_sha256"],
        field="development grid rows SHA-256",
    )
    grid_path, grid_bytes = _require_policy_artifact_snapshot(
        {"path": grid_record["path"], "sha256": grid_record["sha256"]},
        field="development grid",
        project_root=root,
    )
    from scripts.freeze_division_policy import (  # noqa: PLC0415
        development_result_payload,
        policy_source_sha256,
        select_grid_policy,
    )
    from scripts.validate_division_model import (  # noqa: PLC0415
        read_validation_grid,
        replay_selected_policy,
    )

    if policy["policy_source_sha256"] != policy_source_sha256():
        raise ValueError("evaluation policy freezer source has changed")
    grid = read_validation_grid(
        grid_path,
        require_complete=True,
        verify_current_sources=True,
    )
    if grid_path.read_bytes() != grid_bytes:
        raise ValueError("development grid changed while reading")
    if grid["rows_sha256"] != grid_rows_hash:
        raise ValueError("development grid row hash does not match the frozen policy")
    selection = select_grid_policy(
        grid,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    expected_result = development_result_payload(grid, selection)
    if development_result != expected_result:
        raise ValueError("development result is not the deterministic grid selection")
    if (
        selection.evaluation_unlocked is not True
        or selection.model_name != policy["model_config"]["model_name"]
        or selection.score_threshold != policy["score_threshold"]
        or dict(selection.guard_results) != dict(policy["development_guard_results"])
    ):
        raise ValueError("evaluation policy does not match the deterministic grid selection")

    upstream_manifest = _require_policy_artifact(
        grid["upstream_manifest"],
        field="grid upstream manifest",
        project_root=root,
    )
    if read_division_manifest(upstream_manifest) != tuple(manifest_rows):
        raise ValueError("development grid and cache command use different manifests")
    summary_path = _require_policy_artifact(
        grid["summary"],
        field="grid manifest summary",
        project_root=root,
    )
    base_checkpoint = _require_policy_artifact(
        grid["base_checkpoint"],
        field="grid base checkpoint",
        project_root=root,
    )
    fit_manifest = _require_policy_artifact(
        grid["fit_dataset_manifest"],
        field="grid fit dataset manifest",
        project_root=root,
    )
    data_root = _resolve_relative_directory(
        grid["data_root"],
        field="grid data root",
        project_root=root,
    )
    cache_dir = _resolve_relative_directory(
        grid["cache_directory"],
        field="grid cache directory",
        project_root=root,
    )
    current_cache_source = source_bundle_sha256(CACHE_SOURCE_PATHS)
    if grid["cache_source_sha256"] != current_cache_source:
        raise ValueError("development grid cache source does not match current cache code")
    replay_selected_policy(
        development_result,
        project_root=root,
        data_root=data_root,
        cache_dir=cache_dir,
        fit_dataset=fit_manifest.parent,
        base_checkpoint=base_checkpoint,
        config_dir=Path(config_dir),
        upstream_manifest=upstream_manifest,
        summary=summary_path,
        checkpoint_snapshot=checkpoint_bytes,
    )
    return _EvaluationAuthorization(
        evaluation_datasets=contracts["split"]["role_datasets"]["evaluation"],
        upstream_manifest_sha256=file_sha256(upstream_manifest),
        policy_sha256=expected_policy_hash,
        cache_source_sha256=current_cache_source,
        _key=_EVALUATION_AUTHORIZATION_KEY,
    )


def require_approved_cache_backend(device: str | torch.device) -> torch.device:
    """Require the one backend approved for real Phase 11 cache artifacts."""
    target = torch.device(device)
    if target.type != CACHE_DEVICE or target.index is not None:
        raise ValueError("real Phase 11 cache builds require the approved MPS backend")
    if not torch.backends.mps.is_available():
        raise ValueError("approved MPS backend is unavailable on this machine")
    return target


def load_trusted_cellness_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[CellnessCNN3D, dict[str, Any]]:
    """Load the accepted checkpoint only after file and logical-state verification."""
    path = Path(checkpoint_path)
    if file_sha256(path) != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 is not the accepted Phase 10 checkpoint")
    model, checkpoint = load_cellness_checkpoint(path, device=device)
    state = checkpoint.get("model_state_dict")
    try:
        state_hash = canonical_model_state_sha256(state)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint model-state fingerprint is invalid: {exc}") from exc
    if state_hash != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("checkpoint model-state fingerprint is not the accepted Phase 10 state")
    if tuple(checkpoint.get("patch_shape", ())) != NODE_CACHE_PATCH_SHAPE:
        raise ValueError("accepted checkpoint does not use exact patch shape 9x33x33")
    if checkpoint.get("normalization") != NODE_CACHE_NORMALIZATION:
        raise ValueError("accepted checkpoint does not use the frozen patch normalization")
    if checkpoint.get("detector_threshold_k") != NODE_CACHE_DETECTOR_THRESHOLD_K:
        raise ValueError("accepted checkpoint does not use exact DoG k=3")
    if int(model.config.base_channels) * 4 != NODE_CACHE_EMBEDDING_DIM:
        raise ValueError("accepted checkpoint does not expose the required 64-D embedding")
    return model, checkpoint


def cache_lock_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(f"{path.name}.lock")


@contextmanager
def exclusive_cache_builder_lock(output: str | Path) -> Iterator[None]:
    """Hold a nonblocking OS lock for one output through validation and promotion."""
    lock_path = cache_lock_path(output)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(f"cache builder is already active for {Path(output)}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _image_layout(image_array: Any) -> tuple[tuple[int, int, int, int], np.dtype[Any]]:
    try:
        shape = tuple(int(value) for value in image_array.shape)
        dtype = np.dtype(image_array.dtype)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("image must expose a valid tzyx shape and numeric dtype") from exc
    if len(shape) != 4 or any(value < 1 for value in shape):
        raise ValueError(f"image must have nonempty tzyx shape, got {shape!r}")
    if not np.issubdtype(dtype, np.number):
        raise ValueError("image must use a numeric dtype")
    return shape, dtype.newbyteorder("<")


def _new_image_digest(shape: tuple[int, ...], dtype: np.dtype[Any]) -> Any:
    digest = hashlib.sha256()
    digest.update(IMAGE_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0dtype\0")
    digest.update(dtype.str.encode("ascii"))
    digest.update(b"\0shape\0")
    digest.update(",".join(str(value) for value in shape).encode("ascii"))
    digest.update(b"\0data\0")
    return digest


def _canonical_image_frame(
    image_array: Any,
    timepoint: int,
    *,
    frame_shape: tuple[int, int, int],
    dtype: np.dtype[Any],
) -> np.ndarray:
    frame = np.asarray(image_array[timepoint])
    if frame.shape != frame_shape:
        raise ValueError(
            f"image timepoint {timepoint} has shape {frame.shape}, expected {frame_shape}"
        )
    return np.ascontiguousarray(frame, dtype=dtype)


def canonical_image_sha256(image_array: Any) -> str:
    """Hash logical tzyx image values independently of the Zarr container."""
    shape, dtype = _image_layout(image_array)
    digest = _new_image_digest(shape, dtype)
    for timepoint in range(shape[0]):
        frame = _canonical_image_frame(
            image_array,
            timepoint,
            frame_shape=shape[1:],
            dtype=dtype,
        )
        digest.update(frame.tobytes(order="C"))
    digest.update(b"\0")
    return digest.hexdigest()


def pad_frame_for_patches(
    volume: np.ndarray,
    patch_shape: tuple[int, int, int],
) -> np.ndarray:
    """Pad one frame exactly as frozen appearance patch extraction requires."""
    shape = validate_patch_shape(patch_shape)
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {array.shape!r}")
    radii = np.asarray(shape) // 2
    padding = tuple((int(radius), int(radius)) for radius in radii)
    mode = "reflect" if all(size > 1 for size in (*array.shape, *shape)) else "edge"
    return np.pad(array, padding, mode=mode)


def gather_patches_from_padded_frame(
    padded: np.ndarray,
    centers_zyx: np.ndarray,
    *,
    original_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> np.ndarray:
    """Gather one bounded patch batch from a frame padded exactly once."""
    shape = validate_patch_shape(patch_shape)
    original = tuple(int(value) for value in original_shape)
    array = np.asarray(padded)
    radii = np.asarray(shape) // 2
    expected_padded_shape = tuple(
        size + 2 * int(radius) for size, radius in zip(original, radii, strict=True)
    )
    if len(original) != 3 or array.shape != expected_padded_shape:
        raise ValueError("padded frame shape is inconsistent with the original and patch shapes")
    centers = np.asarray(centers_zyx, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or not np.all(np.isfinite(centers)):
        raise ValueError("centers_zyx must contain finite values with shape [n,3]")
    rounded = np.rint(centers).astype(np.int64)
    if any(
        np.any(rounded[:, axis] < 0) or np.any(rounded[:, axis] >= size)
        for axis, size in enumerate(original)
    ):
        raise ValueError(f"at least one center is outside volume shape {original}")
    if len(rounded) == 0:
        return np.empty((0, *shape), dtype=array.dtype)
    z_indices = rounded[:, 0, None, None, None] + np.arange(shape[0])[None, :, None, None]
    y_indices = rounded[:, 1, None, None, None] + np.arange(shape[1])[None, None, :, None]
    x_indices = rounded[:, 2, None, None, None] + np.arange(shape[2])[None, None, None, :]
    return np.asarray(array[z_indices, y_indices, x_indices])


@torch.inference_mode()
def compute_candidate_features(
    model: CellnessCNN3D,
    image_array: Any,
    candidates: TrackingGraph,
    *,
    patch_shape: tuple[int, int, int] = NODE_CACHE_PATCH_SHAPE,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Compute each embedding and its cellness from the same bounded backbone pass."""
    if not isinstance(model, CellnessCNN3D):
        raise TypeError("candidate feature extraction requires a CellnessCNN3D")
    shape = validate_patch_shape(patch_shape)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    image_shape, image_dtype = _image_layout(image_array)
    node_ids = np.asarray(candidates.node_ids)
    count = len(node_ids)
    expected = (count,)
    if count < 1 or node_ids.ndim != 1 or len(np.unique(node_ids)) != count:
        raise ValueError("candidate graph must contain unique one-dimensional node identifiers")
    try:
        times = np.asarray(candidates.t, dtype=np.float64)
        zyx = np.column_stack([candidates.z, candidates.y, candidates.x]).astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate graph coordinates must be numeric") from exc
    if (
        times.shape != expected
        or zyx.shape != (count, 3)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(zyx))
        or not np.all(times == np.floor(times))
        or np.any(times < 0.0)
        or np.any(times >= image_shape[0])
        or np.any(zyx < 0.0)
    ):
        raise ValueError("candidate graph must contain valid non-negative tzyx coordinates")
    target_device = torch.device(device)
    encoder = FrozenCellnessEncoder(model.to(target_device)).to(target_device)
    encoder.eval()
    embeddings = np.empty((count, encoder.output_dim), dtype=np.float32)
    cellness = np.empty(count, dtype=np.float32)
    intensity_summaries = np.empty((count, 5), dtype=np.float32)
    integer_times = times.astype(np.int64)
    image_digest = _new_image_digest(image_shape, image_dtype)
    processed_candidates = 0
    for timepoint in range(image_shape[0]):
        volume = _canonical_image_frame(
            image_array,
            timepoint,
            frame_shape=image_shape[1:],
            dtype=image_dtype,
        )
        image_digest.update(volume.tobytes(order="C"))
        frame_indices = np.flatnonzero(integer_times == timepoint)
        if len(frame_indices):
            padded = pad_frame_for_patches(volume, shape)
            for start in range(0, len(frame_indices), batch_size):
                indices = frame_indices[start : start + batch_size]
                raw_patches = gather_patches_from_padded_frame(
                    padded,
                    zyx[indices],
                    original_shape=volume.shape,
                    patch_shape=shape,
                )
                intensity_summaries[indices] = raw_patch_intensity_summaries(raw_patches)
                normalized = normalize_patches(raw_patches)
                inputs = torch.from_numpy(normalized).unsqueeze(1).to(target_device)
                batch_embeddings, batch_cellness = encoder.forward_with_cellness(inputs)
                embeddings[indices] = batch_embeddings.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
                cellness[indices] = batch_cellness.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
                processed_candidates += len(indices)
        completed_frames = timepoint + 1
        if progress_callback is not None and (
            completed_frames == image_shape[0] or completed_frames % 10 == 0
        ):
            progress_callback(
                completed_frames,
                image_shape[0],
                processed_candidates,
                count,
            )
    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(cellness)):
        raise RuntimeError("frozen candidate feature extraction produced non-finite values")
    image_digest.update(b"\0")
    return embeddings, cellness, intensity_summaries, image_digest.hexdigest()


def _phase10_score_cache_path(directory: str | Path, dataset: str) -> Path:
    return Path(directory) / (
        f"{dataset}.{EXPECTED_BASE_CHECKPOINT_SHA256[:12]}."
        f"{EXPECTED_SCORING_SOURCE_SHA256[:12]}.npz"
    )


def _graphs_match_exactly(left: TrackingGraph, right: TrackingGraph) -> bool:
    return all(
        np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))
        for name in ("node_ids", "t", "z", "y", "x")
    )


def _phase10_reference(
    score_cache_dir: str | Path,
    dataset: str,
) -> tuple[Path, TrackingGraph, np.ndarray] | None:
    path = _phase10_score_cache_path(score_cache_dir, dataset)
    if not path.is_file():
        return None
    graph, scores, _ = read_score_cache(
        path,
        expected_dataset=dataset,
        expected_data_split="train",
        expected_checkpoint_sha256=EXPECTED_BASE_CHECKPOINT_SHA256,
        expected_scoring_source_sha256=EXPECTED_SCORING_SOURCE_SHA256,
        expected_detector_threshold_k=3.0,
        expected_patch_shape=NODE_CACHE_PATCH_SHAPE,
    )
    return path, graph, scores


def _compare_with_phase10(
    graph: TrackingGraph,
    cellness: np.ndarray,
    reference: tuple[Path, TrackingGraph, np.ndarray] | None,
) -> tuple[str | None, float | None]:
    if reference is None:
        return None, None
    path, phase10_graph, phase10_scores = reference
    if not _graphs_match_exactly(graph, phase10_graph):
        raise ValueError("division cache candidates do not exactly match the Phase 10 score cache")
    if cellness.shape != phase10_scores.shape:
        raise ValueError("division cache cellness shape does not match the Phase 10 score cache")
    maximum_difference = float(
        np.max(np.abs(cellness.astype(np.float64) - phase10_scores.astype(np.float64)))
    )
    if maximum_difference > CELLNESS_MATCH_ATOL:
        raise ValueError(
            "division cache cellness differs from the accepted Phase 10 scoring path: "
            f"max_abs_diff={maximum_difference:.9g}, atol={CELLNESS_MATCH_ATOL:.9g}"
        )
    return file_sha256(path), maximum_difference


def validate_resume_reference(
    metadata: Mapping[str, Any],
    *,
    phase10_sha256: str | None,
    maximum_difference: float | None,
) -> None:
    """Require an existing cache to match the currently discovered reference exactly."""
    if metadata.get("phase10_score_cache_sha256") != phase10_sha256:
        raise ValueError("existing division cache references a different Phase 10 score cache")
    if metadata.get("phase10_max_cellness_abs_diff") != maximum_difference:
        raise ValueError("existing division cache records a stale Phase 10 cellness comparison")


def build_movie_cache(
    sample: SamplePaths,
    *,
    role: str,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    batch_size: int,
    device: str | torch.device,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
    phase10_score_cache_dir: str | Path,
    overwrite: bool = False,
    evaluation_authorization: _EvaluationAuthorization | None = None,
) -> CacheBuildResult:
    """Build or validate one cache under its exclusive per-output lock."""
    if role == "evaluation":
        if not isinstance(evaluation_authorization, _EvaluationAuthorization):
            raise PermissionError(
                "evaluation cache access requires strict development replay authorization"
            )
        if sample.dataset not in evaluation_authorization.evaluation_datasets:
            raise PermissionError("evaluation authorization does not cover this dataset")
        if upstream_manifest_sha256 != (
            evaluation_authorization.upstream_manifest_sha256
        ):
            raise PermissionError("evaluation authorization uses another manifest")
        if cache_source_sha256 != evaluation_authorization.cache_source_sha256:
            raise PermissionError("evaluation authorization uses another cache source")
    elif evaluation_authorization is not None:
        raise ValueError("evaluation authorization cannot be used for fit or development")
    if sample.embryo != "6bba" or not sample.dataset.startswith("6bba_"):
        raise ValueError("division cache construction accepts only 6bba datasets")
    if role not in ROLE_NAMES:
        raise ValueError("division cache construction requires a frozen movie role")
    output = Path(output_dir) / f"{sample.dataset}.npz"
    target_device = require_approved_cache_backend(device)
    with exclusive_cache_builder_lock(output):
        model, _ = load_trusted_cellness_model(checkpoint_path, device=target_device)
        return _build_movie_cache_locked(
            sample,
            role=role,
            output=output,
            model=model,
            batch_size=batch_size,
            device=target_device,
            cache_source_sha256=cache_source_sha256,
            contract_sha256=contract_sha256,
            upstream_manifest_sha256=upstream_manifest_sha256,
            phase10_score_cache_dir=phase10_score_cache_dir,
            overwrite=overwrite,
        )


def _build_movie_cache_locked(
    sample: SamplePaths,
    *,
    role: str,
    output: Path,
    model: CellnessCNN3D,
    batch_size: int,
    device: torch.device,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
    phase10_score_cache_dir: str | Path,
    overwrite: bool,
) -> CacheBuildResult:
    reference = _phase10_reference(phase10_score_cache_dir, sample.dataset)
    image = open_image_array(sample.image)
    if output.exists() and not overwrite:
        image_hash = canonical_image_sha256(image)
        graph, cellness, _, _, metadata = read_division_node_cache(
            output,
            expected_dataset=sample.dataset,
            expected_role=role,
            expected_cache_source_sha256=cache_source_sha256,
            expected_contract_sha256=contract_sha256,
            expected_upstream_manifest_sha256=upstream_manifest_sha256,
            expected_image_sha256=image_hash,
        )
        phase10_hash, maximum_difference = _compare_with_phase10(graph, cellness, reference)
        validate_resume_reference(
            metadata,
            phase10_sha256=phase10_hash,
            maximum_difference=maximum_difference,
        )
        return CacheBuildResult(
            dataset=sample.dataset,
            role=role,
            path=output,
            candidate_count=len(cellness),
            content_sha256=metadata["content_sha256"],
            file_sha256=file_sha256(output),
            max_cellness_abs_diff=maximum_difference,
            skipped=True,
        )

    candidates = detect_movie(
        image,
        DetectorConfig(method="dog", threshold_k=NODE_CACHE_DETECTOR_THRESHOLD_K),
    )
    embeddings, cellness, intensity_summaries, image_hash = compute_candidate_features(
        model,
        image,
        candidates,
        patch_shape=NODE_CACHE_PATCH_SHAPE,
        batch_size=batch_size,
        device=device,
        progress_callback=lambda completed, total, processed, candidate_count: print(
            f"  {sample.dataset} feature_progress={completed}/{total}_frames "
            f"candidates={processed:,}/{candidate_count:,}",
            flush=True,
        ),
    )
    if embeddings.shape[1] != NODE_CACHE_EMBEDDING_DIM:
        raise ValueError(
            f"accepted cellness backbone must emit {NODE_CACHE_EMBEDDING_DIM} features, "
            f"got {embeddings.shape[1]}"
        )
    phase10_hash, maximum_difference = _compare_with_phase10(
        candidates,
        cellness,
        reference,
    )
    content_hash = write_division_node_cache(
        output,
        candidates,
        cellness,
        embeddings,
        intensity_summaries,
        dataset=sample.dataset,
        role=role,
        checkpoint_sha256=EXPECTED_BASE_CHECKPOINT_SHA256,
        scoring_source_sha256=EXPECTED_SCORING_SOURCE_SHA256,
        cache_source_sha256=cache_source_sha256,
        contract_sha256=contract_sha256,
        upstream_manifest_sha256=upstream_manifest_sha256,
        image_sha256=image_hash,
        model_state_sha256=EXPECTED_MODEL_STATE_SHA256,
        phase10_score_cache_sha256=phase10_hash,
        phase10_max_cellness_abs_diff=maximum_difference,
    )
    _, checked_cellness, _, _, checked_metadata = read_division_node_cache(
        output,
        expected_dataset=sample.dataset,
        expected_role=role,
        expected_cache_source_sha256=cache_source_sha256,
        expected_contract_sha256=contract_sha256,
        expected_upstream_manifest_sha256=upstream_manifest_sha256,
        expected_image_sha256=image_hash,
    )
    return CacheBuildResult(
        dataset=sample.dataset,
        role=role,
        path=output,
        candidate_count=len(checked_cellness),
        content_sha256=content_hash,
        file_sha256=file_sha256(output),
        max_cellness_abs_diff=checked_metadata["phase10_max_cellness_abs_diff"],
        skipped=False,
    )


def _selected_rows(
    rows: Sequence[DivisionManifestRow],
    *,
    datasets: Sequence[str] | None,
    role: str | None,
    evaluation_datasets: Sequence[str],
) -> list[DivisionManifestRow]:
    by_dataset = {row.dataset: row for row in rows}
    frozen_evaluation = tuple(evaluation_datasets)
    evaluation_set = set(frozen_evaluation)
    if len(frozen_evaluation) != 20 or len(evaluation_set) != 20:
        raise ValueError("frozen split must define exactly 20 unique evaluation movies")
    if datasets is not None:
        requested = list(datasets)
        if len(set(requested)) != len(requested):
            raise ValueError("--dataset values must be unique")
        missing = sorted(set(requested) - set(by_dataset))
        if missing:
            raise ValueError(
                "requested datasets are outside the exact frozen 6bba manifest: "
                + ", ".join(missing)
            )
        selected = [by_dataset[dataset] for dataset in requested]
        if any(row.dataset in evaluation_set for row in selected):
            raise ValueError("evaluation movies may be selected only by exact --role evaluation")
        return selected
    if role is None:
        raise ValueError("select exact --dataset values or one --role")
    selected = [row for row in rows if row.role == role]
    if not selected:
        raise ValueError(f"the frozen manifest contains no {role} movies")
    if role == "evaluation":
        expected = frozen_evaluation
        observed = tuple(row.dataset for row in selected)
        if len(expected) != 20 or len(set(expected)) != 20 or observed != expected:
            raise ValueError("--role evaluation requires the exact frozen 20-movie cohort")
    elif any(row.dataset in evaluation_set for row in selected):
        raise ValueError("non-evaluation role selection contains a frozen evaluation movie")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--dataset",
        action="append",
        help="exact frozen 6bba dataset; repeat to build more than one",
    )
    selection.add_argument(
        "--role",
        choices=ROLE_NAMES,
        help="build the exact frozen fit or development role; evaluation is guarded",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "working" / "phase11_division_cache",
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
        "--phase10-score-cache-dir",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_scores" / "phase10",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=(CACHE_DEVICE,), default=CACHE_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--policy",
        type=Path,
        help="frozen development policy required for evaluation-role image access",
    )
    parser.add_argument(
        "--policy-sha256",
        help="exact externally pinned SHA-256 of --policy",
    )
    parser.add_argument(
        "--confirm-evaluation",
        action="store_true",
        help="confirm the one locked evaluation cache construction",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        device = require_approved_cache_backend(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    # This complete frozen-bundle verification deliberately precedes dataset discovery
    # and every call that can open a Zarr image.
    try:
        manifest_snapshot = args.upstream_manifest.read_bytes()
        verified_contracts = verify_division_contract_bundle(
            args.config_dir,
            manifest_path=args.upstream_manifest,
            summary_path=args.summary,
            project_root=ROOT,
            checkpoint_path=args.checkpoint,
        )
        if args.upstream_manifest.read_bytes() != manifest_snapshot:
            raise ValueError("upstream manifest changed during contract verification")
        manifest_snapshot_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
        if manifest_snapshot_sha256 != verified_contracts["split"]["manifest"]["sha256"]:
            raise ValueError("upstream manifest snapshot does not match the frozen split")
        contracts = current_division_contract_hashes(args.config_dir)
        manifest_rows = read_division_manifest(args.upstream_manifest)
        if args.upstream_manifest.read_bytes() != manifest_snapshot:
            raise ValueError("upstream manifest changed while parsing selected rows")
        selected_rows = _selected_rows(
            manifest_rows,
            datasets=args.dataset,
            role=args.role,
            evaluation_datasets=verified_contracts["split"]["role_datasets"]["evaluation"],
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(f"frozen Task 1 contract verification failed: {exc}")

    evaluation_selected = args.role == "evaluation"
    evaluation_authorization: _EvaluationAuthorization | None = None
    if evaluation_selected:
        if not args.confirm_evaluation:
            parser.error("evaluation cache construction requires --confirm-evaluation")
        if args.policy is None or args.policy_sha256 is None:
            parser.error("evaluation cache construction requires --policy and --policy-sha256")
        try:
            evaluation_authorization = verify_evaluation_unlock_policy(
                args.policy,
                expected_policy_sha256=args.policy_sha256,
                config_dir=args.config_dir,
                manifest_rows=manifest_rows,
                project_root=ROOT,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            parser.error(f"evaluation remains locked: {exc}")
    elif args.confirm_evaluation or args.policy is not None or args.policy_sha256 is not None:
        parser.error(
            "evaluation policy flags are accepted only for the exact --role evaluation cohort"
        )

    if args.upstream_manifest.read_bytes() != manifest_snapshot:
        parser.error("upstream manifest changed before dataset discovery")
    samples = discover_split(args.data_root, "train")
    by_dataset = {sample.dataset: sample for sample in samples if sample.embryo == "6bba"}
    missing = [row.dataset for row in selected_rows if row.dataset not in by_dataset]
    if missing:
        parser.error("selected frozen datasets are missing from train: " + ", ".join(missing))
    selected_samples = [by_dataset[row.dataset] for row in selected_rows]
    if any(sample.embryo != "6bba" for sample in selected_samples):
        parser.error("division caches cannot access 44b6")

    cache_source_hash = source_bundle_sha256(CACHE_SOURCE_PATHS)
    upstream_manifest_hash = manifest_snapshot_sha256
    print(
        f"Movies: {len(selected_rows)}   device={device}   DoG k=3   "
        f"patch={NODE_CACHE_PATCH_SHAPE}",
        flush=True,
    )
    print(f"Output: {args.output_dir}", flush=True)
    for index, row in enumerate(selected_rows, start=1):
        start = time.time()
        try:
            if args.upstream_manifest.read_bytes() != manifest_snapshot:
                raise ValueError("upstream manifest changed during cache construction")
            result = build_movie_cache(
                by_dataset[row.dataset],
                role=str(row.role),
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                batch_size=args.batch_size,
                device=device,
                cache_source_sha256=cache_source_hash,
                contract_sha256=contracts,
                upstream_manifest_sha256=upstream_manifest_hash,
                phase10_score_cache_dir=args.phase10_score_cache_dir,
                overwrite=args.overwrite,
                evaluation_authorization=evaluation_authorization,
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(f"cache build failed for {row.dataset}: {exc}")
        difference = (
            "unavailable"
            if result.max_cellness_abs_diff is None
            else f"{result.max_cellness_abs_diff:.9g}"
        )
        status = "validated" if result.skipped else "built"
        print(
            f"[{index}/{len(selected_rows)}] {row.dataset} role={row.role} {status} "
            f"candidates={result.candidate_count:,} max_cellness_diff={difference} "
            f"content_sha256={result.content_sha256} file_sha256={result.file_sha256} "
            f"elapsed={time.time() - start:.1f}s path={result.path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
