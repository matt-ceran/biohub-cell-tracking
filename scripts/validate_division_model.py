#!/usr/bin/env python3
"""Run the exact resumable Phase 11 development model and threshold grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.division_data import (  # noqa: E402
    DivisionManifestRow,
    canonical_json_sha256,
    file_sha256,
    logical_graph_sha256,
    read_canonical_json,
    read_division_contracts,
    read_division_manifest,
    source_bundle_sha256,
    truth_divisions,
    verify_division_contract_bundle,
    write_canonical_json,
)
from biohub.division_dataset import (  # noqa: E402
    EXPECTED_DEVELOPMENT_MOVIES,
    InferenceMovie,
    LoadedDivisionDataset,
    division_cache_source_sha256,
    iter_inference_proposal_batches,
    prepare_inference_movie,
    read_division_dataset,
)
from biohub.division_inference import (  # noqa: E402
    PROBABILITY_THRESHOLDS,
    AcceptedFork,
    MovieEvaluation,
    ScoredProposal,
    apply_division_forks,
    evaluate_division_movie,
    select_division_forks,
)
from biohub.division_model import (  # noqa: E402
    DIVISION_MODEL_NAMES,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    FIXED_DIVISION_MODEL_CONFIGS,
    DivisionModel,
    current_division_contract_hashes,
    load_division_checkpoint,
    read_division_node_cache,
)
from biohub.io import discover_split, read_geff_graph  # noqa: E402
from biohub.metric import MetricResult, TrackingGraph  # noqa: E402
from scripts.train_division import trainer_source_sha256  # noqa: E402

VALIDATION_GRID_SCHEMA_VERSION = 1
DEVELOPMENT_RESULT_SCHEMA_VERSION = 3
EXPECTED_GRID_ROWS = (
    EXPECTED_DEVELOPMENT_MOVIES
    * len(DIVISION_MODEL_NAMES)
    * len(PROBABILITY_THRESHOLDS)
)
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"
DEFAULT_CACHE_DIR = ROOT / "data" / "working" / "phase11_division_cache"
DEFAULT_FIT_DATASET = (
    ROOT / "data" / "working" / "phase11_division_dataset" / "fit_complete"
)
DEFAULT_CHECKPOINT_DIR = ROOT / "data" / "working" / "phase11_division_checkpoints"
DEFAULT_OUTPUT = ROOT / "data" / "working" / "phase11_division_development_grid.json"
DEFAULT_BASE_CHECKPOINT = (
    ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn_phase10_full.pt"
)
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_UPSTREAM_MANIFEST = ROOT / "data" / "working" / "phase11_division_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "working" / "phase11_division_manifest_summary.json"
VALIDATION_SOURCE_PATHS = (
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "division.py",
    SRC / "biohub" / "division_data.py",
    SRC / "biohub" / "division_dataset.py",
    SRC / "biohub" / "division_inference.py",
    SRC / "biohub" / "division_model.py",
    SRC / "biohub" / "link.py",
    SRC / "biohub" / "metric.py",
    Path(__file__).resolve(),
)

SELECTED_ROW_FIELDS = {
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
GRID_ROW_FIELDS = SELECTED_ROW_FIELDS | {
    "accepted_fork_records",
    "base_checkpoint_sha256",
    "baseline_metric",
    "baseline_triplets",
    "cache_content_sha256",
    "cache_file_sha256",
    "cache_image_sha256",
    "candidate_metric",
    "candidate_triplets",
    "checkpoint_sha256",
    "contract_sha256",
    "dataset_manifest_sha256",
    "feature_schema_sha256",
    "model_config",
    "model_inputs_sha256",
    "model_name",
    "runtime_seconds",
    "score_threshold",
    "trainer_source_sha256",
    "validation_source_sha256",
}
GRID_FIELDS = {
    "assignment_sha256",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "cache_directory",
    "cache_source_sha256",
    "checkpoints",
    "complete",
    "contract_sha256",
    "dataset_manifest_sha256",
    "datasets",
    "data_root",
    "expected_row_count",
    "feature_schema_sha256",
    "fit_dataset_manifest",
    "model_inputs_sha256",
    "role",
    "rows",
    "rows_sha256",
    "thresholds",
    "trainer_source_sha256",
    "summary",
    "upstream_manifest",
    "upstream_manifest_sha256",
    "validation_grid_schema_version",
    "validation_source_sha256",
}
DEVELOPMENT_RESULT_FIELDS = {
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
METRIC_FIELDS = {
    "division_fn",
    "division_fp",
    "division_jaccard",
    "division_tp",
    "edge_fn",
    "edge_fp",
    "edge_ignored",
    "edge_jaccard",
    "edge_tp",
    "matched_nodes",
    "predicted_nodes",
    "score",
    "truth_nodes",
}
TRIPLET_FIELDS = {
    "fn",
    "fp",
    "fully_matched_wrong_forks",
    "matched_division_false_positives",
    "tp",
    "unmatched_forks",
}


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


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return int(value)


def _require_finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite and numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _project_relative(path: str | Path, *, project_root: Path) -> str:
    source = Path(path)
    if source.is_absolute():
        resolved = source.resolve()
    else:
        resolved = (project_root / source).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("artifact path must remain inside the project root") from exc


def _require_logical_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{field} must be a nonempty project-relative path")
    if ".." in Path(value).parts:
        raise ValueError(f"{field} cannot escape the project root")
    return value


def _validate_artifact_record(record: Any, *, field: str) -> dict[str, str]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{field} artifact record is invalid")
    return {
        "path": _require_logical_path(record["path"], field=f"{field} path"),
        "sha256": _require_sha256(record["sha256"], field=f"{field} SHA-256"),
    }


def validation_source_sha256() -> str:
    return source_bundle_sha256(VALIDATION_SOURCE_PATHS)


def selected_row_from_grid_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one complete grid row into the strict development gate schema."""
    if not isinstance(row, Mapping) or set(row) != GRID_ROW_FIELDS:
        raise ValueError("validation grid row has incomplete or unexpected fields")
    return {name: row[name] for name in sorted(SELECTED_ROW_FIELDS)}


def _metric_payload(metric: MetricResult) -> dict[str, Any]:
    payload = asdict(metric)
    if set(payload) != METRIC_FIELDS:
        raise RuntimeError("metric payload disagrees with the frozen local metric")
    return payload


def _triplet_payload(evaluation: Any) -> dict[str, int]:
    payload = {
        "fn": evaluation.fn,
        "fp": evaluation.fp,
        "fully_matched_wrong_forks": evaluation.fully_matched_wrong_forks,
        "matched_division_false_positives": evaluation.matched_division_false_positives,
        "tp": evaluation.tp,
        "unmatched_forks": evaluation.unmatched_forks,
    }
    if set(payload) != TRIPLET_FIELDS:
        raise RuntimeError("strict triplet payload is incomplete")
    return payload


def _fork_payload(fork: AcceptedFork) -> dict[str, Any]:
    return {
        "candidate_1_id": fork.candidate_1_id,
        "candidate_2_id": fork.candidate_2_id,
        "edge_removal_count": fork.edge_removal_count,
        "logit": fork.logit,
        "mother_id": fork.mother_id,
        "probability": fork.probability,
        "restored_node_count": fork.restored_node_count,
    }


def validation_row(
    evaluation: MovieEvaluation,
    *,
    forks: Sequence[AcceptedFork],
    model_name: str,
    model_config: Mapping[str, Any],
    score_threshold: float,
    runtime_seconds: float,
    checkpoint_sha256: str,
    cache_metadata: Mapping[str, Any],
    cache_file_sha256: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize one deterministic policy/movie result plus diagnostic provenance."""
    row = {
        "accepted_fork_records": [_fork_payload(fork) for fork in forks],
        "accepted_forks": evaluation.accepted_forks,
        "added_edge_count": evaluation.added_edge_count,
        "baseline_edge_jaccard": evaluation.baseline_metric.edge_jaccard,
        "baseline_full_score": evaluation.baseline_metric.score,
        "baseline_metric": _metric_payload(evaluation.baseline_metric),
        "baseline_predicted_nodes": evaluation.baseline_metric.predicted_nodes,
        "baseline_triplets": _triplet_payload(evaluation.baseline_triplets),
        "base_checkpoint_sha256": identity["base_checkpoint_sha256"],
        "cache_content_sha256": cache_metadata["content_sha256"],
        "cache_file_sha256": cache_file_sha256,
        "cache_image_sha256": cache_metadata["image_sha256"],
        "candidate_edge_jaccard": evaluation.candidate_metric.edge_jaccard,
        "candidate_full_score": evaluation.candidate_metric.score,
        "candidate_graph_sha256": evaluation.candidate_graph_sha256,
        "candidate_metric": _metric_payload(evaluation.candidate_metric),
        "candidate_predicted_nodes": evaluation.candidate_metric.predicted_nodes,
        "candidate_triplets": _triplet_payload(evaluation.candidate_triplets),
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": dict(identity["contract_sha256"]),
        "dataset": evaluation.dataset,
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "edge_removal_count": evaluation.removed_edge_count,
        "estimated_nodes": evaluation.estimated_nodes,
        "fallback_graph_sha256": evaluation.fallback_graph_sha256,
        "feature_schema_sha256": identity["feature_schema_sha256"],
        "fully_matched_wrong_forks": (
            evaluation.candidate_triplets.fully_matched_wrong_forks
        ),
        "matched_division_false_positives": (
            evaluation.candidate_triplets.matched_division_false_positives
        ),
        "model_config": dict(model_config),
        "model_inputs_sha256": identity["model_inputs_sha256"],
        "model_name": model_name,
        "restored_node_count": evaluation.restored_node_count,
        "runtime_seconds": _require_finite(
            runtime_seconds,
            field="runtime_seconds",
            minimum=0.0,
        ),
        "score_threshold": score_threshold,
        "strict_true_recoveries": evaluation.strict_true_recoveries,
        "trainer_source_sha256": identity["trainer_source_sha256"],
        "truth_division_count": evaluation.truth_division_count,
        "truth_graph_sha256": evaluation.truth_graph_sha256,
        "validation_source_sha256": identity["validation_source_sha256"],
    }
    if set(row) != GRID_ROW_FIELDS:
        raise RuntimeError("validation row implementation is incomplete")
    return row


def _best_key(proposal: ScoredProposal) -> tuple[float, int, int, int, int]:
    return (
        -proposal.logit,
        proposal.edge_removal_count,
        proposal.restored_node_count,
        proposal.candidate_1_id,
        proposal.candidate_2_id,
    )


def retain_best_per_mother(
    proposals: Sequence[ScoredProposal],
) -> tuple[ScoredProposal, ...]:
    """Retain the exact best scored pair per mother without applying a threshold."""
    best: dict[int, ScoredProposal] = {}
    for proposal in proposals:
        if not isinstance(proposal, ScoredProposal):
            raise TypeError("best-pair retention requires ScoredProposal values")
        current = best.get(proposal.mother_id)
        if current is None or _best_key(proposal) < _best_key(current):
            best[proposal.mother_id] = proposal
    return tuple(best[mother] for mother in sorted(best))


def _expected_row_keys(identity: Mapping[str, Any]) -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (dataset, model_name, float(threshold))
        for dataset in identity["datasets"]
        for model_name in DIVISION_MODEL_NAMES
        for threshold in PROBABILITY_THRESHOLDS
    )


def _validate_grid_row(
    row: Any,
    *,
    identity: Mapping[str, Any],
    key: tuple[str, str, float],
) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != GRID_ROW_FIELDS:
        raise ValueError("validation grid row has incomplete or unexpected fields")
    dataset, model_name, threshold = key
    if (
        row["dataset"] != dataset
        or row["model_name"] != model_name
        or row["score_threshold"] != threshold
    ):
        raise ValueError("validation grid rows are not in the exact frozen order")
    checkpoint = identity["checkpoints"][model_name]
    repeated = {
        "base_checkpoint_sha256": identity["base_checkpoint_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "contract_sha256": identity["contract_sha256"],
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "feature_schema_sha256": identity["feature_schema_sha256"],
        "model_config": checkpoint["model_config"],
        "model_inputs_sha256": identity["model_inputs_sha256"],
        "trainer_source_sha256": identity["trainer_source_sha256"],
        "validation_source_sha256": identity["validation_source_sha256"],
    }
    if any(row[field] != value for field, value in repeated.items()):
        raise ValueError("validation grid row provenance disagrees with its grid identity")
    _require_finite(row["runtime_seconds"], field="runtime_seconds", minimum=0.0)
    for field in (
        "cache_content_sha256",
        "cache_file_sha256",
        "cache_image_sha256",
        "candidate_graph_sha256",
        "fallback_graph_sha256",
        "truth_graph_sha256",
    ):
        _require_sha256(row[field], field=field)
    if not isinstance(row["baseline_metric"], dict) or set(row["baseline_metric"]) != METRIC_FIELDS:
        raise ValueError("validation row baseline metric is incomplete")
    if (
        not isinstance(row["candidate_metric"], dict)
        or set(row["candidate_metric"]) != METRIC_FIELDS
    ):
        raise ValueError("validation row candidate metric is incomplete")
    if (
        not isinstance(row["baseline_triplets"], dict)
        or set(row["baseline_triplets"]) != TRIPLET_FIELDS
    ):
        raise ValueError("validation row baseline triplets are incomplete")
    if (
        not isinstance(row["candidate_triplets"], dict)
        or set(row["candidate_triplets"]) != TRIPLET_FIELDS
    ):
        raise ValueError("validation row candidate triplets are incomplete")
    selected = selected_row_from_grid_row(row)
    if (
        row["baseline_metric"]["score"] != selected["baseline_full_score"]
        or row["baseline_metric"]["edge_jaccard"]
        != selected["baseline_edge_jaccard"]
        or row["baseline_metric"]["predicted_nodes"]
        != selected["baseline_predicted_nodes"]
        or row["candidate_metric"]["score"] != selected["candidate_full_score"]
        or row["candidate_metric"]["edge_jaccard"]
        != selected["candidate_edge_jaccard"]
        or row["candidate_metric"]["predicted_nodes"]
        != selected["candidate_predicted_nodes"]
    ):
        raise ValueError("validation row metric detail disagrees with selected metrics")
    baseline_triplets = row["baseline_triplets"]
    candidate_triplets = row["candidate_triplets"]
    if (
        baseline_triplets["tp"] != 0
        or baseline_triplets["fp"] != 0
        or baseline_triplets["tp"] + baseline_triplets["fn"]
        != selected["truth_division_count"]
        or candidate_triplets["tp"] + candidate_triplets["fn"]
        != selected["truth_division_count"]
        or candidate_triplets["tp"] != selected["strict_true_recoveries"]
        or candidate_triplets["fully_matched_wrong_forks"]
        != selected["fully_matched_wrong_forks"]
        or candidate_triplets["matched_division_false_positives"]
        != selected["matched_division_false_positives"]
    ):
        raise ValueError("validation row strict triplet detail is inconsistent")
    fork_records = row["accepted_fork_records"]
    if (
        not isinstance(fork_records, list)
        or len(fork_records) != selected["accepted_forks"]
    ):
        raise ValueError("validation row accepted fork records are incomplete")
    fork_fields = {
        "candidate_1_id",
        "candidate_2_id",
        "edge_removal_count",
        "logit",
        "mother_id",
        "probability",
        "restored_node_count",
    }
    for fork in fork_records:
        if not isinstance(fork, dict) or set(fork) != fork_fields:
            raise ValueError("validation row accepted fork record is invalid")
        for field in ("mother_id", "candidate_1_id", "candidate_2_id"):
            _require_int(fork[field], field=f"accepted fork {field}")
        if not fork["candidate_1_id"] < fork["candidate_2_id"]:
            raise ValueError("validation row accepted fork daughters are not canonical")
        _require_int(fork["edge_removal_count"], field="accepted fork removals")
        _require_int(fork["restored_node_count"], field="accepted fork restorations")
        logit = _require_finite(fork["logit"], field="accepted fork logit")
        probability = _require_finite(
            fork["probability"],
            field="accepted fork probability",
            minimum=0.0,
        )
        if logit >= 0.0:
            expected_probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            expected_probability = exponential / (1.0 + exponential)
        if probability > 1.0 or probability != expected_probability:
            raise ValueError("validation row accepted fork probability is inconsistent")
    if len({fork["mother_id"] for fork in fork_records}) != len(fork_records):
        raise ValueError("validation row accepted fork records reuse a mother")
    return row


def _validate_grid_identity(identity: Mapping[str, Any]) -> None:
    datasets = identity.get("datasets")
    if (
        not isinstance(datasets, list)
        or len(datasets) != EXPECTED_DEVELOPMENT_MOVIES
        or datasets != sorted(set(datasets))
        or any(
            not isinstance(dataset, str) or not dataset.startswith("6bba_")
            for dataset in datasets
        )
    ):
        raise ValueError("validation grid requires 20 unique sorted 6bba development movies")
    if identity.get("thresholds") != list(PROBABILITY_THRESHOLDS):
        raise ValueError("validation grid threshold sequence is not frozen")
    if identity.get("role") != "development":
        raise ValueError("validation grid must use only development")
    contracts = identity.get("contract_sha256")
    if (
        not isinstance(contracts, dict)
        or set(contracts) != set(EXPECTED_DIVISION_CONTRACT_SHA256)
        or any(not _is_sha256(value) for value in contracts.values())
        or identity.get("assignment_sha256") != contracts["assignment"]
    ):
        raise ValueError("validation grid contract identity is incomplete")
    for field in (
        "base_checkpoint_sha256",
        "cache_source_sha256",
        "dataset_manifest_sha256",
        "feature_schema_sha256",
        "model_inputs_sha256",
        "trainer_source_sha256",
        "upstream_manifest_sha256",
        "validation_source_sha256",
    ):
        _require_sha256(identity.get(field), field=f"validation grid {field}")
    for field in ("cache_directory", "data_root"):
        _require_logical_path(identity.get(field), field=f"validation grid {field}")
    base = _validate_artifact_record(
        identity.get("base_checkpoint"),
        field="base checkpoint",
    )
    upstream = _validate_artifact_record(
        identity.get("upstream_manifest"),
        field="upstream manifest",
    )
    _validate_artifact_record(identity.get("summary"), field="manifest summary")
    _validate_artifact_record(
        identity.get("fit_dataset_manifest"),
        field="fit dataset manifest",
    )
    if (
        base["sha256"] != identity["base_checkpoint_sha256"]
        or upstream["sha256"] != identity["upstream_manifest_sha256"]
    ):
        raise ValueError("validation grid artifact hashes disagree with their identities")
    checkpoints = identity.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(DIVISION_MODEL_NAMES):
        raise ValueError("validation grid requires the exact three checkpoint records")
    for model_name in DIVISION_MODEL_NAMES:
        record = checkpoints[model_name]
        if not isinstance(record, dict) or set(record) != {
            "model_config",
            "path",
            "sha256",
        }:
            raise ValueError("validation grid checkpoint record is invalid")
        _require_logical_path(record["path"], field=f"{model_name} checkpoint path")
        _require_sha256(record["sha256"], field=f"{model_name} checkpoint SHA-256")
        if record["model_config"] != asdict(FIXED_DIVISION_MODEL_CONFIGS[model_name]):
            raise ValueError("validation grid checkpoint model config is not frozen")


def grid_payload(identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build one canonical partial or complete grid payload."""
    materialized = [dict(row) for row in rows]
    _validate_grid_identity(identity)
    expected_keys = _expected_row_keys(identity)
    if len(materialized) > len(expected_keys):
        raise ValueError("validation grid contains too many rows")
    for row, key in zip(materialized, expected_keys, strict=False):
        _validate_grid_row(row, identity=identity, key=key)
    rows_per_movie = len(DIVISION_MODEL_NAMES) * len(PROBABILITY_THRESHOLDS)
    if len(materialized) % rows_per_movie:
        raise ValueError("resumable validation grids must end on a complete movie boundary")
    complete = len(materialized) == EXPECTED_GRID_ROWS
    payload = {
        **{name: identity[name] for name in GRID_FIELDS if name in identity},
        "complete": complete,
        "expected_row_count": EXPECTED_GRID_ROWS,
        "rows": materialized,
        "rows_sha256": canonical_json_sha256(materialized),
        "validation_grid_schema_version": VALIDATION_GRID_SCHEMA_VERSION,
    }
    if set(payload) != GRID_FIELDS:
        missing = sorted(GRID_FIELDS - set(payload))
        raise ValueError("validation grid identity is incomplete: " + ", ".join(missing))
    return payload


def read_validation_grid(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    require_complete: bool,
    verify_current_sources: bool = False,
) -> dict[str, Any]:
    """Read and structurally verify one canonical resumable development grid."""
    payload = read_canonical_json(path)
    if not isinstance(payload, dict) or set(payload) != GRID_FIELDS:
        raise ValueError("validation grid has incomplete or unexpected fields")
    if payload["validation_grid_schema_version"] != VALIDATION_GRID_SCHEMA_VERSION:
        raise ValueError("unsupported validation grid schema")
    if payload["role"] != "development":
        raise ValueError("validation grid must use only development")
    if payload["expected_row_count"] != EXPECTED_GRID_ROWS:
        raise ValueError("validation grid expected row count is not frozen")
    dynamic_fields = {
        "complete",
        "expected_row_count",
        "rows",
        "rows_sha256",
        "validation_grid_schema_version",
    }
    identity = {name: payload[name] for name in GRID_FIELDS - dynamic_fields}
    _validate_grid_identity(identity)
    if verify_current_sources and identity["validation_source_sha256"] != (
        validation_source_sha256()
    ):
        raise ValueError("validation grid source code has changed")
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError("validation grid identity has changed")
    rows = payload["rows"]
    if not isinstance(rows, list) or payload["rows_sha256"] != canonical_json_sha256(rows):
        raise ValueError("validation grid row hash does not match its rows")
    rebuilt = grid_payload(identity, rows)
    if rebuilt != payload:
        raise ValueError("validation grid completion state or identity is inconsistent")
    if require_complete and not payload["complete"]:
        raise ValueError("validation grid is incomplete")
    return payload


def _artifact_record(path: str | Path, *, project_root: Path) -> dict[str, str]:
    source = Path(path)
    return {
        "path": _project_relative(source, project_root=project_root),
        "sha256": file_sha256(source),
    }


def _resolve_artifact(record: Any, *, project_root: Path, field: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{field} artifact record is invalid")
    logical_path = record["path"]
    if not isinstance(logical_path, str) or Path(logical_path).is_absolute():
        raise ValueError(f"{field} artifact path must be project-relative")
    expected_hash = _require_sha256(record["sha256"], field=f"{field} SHA-256")
    path = (project_root / logical_path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} artifact path escapes the project root") from exc
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"{field} artifact is missing or has changed")
    return path


def _load_validation_identity(
    *,
    project_root: Path,
    data_root: Path,
    cache_dir: Path,
    fit_dataset: Path,
    checkpoint_dir: Path,
    base_checkpoint: Path,
    config_dir: Path,
    upstream_manifest: Path,
    summary: Path,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    LoadedDivisionDataset,
    dict[str, DivisionModel],
    tuple[DivisionManifestRow, ...],
    dict[str, Mapping[str, Any]],
]:
    verified = verify_division_contract_bundle(
        config_dir,
        manifest_path=upstream_manifest,
        summary_path=summary,
        project_root=project_root,
        checkpoint_path=base_checkpoint,
    )
    manifest_rows = read_division_manifest(upstream_manifest)
    development = tuple(verified["split"]["role_datasets"]["development"])
    if len(development) != EXPECTED_DEVELOPMENT_MOVIES:
        raise ValueError("frozen split does not contain exactly 20 development movies")
    loaded = read_division_dataset(
        fit_dataset,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest,
        summary_path=summary,
        cache_dir=cache_dir,
        checkpoint_path=base_checkpoint,
    )
    trainer_hash = trainer_source_sha256()
    checkpoint_records: dict[str, dict[str, Any]] = {}
    models: dict[str, DivisionModel] = {}
    checkpoint_metadata: dict[str, Mapping[str, Any]] = {}
    for model_name in DIVISION_MODEL_NAMES:
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        model, metadata = load_division_checkpoint(
            checkpoint_path,
            base_checkpoint_path=base_checkpoint,
            config_dir=config_dir,
            feature_schema=loaded.feature_schema,
            expected_dataset_manifest_sha256=loaded.dataset_manifest_sha256,
            expected_model_inputs_sha256=loaded.model_inputs_sha256,
            expected_trainer_source_sha256=trainer_hash,
            expected_model_config=FIXED_DIVISION_MODEL_CONFIGS[model_name],
            device=device,
        )
        if metadata["epoch"] != 100:
            raise ValueError(f"{model_name} checkpoint is not exact production epoch 100")
        checkpoint_records[model_name] = {
            **_artifact_record(checkpoint_path, project_root=project_root),
            "model_config": asdict(FIXED_DIVISION_MODEL_CONFIGS[model_name]),
        }
        models[model_name] = model
        checkpoint_metadata[model_name] = metadata
    base_record = _artifact_record(base_checkpoint, project_root=project_root)
    upstream_record = _artifact_record(upstream_manifest, project_root=project_root)
    identity = {
        "assignment_sha256": current_division_contract_hashes(config_dir)["assignment"],
        "base_checkpoint": base_record,
        "base_checkpoint_sha256": base_record["sha256"],
        "cache_directory": _project_relative(cache_dir, project_root=project_root),
        "cache_source_sha256": division_cache_source_sha256(project_root),
        "checkpoints": checkpoint_records,
        "contract_sha256": current_division_contract_hashes(config_dir),
        "dataset_manifest_sha256": loaded.dataset_manifest_sha256,
        "datasets": list(development),
        "data_root": _project_relative(data_root, project_root=project_root),
        "feature_schema_sha256": loaded.feature_schema.sha256,
        "fit_dataset_manifest": _artifact_record(
            fit_dataset / "manifest.json",
            project_root=project_root,
        ),
        "model_inputs_sha256": loaded.model_inputs_sha256,
        "role": "development",
        "summary": _artifact_record(summary, project_root=project_root),
        "thresholds": list(PROBABILITY_THRESHOLDS),
        "trainer_source_sha256": trainer_hash,
        "upstream_manifest": upstream_record,
        "upstream_manifest_sha256": upstream_record["sha256"],
        "validation_source_sha256": validation_source_sha256(),
    }
    if set(identity) != GRID_FIELDS - {
        "complete",
        "expected_row_count",
        "rows",
        "rows_sha256",
        "validation_grid_schema_version",
    }:
        raise RuntimeError("validation grid identity implementation is incomplete")
    return identity, loaded, models, manifest_rows, checkpoint_metadata


def _estimated_node_count(labels: Path) -> int:
    try:
        metadata = json.loads((labels / "zarr.json").read_text())
        count = metadata["attributes"]["geff"]["extra"]["estimated_number_of_nodes"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid estimated-node metadata for {labels}") from exc
    return _require_int(count, field="estimated node count", minimum=1)


def _load_inference_movie(
    dataset: str,
    *,
    role: str,
    cache_dir: Path,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
    proposal_contract: Mapping[str, Any],
) -> tuple[InferenceMovie, dict[str, Any], str]:
    cache_path = cache_dir / f"{dataset}.npz"
    cache_file_hash = file_sha256(cache_path)
    raw, scores, embeddings, intensity, metadata = read_division_node_cache(
        cache_path,
        expected_dataset=dataset,
        expected_role=role,
        expected_cache_source_sha256=cache_source_sha256,
        expected_contract_sha256=contract_sha256,
        expected_upstream_manifest_sha256=upstream_manifest_sha256,
    )
    if file_sha256(cache_path) != cache_file_hash:
        raise ValueError(f"division node cache changed while reading {dataset}")
    movie = prepare_inference_movie(
        dataset,
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        proposal_contract=proposal_contract,
        source_role=role,
    )
    if truth_divisions(movie.fallback):
        raise ValueError(f"divisions-off fallback unexpectedly contains a fork for {dataset}")
    return movie, metadata, cache_file_hash


def _load_truth(
    dataset: str,
    *,
    sample_by_dataset: Mapping[str, Any],
    manifest_by_dataset: Mapping[str, DivisionManifestRow],
) -> tuple[TrackingGraph, int]:
    sample = sample_by_dataset.get(dataset)
    if sample is None or sample.labels is None:
        raise FileNotFoundError(f"development labels are missing for {dataset}")
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    observed_hash = logical_graph_sha256(truth)
    expected_hash = manifest_by_dataset[dataset].truth_graph_sha256
    if observed_hash != expected_hash:
        raise ValueError(f"development truth graph changed for {dataset}")
    return truth, _estimated_node_count(sample.labels)


def score_movie_once_per_model(
    movie: InferenceMovie,
    *,
    models: Mapping[str, DivisionModel],
    feature_schema: Any,
    proposal_contract: Mapping[str, Any],
    device: torch.device,
    max_mothers_per_batch: int,
    require_full_ladder: bool = True,
) -> tuple[dict[str, tuple[ScoredProposal, ...]], dict[str, float]]:
    """Score every pair once per model and retain only each mother's best pair."""
    if require_full_ladder and set(models) != set(DIVISION_MODEL_NAMES):
        raise ValueError("development validation requires the exact three-model ladder")
    model_names = tuple(name for name in DIVISION_MODEL_NAMES if name in models)
    if not model_names or set(model_names) != set(models):
        raise ValueError("model scoring received an unsupported model selection")
    fallback_hash = logical_graph_sha256(movie.fallback)
    best: dict[str, dict[int, ScoredProposal]] = {
        model_name: {} for model_name in model_names
    }
    runtimes = {model_name: 0.0 for model_name in model_names}
    for batch in iter_inference_proposal_batches(
        movie,
        feature_schema=feature_schema,
        proposal_contract=proposal_contract,
        max_mothers_per_batch=max_mothers_per_batch,
    ):
        image = torch.tensor(batch.image_features, dtype=torch.float32, device=device)
        context = torch.tensor(batch.normalized_context, dtype=torch.float32, device=device)
        for model_name in model_names:
            started = time.perf_counter()
            with torch.inference_mode():
                logits = (
                    models[model_name]
                    .forward_cached_features(image, context)
                    .detach()
                    .cpu()
                    .numpy()
                )
            runtimes[model_name] += time.perf_counter() - started
            retained = best[model_name]
            for index, (proposal, logit) in enumerate(
                zip(batch.proposals, logits, strict=True)
            ):
                scored = ScoredProposal(
                    mother_id=proposal.mother_id,
                    candidate_1_id=proposal.candidate_1.node_id,
                    candidate_2_id=proposal.candidate_2.node_id,
                    logit=float(logit),
                    edge_removal_count=int(batch.edge_removal_counts[index]),
                    restored_node_count=int(batch.restored_node_counts[index]),
                    fallback_graph_sha256=fallback_hash,
                )
                current = retained.get(scored.mother_id)
                if current is None or _best_key(scored) < _best_key(current):
                    retained[scored.mother_id] = scored
    if logical_graph_sha256(movie.fallback) != fallback_hash:
        raise RuntimeError("fallback graph changed during model scoring")
    results = {
        model_name: tuple(best[model_name][mother] for mother in sorted(best[model_name]))
        for model_name in model_names
    }
    return results, runtimes


def _evaluate_movie_grid(
    movie: InferenceMovie,
    truth: TrackingGraph,
    *,
    estimated_nodes: int,
    scored_by_model: Mapping[str, Sequence[ScoredProposal]],
    scoring_seconds: Mapping[str, float],
    cache_metadata: Mapping[str, Any],
    cache_file_sha256: str,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluation_cache: dict[tuple[tuple[int, int, int], ...], MovieEvaluation] = {}
    for model_name in DIVISION_MODEL_NAMES:
        checkpoint = identity["checkpoints"][model_name]
        for threshold in PROBABILITY_THRESHOLDS:
            started = time.perf_counter()
            forks = select_division_forks(
                scored_by_model[model_name],
                probability_threshold=threshold,
            )
            graph_key = tuple(
                (fork.mother_id, fork.candidate_1_id, fork.candidate_2_id)
                for fork in forks
            )
            evaluation = evaluation_cache.get(graph_key)
            if evaluation is None:
                mutation = apply_division_forks(movie.fallback, movie.raw, forks)
                if mutation.rolled_back:
                    raise RuntimeError(
                        f"division graph transaction failed for {movie.dataset}: "
                        f"{mutation.diagnostic}"
                    )
                evaluation = evaluate_division_movie(
                    movie.dataset,
                    movie.fallback,
                    mutation,
                    truth,
                    estimated_nodes=estimated_nodes,
                )
                evaluation_cache[graph_key] = evaluation
            runtime = (
                scoring_seconds[model_name] / len(PROBABILITY_THRESHOLDS)
                + time.perf_counter()
                - started
            )
            rows.append(
                validation_row(
                    evaluation,
                    forks=forks,
                    model_name=model_name,
                    model_config=checkpoint["model_config"],
                    score_threshold=threshold,
                    runtime_seconds=runtime,
                    checkpoint_sha256=checkpoint["sha256"],
                    cache_metadata=cache_metadata,
                    cache_file_sha256=cache_file_sha256,
                    identity=identity,
                )
            )
    return rows


def _validate_existing_grid_artifacts(
    grid: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    _resolve_artifact(grid["base_checkpoint"], project_root=project_root, field="base checkpoint")
    _resolve_artifact(
        grid["fit_dataset_manifest"],
        project_root=project_root,
        field="fit dataset manifest",
    )
    _resolve_artifact(grid["summary"], project_root=project_root, field="manifest summary")
    _resolve_artifact(
        grid["upstream_manifest"],
        project_root=project_root,
        field="upstream manifest",
    )
    for model_name in DIVISION_MODEL_NAMES:
        record = grid["checkpoints"][model_name]
        _resolve_artifact(
            {"path": record["path"], "sha256": record["sha256"]},
            project_root=project_root,
            field=f"{model_name} checkpoint",
        )
    cache_dir = project_root / grid["cache_directory"]
    by_dataset: dict[str, str] = {}
    for row in grid["rows"]:
        dataset = row["dataset"]
        observed = by_dataset.setdefault(dataset, row["cache_file_sha256"])
        if observed != row["cache_file_sha256"]:
            raise ValueError(f"resumed rows disagree on the cache hash for {dataset}")
    for dataset, expected_hash in by_dataset.items():
        if file_sha256(cache_dir / f"{dataset}.npz") != expected_hash:
            raise ValueError(f"resumed development cache changed for {dataset}")


def run_development_grid(
    *,
    project_root: Path = ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fit_dataset: Path = DEFAULT_FIT_DATASET,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    upstream_manifest: Path = DEFAULT_UPSTREAM_MANIFEST,
    summary: Path = DEFAULT_SUMMARY,
    output: Path = DEFAULT_OUTPUT,
    device: str = "cpu",
    max_mothers_per_batch: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run or resume the complete 3 by 11 by 20 development grid."""
    target_device = torch.device(device)
    identity, loaded, models, manifest_rows, _ = _load_validation_identity(
        project_root=project_root,
        data_root=data_root,
        cache_dir=cache_dir,
        fit_dataset=fit_dataset,
        checkpoint_dir=checkpoint_dir,
        base_checkpoint=base_checkpoint,
        config_dir=config_dir,
        upstream_manifest=upstream_manifest,
        summary=summary,
        device=target_device,
    )
    rows: list[dict[str, Any]] = []
    if output.exists() and not overwrite:
        existing = read_validation_grid(
            output,
            expected_identity=identity,
            require_complete=False,
        )
        _validate_existing_grid_artifacts(existing, project_root=project_root)
        rows = list(existing["rows"])
        if existing["complete"]:
            return existing
    contracts = read_division_contracts(config_dir)
    manifest_by_dataset = {row.dataset: row for row in manifest_rows}
    samples = discover_split(data_root, "train")
    sample_by_dataset = {sample.dataset: sample for sample in samples}
    rows_per_movie = len(DIVISION_MODEL_NAMES) * len(PROBABILITY_THRESHOLDS)
    completed_movies = len(rows) // rows_per_movie
    remaining = identity["datasets"][completed_movies:]
    for index, dataset in enumerate(remaining, start=completed_movies):
        movie, cache_metadata, cache_file_hash = _load_inference_movie(
            dataset,
            role="development",
            cache_dir=cache_dir,
            cache_source_sha256=identity["cache_source_sha256"],
            contract_sha256=identity["contract_sha256"],
            upstream_manifest_sha256=identity["upstream_manifest_sha256"],
            proposal_contract=contracts["proposal"]["proposal_rule"],
        )
        fallback_hash = logical_graph_sha256(movie.fallback)
        truth, estimated_nodes = _load_truth(
            dataset,
            sample_by_dataset=sample_by_dataset,
            manifest_by_dataset=manifest_by_dataset,
        )
        scored, scoring_seconds = score_movie_once_per_model(
            movie,
            models=models,
            feature_schema=loaded.feature_schema,
            proposal_contract=contracts["proposal"]["proposal_rule"],
            device=target_device,
            max_mothers_per_batch=max_mothers_per_batch,
        )
        if logical_graph_sha256(movie.fallback) != fallback_hash:
            raise RuntimeError(f"fallback identity changed before evaluation for {dataset}")
        movie_rows = _evaluate_movie_grid(
            movie,
            truth,
            estimated_nodes=estimated_nodes,
            scored_by_model=scored,
            scoring_seconds=scoring_seconds,
            cache_metadata=cache_metadata,
            cache_file_sha256=cache_file_hash,
            identity=identity,
        )
        if len(movie_rows) != rows_per_movie:
            raise RuntimeError("development movie did not produce its exact policy grid")
        rows.extend(movie_rows)
        payload = grid_payload(identity, rows)
        write_canonical_json(output, payload)
        print(
            f"[{index + 1}/{len(identity['datasets'])}] {dataset} "
            f"rows={len(rows)}/{EXPECTED_GRID_ROWS} fallback_sha256={fallback_hash}",
            flush=True,
        )
    return read_validation_grid(
        output,
        expected_identity=identity,
        require_complete=True,
    )


def _selected_evaluation_row(evaluation: MovieEvaluation) -> dict[str, Any]:
    return {
        "accepted_forks": evaluation.accepted_forks,
        "added_edge_count": evaluation.added_edge_count,
        "baseline_edge_jaccard": evaluation.baseline_metric.edge_jaccard,
        "baseline_full_score": evaluation.baseline_metric.score,
        "baseline_predicted_nodes": evaluation.baseline_metric.predicted_nodes,
        "candidate_edge_jaccard": evaluation.candidate_metric.edge_jaccard,
        "candidate_full_score": evaluation.candidate_metric.score,
        "candidate_graph_sha256": evaluation.candidate_graph_sha256,
        "candidate_predicted_nodes": evaluation.candidate_metric.predicted_nodes,
        "dataset": evaluation.dataset,
        "edge_removal_count": evaluation.removed_edge_count,
        "estimated_nodes": evaluation.estimated_nodes,
        "fallback_graph_sha256": evaluation.fallback_graph_sha256,
        "fully_matched_wrong_forks": (
            evaluation.candidate_triplets.fully_matched_wrong_forks
        ),
        "matched_division_false_positives": (
            evaluation.candidate_triplets.matched_division_false_positives
        ),
        "restored_node_count": evaluation.restored_node_count,
        "strict_true_recoveries": evaluation.strict_true_recoveries,
        "truth_division_count": evaluation.truth_division_count,
        "truth_graph_sha256": evaluation.truth_graph_sha256,
    }


def replay_selected_policy(
    development_result: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fit_dataset: Path = DEFAULT_FIT_DATASET,
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    upstream_manifest: Path = DEFAULT_UPSTREAM_MANIFEST,
    summary: Path = DEFAULT_SUMMARY,
    device: str = "cpu",
    max_mothers_per_batch: int = 128,
    checkpoint_snapshot: bytes | None = None,
) -> tuple[dict[str, Any], ...]:
    """Recompute every selected development row from caches, checkpoint, and truth."""
    if not isinstance(development_result, Mapping) or set(development_result) != (
        DEVELOPMENT_RESULT_FIELDS
    ):
        raise ValueError("development result has incomplete or unexpected fields")
    if development_result["development_result_schema_version"] != (
        DEVELOPMENT_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported development result schema")
    if development_result["role"] != "development":
        raise ValueError("selected policy replay accepts only development rows")
    rows = development_result["rows"]
    if not isinstance(rows, list) or development_result["rows_sha256"] != (
        canonical_json_sha256(rows)
    ):
        raise ValueError("development result row hash does not match its rows")
    if any(not isinstance(row, dict) or set(row) != SELECTED_ROW_FIELDS for row in rows):
        raise ValueError("development result contains an invalid selected row")
    verified = verify_division_contract_bundle(
        config_dir,
        manifest_path=upstream_manifest,
        summary_path=summary,
        project_root=project_root,
        checkpoint_path=base_checkpoint,
    )
    contracts = read_division_contracts(config_dir)
    contract_hashes = current_division_contract_hashes(config_dir)
    datasets = tuple(verified["split"]["role_datasets"]["development"])
    if tuple(row["dataset"] for row in rows) != datasets:
        raise ValueError("development result does not contain the exact frozen cohort")
    loaded = read_division_dataset(
        fit_dataset,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest,
        summary_path=summary,
        cache_dir=cache_dir,
        checkpoint_path=base_checkpoint,
    )
    trainer_hash = trainer_source_sha256()
    if (
        development_result["base_checkpoint_sha256"] != file_sha256(base_checkpoint)
        or development_result["contract_sha256"] != contract_hashes
        or development_result["assignment_sha256"] != contract_hashes["assignment"]
        or development_result["dataset_manifest_sha256"]
        != loaded.dataset_manifest_sha256
        or development_result["feature_schema_sha256"] != loaded.feature_schema.sha256
        or development_result["model_inputs_sha256"] != loaded.model_inputs_sha256
        or development_result["trainer_source_sha256"] != trainer_hash
    ):
        raise ValueError("development result identity does not match current fit artifacts")
    model_config = development_result["model_config"]
    model_name = model_config.get("model_name") if isinstance(model_config, Mapping) else None
    if model_name not in DIVISION_MODEL_NAMES or dict(model_config) != asdict(
        FIXED_DIVISION_MODEL_CONFIGS[model_name]
    ):
        raise ValueError("development result model config is not in the fixed ladder")
    threshold = development_result["score_threshold"]
    if threshold not in PROBABILITY_THRESHOLDS:
        raise ValueError("development result threshold is outside the frozen grid")
    checkpoint_path = _resolve_artifact(
        development_result["division_checkpoint"],
        project_root=project_root,
        field="division checkpoint",
    )
    load_kwargs = {
        "base_checkpoint_path": base_checkpoint,
        "config_dir": config_dir,
        "feature_schema": loaded.feature_schema,
        "expected_dataset_manifest_sha256": loaded.dataset_manifest_sha256,
        "expected_model_inputs_sha256": loaded.model_inputs_sha256,
        "expected_trainer_source_sha256": trainer_hash,
        "expected_model_config": FIXED_DIVISION_MODEL_CONFIGS[model_name],
        "device": device,
    }
    if checkpoint_snapshot is None:
        model, checkpoint_metadata = load_division_checkpoint(
            checkpoint_path,
            **load_kwargs,
        )
    else:
        if not isinstance(checkpoint_snapshot, bytes) or hashlib.sha256(
            checkpoint_snapshot
        ).hexdigest() != development_result["division_checkpoint"]["sha256"]:
            raise ValueError("selected checkpoint snapshot does not match its frozen hash")
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            handle.write(checkpoint_snapshot)
            handle.flush()
            model, checkpoint_metadata = load_division_checkpoint(
                handle.name,
                **load_kwargs,
            )
    if checkpoint_metadata["epoch"] != 100:
        raise ValueError("selected division checkpoint is not production epoch 100")
    manifest_rows = read_division_manifest(upstream_manifest)
    manifest_by_dataset = {row.dataset: row for row in manifest_rows}
    sample_by_dataset = {
        sample.dataset: sample for sample in discover_split(data_root, "train")
    }
    cache_source_hash = division_cache_source_sha256(project_root)
    upstream_hash = file_sha256(upstream_manifest)
    target_device = torch.device(device)
    observed_rows: list[dict[str, Any]] = []
    for expected_row in rows:
        dataset = expected_row["dataset"]
        movie, _, _ = _load_inference_movie(
            dataset,
            role="development",
            cache_dir=cache_dir,
            cache_source_sha256=cache_source_hash,
            contract_sha256=contract_hashes,
            upstream_manifest_sha256=upstream_hash,
            proposal_contract=contracts["proposal"]["proposal_rule"],
        )
        truth, estimated_nodes = _load_truth(
            dataset,
            sample_by_dataset=sample_by_dataset,
            manifest_by_dataset=manifest_by_dataset,
        )
        scored, _ = score_movie_once_per_model(
            movie,
            models={model_name: model},
            feature_schema=loaded.feature_schema,
            proposal_contract=contracts["proposal"]["proposal_rule"],
            device=target_device,
            max_mothers_per_batch=max_mothers_per_batch,
            require_full_ladder=False,
        )
        forks = select_division_forks(
            scored[model_name],
            probability_threshold=float(threshold),
        )
        mutation = apply_division_forks(movie.fallback, movie.raw, forks)
        if mutation.rolled_back:
            raise RuntimeError(
                f"selected policy replay transaction failed for {dataset}: "
                f"{mutation.diagnostic}"
            )
        evaluation = evaluate_division_movie(
            dataset,
            movie.fallback,
            mutation,
            truth,
            estimated_nodes=estimated_nodes,
        )
        observed = _selected_evaluation_row(evaluation)
        if observed != expected_row:
            raise ValueError(f"development result replay disagrees for {dataset}")
        observed_rows.append(observed)
    return tuple(observed_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("development",), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fit-dataset", type=Path, default=DEFAULT_FIT_DATASET)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--upstream-manifest", type=Path, default=DEFAULT_UPSTREAM_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-mothers-per-batch", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_mothers_per_batch < 1:
        parser.error("--max-mothers-per-batch must be positive")
    try:
        result = run_development_grid(
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            fit_dataset=args.fit_dataset,
            checkpoint_dir=args.checkpoint_dir,
            base_checkpoint=args.base_checkpoint,
            config_dir=args.config_dir,
            upstream_manifest=args.upstream_manifest,
            summary=args.summary,
            output=args.output,
            device=args.device,
            max_mothers_per_batch=args.max_mothers_per_batch,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"complete development grid rows={len(result['rows'])} "
        f"rows_sha256={result['rows_sha256']} file_sha256={file_sha256(args.output)} "
        f"path={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
