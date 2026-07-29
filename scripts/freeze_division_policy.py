#!/usr/bin/env python3
"""Freeze the best guard-eligible Phase 11 development policy or a rejection."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    read_division_contracts,
    read_division_manifest,
    source_bundle_sha256,
    write_canonical_json,
)
from biohub.division_model import DIVISION_MODEL_NAMES  # noqa: E402
from scripts.build_division_cache import (  # noqa: E402
    DEVELOPMENT_RESULT_SCHEMA_VERSION,
    EVALUATION_UNLOCK_POLICY_SCHEMA_VERSION,
    recompute_development_guards,
)
from scripts.validate_division_model import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_GRID,
)
from scripts.validate_division_model import (  # noqa: E402
    PROBABILITY_THRESHOLDS,
    read_validation_grid,
    selected_row_from_grid_row,
)

DEFAULT_RESULT = ROOT / "data" / "working" / "phase11_division_development_result.json"
DEFAULT_POLICY = ROOT / "data" / "working" / "phase11_division_policy.json"
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_MANIFEST = ROOT / "data" / "working" / "phase11_division_manifest.csv"
POLICY_SOURCE_PATHS = (
    SRC / "biohub" / "division_data.py",
    SRC / "biohub" / "division_inference.py",
    Path(__file__).resolve(),
    ROOT / "scripts" / "build_division_cache.py",
    ROOT / "scripts" / "validate_division_model.py",
)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    model_name: str
    score_threshold: float
    rows: tuple[dict[str, Any], ...]
    guard_results: Mapping[str, bool]
    guard_metrics: Mapping[str, float | int | bool]
    mean_full_score: float
    accepted_forks: int
    evaluation_unlocked: bool
    checkpoint: Mapping[str, Any]
    model_config: Mapping[str, Any]


def policy_source_sha256() -> str:
    return source_bundle_sha256(POLICY_SOURCE_PATHS)


def _selection_key(selection: PolicySelection) -> tuple[float, int, int, float, str]:
    return (
        -selection.mean_full_score,
        selection.accepted_forks,
        DIVISION_MODEL_NAMES.index(selection.model_name),
        -selection.score_threshold,
        str(selection.checkpoint["sha256"]),
    )


def select_grid_policy(
    grid: Mapping[str, Any],
    *,
    contracts: Mapping[str, Mapping[str, Any]],
    manifest_rows: Sequence[DivisionManifestRow],
) -> PolicySelection:
    """Select the exact best passing policy with the approved stable tie-breaks."""
    if grid.get("complete") is not True:
        raise ValueError("policy selection requires a complete validation grid")
    expected_datasets = tuple(contracts["split"]["role_datasets"]["development"])
    if tuple(grid.get("datasets", ())) != expected_datasets:
        raise ValueError("validation grid does not contain the exact development cohort")
    rows = grid.get("rows")
    if not isinstance(rows, list):
        raise ValueError("validation grid rows must be a list")
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_name"], float(row["score_threshold"]))
        grouped.setdefault(key, []).append(selected_row_from_grid_row(row))
    expected_keys = {
        (model_name, float(threshold))
        for model_name in DIVISION_MODEL_NAMES
        for threshold in PROBABILITY_THRESHOLDS
    }
    if set(grouped) != expected_keys:
        raise ValueError("validation grid does not contain the exact model-threshold grid")
    candidates: list[PolicySelection] = []
    for model_name in DIVISION_MODEL_NAMES:
        for threshold in PROBABILITY_THRESHOLDS:
            selected_rows = grouped[(model_name, threshold)]
            if tuple(row["dataset"] for row in selected_rows) != expected_datasets:
                raise ValueError("one policy does not contain the exact development cohort")
            guards, metrics = recompute_development_guards(
                selected_rows,
                contracts=contracts,
                manifest_rows=manifest_rows,
            )
            candidates.append(
                PolicySelection(
                    model_name=model_name,
                    score_threshold=threshold,
                    rows=tuple(selected_rows),
                    guard_results=guards,
                    guard_metrics=metrics,
                    mean_full_score=sum(
                        float(row["candidate_full_score"]) for row in selected_rows
                    )
                    / len(selected_rows),
                    accepted_forks=sum(int(row["accepted_forks"]) for row in selected_rows),
                    evaluation_unlocked=all(guards.values()),
                    checkpoint=grid["checkpoints"][model_name],
                    model_config=grid["checkpoints"][model_name]["model_config"],
                )
            )
    eligible = [candidate for candidate in candidates if candidate.evaluation_unlocked]
    pool = eligible if eligible else candidates
    selected = min(pool, key=_selection_key)
    if not eligible and selected.evaluation_unlocked:
        raise RuntimeError("rejection policy selection became internally inconsistent")
    return selected


def development_result_payload(
    grid: Mapping[str, Any],
    selection: PolicySelection,
) -> dict[str, Any]:
    rows = [dict(row) for row in selection.rows]
    checkpoint = {
        "path": selection.checkpoint["path"],
        "sha256": selection.checkpoint["sha256"],
    }
    return {
        "assignment_sha256": grid["assignment_sha256"],
        "base_checkpoint_sha256": grid["base_checkpoint_sha256"],
        "contract_sha256": grid["contract_sha256"],
        "dataset_manifest_sha256": grid["dataset_manifest_sha256"],
        "development_result_schema_version": DEVELOPMENT_RESULT_SCHEMA_VERSION,
        "division_checkpoint": checkpoint,
        "feature_schema_sha256": grid["feature_schema_sha256"],
        "model_config": dict(selection.model_config),
        "model_inputs_sha256": grid["model_inputs_sha256"],
        "role": "development",
        "rows": rows,
        "rows_sha256": canonical_json_sha256(rows),
        "score_threshold": selection.score_threshold,
        "trainer_source_sha256": grid["trainer_source_sha256"],
    }


def policy_payload(
    grid: Mapping[str, Any],
    selection: PolicySelection,
    *,
    grid_record: Mapping[str, str],
    result_record: Mapping[str, str],
    rows_sha256: str,
) -> dict[str, Any]:
    checkpoint = {
        "path": selection.checkpoint["path"],
        "sha256": selection.checkpoint["sha256"],
    }
    return {
        "assignment_sha256": grid["assignment_sha256"],
        "base_checkpoint_sha256": grid["base_checkpoint_sha256"],
        "contract_sha256": grid["contract_sha256"],
        "dataset_manifest_sha256": grid["dataset_manifest_sha256"],
        "development_grid": {
            "path": grid_record["path"],
            "rows_sha256": grid["rows_sha256"],
            "sha256": grid_record["sha256"],
        },
        "development_guard_results": dict(selection.guard_results),
        "development_result": {
            "path": result_record["path"],
            "rows_sha256": rows_sha256,
            "sha256": result_record["sha256"],
        },
        "division_checkpoint": checkpoint,
        "division_policy_schema_version": EVALUATION_UNLOCK_POLICY_SCHEMA_VERSION,
        "evaluation_unlocked": selection.evaluation_unlocked,
        "feature_schema_sha256": grid["feature_schema_sha256"],
        "model_config": dict(selection.model_config),
        "model_inputs_sha256": grid["model_inputs_sha256"],
        "policy_source_sha256": policy_source_sha256(),
        "score_threshold": selection.score_threshold,
        "selection_role": "development",
        "trainer_source_sha256": grid["trainer_source_sha256"],
    }


def _relative_artifact(path: Path, *, project_root: Path) -> dict[str, str]:
    try:
        logical_path = path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("policy artifacts must remain inside the project root") from exc
    return {"path": logical_path, "sha256": file_sha256(path)}


def freeze_policy(
    *,
    grid_path: Path = DEFAULT_GRID,
    result_path: Path = DEFAULT_RESULT,
    policy_path: Path = DEFAULT_POLICY,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    project_root: Path = ROOT,
    overwrite: bool = False,
) -> tuple[dict[str, Any], str]:
    """Verify the complete grid and atomically freeze one selected result and policy."""
    grid = read_validation_grid(
        grid_path,
        require_complete=True,
        verify_current_sources=True,
    )
    contracts = read_division_contracts(config_dir)
    manifest_rows = read_division_manifest(manifest_path)
    selection = select_grid_policy(
        grid,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    result = development_result_payload(grid, selection)
    if result_path.exists() and not overwrite:
        existing = read_division_result(result_path)
        if existing != result:
            raise FileExistsError("development result already exists with different content")
    else:
        write_canonical_json(result_path, result)
    grid_record = _relative_artifact(grid_path, project_root=project_root)
    result_record = _relative_artifact(result_path, project_root=project_root)
    policy = policy_payload(
        grid,
        selection,
        grid_record=grid_record,
        result_record=result_record,
        rows_sha256=result["rows_sha256"],
    )
    if policy_path.exists() and not overwrite:
        existing = read_policy(policy_path)
        if existing != policy:
            raise FileExistsError("division policy already exists with different content")
    else:
        write_canonical_json(policy_path, policy)
    return policy, file_sha256(policy_path)


def read_division_result(path: str | Path) -> dict[str, Any]:
    from biohub.division_data import read_canonical_json

    payload = read_canonical_json(path)
    expected = {
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
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("development result has incomplete or unexpected fields")
    return payload


def read_policy(path: str | Path) -> dict[str, Any]:
    from biohub.division_data import read_canonical_json

    payload = read_canonical_json(path)
    expected = {
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
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("division policy has incomplete or unexpected fields")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        policy, digest = freeze_policy(
            grid_path=args.grid,
            result_path=args.result,
            policy_path=args.policy,
            config_dir=args.config_dir,
            manifest_path=args.manifest,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"evaluation_unlocked={str(policy['evaluation_unlocked']).lower()} "
        f"model={policy['model_config']['model_name']} "
        f"threshold={policy['score_threshold']} policy_sha256={digest} "
        f"path={args.policy}",
        flush=True,
    )


if __name__ == "__main__":
    main()
