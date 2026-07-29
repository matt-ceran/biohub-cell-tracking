from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from biohub.division_data import (
    DivisionManifestRow,
    read_division_contracts,
    write_canonical_json,
)
from biohub.division_inference import PROBABILITY_THRESHOLDS, ScoredProposal
from biohub.division_model import (
    DIVISION_MODEL_NAMES,
    EXPECTED_BASE_CHECKPOINT_SHA256,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    FIXED_DIVISION_MODEL_CONFIGS,
)
from scripts.freeze_division_policy import select_grid_policy
from scripts.validate_division_model import (
    EXPECTED_GRID_ROWS,
    GRID_FIELDS,
    GRID_ROW_FIELDS,
    grid_payload,
    read_validation_grid,
    retain_best_per_mother,
)

ROOT = Path(__file__).resolve().parents[1]


def _manifest_rows() -> tuple[DivisionManifestRow, ...]:
    split = read_division_contracts(ROOT / "configs")["split"]
    exposed = set(split["forced_fit"]["selection_datasets"])
    rows = [
        DivisionManifestRow(
            dataset=dataset,
            embryo="6bba",
            division_count=int(division_count),
            no_division=int(division_count) == 0,
            proposal_exposed=dataset in exposed,
            role=role,
            truth_graph_sha256=hashlib.sha256(dataset.encode()).hexdigest(),
        )
        for division_count, roles in split["strata"].items()
        for role, datasets in roles.items()
        for dataset in datasets
    ]
    return tuple(sorted(rows, key=lambda row: row.dataset))


def _identity() -> dict[str, object]:
    contracts = read_division_contracts(ROOT / "configs")
    datasets = contracts["split"]["role_datasets"]["development"]
    checkpoints = {
        model_name: {
            "model_config": asdict(FIXED_DIVISION_MODEL_CONFIGS[model_name]),
            "path": f"data/working/checkpoints/{model_name}.pt",
            "sha256": hashlib.sha256(model_name.encode()).hexdigest(),
        }
        for model_name in DIVISION_MODEL_NAMES
    }
    identity = {
        "assignment_sha256": EXPECTED_DIVISION_CONTRACT_SHA256["assignment"],
        "base_checkpoint": {
            "path": "data/working/base.pt",
            "sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        },
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        "cache_directory": "data/working/cache",
        "cache_source_sha256": "1" * 64,
        "checkpoints": checkpoints,
        "contract_sha256": EXPECTED_DIVISION_CONTRACT_SHA256,
        "dataset_manifest_sha256": "2" * 64,
        "datasets": datasets,
        "data_root": "data/raw",
        "feature_schema_sha256": "3" * 64,
        "fit_dataset_manifest": {
            "path": "data/working/fit/manifest.json",
            "sha256": "4" * 64,
        },
        "model_inputs_sha256": "5" * 64,
        "role": "development",
        "summary": {
            "path": "data/working/summary.json",
            "sha256": "6" * 64,
        },
        "thresholds": list(PROBABILITY_THRESHOLDS),
        "trainer_source_sha256": "7" * 64,
        "upstream_manifest": {
            "path": "data/working/manifest.csv",
            "sha256": "8" * 64,
        },
        "upstream_manifest_sha256": "8" * 64,
        "validation_source_sha256": "9" * 64,
    }
    dynamic = {
        "complete",
        "expected_row_count",
        "rows",
        "rows_sha256",
        "validation_grid_schema_version",
    }
    assert set(identity) == GRID_FIELDS - dynamic
    return identity


def _row(
    identity: dict[str, object],
    dataset: str,
    model_name: str,
    threshold: float,
    *,
    improvement: float = 0.03,
) -> dict[str, object]:
    contracts = read_division_contracts(ROOT / "configs")
    division_counts = {
        value: int(count)
        for count, roles in contracts["split"]["strata"].items()
        for value in roles["development"]
    }
    division_dataset = next(
        value for value in identity["datasets"] if division_counts[value] > 0
    )
    accepted = int(dataset == division_dataset and improvement > 0.0)
    fallback_hash = hashlib.sha256(f"{dataset}:fallback".encode()).hexdigest()
    candidate_hash = (
        hashlib.sha256(f"{dataset}:{model_name}:{threshold}".encode()).hexdigest()
        if accepted
        else fallback_hash
    )
    checkpoint = identity["checkpoints"][model_name]
    baseline_metric = {
        "division_fn": division_counts[dataset],
        "division_fp": 0,
        "division_jaccard": float(division_counts[dataset] == 0),
        "division_tp": 0,
        "edge_fn": 1,
        "edge_fp": 1,
        "edge_ignored": 0,
        "edge_jaccard": 0.5,
        "edge_tp": 1,
        "matched_nodes": 4,
        "predicted_nodes": 100,
        "score": 0.5,
        "truth_nodes": 4,
    }
    candidate_metric = {
        **baseline_metric,
        "score": 0.5 + improvement if accepted else 0.5,
    }
    triplets = {
        "fn": division_counts[dataset],
        "fp": 0,
        "fully_matched_wrong_forks": 0,
        "matched_division_false_positives": 0,
        "tp": 0,
        "unmatched_forks": 0,
    }
    candidate_triplets = {
        **triplets,
        "fn": division_counts[dataset] - accepted,
        "tp": accepted,
    }
    logit = 10.0
    accepted_records = (
        [
            {
                "candidate_1_id": 20,
                "candidate_2_id": 21,
                "edge_removal_count": 1,
                "logit": logit,
                "mother_id": 10,
                "probability": 1.0 / (1.0 + math.exp(-logit)),
                "restored_node_count": 0,
            }
        ]
        if accepted
        else []
    )
    row = {
        "accepted_fork_records": accepted_records,
        "accepted_forks": accepted,
        "added_edge_count": 2 * accepted,
        "base_checkpoint_sha256": identity["base_checkpoint_sha256"],
        "baseline_edge_jaccard": 0.5,
        "baseline_full_score": 0.5,
        "baseline_metric": baseline_metric,
        "baseline_predicted_nodes": 100,
        "baseline_triplets": triplets,
        "cache_content_sha256": hashlib.sha256(f"{dataset}:content".encode()).hexdigest(),
        "cache_file_sha256": hashlib.sha256(f"{dataset}:file".encode()).hexdigest(),
        "cache_image_sha256": hashlib.sha256(f"{dataset}:image".encode()).hexdigest(),
        "candidate_edge_jaccard": 0.5,
        "candidate_full_score": 0.5 + improvement if accepted else 0.5,
        "candidate_graph_sha256": candidate_hash,
        "candidate_metric": candidate_metric,
        "candidate_predicted_nodes": 100,
        "candidate_triplets": candidate_triplets,
        "checkpoint_sha256": checkpoint["sha256"],
        "contract_sha256": identity["contract_sha256"],
        "dataset": dataset,
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "edge_removal_count": accepted,
        "estimated_nodes": 1000,
        "fallback_graph_sha256": fallback_hash,
        "feature_schema_sha256": identity["feature_schema_sha256"],
        "fully_matched_wrong_forks": 0,
        "matched_division_false_positives": 0,
        "model_config": checkpoint["model_config"],
        "model_inputs_sha256": identity["model_inputs_sha256"],
        "model_name": model_name,
        "restored_node_count": 0,
        "runtime_seconds": 0.1,
        "score_threshold": threshold,
        "strict_true_recoveries": accepted,
        "trainer_source_sha256": identity["trainer_source_sha256"],
        "truth_division_count": division_counts[dataset],
        "truth_graph_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
        "validation_source_sha256": identity["validation_source_sha256"],
    }
    assert set(row) == GRID_ROW_FIELDS
    return row


def _grid_rows(
    identity: dict[str, object],
    *,
    preferred: tuple[str, float] | None = None,
    rejection: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for dataset in identity["datasets"]:
        for model_name in DIVISION_MODEL_NAMES:
            for threshold in PROBABILITY_THRESHOLDS:
                improvement = 0.0 if rejection else 0.03
                if preferred == (model_name, threshold):
                    improvement = 0.04
                rows.append(
                    _row(
                        identity,
                        dataset,
                        model_name,
                        threshold,
                        improvement=improvement,
                    )
                )
    return rows


def test_resumable_grid_requires_complete_movie_prefix_and_exact_identity(
    tmp_path: Path,
) -> None:
    identity = _identity()
    rows = _grid_rows(identity)
    rows_per_movie = len(DIVISION_MODEL_NAMES) * len(PROBABILITY_THRESHOLDS)

    with pytest.raises(ValueError, match="complete movie boundary"):
        grid_payload(identity, rows[: rows_per_movie - 1])

    partial = grid_payload(identity, rows[:rows_per_movie])
    assert partial["complete"] is False
    output = tmp_path / "grid.json"
    write_canonical_json(output, partial)
    assert read_validation_grid(
        output,
        expected_identity=identity,
        require_complete=False,
    ) == partial
    with pytest.raises(ValueError, match="incomplete"):
        read_validation_grid(output, require_complete=True)

    changed = dict(identity)
    changed["validation_source_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="identity has changed"):
        read_validation_grid(
            output,
            expected_identity=changed,
            require_complete=False,
        )


def test_complete_grid_has_exact_660_rows_and_selects_best_passing_policy() -> None:
    identity = _identity()
    preferred = ("embedding_linear", 0.9)
    grid = grid_payload(identity, _grid_rows(identity, preferred=preferred))

    assert grid["complete"] is True
    assert len(grid["rows"]) == EXPECTED_GRID_ROWS == 660
    selection = select_grid_policy(
        grid,
        contracts=read_division_contracts(ROOT / "configs"),
        manifest_rows=_manifest_rows(),
    )
    assert (selection.model_name, selection.score_threshold) == preferred
    assert selection.evaluation_unlocked is True
    assert all(selection.guard_results.values())


def test_policy_tie_break_uses_model_order_then_higher_threshold() -> None:
    identity = _identity()
    grid = grid_payload(identity, _grid_rows(identity))

    selection = select_grid_policy(
        grid,
        contracts=read_division_contracts(ROOT / "configs"),
        manifest_rows=_manifest_rows(),
    )
    assert selection.model_name == "scalar"
    assert selection.score_threshold == 0.9999


def test_no_passing_policy_freezes_a_rejection_selection() -> None:
    identity = _identity()
    grid = grid_payload(identity, _grid_rows(identity, rejection=True))

    selection = select_grid_policy(
        grid,
        contracts=read_division_contracts(ROOT / "configs"),
        manifest_rows=_manifest_rows(),
    )
    assert selection.evaluation_unlocked is False
    assert not all(selection.guard_results.values())


def test_best_pair_retention_matches_logit_cost_and_identifier_tie_breaks() -> None:
    fallback_hash = "f" * 64
    proposals = [
        ScoredProposal(1, 10, 11, 2.0, 1, 0, fallback_hash),
        ScoredProposal(1, 12, 13, 2.0, 0, 1, fallback_hash),
        ScoredProposal(1, 14, 15, 1.0, 0, 0, fallback_hash),
        ScoredProposal(2, 20, 21, 0.5, 0, 0, fallback_hash),
    ]

    retained = retain_best_per_mother(proposals)

    assert [(row.mother_id, row.candidate_ids) for row in retained] == [
        (1, (12, 13)),
        (2, (20, 21)),
    ]
