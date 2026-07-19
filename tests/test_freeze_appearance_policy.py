from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from biohub.appearance_evaluation import (
    CANDIDATE_POLICY_SCHEMA_VERSION,
    CANDIDATE_SELECTION_RULE,
    verify_frozen_policy_results,
)
from scripts import freeze_appearance_policy


def row(
    dataset: str,
    threshold: float,
    min_track_length: int,
    *,
    edge_jaccard: float,
    predicted_nodes: int,
    estimated_nodes: int = 100,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "score_threshold": threshold,
        "min_track_length": min_track_length,
        "edge_jaccard": edge_jaccard,
        "predicted_nodes": predicted_nodes,
        "estimated_nodes": estimated_nodes,
        "recall": edge_jaccard,
        "runtime_seconds": 10.0,
    }


def complete_grid() -> list[dict[str, object]]:
    return [
        row("6bba_a", 0.0, 8, edge_jaccard=0.60, predicted_nodes=140),
        row("6bba_b", 0.0, 8, edge_jaccard=0.70, predicted_nodes=140),
        row("6bba_a", 0.4, 8, edge_jaccard=0.66, predicted_nodes=100),
        row("6bba_b", 0.4, 8, edge_jaccard=0.74, predicted_nodes=100),
        row("6bba_a", 0.6, 8, edge_jaccard=0.64, predicted_nodes=70),
        row("6bba_b", 0.6, 8, edge_jaccard=0.72, predicted_nodes=70),
    ]


def test_select_policy_requires_improvement_and_density() -> None:
    aggregates = freeze_appearance_policy.aggregate_policy_rows(complete_grid())

    selected, baseline = freeze_appearance_policy.select_policy(
        aggregates,
        density_limit=1.1,
    )

    assert selected["score_threshold"] == 0.4
    assert selected["mean_edge_jaccard"] == pytest.approx(0.70)
    assert selected["aggregate_node_ratio"] == pytest.approx(1.0)
    assert baseline["mean_edge_jaccard"] == pytest.approx(0.65)


def test_aggregate_policy_rows_rejects_incomplete_grid() -> None:
    rows = complete_grid()
    rows.pop()

    with pytest.raises(ValueError, match="same complete movie set"):
        freeze_appearance_policy.aggregate_policy_rows(rows)


def test_select_policy_does_not_unlock_holdout_without_gain() -> None:
    rows = complete_grid()
    for item in rows:
        if item["score_threshold"] > 0:
            item["edge_jaccard"] = 0.5
    aggregates = freeze_appearance_policy.aggregate_policy_rows(rows)

    with pytest.raises(ValueError, match="do not unlock 44b6"):
        freeze_appearance_policy.select_policy(aggregates, density_limit=1.1)


def test_select_policy_rejects_all_over_density() -> None:
    aggregates = freeze_appearance_policy.aggregate_policy_rows(complete_grid())

    with pytest.raises(ValueError, match="node-density limit"):
        freeze_appearance_policy.select_policy(aggregates, density_limit=0.5)


def test_freezing_requires_the_checkpoint_complete_dev_split() -> None:
    aggregates = freeze_appearance_policy.aggregate_policy_rows(complete_grid())
    checkpoint = {"dev_shards": ["6bba_a", "6bba_b"]}

    assert freeze_appearance_policy.require_complete_checkpoint_dev(
        aggregates,
        checkpoint,
    ) == ["6bba_a", "6bba_b"]

    checkpoint["dev_shards"].append("6bba_c")
    with pytest.raises(ValueError, match="exact complete movie split"):
        freeze_appearance_policy.require_complete_checkpoint_dev(aggregates, checkpoint)


def test_frozen_policy_is_recomputed_from_its_source_results() -> None:
    rows = complete_grid()
    policy = {
        "candidate_policy_schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "selection_cohort": "internal-dev",
        "selection_datasets": ["6bba_a", "6bba_b"],
        "checkpoint_sha256": "a" * 64,
        "evaluation_sha256": "b" * 64,
        "source_results_sha256": "c" * 64,
        "score_threshold": 0.4,
        "min_track_length": 8,
        "density_limit": 1.1,
        "selection_rule": CANDIDATE_SELECTION_RULE,
        "internal_improvement_supported": True,
        "selected_metrics": {
            "mean_edge_jaccard": 0.70,
            "median_edge_jaccard": 0.70,
            "aggregate_node_ratio": 1.0,
            "mean_recall": 0.70,
            "mean_runtime_seconds": 10.0,
        },
        "baseline_metrics": {
            "mean_edge_jaccard": 0.65,
            "median_edge_jaccard": 0.65,
            "aggregate_node_ratio": 1.4,
            "mean_recall": 0.65,
            "mean_runtime_seconds": 10.0,
        },
        "mean_edge_jaccard_gain": 0.05,
        "per_movie_wins": 2,
        "per_movie_ties": 0,
        "per_movie_losses": 0,
    }
    checkpoint = {"dev_shards": ["6bba_a", "6bba_b"]}

    verify_frozen_policy_results(policy, rows, checkpoint)

    policy["score_threshold"] = 0.6
    with pytest.raises(ValueError, match="not the winning policy"):
        verify_frozen_policy_results(policy, rows, checkpoint)


@pytest.mark.parametrize(
    "script_name",
    ["freeze_appearance_policy.py", "summarize_appearance_holdout.py"],
)
def test_policy_scripts_are_executable_entry_points(script_name: str) -> None:
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script_name), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
