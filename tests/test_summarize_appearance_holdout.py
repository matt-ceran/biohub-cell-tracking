from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_appearance_holdout


def row(
    dataset: str,
    edge_jaccard: float,
    predicted_nodes: int,
    *,
    estimated_nodes: int = 100,
    recall: float = 0.8,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "edge_jaccard": edge_jaccard,
        "predicted_nodes": predicted_nodes,
        "estimated_nodes": estimated_nodes,
        "recall": recall,
        "runtime_seconds": 10.0,
    }


def test_compare_rows_reports_exact_success_contract() -> None:
    baseline = [
        row("44b6_a", 0.6, 140),
        row("44b6_b", 0.7, 140),
        row("44b6_c", 0.8, 140),
    ]
    learned = [
        row("44b6_a", 0.7, 120),
        row("44b6_b", 0.7, 120),
        row("44b6_c", 0.75, 120),
    ]

    comparison = summarize_appearance_holdout.compare_rows(learned, baseline)

    assert comparison["baseline"]["mean_edge_jaccard"] == pytest.approx(0.7)
    assert comparison["learned"]["mean_edge_jaccard"] == pytest.approx(0.7166666667)
    assert comparison["mean_edge_jaccard_gain"] == pytest.approx(0.0166666667)
    assert comparison["per_movie_wins"] == 1
    assert comparison["per_movie_ties"] == 1
    assert comparison["per_movie_losses"] == 1
    assert comparison["density_within_guardrail"] is True
    assert comparison["phase10_success"] is True


def test_compare_rows_rejects_different_movie_sets() -> None:
    learned = [row("44b6_a", 0.7, 100)]
    baseline = [row("44b6_b", 0.7, 100)]

    with pytest.raises(ValueError, match="identical movies"):
        summarize_appearance_holdout.compare_rows(learned, baseline)


def test_aggregate_rows_uses_aggregate_node_density() -> None:
    rows = [
        row("44b6_a", 0.5, 50, estimated_nodes=100),
        row("44b6_b", 0.7, 300, estimated_nodes=200),
    ]

    result = summarize_appearance_holdout.aggregate_rows(rows)

    assert result["mean_edge_jaccard"] == pytest.approx(0.6)
    assert result["median_edge_jaccard"] == pytest.approx(0.6)
    assert result["aggregate_node_ratio"] == pytest.approx(350 / 300)


def test_summary_write_is_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {"phase10_success": True}

    summarize_appearance_holdout.write_summary(path, payload)

    assert json.loads(path.read_text()) == payload
    assert not path.with_suffix(".json.tmp").exists()
