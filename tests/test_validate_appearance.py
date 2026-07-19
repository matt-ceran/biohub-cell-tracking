from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from biohub.io import SamplePaths
from scripts import validate_appearance


def sample(dataset: str, embryo: str) -> SamplePaths:
    return SamplePaths(
        dataset=dataset,
        image=Path(f"{dataset}.zarr"),
        labels=Path(f"{dataset}.geff"),
        embryo=embryo,
    )


def result_row(dataset: str, threshold: float, min_track_length: int) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "cohort": "internal-dev",
        "checkpoint_sha256": "a" * 64,
        "evaluation_sha256": "b" * 64,
        "score_threshold": threshold,
        "min_track_length": min_track_length,
        "raw_candidates": 100,
        "kept_candidates": 40,
        "kept_fraction": 0.4,
        "detect_seconds": 1.0,
        "score_seconds": 2.0,
        "link_seconds": 3.0,
        "runtime_seconds": 6.0,
        "edge_jaccard": 0.7,
        "score": 0.63,
        "division_jaccard": 0.0,
        "edge_tp": 7,
        "edge_fp": 1,
        "edge_fn": 2,
        "edge_ignored": 3,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 1,
        "matched_nodes": 8,
        "predicted_nodes": 40,
        "truth_nodes": 9,
        "estimated_nodes": 50,
        "predicted_to_estimated": 0.8,
        "recall": 8 / 9,
    }
    return row


def test_result_checkpoint_round_trip_is_atomic_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    rows = [result_row("6bba_b", 0.5, 8), result_row("6bba_a", 0.2, 3)]

    validate_appearance.write_result_rows(path, rows)
    loaded = validate_appearance.read_result_rows(path)

    assert [row["dataset"] for row in loaded] == ["6bba_a", "6bba_b"]
    assert loaded[0]["score_threshold"] == 0.2
    assert loaded[0]["predicted_nodes"] == 40
    assert not path.with_suffix(".csv.tmp").exists()


def test_internal_dev_samples_follows_checkpoint_movie_order(monkeypatch) -> None:
    samples = [
        sample("6bba_train", "6bba"),
        sample("6bba_dev_b", "6bba"),
        sample("6bba_dev_a", "6bba"),
        sample("44b6_holdout", "44b6"),
    ]
    monkeypatch.setattr(validate_appearance, "discover_split", lambda *_: samples)
    checkpoint = {
        "train_embryo": "6bba",
        "dev_shards": ["6bba_dev_a", "6bba_dev_b"],
    }

    selected = validate_appearance.internal_dev_samples(Path("unused"), checkpoint)

    assert [item.dataset for item in selected] == ["6bba_dev_a", "6bba_dev_b"]


def test_locked_holdout_requires_the_exact_complete_cohort(monkeypatch) -> None:
    holdout = [sample(f"44b6_{index:02d}", "44b6") for index in range(71)]
    monkeypatch.setattr(validate_appearance, "discover_split", lambda *_: holdout)
    checkpoint = {"holdout_embryo": "44b6"}

    selected = validate_appearance.locked_holdout_samples(Path("unused"), checkpoint)

    assert len(selected) == 71

    monkeypatch.setattr(validate_appearance, "discover_split", lambda *_: holdout[:-1])
    with pytest.raises(ValueError, match="requires 71"):
        validate_appearance.locked_holdout_samples(Path("unused"), checkpoint)


def test_policy_grid_parsers_reject_invalid_values() -> None:
    assert validate_appearance.parse_float_list("0,0.5,1") == [0.0, 0.5, 1.0]
    assert validate_appearance.parse_int_list("1,3,8") == [1, 3, 8]

    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        validate_appearance.parse_float_list("0.5,0.5")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        validate_appearance.parse_int_list("0,3")
