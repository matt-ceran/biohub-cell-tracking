from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from biohub.io import SamplePaths
from scripts import validate_pruning


def result_row(dataset: str, min_track_length: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "threshold_k": 3.0,
        "min_track_length": min_track_length,
        "runtime_seconds": 12.5,
        "edge_jaccard": 0.75,
        "score": 0.675,
        "division_jaccard": 0.0,
        "edge_tp": 3,
        "edge_fp": 1,
        "edge_fn": 2,
        "edge_ignored": 4,
        "division_tp": 0,
        "division_fp": 0,
        "division_fn": 1,
        "matched_nodes": 8,
        "predicted_nodes": 10,
        "truth_nodes": 9,
        "estimated_nodes": 20,
        "predicted_to_estimated": 0.5,
        "recall": 8 / 9,
    }


def test_result_checkpoint_round_trip_is_sorted_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    rows = [result_row("movie_b", 8), result_row("movie_a", 3)]

    validate_pruning.write_result_rows(path, rows)
    loaded = validate_pruning.read_result_rows(path)

    assert [(row["dataset"], row["min_track_length"]) for row in loaded] == [
        ("movie_a", 3),
        ("movie_b", 8),
    ]
    assert loaded[0]["predicted_nodes"] == 10
    assert loaded[0]["edge_jaccard"] == 0.75
    assert not path.with_suffix(".csv.tmp").exists()


def test_estimated_node_count_reads_geff_metadata(tmp_path: Path) -> None:
    labels = tmp_path / "movie.geff"
    labels.mkdir()
    metadata = {
        "attributes": {
            "geff": {
                "extra": {
                    "estimated_number_of_nodes": 12345,
                }
            }
        }
    }
    (labels / "zarr.json").write_text(json.dumps(metadata))

    assert validate_pruning.estimated_node_count(labels) == 12345


def test_validation_samples_applies_deterministic_start_offset(monkeypatch) -> None:
    samples = [
        SamplePaths(
            dataset=f"44b6_movie_{index}",
            image=Path(f"movie_{index}.zarr"),
            labels=Path(f"movie_{index}.geff"),
            embryo="44b6",
        )
        for index in range(5)
    ]
    samples.append(
        SamplePaths(
            dataset="6bba_train",
            image=Path("train.zarr"),
            labels=Path("train.geff"),
            embryo="6bba",
        )
    )
    monkeypatch.setattr(validate_pruning, "discover_split", lambda *_: samples)
    validation_ids = [sample.dataset for sample in samples[:5]]
    monkeypatch.setattr(
        validate_pruning,
        "make_embryo_holdout",
        lambda *_args, **_kwargs: SimpleNamespace(validation=validation_ids),
    )

    selected = validate_pruning.validation_samples(Path("unused"), n=2, start_index=1)

    assert [sample.dataset for sample in selected] == ["44b6_movie_1", "44b6_movie_2"]
