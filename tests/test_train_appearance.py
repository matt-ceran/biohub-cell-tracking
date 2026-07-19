from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import train_appearance


def test_split_shards_is_deterministic_and_movie_level() -> None:
    shards = [Path(f"6bba_movie_{index}") for index in range(10)]

    train_a, dev_a = train_appearance.split_shards(shards, dev_fraction=0.2, seed=7)
    train_b, dev_b = train_appearance.split_shards(shards, dev_fraction=0.2, seed=7)

    assert train_a == train_b
    assert dev_a == dev_b
    assert len(train_a) == 8
    assert len(dev_a) == 2
    assert set(train_a).isdisjoint(dev_a)
    assert set(train_a) | set(dev_a) == set(shards)


def test_selection_contract_tracks_ranking_in_the_correct_direction() -> None:
    assert train_appearance.selection_contract("pu-auc") == ("positive_unlabeled_auc", "max")
    assert train_appearance.selection_contract("nnpu-risk") == ("loss", "min")
    assert train_appearance.selection_improved(0.901, 0.897, mode="max", min_delta=0.001)
    assert not train_appearance.selection_improved(0.8975, 0.897, mode="max", min_delta=0.001)
    assert train_appearance.selection_improved(0.1, 0.2, mode="min", min_delta=0.01)


def test_training_history_is_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "model.history.json"
    payload = {"completed_epochs": 2, "history": [{"epoch": 1}, {"epoch": 2}]}

    train_appearance.write_training_history(path, payload)

    assert json.loads(path.read_text()) == payload
    assert not path.with_suffix(".json.tmp").exists()


def test_training_state_round_trip_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "model.state.pt"
    payload = {
        "training_state_schema_version": train_appearance.TRAINING_STATE_SCHEMA_VERSION,
        "last_epoch": 3,
    }
    train_appearance.save_training_state(path, payload)

    assert train_appearance.load_training_state(path)["last_epoch"] == 3

    bad_path = tmp_path / "bad.state.pt"
    train_appearance.save_training_state(
        bad_path,
        {"training_state_schema_version": 999},
    )
    with pytest.raises(ValueError, match="unsupported training state schema"):
        train_appearance.load_training_state(bad_path)


def test_sidecar_paths_keep_the_checkpoint_stem() -> None:
    checkpoint = Path("checkpoints/cellness.pt")

    assert train_appearance._sidecar_path(checkpoint, ".history.json") == Path(
        "checkpoints/cellness.history.json"
    )
    assert train_appearance._sidecar_path(checkpoint, ".state.pt") == Path(
        "checkpoints/cellness.state.pt"
    )
