from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biohub.appearance_evaluation import (
    CANDIDATE_POLICY_SCHEMA_VERSION,
    CANDIDATE_SELECTION_RULE,
    LOCKED_NODE_DENSITY_LIMIT,
    checkpoint_dev_datasets,
    file_sha256,
    files_sha256,
    read_frozen_policy,
    read_score_cache,
    write_frozen_policy,
    write_score_cache,
)
from biohub.metric import TrackingGraph


def candidate_graph() -> TrackingGraph:
    return TrackingGraph(
        node_ids=np.array([10, 20], dtype=np.int64),
        t=np.array([0, 1], dtype=np.int64),
        z=np.array([4.0, 4.5]),
        y=np.array([16.0, 17.0]),
        x=np.array([16.0, 17.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_checkpoint_dev_datasets_requires_a_unique_6bba_split() -> None:
    assert checkpoint_dev_datasets(
        {"dev_shards": ["6bba_b", "6bba_a"]}
    ) == ["6bba_a", "6bba_b"]

    with pytest.raises(ValueError, match="duplicate"):
        checkpoint_dev_datasets({"dev_shards": ["6bba_a", "6bba_a"]})
    with pytest.raises(ValueError, match="only 6bba"):
        checkpoint_dev_datasets({"dev_shards": ["44b6_holdout"]})


def test_score_cache_round_trip_and_contract_validation(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    checkpoint_sha256 = "a" * 64
    scoring_source_sha256 = "b" * 64
    write_score_cache(
        path,
        candidate_graph(),
        np.array([0.2, 0.9], dtype=np.float32),
        dataset="6bba_movie",
        data_split="train",
        checkpoint_sha256=checkpoint_sha256,
        scoring_source_sha256=scoring_source_sha256,
        detector_threshold_k=3.0,
        patch_shape=(9, 33, 33),
        detect_seconds=1.25,
        score_seconds=2.5,
    )

    graph, scores, metadata = read_score_cache(
        path,
        expected_dataset="6bba_movie",
        expected_data_split="train",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_scoring_source_sha256=scoring_source_sha256,
        expected_detector_threshold_k=3.0,
        expected_patch_shape=(9, 33, 33),
    )

    assert graph.node_ids.tolist() == [10, 20]
    assert scores.tolist() == pytest.approx([0.2, 0.9])
    assert metadata["detect_seconds"] == 1.25
    assert not path.with_suffix(".npz.tmp").exists()

    with pytest.raises(ValueError, match="different DoG threshold"):
        read_score_cache(path, expected_detector_threshold_k=4.0)


def test_score_cache_rejects_invalid_scores(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        write_score_cache(
            tmp_path / "bad.npz",
            candidate_graph(),
            np.array([0.2, 1.1]),
            dataset="6bba_movie",
            data_split="train",
            checkpoint_sha256="b" * 64,
            scoring_source_sha256="c" * 64,
            detector_threshold_k=3.0,
            patch_shape=(9, 33, 33),
            detect_seconds=1.0,
            score_seconds=2.0,
        )


def test_frozen_policy_requires_internal_dev_provenance(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    policy = {
        "candidate_policy_schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "selection_cohort": "internal-dev",
        "selection_datasets": ["6bba_movie_a", "6bba_movie_b"],
        "checkpoint_sha256": "c" * 64,
        "evaluation_sha256": "d" * 64,
        "source_results_sha256": "e" * 64,
        "score_threshold": 0.4,
        "min_track_length": 8,
        "density_limit": LOCKED_NODE_DENSITY_LIMIT,
        "selection_rule": CANDIDATE_SELECTION_RULE,
        "internal_improvement_supported": True,
        "selected_metrics": {
            "mean_edge_jaccard": 0.7,
            "median_edge_jaccard": 0.7,
            "aggregate_node_ratio": 1.0,
            "mean_recall": 0.8,
            "mean_runtime_seconds": 10.0,
        },
        "baseline_metrics": {
            "mean_edge_jaccard": 0.6,
            "median_edge_jaccard": 0.6,
            "aggregate_node_ratio": 1.4,
            "mean_recall": 0.85,
            "mean_runtime_seconds": 8.0,
        },
        "mean_edge_jaccard_gain": 0.1,
        "per_movie_wins": 1,
        "per_movie_ties": 1,
        "per_movie_losses": 0,
    }
    write_frozen_policy(path, policy)

    assert read_frozen_policy(path) == policy
    assert json.loads(path.read_text()) == policy

    policy["selection_cohort"] = "locked-holdout"
    with pytest.raises(ValueError, match="internal-dev"):
        write_frozen_policy(tmp_path / "bad.json", policy)


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"biohub")

    assert file_sha256(path) == "560cc3b66fc8dc55a331b54170cecff643fbcb42329530e8d19fd1ab557c74cb"


def test_source_bundle_hash_depends_on_file_identity_and_order(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n")
    second.write_text("value = 2\n")

    assert files_sha256([first, second]) == files_sha256([first, second])
    assert files_sha256([first, second]) != files_sha256([second, first])
