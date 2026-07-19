"""Synthetic coverage for the learned DoG-candidate appearance boundary."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from biohub.appearance import (
    APPEARANCE_SCHEMA_VERSION,
    APPEARANCE_TRAIN_EMBRYO,
    SAMPLE_KNOWN_POSITIVE,
    SAMPLE_UNLABELED,
    AppearanceShardSummary,
    appearance_shards_sha256,
    build_appearance_shard,
    discover_appearance_shards,
    extract_patch,
    extract_patches,
    filter_candidates,
    known_positive_mask,
    normalize_patch,
    normalize_patches,
    read_appearance_shard,
)
from biohub.io import SamplePaths
from biohub.metric import TrackingGraph


def _graph(rows, edges=()):
    rows = np.asarray(rows, dtype=float)
    return TrackingGraph(
        node_ids=rows[:, 0].astype(np.int64),
        t=rows[:, 1].astype(int),
        z=rows[:, 2],
        y=rows[:, 3],
        x=rows[:, 4],
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_extract_patch_has_an_exact_center_and_reflects_boundaries():
    volume = np.arange(5 * 7 * 9, dtype=np.uint16).reshape(5, 7, 9)
    centered = extract_patch(volume, (2, 3, 4), patch_shape=(3, 5, 5))
    assert centered.shape == (3, 5, 5)
    assert centered[1, 2, 2] == volume[2, 3, 4]

    boundary = extract_patch(volume, (0, 0, 0), patch_shape=(3, 5, 5))
    assert boundary.shape == (3, 5, 5)
    assert boundary[1, 2, 2] == volume[0, 0, 0]


def test_extract_patch_rejects_even_shapes_and_outside_centers():
    volume = np.zeros((5, 7, 9), dtype=np.uint16)
    with pytest.raises(ValueError, match="odd"):
        extract_patch(volume, (2, 3, 4), patch_shape=(4, 5, 5))
    with pytest.raises(ValueError, match="outside"):
        extract_patch(volume, (-1, 3, 4), patch_shape=(3, 5, 5))


def test_batched_patch_extraction_and_normalization_match_single_patch_operations():
    rng = np.random.default_rng(7)
    volume = rng.integers(0, 4096, size=(5, 7, 9), dtype=np.uint16)
    centers = np.array([(0, 0, 0), (2, 3, 4), (4, 6, 8)], dtype=float)
    expected = np.stack(
        [extract_patch(volume, center, patch_shape=(3, 5, 5)) for center in centers]
    )
    actual = extract_patches(volume, centers, patch_shape=(3, 5, 5))

    np.testing.assert_array_equal(actual, expected)
    expected_normalized = np.stack([normalize_patch(patch) for patch in expected])
    np.testing.assert_allclose(normalize_patches(actual), expected_normalized, rtol=0, atol=1e-6)


def test_known_positive_mask_does_not_call_unmatched_candidates_negative():
    candidates = _graph(
        [
            (10, 0, 1, 10, 10),
            (20, 0, 1, 100, 100),
        ]
    )
    truth = _graph([(1, 0, 1, 11, 10)])
    mask = known_positive_mask(candidates, truth)
    assert mask.tolist() == [True, False]


def test_filter_candidates_drops_incident_edges_and_preserves_node_ids():
    graph = _graph(
        [(10, 0, 1, 2, 3), (20, 1, 1, 3, 3), (30, 2, 1, 4, 3)],
        [(10, 20), (20, 30)],
    )
    filtered = filter_candidates(graph, np.array([0.9, 0.2, 0.8]), min_score=0.5)
    assert filtered.node_ids.tolist() == [10, 30]
    assert filtered.edges.shape == (0, 2)


def test_normalize_patch_is_finite_for_a_constant_patch():
    normalized = normalize_patch(np.full((3, 5, 5), 42, dtype=np.uint16))
    assert normalized.dtype == np.float32
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized == 0)


def test_training_shard_builder_rejects_the_holdout_before_reading_data(tmp_path):
    sample = SamplePaths(
        dataset="44b6_forbidden",
        image=tmp_path / "missing.zarr",
        labels=tmp_path / "missing.geff",
        embryo="44b6",
    )
    with pytest.raises(ValueError, match=APPEARANCE_TRAIN_EMBRYO):
        build_appearance_shard(sample, tmp_path / "output")


def _write_synthetic_shard(root: Path, dataset: str = "6bba_synthetic") -> Path:
    shard = root / dataset
    shard.mkdir(parents=True)
    patches = np.zeros((3, 9, 33, 33), dtype=np.uint16)
    patches[0, 4, 16, 16] = 1000
    patches[1, 4, 10, 10] = 500
    patches[2, 4, 20, 20] = 500
    roles = np.array([SAMPLE_KNOWN_POSITIVE, SAMPLE_UNLABELED, SAMPLE_UNLABELED], dtype=np.uint8)
    summary = AppearanceShardSummary(
        schema_version=APPEARANCE_SCHEMA_VERSION,
        dataset=dataset,
        embryo=APPEARANCE_TRAIN_EMBRYO,
        patch_shape=(9, 33, 33),
        detector_threshold_k=3.0,
        unlabeled_ratio=2.0,
        seed=0,
        raw_candidate_count=10,
        truth_node_count=1,
        matched_candidate_count=1,
        sparse_label_recall=1.0,
        estimated_node_count=4,
        estimated_candidate_prior=0.4,
        known_positive_samples=1,
        unlabeled_samples=2,
    )
    np.save(shard / "patches.npy", patches, allow_pickle=False)
    np.save(shard / "sample_roles.npy", roles, allow_pickle=False)
    np.save(shard / "candidate_tzyx.npy", np.zeros((3, 4), dtype=np.float32))
    (shard / "metadata.json").write_text(json.dumps(asdict(summary)))
    return shard


def test_appearance_shard_and_torch_dataset_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from biohub.appearance_model import CandidatePatchDataset

    shard = _write_synthetic_shard(tmp_path)
    summary = read_appearance_shard(shard)
    assert summary.dataset == "6bba_synthetic"
    dataset = CandidatePatchDataset([shard])
    assert len(dataset) == 3
    assert dataset.positive_indices.tolist() == [0]
    assert dataset.unlabeled_indices.tolist() == [1, 2]
    assert dataset.class_prior == pytest.approx(0.4)
    patch, role = dataset[0]
    assert patch.shape == (1, 9, 33, 33)
    assert patch.dtype == torch.float32
    assert role.item() == SAMPLE_KNOWN_POSITIVE


def test_appearance_shard_validation_rejects_inconsistent_arrays(tmp_path):
    shard = _write_synthetic_shard(tmp_path)
    np.save(
        shard / "candidate_tzyx.npy",
        np.zeros((2, 4), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="coordinates have shape"):
        read_appearance_shard(shard)


def test_appearance_shard_validation_rejects_a_mismatched_dataset_name(tmp_path):
    shard = _write_synthetic_shard(tmp_path)
    metadata = json.loads((shard / "metadata.json").read_text())
    metadata["dataset"] = "6bba_different"
    (shard / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="directory name must match"):
        read_appearance_shard(shard)


def test_appearance_shard_discovery_ignores_atomic_temporary_directories(tmp_path):
    shard = _write_synthetic_shard(tmp_path)
    (tmp_path / ".6bba_in_progress.tmp").mkdir()

    assert discover_appearance_shards(tmp_path) == [shard]


def test_appearance_shard_fingerprint_covers_array_contents(tmp_path):
    shard = _write_synthetic_shard(tmp_path)
    original = appearance_shards_sha256([shard])
    patches = np.load(shard / "patches.npy", allow_pickle=False)
    patches[0, 0, 0, 0] = 1
    np.save(shard / "patches.npy", patches, allow_pickle=False)

    assert appearance_shards_sha256([shard]) != original


def test_patch_dataset_bounds_its_open_memmap_cache(tmp_path):
    pytest.importorskip("torch")
    from biohub.appearance_model import PATCH_MEMMAP_CACHE_SIZE, CandidatePatchDataset

    shards = [
        _write_synthetic_shard(tmp_path, f"6bba_synthetic_{index:02d}")
        for index in range(PATCH_MEMMAP_CACHE_SIZE + 3)
    ]
    dataset = CandidatePatchDataset(shards)
    for shard_index in range(len(shards)):
        dataset[shard_index * 3]

    assert len(dataset._patch_cache) == PATCH_MEMMAP_CACHE_SIZE
    assert list(dataset._patch_cache) == shards[-PATCH_MEMMAP_CACHE_SIZE:]


def test_cellness_model_pu_loss_checkpoint_and_scoring(tmp_path):
    torch = pytest.importorskip("torch")
    from biohub.appearance_model import (
        CellnessCNN3D,
        CellnessModelConfig,
        load_cellness_checkpoint,
        nnpu_training_objective,
        non_negative_pu_loss,
        positive_unlabeled_auc,
        save_cellness_checkpoint,
        score_candidates,
    )

    model = CellnessCNN3D(CellnessModelConfig(base_channels=4, dropout=0.0))
    inputs = torch.randn(3, 1, 9, 33, 33)
    logits = model(inputs)
    assert logits.shape == (3,)
    loss, risks = non_negative_pu_loss(logits[:1], logits[1:], class_prior=0.4)
    assert loss.item() >= 0.0
    assert set(risks) == {
        "positive_risk",
        "raw_negative_risk",
        "corrected_negative_risk",
    }
    assert positive_unlabeled_auc(np.array([0.8, 0.9]), np.array([0.1, 0.2])) == 1.0
    loss.backward()

    positive_logits = torch.tensor([10.0], requires_grad=True)
    unlabeled_logits = torch.tensor([-10.0], requires_grad=True)
    objective, estimated_risk, corrected = nnpu_training_objective(
        positive_logits,
        unlabeled_logits,
        class_prior=0.4,
    )
    assert corrected["raw_negative_risk"].item() < 0.0
    assert corrected["correction_active"].item() == 1.0
    assert objective.item() == pytest.approx(-corrected["raw_negative_risk"].item())
    assert estimated_risk.item() == pytest.approx(corrected["positive_risk"].item())
    objective.backward()
    assert positive_logits.grad is not None
    assert unlabeled_logits.grad is not None

    regular_objective, regular_risk, regular = nnpu_training_objective(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        class_prior=0.4,
    )
    assert regular["raw_negative_risk"].item() > 0.0
    assert regular["correction_active"].item() == 0.0
    assert regular_objective.item() == pytest.approx(regular_risk.item())

    checkpoint = tmp_path / "cellness.pt"
    save_cellness_checkpoint(
        checkpoint,
        model,
        patch_shape=(9, 33, 33),
        class_prior=0.4,
        detector_threshold_k=3.0,
        epoch=1,
        train_shards=["6bba_train"],
        dev_shards=["6bba_dev"],
        metrics={"dev_loss": float(loss.detach())},
        training_config={"seed": 0},
    )
    loaded, metadata = load_cellness_checkpoint(checkpoint)
    assert metadata["train_embryo"] == APPEARANCE_TRAIN_EMBRYO

    class FakeArray:
        volume = np.random.default_rng(11).integers(
            0,
            4096,
            size=(11, 40, 40),
            dtype=np.uint16,
        )
        shape = (1, *volume.shape)

        def __getitem__(self, timepoint):
            assert timepoint == 0
            return self.volume

    candidates = _graph([(1, 0, 0, 0, 0), (2, 0, 5, 20, 20)])
    scores = score_candidates(
        loaded,
        FakeArray(),
        candidates,
        patch_shape=(9, 33, 33),
        batch_size=2,
    )
    assert scores.shape == (2,)
    assert np.all((0.0 <= scores) & (scores <= 1.0))
    scalar_patches = np.stack(
        [
            normalize_patch(extract_patch(FakeArray.volume, center, (9, 33, 33)))
            for center in ((0, 0, 0), (5, 20, 20))
        ]
    )
    with torch.inference_mode():
        scalar_scores = torch.sigmoid(
            loaded(torch.from_numpy(scalar_patches).unsqueeze(1))
        ).numpy()
    np.testing.assert_allclose(scores, scalar_scores, rtol=0, atol=1e-6)
