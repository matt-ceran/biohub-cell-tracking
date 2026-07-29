from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from biohub.appearance import extract_patches, normalize_patches
from biohub.appearance_model import (
    CellnessCNN3D,
    CellnessModelConfig,
    save_cellness_checkpoint,
    score_candidates,
)
from biohub.division_data import (
    DivisionManifestRow,
    canonical_json_bytes,
    canonical_json_sha256,
    read_division_contracts,
    write_canonical_json,
    write_division_manifest,
)
from biohub.division_inference import PROBABILITY_THRESHOLDS
from biohub.division_model import (
    CACHE_BACKEND,
    CACHE_BACKEND_POLICY,
    CACHE_DEVICE,
    CACHE_INFERENCE_DTYPE,
    CACHE_TORCH_VERSION,
    CELLNESS_MATCH_ATOL,
    CONTEXT_CONTINUOUS_MASK,
    CONTEXT_FEATURE_KINDS,
    CONTEXT_FEATURE_NAMES,
    DIVISION_MODEL_NAMES,
    EXPECTED_BASE_CHECKPOINT_SHA256,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    EXPECTED_MODEL_STATE_SHA256,
    EXPECTED_SCORING_SOURCE_SHA256,
    FIXED_DIVISION_MODEL_CONFIGS,
    INTENSITY_SUMMARY_NAMES,
    DivisionModel,
    DivisionModelConfig,
    FeatureSchema,
    FrozenCellnessEncoder,
    build_symmetric_context_features,
    canonical_model_state_sha256,
    daughter_symmetric_features,
    load_division_checkpoint,
    raw_patch_intensity_summaries,
    read_division_node_cache,
    read_division_node_cache_metadata,
    save_division_checkpoint,
    validate_division_node_cache_cohort,
    write_division_node_cache,
)
from biohub.metric import TrackingGraph
from scripts import build_division_cache
from scripts.validate_division_model import (
    GRID_ROW_FIELDS,
    grid_payload,
    validation_source_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DATASET_MANIFEST_SHA256 = "a" * 64
CHECKPOINT_MODEL_INPUTS_SHA256 = "b" * 64
CHECKPOINT_TRAINER_SOURCE_SHA256 = "c" * 64

EXPECTED_CONTEXT_FEATURES = (
    ("mother_cellness", "continuous"),
    ("daughter_cellness_min", "continuous"),
    ("daughter_cellness_max", "continuous"),
    ("daughter_cellness_mean", "continuous"),
    ("daughter_cellness_abs_diff", "continuous"),
    ("daughter_distance_um_min", "continuous"),
    ("daughter_distance_um_max", "continuous"),
    ("daughter_distance_um_mean", "continuous"),
    ("daughter_distance_um_abs_diff", "continuous"),
    ("daughter_angle_deg", "continuous"),
    ("present_in_linked_graph_count", "count"),
    ("present_in_linked_graph_any", "binary"),
    ("present_in_linked_graph_both", "binary"),
    ("is_current_child_count", "count"),
    ("is_current_child_any", "binary"),
    ("is_current_child_both", "binary"),
    ("is_parentless_count", "count"),
    ("is_parentless_any", "binary"),
    ("is_parentless_both", "binary"),
    ("has_wrong_parent_count", "count"),
    ("has_wrong_parent_any", "binary"),
    ("has_wrong_parent_both", "binary"),
    ("has_future_count", "count"),
    ("has_future_any", "binary"),
    ("has_future_both", "binary"),
    ("survives_pruning_count", "count"),
    ("survives_pruning_any", "binary"),
    ("survives_pruning_both", "binary"),
    ("mother_has_history", "binary"),
    ("mother_has_current_child", "binary"),
    ("mother_survives_pruning", "binary"),
    ("restored_node_count", "count"),
    ("edge_removal_count", "count"),
    ("mother_intensity_mean", "continuous"),
    ("daughter_intensity_mean_min", "continuous"),
    ("daughter_intensity_mean_max", "continuous"),
    ("daughter_intensity_mean_mean", "continuous"),
    ("daughter_intensity_mean_abs_diff", "continuous"),
    ("mother_intensity_std", "continuous"),
    ("daughter_intensity_std_min", "continuous"),
    ("daughter_intensity_std_max", "continuous"),
    ("daughter_intensity_std_mean", "continuous"),
    ("daughter_intensity_std_abs_diff", "continuous"),
    ("mother_intensity_center_background_contrast", "continuous"),
    ("daughter_intensity_center_background_contrast_min", "continuous"),
    ("daughter_intensity_center_background_contrast_max", "continuous"),
    ("daughter_intensity_center_background_contrast_mean", "continuous"),
    ("daughter_intensity_center_background_contrast_abs_diff", "continuous"),
    ("mother_intensity_min", "continuous"),
    ("daughter_intensity_min_min", "continuous"),
    ("daughter_intensity_min_max", "continuous"),
    ("daughter_intensity_min_mean", "continuous"),
    ("daughter_intensity_min_abs_diff", "continuous"),
    ("mother_intensity_max", "continuous"),
    ("daughter_intensity_max_min", "continuous"),
    ("daughter_intensity_max_max", "continuous"),
    ("daughter_intensity_max_mean", "continuous"),
    ("daughter_intensity_max_abs_diff", "continuous"),
)


def _graph() -> TrackingGraph:
    return TrackingGraph(
        node_ids=np.array([7, 9, 12], dtype=np.int64),
        t=np.array([0, 0, 1], dtype=np.int64),
        z=np.array([2.0, 3.0, 4.0], dtype=np.float64),
        y=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        x=np.array([6.0, 7.0, 8.0], dtype=np.float64),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def _context_inputs() -> dict[str, object]:
    return {
        "mother_cellness": np.array([0.7, 0.8], dtype=np.float32),
        "daughter_cellness": np.array([[0.2, 0.9], [0.4, 0.6]], dtype=np.float32),
        "daughter_distance_um": np.array([[3.0, 7.0], [4.0, 5.0]], dtype=np.float32),
        "daughter_angle_deg": np.array([120.0, 90.0], dtype=np.float32),
        "candidate_flags": {
            "present_in_linked_graph": np.array([[0, 1], [1, 1]], dtype=np.uint8),
            "is_current_child": np.array([[0, 1], [1, 0]], dtype=np.uint8),
            "is_parentless": np.array([[0, 0], [0, 1]], dtype=np.uint8),
            "has_wrong_parent": np.array([[0, 0], [1, 0]], dtype=np.uint8),
            "has_future": np.array([[1, 0], [0, 1]], dtype=np.uint8),
            "survives_pruning": np.array([[0, 1], [1, 0]], dtype=np.uint8),
        },
        "mother_flags": {
            "mother_has_history": np.array([1, 0], dtype=np.uint8),
            "mother_has_current_child": np.array([1, 1], dtype=np.uint8),
            "mother_survives_pruning": np.array([0, 1], dtype=np.uint8),
        },
        "restored_node_count": np.array([2, 0], dtype=np.float32),
        "edge_removal_count": np.array([1, 2], dtype=np.float32),
        "mother_intensity_summaries": np.arange(10, dtype=np.float32).reshape(2, 5),
        "daughter_intensity_summaries": np.arange(20, dtype=np.float32).reshape(2, 2, 5),
    }


def _swap_daughters(values: dict[str, object]) -> dict[str, object]:
    swapped = dict(values)
    for name in ("daughter_cellness", "daughter_distance_um"):
        swapped[name] = np.asarray(values[name])[:, ::-1].copy()
    swapped["candidate_flags"] = {
        name: np.asarray(array)[:, ::-1].copy()
        for name, array in dict(values["candidate_flags"]).items()
    }
    swapped["daughter_intensity_summaries"] = np.asarray(values["daughter_intensity_summaries"])[
        :, ::-1
    ].copy()
    return swapped


def test_frozen_encoder_is_exact_and_remains_frozen_under_outer_training() -> None:
    torch.manual_seed(4)
    backbone = CellnessCNN3D(CellnessModelConfig(base_channels=4, dropout=0.3))
    before = {name: value.detach().clone() for name, value in backbone.state_dict().items()}
    encoder = FrozenCellnessEncoder(backbone)
    outer = nn.Sequential(encoder, nn.Linear(encoder.output_dim, 1))
    outer.train()
    inputs = torch.randn(3, 1, 5, 9, 9, requires_grad=True)

    embeddings, cellness = encoder.forward_with_cellness(inputs)
    with torch.no_grad():
        explicit = torch.flatten(
            backbone.pool(backbone.features(backbone.stem(inputs))),
            start_dim=1,
        )
        explicit_cellness = torch.sigmoid(backbone(inputs))

    assert encoder.output_dim == backbone.config.base_channels * 4
    assert torch.equal(embeddings, explicit)
    assert torch.equal(cellness, explicit_cellness)
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert not embeddings.requires_grad
    assert not cellness.requires_grad

    outer(inputs).sum().backward()
    assert inputs.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in outer[-1].parameters())
    assert all(torch.equal(before[name], value) for name, value in backbone.state_dict().items())


@pytest.mark.parametrize(
    "shape",
    [
        (2, 9, 33, 33),
        (2, 2, 9, 33, 33),
        (1, 1, 9, 33, 33, 1),
    ],
)
def test_frozen_encoder_rejects_non_b1zyx_inputs(shape: tuple[int, ...]) -> None:
    encoder = FrozenCellnessEncoder(CellnessCNN3D(CellnessModelConfig(base_channels=4)))

    with pytest.raises(ValueError, match=r"\[batch, 1, z, y, x\]"):
        encoder(torch.zeros(shape))


def test_exact_daughter_symmetric_features_and_fixed_model_ladder() -> None:
    small_mother = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    small_first = torch.tensor([[2.0, 8.0], [5.0, 9.0]])
    small_second = torch.tensor([[6.0, 4.0], [7.0, 1.0]])
    expected = torch.cat(
        [
            small_mother,
            (small_first + small_second) / 2.0,
            torch.abs(small_first - small_second),
            small_first * small_second,
        ],
        dim=1,
    )

    assert torch.equal(
        daughter_symmetric_features(small_mother, small_first, small_second), expected
    )
    assert torch.equal(
        daughter_symmetric_features(small_mother, small_first, small_second),
        daughter_symmetric_features(small_mother, small_second, small_first),
    )
    assert DIVISION_MODEL_NAMES == ("scalar", "embedding_linear", "embedding_mlp32")

    mother = torch.randn(2, 64)
    first = torch.randn(2, 64)
    second = torch.randn(2, 64)
    context = torch.randn(2, len(CONTEXT_FEATURE_NAMES))
    for name in DIVISION_MODEL_NAMES:
        torch.manual_seed(10)
        config = DivisionModelConfig(model_name=name)
        model = DivisionModel(config).eval()
        forward = model(mother, first, second, context)
        swapped = model(mother, second, first, context)
        assert torch.equal(forward, swapped)

    scalar = DivisionModel(DivisionModelConfig(model_name="scalar")).eval()
    assert torch.equal(
        scalar(mother, first, second, context),
        scalar(mother * 100, first * -20, second * 30, context),
    )

    mlp = DivisionModel(DivisionModelConfig(model_name="embedding_mlp32"))
    assert [type(layer) for layer in mlp.head] == [
        nn.LayerNorm,
        nn.Linear,
        nn.SiLU,
        nn.Dropout,
        nn.Linear,
    ]
    assert mlp.head[1].out_features == 32
    assert mlp.head[3].p == pytest.approx(0.1)
    assert mlp.config.hidden_dim == 32
    assert mlp.config.dropout == pytest.approx(0.1)


@pytest.mark.parametrize("model_name", DIVISION_MODEL_NAMES)
def test_cached_feature_forward_matches_embedding_forward_and_gradients(
    model_name: str,
) -> None:
    torch.manual_seed(23)
    model = DivisionModel(DivisionModelConfig(model_name)).eval()
    mother = torch.randn(3, 64, requires_grad=True)
    first = torch.randn(3, 64, requires_grad=True)
    second = torch.randn(3, 64, requires_grad=True)
    context = torch.randn(3, len(CONTEXT_FEATURE_NAMES), requires_grad=True)
    cached_image = daughter_symmetric_features(mother, first, second).detach().requires_grad_()

    direct = model(mother, first, second, context)
    cached = model.forward_cached_features(cached_image, context)

    assert torch.equal(direct, cached)
    direct.sum().backward()
    direct_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    direct_context_gradient = context.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    context.grad = None
    cached.sum().backward()

    assert torch.equal(context.grad, direct_context_gradient)
    assert all(
        torch.equal(parameter.grad, direct_parameter_gradients[name])
        for name, parameter in model.named_parameters()
    )
    if model_name == "scalar":
        assert cached_image.grad is None
    else:
        assert cached_image.grad is not None
        assert bool(torch.isfinite(cached_image.grad).all())


def test_cached_feature_forward_rejects_wrong_types_shapes_values_and_alignment() -> None:
    model = DivisionModel(DivisionModelConfig("embedding_linear"))
    image = torch.zeros(2, 256)
    context = torch.zeros(2, len(CONTEXT_FEATURE_NAMES))

    with pytest.raises(TypeError, match="torch tensors"):
        model.forward_cached_features(image.numpy(), context)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image features"):
        model.forward_cached_features(image[:, :-1], context)
    with pytest.raises(ValueError, match="normalized context"):
        model.forward_cached_features(image, context[:, :-1])
    with pytest.raises(ValueError, match="finite"):
        model.forward_cached_features(image.clone().fill_(float("nan")), context)
    with pytest.raises(ValueError, match="floating-point"):
        model.forward_cached_features(image.to(torch.int64), context.to(torch.int64))
    with pytest.raises(ValueError, match="dtype and device"):
        model.forward_cached_features(image.to(torch.float64), context)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": "other"},
        {"model_name": "scalar", "embedding_dim": 63},
        {"model_name": "scalar", "context_dim": len(CONTEXT_FEATURE_NAMES) - 1},
        {"model_name": "scalar", "hidden_dim": 32},
        {"model_name": "embedding_linear", "dropout": 0.1},
        {"model_name": "embedding_mlp32", "hidden_dim": 16},
        {"model_name": "embedding_mlp32", "dropout": 0.2},
    ],
)
def test_division_model_config_rejects_every_nonfixed_architecture(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DivisionModelConfig(**kwargs)


def test_symmetric_context_has_exact_frozen_order_types_and_swap_invariance() -> None:
    inputs = _context_inputs()
    observed = build_symmetric_context_features(**inputs)
    swapped = build_symmetric_context_features(**_swap_daughters(inputs))

    assert observed.shape == (2, len(CONTEXT_FEATURE_NAMES))
    assert observed.dtype == np.float32
    assert np.array_equal(observed, swapped)
    assert tuple(zip(CONTEXT_FEATURE_NAMES, CONTEXT_FEATURE_KINDS, strict=True)) == (
        EXPECTED_CONTEXT_FEATURES
    )
    assert CONTEXT_CONTINUOUS_MASK == tuple(
        kind == "continuous" for _, kind in EXPECTED_CONTEXT_FEATURES
    )
    assert not any("product" in name for name in CONTEXT_FEATURE_NAMES)

    row = dict(zip(CONTEXT_FEATURE_NAMES, observed[0], strict=True))
    assert row["daughter_cellness_min"] == pytest.approx(0.2)
    assert row["daughter_cellness_max"] == pytest.approx(0.9)
    assert row["daughter_cellness_mean"] == pytest.approx(0.55)
    assert row["daughter_cellness_abs_diff"] == pytest.approx(0.7)
    assert row["present_in_linked_graph_count"] == 1.0
    assert row["present_in_linked_graph_any"] == 1.0
    assert row["present_in_linked_graph_both"] == 0.0


def _fit_feature_schema() -> FeatureSchema:
    first = build_symmetric_context_features(**_context_inputs())
    second_inputs = _context_inputs()
    second_inputs["mother_cellness"] = np.array([0.2, 0.4], dtype=np.float32)
    second = build_symmetric_context_features(**second_inputs)
    return FeatureSchema.fit(
        CONTEXT_FEATURE_NAMES,
        np.concatenate([first, second]),
        kinds=CONTEXT_FEATURE_KINDS,
        source_role="fit",
    )


def test_feature_schema_is_fit_only_ordered_stable_and_standardizes() -> None:
    rows = build_symmetric_context_features(**_context_inputs())
    schema = FeatureSchema.fit(
        CONTEXT_FEATURE_NAMES,
        rows,
        kinds=CONTEXT_FEATURE_KINDS,
        source_role="fit",
    )
    rebuilt = FeatureSchema.from_dict(schema.to_dict())

    assert schema.names == CONTEXT_FEATURE_NAMES
    assert schema.kinds == CONTEXT_FEATURE_KINDS
    assert schema.continuous_mask == CONTEXT_CONTINUOUS_MASK
    assert schema.sha256 == rebuilt.sha256
    assert schema.sha256 == hashlib.sha256(schema.canonical_bytes()).hexdigest()
    standardized = schema.standardize(rows)
    assert np.all(np.isfinite(standardized))
    assert np.array_equal(
        standardized[:, np.logical_not(CONTEXT_CONTINUOUS_MASK)],
        rows[:, np.logical_not(CONTEXT_CONTINUOUS_MASK)],
    )
    assert all(
        mean == 0.0 and scale == 1.0
        for mean, scale, continuous in zip(
            schema.means,
            schema.scales,
            schema.continuous_mask,
            strict=True,
        )
        if not continuous
    )

    with pytest.raises(ValueError, match="fit role"):
        FeatureSchema.fit(
            CONTEXT_FEATURE_NAMES,
            rows,
            kinds=CONTEXT_FEATURE_KINDS,
            source_role="development",
        )
    with pytest.raises(ValueError, match="exact frozen context"):
        FeatureSchema.fit(
            CONTEXT_FEATURE_NAMES[::-1],
            rows[:, ::-1],
            kinds=CONTEXT_FEATURE_KINDS[::-1],
            source_role="fit",
        )
    changed = schema.to_dict()
    changed["continuous_mask"][0] = False
    with pytest.raises(ValueError, match="continuous mask"):
        FeatureSchema.from_dict(changed)


def test_division_checkpoint_saves_only_head_and_loads_without_touching_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"immutable external checkpoint")
    base_hash = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr("biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256", base_hash)
    schema = _fit_feature_schema()
    config = DivisionModelConfig(model_name="embedding_linear")
    model = DivisionModel(config)
    output = tmp_path / "division.pt"
    base_before = base_checkpoint.read_bytes()

    save_division_checkpoint(
        output,
        model,
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        feature_schema=schema,
        dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
        model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
        trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
        epoch=12,
        metrics={"loss": 0.25, "accuracy": 0.75},
    )

    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "base_checkpoint_sha256",
        "contract_sha256",
        "dataset_manifest_sha256",
        "division_checkpoint_schema_version",
        "epoch",
        "feature_schema_sha256",
        "head_state_dict",
        "metrics",
        "model_config",
        "model_inputs_sha256",
        "trainer_source_sha256",
    }
    assert "model_state_dict" not in payload
    assert all(not name.startswith("backbone") for name in payload["head_state_dict"])

    loaded, metadata = load_division_checkpoint(
        output,
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        feature_schema=schema,
        expected_dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
        expected_model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
        expected_trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
        expected_model_config=config,
    )
    assert metadata["epoch"] == 12
    assert loaded.config == config
    assert all(
        torch.equal(model.head.state_dict()[name], loaded.head.state_dict()[name])
        for name in model.head.state_dict()
    )
    assert base_checkpoint.read_bytes() == base_before


def test_division_checkpoint_save_rejects_a_replaced_nonfixed_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    monkeypatch.setattr(
        "biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256",
        hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
    )
    model = DivisionModel(DivisionModelConfig("embedding_mlp32"))
    model.head = nn.Linear(64 * 4 + len(CONTEXT_FEATURE_NAMES), 1)

    with pytest.raises(ValueError, match="fixed architecture"):
        save_division_checkpoint(
            tmp_path / "invalid.pt",
            model,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            feature_schema=_fit_feature_schema(),
            dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
            model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
            trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
            epoch=1,
            metrics={"loss": 1.0},
        )


def test_division_checkpoint_rejects_nonfinite_head_without_touching_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    monkeypatch.setattr(
        "biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256",
        hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
    )
    model = DivisionModel(DivisionModelConfig("scalar"))
    with torch.no_grad():
        next(model.head.parameters()).fill_(float("inf"))
    destination = tmp_path / "division.pt"
    destination.write_bytes(b"known-good-destination")

    with pytest.raises(ValueError, match="non-finite"):
        save_division_checkpoint(
            destination,
            model,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            feature_schema=_fit_feature_schema(),
            dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
            model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
            trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
            epoch=1,
            metrics={"loss": 1.0},
        )

    assert destination.read_bytes() == b"known-good-destination"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base.pt", "division.pt"]


def test_failed_atomic_torch_save_preserves_destination_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    monkeypatch.setattr(
        "biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256",
        hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
    )
    destination = tmp_path / "division.pt"
    destination.write_bytes(b"known-good-destination")

    def fail_after_partial_write(payload, handle) -> None:
        handle.write(b"partial")
        raise OSError("injected torch save failure")

    monkeypatch.setattr("biohub.division_model.torch.save", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        save_division_checkpoint(
            destination,
            DivisionModel(DivisionModelConfig("scalar")),
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            feature_schema=_fit_feature_schema(),
            dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
            model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
            trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
            epoch=1,
            metrics={"loss": 1.0},
        )

    assert destination.read_bytes() == b"known-good-destination"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base.pt", "division.pt"]


@pytest.mark.parametrize(
    "field",
    [
        "base_checkpoint_sha256",
        "split",
        "proposal",
        "sampling",
        "acceptance",
        "assignment",
        "dataset_manifest_sha256",
        "feature_schema_sha256",
        "model_inputs_sha256",
        "model_config",
        "trainer_source_sha256",
        "epoch",
        "legacy_v1",
        "metrics",
    ],
)
def test_division_checkpoint_loader_rejects_every_identity_mismatch(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    base_hash = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr("biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256", base_hash)
    schema = _fit_feature_schema()
    config = DivisionModelConfig("scalar")
    valid = tmp_path / "valid.pt"
    save_division_checkpoint(
        valid,
        DivisionModel(config),
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        feature_schema=schema,
        dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
        model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
        trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
        epoch=2,
        metrics={"loss": 1.0},
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    if field == "legacy_v1":
        payload["division_checkpoint_schema_version"] = 1
        payload["source_sha256"] = payload.pop("trainer_source_sha256")
        payload.pop("dataset_manifest_sha256")
        payload.pop("model_inputs_sha256")
    elif field in EXPECTED_DIVISION_CONTRACT_SHA256:
        payload["contract_sha256"][field] = "f" * 64
    elif field == "model_config":
        payload[field]["dropout"] = 0.5
    elif field == "epoch":
        payload[field] = -1
    elif field == "metrics":
        payload[field] = {"loss": float("nan")}
    else:
        payload[field] = "f" * 64
    changed = tmp_path / f"wrong-{field}.pt"
    torch.save(payload, changed)
    before = base_checkpoint.read_bytes()

    with pytest.raises(ValueError, match=field.replace("_", " ") + "|checkpoint|contract|schema"):
        load_division_checkpoint(
            changed,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            feature_schema=schema,
            expected_dataset_manifest_sha256=CHECKPOINT_DATASET_MANIFEST_SHA256,
            expected_model_inputs_sha256=CHECKPOINT_MODEL_INPUTS_SHA256,
            expected_trainer_source_sha256=CHECKPOINT_TRAINER_SOURCE_SHA256,
            expected_model_config=config,
        )
    assert base_checkpoint.read_bytes() == before


def _write_valid_cache(path: Path) -> str:
    return write_division_node_cache(
        path,
        _graph(),
        np.array([0.1, 0.5, 0.9], dtype=np.float32),
        np.arange(3 * 64, dtype=np.float32).reshape(3, 64),
        np.array(
            [
                [5.0, 1.0, 2.0, 1.0, 8.0],
                [6.0, 2.0, 1.0, 0.0, 9.0],
                [7.0, 3.0, -1.0, 2.0, 10.0],
            ],
            dtype=np.float32,
        ),
        dataset="6bba_synthetic",
        role="fit",
        checkpoint_sha256=EXPECTED_BASE_CHECKPOINT_SHA256,
        scoring_source_sha256=EXPECTED_SCORING_SOURCE_SHA256,
        cache_source_sha256="c" * 64,
        contract_sha256=EXPECTED_DIVISION_CONTRACT_SHA256,
        image_sha256="1" * 64,
        upstream_manifest_sha256="d" * 64,
        model_state_sha256=EXPECTED_MODEL_STATE_SHA256,
        phase10_score_cache_sha256="e" * 64,
        phase10_max_cellness_abs_diff=0.0,
    )


def _rewrite_cache_metadata(path: Path, **changes: object) -> None:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(changes)
    arrays["metadata_json"] = np.array(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_node_cache_round_trip_has_strict_arrays_provenance_and_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.npz"
    content_hash = _write_valid_cache(path)

    with monkeypatch.context() as context:
        context.setattr(
            "biohub.division_model.np.load",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("metadata preflight must not materialize NPZ payload arrays")
            ),
        )
        preflight_metadata, cache_file_sha256 = read_division_node_cache_metadata(
            path,
            expected_dataset="6bba_synthetic",
            expected_role="fit",
            expected_cache_source_sha256="c" * 64,
            expected_contract_sha256=EXPECTED_DIVISION_CONTRACT_SHA256,
            expected_upstream_manifest_sha256="d" * 64,
        )

    graph, cellness, embeddings, intensity, metadata = read_division_node_cache(
        path,
        expected_dataset="6bba_synthetic",
        expected_role="fit",
        expected_cache_source_sha256="c" * 64,
        expected_contract_sha256=EXPECTED_DIVISION_CONTRACT_SHA256,
        expected_upstream_manifest_sha256="d" * 64,
    )

    assert np.array_equal(graph.node_ids, _graph().node_ids)
    assert preflight_metadata == metadata
    assert cache_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(cellness, np.array([0.1, 0.5, 0.9], dtype=np.float32))
    assert embeddings.shape == (3, 64) and embeddings.dtype == np.float32
    assert intensity.shape == (3, 5) and intensity.dtype == np.float32
    assert metadata["content_sha256"] == content_hash
    assert metadata["detector_threshold_k"] == 3.0
    assert metadata["patch_shape"] == [9, 33, 33]
    assert metadata["normalization"] == "per_patch_mean_std_clip_6"
    assert metadata["dataset"] == "6bba_synthetic"
    assert metadata["role"] == "fit"
    assert metadata["embedding_dim"] == 64
    assert metadata["intensity_summary_names"] == list(INTENSITY_SUMMARY_NAMES)
    assert metadata["backend"] == CACHE_BACKEND
    assert metadata["backend_policy"] == CACHE_BACKEND_POLICY
    assert metadata["device"] == CACHE_DEVICE
    assert metadata["inference_dtype"] == CACHE_INFERENCE_DTYPE
    assert metadata["torch_version"] == CACHE_TORCH_VERSION
    assert metadata["model_state_sha256"] == EXPECTED_MODEL_STATE_SHA256
    assert metadata["image_sha256"] == "1" * 64
    assert metadata["upstream_manifest_sha256"] == "d" * 64
    assert metadata["cellness_match_atol"] == CELLNESS_MATCH_ATOL == 2e-7


def test_failed_atomic_npz_save_preserves_destination_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cache.npz"
    _write_valid_cache(destination)
    before = destination.read_bytes()

    def fail_after_partial_write(handle, **arrays) -> None:
        handle.write(b"partial")
        raise OSError("injected npz save failure")

    monkeypatch.setattr("biohub.division_model.np.savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        _write_valid_cache(destination)

    assert destination.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == ["cache.npz"]


def test_cache_reader_rejects_wrong_image_identity_and_mixed_backend_cohort(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_valid_cache(first)
    _write_valid_cache(second)

    with pytest.raises(ValueError, match="image SHA-256"):
        read_division_node_cache(first, expected_image_sha256="2" * 64)

    validate_division_node_cache_cohort([first, second])
    _rewrite_cache_metadata(second, backend_policy="phase11-cpu-forbidden")
    with pytest.raises(ValueError, match="backend policy"):
        validate_division_node_cache_cohort([first, second])


def test_node_cache_metadata_preflight_rejects_header_and_file_hash_drift(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_valid_cache(valid)
    with np.load(valid, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["embeddings"] = arrays["embeddings"][:, :-1]
    changed = tmp_path / "wrong-width.npz"
    with changed.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ValueError, match="embeddings header"):
        read_division_node_cache_metadata(changed)
    with pytest.raises(ValueError, match="file SHA-256"):
        read_division_node_cache_metadata(valid, expected_file_sha256="f" * 64)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_ids",
        "fractional_time",
        "nonfinite_cellness",
        "embedding_dtype",
        "embedding_width",
        "nonfinite_intensity",
        "detector",
        "patch_shape",
        "normalization",
        "dataset",
        "role",
        "checkpoint",
        "model_state",
        "image",
        "scoring_source",
        "cache_source",
        "upstream_manifest",
        "legacy_manifest_field",
        "backend_policy",
        "backend",
        "device",
        "torch_version",
        "inference_dtype",
        "contract",
        "content_hash",
        "extra_array",
    ],
)
def test_node_cache_rejects_every_array_and_provenance_corruption(
    corruption: str,
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_valid_cache(valid)
    with np.load(valid, allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))

    if corruption == "duplicate_ids":
        arrays["node_ids"][1] = arrays["node_ids"][0]
    elif corruption == "fractional_time":
        arrays["tzyx"][0, 0] = 0.5
    elif corruption == "nonfinite_cellness":
        arrays["cellness"][0] = np.nan
    elif corruption == "embedding_dtype":
        arrays["embeddings"] = arrays["embeddings"].astype(np.float64)
    elif corruption == "embedding_width":
        arrays["embeddings"] = arrays["embeddings"][:, :-1]
    elif corruption == "nonfinite_intensity":
        arrays["intensity_summaries"][0, 0] = np.inf
    elif corruption == "detector":
        metadata["detector_threshold_k"] = 4.0
    elif corruption == "patch_shape":
        metadata["patch_shape"] = [7, 31, 31]
    elif corruption == "normalization":
        metadata["normalization"] = "other"
    elif corruption == "dataset":
        metadata["dataset"] = "44b6_forbidden"
    elif corruption == "role":
        metadata["role"] = "unknown"
    elif corruption == "checkpoint":
        metadata["checkpoint_sha256"] = "f" * 64
    elif corruption == "model_state":
        metadata["model_state_sha256"] = "f" * 64
    elif corruption == "image":
        metadata["image_sha256"] = "f" * 64
    elif corruption == "scoring_source":
        metadata["scoring_source_sha256"] = "f" * 64
    elif corruption == "cache_source":
        metadata["cache_source_sha256"] = "f" * 64
    elif corruption == "upstream_manifest":
        metadata["upstream_manifest_sha256"] = "f" * 64
    elif corruption == "legacy_manifest_field":
        metadata["manifest_sha256"] = metadata.pop("upstream_manifest_sha256")
    elif corruption == "backend_policy":
        metadata["backend_policy"] = "other"
    elif corruption == "backend":
        metadata["backend"] = "pytorch_cpu"
    elif corruption == "device":
        metadata["device"] = "cpu"
    elif corruption == "torch_version":
        metadata["torch_version"] = "0.0"
    elif corruption == "inference_dtype":
        metadata["inference_dtype"] = "float64"
    elif corruption == "contract":
        metadata["contract_sha256"]["sampling"] = "f" * 64
    elif corruption == "content_hash":
        metadata["content_sha256"] = "f" * 64
    elif corruption == "extra_array":
        arrays["unexpected"] = np.array(1)
    arrays["metadata_json"] = np.array(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    changed = tmp_path / f"{corruption}.npz"
    with changed.open("wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(ValueError):
        read_division_node_cache(
            changed,
            expected_dataset="6bba_synthetic",
            expected_role="fit",
            expected_cache_source_sha256="c" * 64,
            expected_contract_sha256=EXPECTED_DIVISION_CONTRACT_SHA256,
            expected_upstream_manifest_sha256="d" * 64,
        )


def test_raw_intensity_summary_has_the_exact_order_and_float32_dtype() -> None:
    patches = np.arange(2 * 3 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3, 3)
    summaries = raw_patch_intensity_summaries(patches)
    flattened = patches.astype(np.float64).reshape(2, -1)
    centers = patches[:, 1, 1, 1].astype(np.float64)
    backgrounds = (flattened.sum(axis=1) - centers) / (flattened.shape[1] - 1)
    expected = np.column_stack(
        [
            flattened.mean(axis=1),
            flattened.std(axis=1),
            centers - backgrounds,
            flattened.min(axis=1),
            flattened.max(axis=1),
        ]
    ).astype(np.float32)

    assert INTENSITY_SUMMARY_NAMES == (
        "mean",
        "std",
        "center_background_contrast",
        "min",
        "max",
    )
    assert summaries.dtype == np.float32
    assert np.array_equal(summaries, expected)


def test_one_pass_candidate_features_match_standard_cellness_scoring_exactly() -> None:
    torch.manual_seed(8)
    image = np.arange(2 * 7 * 13 * 13, dtype=np.uint16).reshape(2, 7, 13, 13)
    candidates = TrackingGraph(
        node_ids=np.array([0, 1, 2], dtype=np.int64),
        t=np.array([0, 0, 1], dtype=np.int64),
        z=np.array([2.0, 4.0, 3.0]),
        y=np.array([4.0, 8.0, 6.0]),
        x=np.array([5.0, 7.0, 6.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    model = CellnessCNN3D(CellnessModelConfig(base_channels=4, dropout=0.2)).eval()
    progress: list[tuple[int, int, int, int]] = []

    embeddings, cellness, intensity, image_hash = build_division_cache.compute_candidate_features(
        model,
        image,
        candidates,
        patch_shape=(5, 9, 9),
        batch_size=2,
        device="cpu",
        progress_callback=lambda *values: progress.append(values),
    )
    standard = score_candidates(
        model,
        image,
        candidates,
        patch_shape=(5, 9, 9),
        batch_size=2,
        device="cpu",
    )

    assert embeddings.shape == (3, 16)
    assert embeddings.dtype == np.float32
    assert intensity.shape == (3, 5)
    assert len(image_hash) == 64
    assert np.array_equal(cellness, standard)
    assert float(np.max(np.abs(cellness - standard))) <= CELLNESS_MATCH_ATOL
    assert progress == [(2, 2, 3, 3)]


def test_padded_frame_patch_gather_is_byte_equivalent_to_frozen_extraction() -> None:
    volume = np.arange(5 * 11 * 13, dtype=np.uint16).reshape(5, 11, 13)
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 10.0, 12.0],
            [2.0, 5.0, 6.0],
            [1.6, 3.7, 8.4],
        ]
    )
    shape = (3, 7, 7)

    padded = build_division_cache.pad_frame_for_patches(volume, shape)
    observed = build_division_cache.gather_patches_from_padded_frame(
        padded,
        centers,
        original_shape=volume.shape,
        patch_shape=shape,
    )
    expected = extract_patches(volume, centers, shape)

    assert np.array_equal(observed, expected)
    assert observed.tobytes() == expected.tobytes()
    assert np.array_equal(normalize_patches(observed), normalize_patches(expected))


def test_candidate_feature_extraction_pads_once_per_candidate_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(3)
    image = np.arange(3 * 7 * 13 * 13, dtype=np.uint16).reshape(3, 7, 13, 13)
    candidates = TrackingGraph(
        node_ids=np.arange(6, dtype=np.int64),
        t=np.array([0, 0, 0, 2, 2, 2], dtype=np.int64),
        z=np.array([1, 2, 3, 2, 3, 4], dtype=float),
        y=np.array([2, 4, 6, 3, 5, 7], dtype=float),
        x=np.array([3, 5, 7, 4, 6, 8], dtype=float),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    calls = 0
    original = build_division_cache.pad_frame_for_patches

    def counted(volume, patch_shape):
        nonlocal calls
        calls += 1
        return original(volume, patch_shape)

    monkeypatch.setattr(build_division_cache, "pad_frame_for_patches", counted)
    build_division_cache.compute_candidate_features(
        CellnessCNN3D(CellnessModelConfig(base_channels=4)).eval(),
        image,
        candidates,
        patch_shape=(5, 9, 9),
        batch_size=1,
        device="cpu",
    )

    assert calls == 2


def test_image_hash_detects_swapped_frames_and_corrupt_voxels() -> None:
    image = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
    swapped = image[[1, 0, 2]].copy()
    corrupt = image.copy()
    corrupt[2, 3, 4, 5] += 1

    original_hash = build_division_cache.canonical_image_sha256(image)

    assert build_division_cache.canonical_image_sha256(swapped) != original_hash
    assert build_division_cache.canonical_image_sha256(corrupt) != original_hash


def test_phase10_comparison_accepts_frozen_cross_backend_tolerance(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npz"
    reference_path.write_bytes(b"reference")
    graph = _graph()
    reference_scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    within = reference_scores.copy()
    within[0] = np.float32(within[0] + 1.1920929e-7)
    outside = reference_scores.copy()
    outside[0] = np.float32(outside[0] + 2.3841858e-7)
    reference = (reference_path, graph, reference_scores)

    _, difference = build_division_cache._compare_with_phase10(graph, within, reference)
    assert difference == pytest.approx(1.1920929e-7)
    with pytest.raises(ValueError, match="atol"):
        build_division_cache._compare_with_phase10(graph, outside, reference)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS is unavailable")
def test_cpu_and_mps_cellness_stay_within_frozen_tolerance() -> None:
    torch.manual_seed(11)
    image = np.arange(2 * 7 * 13 * 13, dtype=np.uint16).reshape(2, 7, 13, 13)
    candidates = TrackingGraph(
        node_ids=np.array([0, 1, 2], dtype=np.int64),
        t=np.array([0, 0, 1], dtype=np.int64),
        z=np.array([2.0, 4.0, 3.0]),
        y=np.array([4.0, 8.0, 6.0]),
        x=np.array([5.0, 7.0, 6.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    cpu_model = CellnessCNN3D(CellnessModelConfig(base_channels=4, dropout=0.2)).eval()
    mps_model = CellnessCNN3D(cpu_model.config).eval()
    mps_model.load_state_dict(cpu_model.state_dict())

    _, cpu_scores, _, cpu_image_hash = build_division_cache.compute_candidate_features(
        cpu_model,
        image,
        candidates,
        patch_shape=(5, 9, 9),
        batch_size=2,
        device="cpu",
    )
    _, mps_scores, _, mps_image_hash = build_division_cache.compute_candidate_features(
        mps_model,
        image,
        candidates,
        patch_shape=(5, 9, 9),
        batch_size=2,
        device="mps",
    )

    assert cpu_image_hash == mps_image_hash
    assert float(np.max(np.abs(cpu_scores - mps_scores))) <= CELLNESS_MATCH_ATOL


def test_trusted_model_loader_rejects_altered_checkpoint_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    valid = tmp_path / "synthetic-valid.pt"
    save_cellness_checkpoint(
        valid,
        CellnessCNN3D(CellnessModelConfig(dropout=0.0)),
        patch_shape=(9, 33, 33),
        class_prior=0.4,
        detector_threshold_k=3.0,
        epoch=1,
        train_shards=["6bba_synthetic_train"],
        dev_shards=["6bba_synthetic_dev"],
        metrics={"dev_loss": 0.25},
        training_config={"seed": 0},
    )
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    trusted_state_hash = canonical_model_state_sha256(payload["model_state_dict"])
    first_name = next(iter(payload["model_state_dict"]))
    payload["model_state_dict"][first_name] = payload["model_state_dict"][first_name].clone()
    payload["model_state_dict"][first_name].view(-1)[0] += 1.0
    altered = tmp_path / "altered.pt"
    torch.save(payload, altered)
    monkeypatch.setattr(
        build_division_cache,
        "EXPECTED_BASE_CHECKPOINT_SHA256",
        hashlib.sha256(altered.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        build_division_cache,
        "EXPECTED_MODEL_STATE_SHA256",
        trusted_state_hash,
    )

    with pytest.raises(ValueError, match="model-state fingerprint"):
        build_division_cache.load_trusted_cellness_model(altered, device="cpu")


def test_canonical_model_state_fingerprint_is_order_independent_and_content_sensitive() -> None:
    state = {
        "stem.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "head.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    reordered = dict(reversed(tuple(state.items())))
    altered = {name: value.clone() for name, value in state.items()}
    altered["stem.weight"][0, 0] += 1.0

    fingerprint = canonical_model_state_sha256(state)

    assert len(fingerprint) == 64
    assert canonical_model_state_sha256(reordered) == fingerprint
    assert canonical_model_state_sha256(altered) != fingerprint


def test_local_artifact_audit_accepted_checkpoint_model_state_fingerprint() -> None:
    checkpoint_path = (
        ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn_phase10_full.pt"
    )
    if not checkpoint_path.is_file():
        pytest.skip("ignored accepted Phase 10 checkpoint is not available locally")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert canonical_model_state_sha256(checkpoint["model_state_dict"]) == (
        EXPECTED_MODEL_STATE_SHA256
    )


def test_cli_has_no_unrestricted_all_and_verifies_contracts_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = build_division_cache.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--all"])
    assert parser.parse_args(["--dataset", "6bba_synthetic"]).device == "mps"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "6bba_synthetic", "--device", "cpu"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "6bba_synthetic", "--manifest", "legacy.csv"])

    image_accessed = False

    def reject_contracts(*args, **kwargs):
        raise ValueError("frozen bundle failed")

    def unexpected_discovery(*args, **kwargs):
        nonlocal image_accessed
        image_accessed = True
        raise AssertionError("dataset discovery must follow contract verification")

    monkeypatch.setattr(build_division_cache, "verify_division_contract_bundle", reject_contracts)
    monkeypatch.setattr(build_division_cache, "discover_split", unexpected_discovery)

    with pytest.raises(SystemExit):
        build_division_cache.main(
            [
                "--dataset",
                "6bba_synthetic",
                "--data-root",
                str(tmp_path),
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--upstream-manifest",
                str(tmp_path / "manifest.csv"),
                "--summary",
                str(tmp_path / "summary.json"),
            ]
        )
    assert not image_accessed


def _patch_cache_cli_contracts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    contracts: dict[str, dict[str, object]],
    manifest_rows: tuple[DivisionManifestRow, ...] | list[DivisionManifestRow],
) -> None:
    monkeypatch.setattr(
        build_division_cache,
        "require_approved_cache_backend",
        lambda device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        build_division_cache,
        "verify_division_contract_bundle",
        lambda *args, **kwargs: contracts,
    )
    monkeypatch.setattr(
        build_division_cache,
        "current_division_contract_hashes",
        lambda *args, **kwargs: EXPECTED_DIVISION_CONTRACT_SHA256,
    )
    monkeypatch.setattr(
        build_division_cache,
        "read_division_manifest",
        lambda *args, **kwargs: tuple(manifest_rows),
    )


def _forbid_cache_cli_calls(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_policy: bool = True,
) -> list[str]:
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called")

        return fail

    names = ["discover_split", "open_image_array", "build_movie_cache"]
    if include_policy:
        names.insert(0, "verify_evaluation_unlock_policy")
    for name in names:
        monkeypatch.setattr(build_division_cache, name, forbidden(name))
    return calls


def _policy_cli_args(
    tmp_path: Path,
    *,
    confirm: bool,
    policy: bool,
    policy_hash: bool,
) -> list[str]:
    args: list[str] = []
    if confirm:
        args.append("--confirm-evaluation")
    if policy:
        args.extend(("--policy", str(tmp_path / "policy.json")))
    if policy_hash:
        args.extend(("--policy-sha256", "a" * 64))
    return args


_POLICY_FLAG_CASES = (
    pytest.param(False, False, False, id="none"),
    pytest.param(True, False, False, id="confirm-only"),
    pytest.param(False, True, False, id="policy-only"),
    pytest.param(False, False, True, id="hash-only"),
    pytest.param(True, True, False, id="missing-hash"),
    pytest.param(True, False, True, id="missing-policy"),
    pytest.param(False, True, True, id="missing-confirm"),
    pytest.param(True, True, True, id="all"),
)


@pytest.mark.parametrize("selection", ["full-ordered", "subset", "reordered", "mixed"])
def test_cli_rejects_every_explicit_evaluation_selection_before_access_or_output(
    selection: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _development_contracts()
    manifest_rows = _manifest_rows()
    evaluation = list(contracts["split"]["role_datasets"]["evaluation"])
    assert len(evaluation) == 20
    if selection == "full-ordered":
        datasets = evaluation
    elif selection == "subset":
        datasets = evaluation[:-1]
    elif selection == "reordered":
        datasets = list(reversed(evaluation))
    else:
        datasets = [*evaluation[:-1], contracts["split"]["role_datasets"]["fit"][0]]
    output_dir = tmp_path / "cache"

    _patch_cache_cli_contracts(
        monkeypatch,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    forbidden_calls = _forbid_cache_cli_calls(monkeypatch)
    dataset_args = [value for dataset in datasets for value in ("--dataset", dataset)]

    with pytest.raises(SystemExit):
        build_division_cache.main(
            [
                *dataset_args,
                *_policy_cli_args(
                    tmp_path,
                    confirm=True,
                    policy=True,
                    policy_hash=True,
                ),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert forbidden_calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize("cohort_error", ["missing", "reordered"])
def test_evaluation_role_rejects_inexact_full_cohort_before_policy(
    cohort_error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _development_contracts()
    manifest_rows = list(_manifest_rows())
    evaluation_indices = [
        index for index, row in enumerate(manifest_rows) if row.role == "evaluation"
    ]
    if cohort_error == "missing":
        manifest_rows.pop(evaluation_indices[-1])
    elif cohort_error == "reordered":
        first, second = evaluation_indices[:2]
        manifest_rows[first], manifest_rows[second] = manifest_rows[second], manifest_rows[first]
    output_dir = tmp_path / "cache"

    _patch_cache_cli_contracts(
        monkeypatch,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    forbidden_calls = _forbid_cache_cli_calls(monkeypatch)

    with pytest.raises(SystemExit):
        build_division_cache.main(
            [
                "--role",
                "evaluation",
                "--confirm-evaluation",
                "--policy",
                str(tmp_path / "policy.json"),
                "--policy-sha256",
                "a" * 64,
                "--output-dir",
                str(output_dir),
            ]
        )

    assert forbidden_calls == []
    assert not output_dir.exists()


def test_exact_evaluation_role_verifies_policy_before_dataset_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _development_contracts()
    manifest_rows = _manifest_rows()
    policy_path = tmp_path / "policy.json"
    policy_hash = "a" * 64
    config_dir = tmp_path / "configs"
    data_root = tmp_path / "raw"
    output_dir = tmp_path / "cache"
    events: list[str] = []
    policy_call: dict[str, object] = {}
    assert tuple(row.dataset for row in manifest_rows if row.role == "evaluation") == tuple(
        contracts["split"]["role_datasets"]["evaluation"]
    )

    _patch_cache_cli_contracts(
        monkeypatch,
        contracts=contracts,
        manifest_rows=manifest_rows,
    )
    forbidden_calls = _forbid_cache_cli_calls(monkeypatch, include_policy=False)

    def verify_policy(path: Path, **kwargs) -> dict[str, object]:
        events.append("policy")
        policy_call.update({"path": path, **kwargs})
        return {}

    def stop_at_discovery(root: Path, split: str):
        events.append("discovery")
        assert root == data_root
        assert split == "train"
        raise RuntimeError("dataset discovery reached")

    monkeypatch.setattr(build_division_cache, "verify_evaluation_unlock_policy", verify_policy)
    monkeypatch.setattr(build_division_cache, "discover_split", stop_at_discovery)

    with pytest.raises(RuntimeError, match="dataset discovery reached"):
        build_division_cache.main(
            [
                "--role",
                "evaluation",
                "--confirm-evaluation",
                "--policy",
                str(policy_path),
                "--policy-sha256",
                policy_hash,
                "--config-dir",
                str(config_dir),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert events == ["policy", "discovery"]
    assert policy_call == {
        "path": policy_path,
        "expected_policy_sha256": policy_hash,
        "config_dir": config_dir,
        "manifest_rows": manifest_rows,
        "project_root": ROOT,
    }
    assert forbidden_calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("confirm", "policy", "policy_hash"),
    _POLICY_FLAG_CASES[:-1],
)
def test_evaluation_role_requires_complete_policy_flag_set_before_access(
    confirm: bool,
    policy: bool,
    policy_hash: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _development_contracts()
    output_dir = tmp_path / "cache"
    _patch_cache_cli_contracts(
        monkeypatch,
        contracts=contracts,
        manifest_rows=_manifest_rows(),
    )
    forbidden_calls = _forbid_cache_cli_calls(monkeypatch)

    with pytest.raises(SystemExit):
        build_division_cache.main(
            [
                "--role",
                "evaluation",
                *_policy_cli_args(
                    tmp_path,
                    confirm=confirm,
                    policy=policy,
                    policy_hash=policy_hash,
                ),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert forbidden_calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("confirm", "policy", "policy_hash"),
    _POLICY_FLAG_CASES[1:],
)
def test_non_evaluation_role_rejects_every_policy_flag_set_before_access(
    confirm: bool,
    policy: bool,
    policy_hash: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _development_contracts()
    output_dir = tmp_path / "cache"
    _patch_cache_cli_contracts(
        monkeypatch,
        contracts=contracts,
        manifest_rows=_manifest_rows(),
    )
    forbidden_calls = _forbid_cache_cli_calls(monkeypatch)

    with pytest.raises(SystemExit):
        build_division_cache.main(
            [
                "--role",
                "development",
                *_policy_cli_args(
                    tmp_path,
                    confirm=confirm,
                    policy=policy,
                    policy_hash=policy_hash,
                ),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert forbidden_calls == []
    assert not output_dir.exists()


def test_cli_fails_clearly_when_approved_mps_backend_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="approved MPS backend is unavailable"):
        build_division_cache.require_approved_cache_backend("mps")


def test_cache_builder_lock_refuses_active_builder_and_reuses_stale_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "movie.npz"
    lock_path = build_division_cache.cache_lock_path(output)

    with build_division_cache.exclusive_cache_builder_lock(output):
        with pytest.raises(RuntimeError, match="already active"):
            with build_division_cache.exclusive_cache_builder_lock(output):
                pass

    assert lock_path.is_file()
    lock_path.write_text("stale but unlocked")
    with build_division_cache.exclusive_cache_builder_lock(output):
        pass


def test_build_movie_cache_holds_lock_through_locked_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = build_division_cache.SamplePaths(
        dataset="6bba_synthetic",
        image=tmp_path / "image.zarr",
        labels=None,
        embryo="6bba",
    )
    output = tmp_path / "6bba_synthetic.npz"

    monkeypatch.setattr(
        build_division_cache,
        "require_approved_cache_backend",
        lambda device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        build_division_cache,
        "load_trusted_cellness_model",
        lambda checkpoint_path, device: (object(), {}),
    )

    def locked_build(*args, **kwargs):
        with pytest.raises(RuntimeError, match="already active"):
            with build_division_cache.exclusive_cache_builder_lock(output):
                pass
        output.write_bytes(b"promoted")
        return build_division_cache.CacheBuildResult(
            dataset=sample.dataset,
            role="fit",
            path=output,
            candidate_count=1,
            content_sha256="a" * 64,
            file_sha256="b" * 64,
            max_cellness_abs_diff=0.0,
            skipped=False,
        )

    monkeypatch.setattr(build_division_cache, "_build_movie_cache_locked", locked_build)

    result = build_division_cache.build_movie_cache(
        sample,
        role="fit",
        output_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        batch_size=1,
        device="mps",
        cache_source_sha256="c" * 64,
        contract_sha256={},
        upstream_manifest_sha256="e" * 64,
        phase10_score_cache_dir=tmp_path,
    )

    assert result.path == output
    assert output.read_bytes() == b"promoted"


def test_direct_evaluation_cache_api_fails_before_backend_or_image_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = build_division_cache.SamplePaths(
        dataset="6bba_synthetic",
        image=tmp_path / "never-opened.zarr",
        labels=None,
        embryo="6bba",
    )
    calls: list[str] = []

    def unexpected_backend(device):
        calls.append(str(device))
        raise AssertionError("backend must not be consulted")

    monkeypatch.setattr(
        build_division_cache,
        "require_approved_cache_backend",
        unexpected_backend,
    )

    with pytest.raises(PermissionError, match="strict development replay"):
        build_division_cache.build_movie_cache(
            sample,
            role="evaluation",
            output_dir=tmp_path,
            checkpoint_path=tmp_path / "checkpoint.pt",
            batch_size=1,
            device="mps",
            cache_source_sha256="a" * 64,
            contract_sha256=EXPECTED_DIVISION_CONTRACT_SHA256,
            upstream_manifest_sha256="b" * 64,
            phase10_score_cache_dir=tmp_path,
        )

    assert calls == []
    assert not (tmp_path / "6bba_synthetic.npz").exists()


def test_evaluation_authorization_cannot_be_forged() -> None:
    with pytest.raises(PermissionError, match="strict policy verification"):
        build_division_cache._EvaluationAuthorization(
            evaluation_datasets=tuple(f"6bba_{index:08x}" for index in range(20)),
            upstream_manifest_sha256="a" * 64,
            policy_sha256="b" * 64,
            cache_source_sha256="c" * 64,
            _key=object(),
        )


def test_policy_artifacts_reject_absolute_paths_even_inside_project(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"artifact")

    with pytest.raises(ValueError, match="project-relative"):
        build_division_cache._require_policy_artifact(
            {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            field="test artifact",
            project_root=tmp_path,
        )


def test_malformed_checkpoint_snapshot_has_one_normalized_gate_error() -> None:
    with pytest.raises(ValueError, match="division checkpoint is invalid"):
        build_division_cache._validate_division_checkpoint_identity(
            b"not a torch checkpoint",
            policy={},
        )


def test_frozen_evaluation_membership_cannot_be_hidden_by_manifest_role() -> None:
    contracts = _development_contracts()
    evaluation = contracts["split"]["role_datasets"]["evaluation"]
    disguised = DivisionManifestRow(
        dataset=evaluation[0],
        embryo="6bba",
        division_count=0,
        no_division=True,
        truth_graph_sha256="a" * 64,
        role="fit",
    )

    with pytest.raises(ValueError, match="only by exact --role evaluation"):
        build_division_cache._selected_rows(
            [disguised],
            datasets=[disguised.dataset],
            role=None,
            evaluation_datasets=evaluation,
        )


def test_resume_rejects_stale_phase10_reference_identity() -> None:
    metadata = {
        "phase10_score_cache_sha256": "a" * 64,
        "phase10_max_cellness_abs_diff": 0.0,
    }

    with pytest.raises(ValueError, match="different Phase 10 score cache"):
        build_division_cache.validate_resume_reference(
            metadata,
            phase10_sha256="b" * 64,
            maximum_difference=0.0,
        )


def _development_contracts() -> dict[str, dict[str, object]]:
    return read_division_contracts(ROOT / "configs")


def _valid_development_rows() -> list[dict[str, object]]:
    split = _development_contracts()["split"]
    division_count_by_dataset = {
        dataset: int(division_count)
        for division_count, roles in split["strata"].items()
        for dataset in roles["development"]
    }
    division_dataset = next(
        dataset
        for dataset in split["role_datasets"]["development"]
        if division_count_by_dataset[dataset] > 0
    )
    rows: list[dict[str, object]] = []
    for dataset in split["role_datasets"]["development"]:
        accepted_forks = int(dataset == division_dataset)
        fallback_hash = hashlib.sha256(f"{dataset}:fallback".encode()).hexdigest()
        candidate_hash = (
            hashlib.sha256(f"{dataset}:candidate".encode()).hexdigest()
            if accepted_forks
            else fallback_hash
        )
        rows.append(
            {
                "accepted_forks": accepted_forks,
                "added_edge_count": 2 * accepted_forks,
                "baseline_edge_jaccard": 0.5,
                "baseline_full_score": 0.5,
                "baseline_predicted_nodes": 100,
                "candidate_edge_jaccard": 0.5,
                "candidate_full_score": 0.53 if accepted_forks else 0.5,
                "candidate_graph_sha256": candidate_hash,
                "candidate_predicted_nodes": 100,
                "dataset": dataset,
                "edge_removal_count": accepted_forks,
                "estimated_nodes": 1000,
                "fallback_graph_sha256": fallback_hash,
                "fully_matched_wrong_forks": 0,
                "matched_division_false_positives": 0,
                "restored_node_count": 0,
                "strict_true_recoveries": accepted_forks,
                "truth_division_count": division_count_by_dataset[dataset],
                "truth_graph_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
            }
        )
    return rows


def _manifest_rows() -> tuple[DivisionManifestRow, ...]:
    split = _development_contracts()["split"]
    rows: list[DivisionManifestRow] = []
    exposed = set(split["forced_fit"]["selection_datasets"])
    for division_count, roles in split["strata"].items():
        for role, datasets in roles.items():
            for dataset in datasets:
                rows.append(
                    DivisionManifestRow(
                        dataset=dataset,
                        embryo="6bba",
                        division_count=int(division_count),
                        no_division=int(division_count) == 0,
                        proposal_exposed=dataset in exposed,
                        role=role,
                        truth_graph_sha256=hashlib.sha256(dataset.encode()).hexdigest(),
                    )
                )
    return tuple(sorted(rows, key=lambda row: row.dataset))


def _mark_row_accepted(row: dict[str, object]) -> None:
    row["accepted_forks"] = 1
    row["added_edge_count"] = 2
    row["candidate_graph_sha256"] = hashlib.sha256(f"{row['dataset']}:changed".encode()).hexdigest()
    row["edge_removal_count"] = 1
    row["restored_node_count"] = 0


def _unlock_grid(
    directory: Path,
    *,
    checkpoint_record: dict[str, str],
    model_config: DivisionModelConfig,
) -> tuple[Path, dict[str, object]]:
    contracts = _development_contracts()
    selected_rows = _valid_development_rows()
    datasets = contracts["split"]["role_datasets"]["development"]
    raw_dir = directory / "raw"
    cache_dir = directory / "cache"
    fit_dir = directory / "fit"
    raw_dir.mkdir()
    cache_dir.mkdir()
    fit_dir.mkdir()
    base = directory / "base.pt"
    base.write_bytes(b"synthetic base")
    fit_manifest = fit_dir / "manifest.json"
    write_canonical_json(fit_manifest, {"synthetic": True})
    summary = directory / "summary.json"
    write_canonical_json(summary, {"synthetic": True})
    upstream = directory / "manifest.csv"
    write_division_manifest(upstream, _manifest_rows())
    checkpoints = {}
    for name in DIVISION_MODEL_NAMES:
        if name == model_config.model_name:
            record = checkpoint_record
        else:
            path = directory / f"{name}.pt"
            path.write_bytes(name.encode())
            record = {
                "path": path.relative_to(directory).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        checkpoints[name] = {
            **record,
            "model_config": asdict(FIXED_DIVISION_MODEL_CONFIGS[name]),
        }
    identity = {
        "assignment_sha256": EXPECTED_DIVISION_CONTRACT_SHA256["assignment"],
        "base_checkpoint": {
            "path": base.relative_to(directory).as_posix(),
            "sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        },
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        "cache_directory": cache_dir.relative_to(directory).as_posix(),
        "cache_source_sha256": build_division_cache.source_bundle_sha256(
            build_division_cache.CACHE_SOURCE_PATHS
        ),
        "checkpoints": checkpoints,
        "contract_sha256": EXPECTED_DIVISION_CONTRACT_SHA256,
        "dataset_manifest_sha256": "d" * 64,
        "datasets": datasets,
        "data_root": raw_dir.relative_to(directory).as_posix(),
        "feature_schema_sha256": "b" * 64,
        "fit_dataset_manifest": {
            "path": fit_manifest.relative_to(directory).as_posix(),
            "sha256": hashlib.sha256(fit_manifest.read_bytes()).hexdigest(),
        },
        "model_inputs_sha256": "e" * 64,
        "role": "development",
        "summary": {
            "path": summary.relative_to(directory).as_posix(),
            "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        },
        "thresholds": list(PROBABILITY_THRESHOLDS),
        "trainer_source_sha256": "c" * 64,
        "upstream_manifest": {
            "path": upstream.relative_to(directory).as_posix(),
            "sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        },
        "upstream_manifest_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        "validation_source_sha256": validation_source_sha256(),
    }
    by_dataset = {row["dataset"]: row for row in selected_rows}
    grid_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for name in DIVISION_MODEL_NAMES:
            for threshold in PROBABILITY_THRESHOLDS:
                selected = dict(by_dataset[dataset])
                if (
                    (name, threshold) != (model_config.model_name, 0.9)
                    and selected["accepted_forks"]
                ):
                    selected["candidate_full_score"] = 0.529
                checkpoint = checkpoints[name]
                truth_count = int(selected["truth_division_count"])
                baseline_metric = {
                    "division_fn": truth_count,
                    "division_fp": 0,
                    "division_jaccard": float(truth_count == 0),
                    "division_tp": 0,
                    "edge_fn": 0,
                    "edge_fp": 0,
                    "edge_ignored": 0,
                    "edge_jaccard": selected["baseline_edge_jaccard"],
                    "edge_tp": 0,
                    "matched_nodes": 0,
                    "predicted_nodes": selected["baseline_predicted_nodes"],
                    "score": selected["baseline_full_score"],
                    "truth_nodes": 0,
                }
                candidate_metric = {
                    **baseline_metric,
                    "edge_jaccard": selected["candidate_edge_jaccard"],
                    "predicted_nodes": selected["candidate_predicted_nodes"],
                    "score": selected["candidate_full_score"],
                }
                baseline_triplets = {
                    "fn": truth_count,
                    "fp": 0,
                    "fully_matched_wrong_forks": 0,
                    "matched_division_false_positives": 0,
                    "tp": 0,
                    "unmatched_forks": 0,
                }
                strict_recoveries = int(selected["strict_true_recoveries"])
                candidate_triplets = {
                    **baseline_triplets,
                    "fn": truth_count - strict_recoveries,
                    "fully_matched_wrong_forks": selected[
                        "fully_matched_wrong_forks"
                    ],
                    "matched_division_false_positives": selected[
                        "matched_division_false_positives"
                    ],
                    "tp": strict_recoveries,
                }
                logit = 10.0
                accepted_fork_records = (
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
                    if selected["accepted_forks"]
                    else []
                )
                row = {
                    **selected,
                    "accepted_fork_records": accepted_fork_records,
                    "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
                    "baseline_metric": baseline_metric,
                    "baseline_triplets": baseline_triplets,
                    "cache_content_sha256": hashlib.sha256(
                        f"{dataset}:content".encode()
                    ).hexdigest(),
                    "cache_file_sha256": hashlib.sha256(
                        f"{dataset}:file".encode()
                    ).hexdigest(),
                    "cache_image_sha256": hashlib.sha256(
                        f"{dataset}:image".encode()
                    ).hexdigest(),
                    "candidate_metric": candidate_metric,
                    "candidate_triplets": candidate_triplets,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "contract_sha256": EXPECTED_DIVISION_CONTRACT_SHA256,
                    "dataset_manifest_sha256": "d" * 64,
                    "feature_schema_sha256": "b" * 64,
                    "model_config": checkpoint["model_config"],
                    "model_inputs_sha256": "e" * 64,
                    "model_name": name,
                    "runtime_seconds": 0.1,
                    "score_threshold": threshold,
                    "trainer_source_sha256": "c" * 64,
                    "validation_source_sha256": validation_source_sha256(),
                }
                assert set(row) == GRID_ROW_FIELDS
                grid_rows.append(row)
    grid = grid_payload(identity, grid_rows)
    path = directory / "grid.json"
    write_canonical_json(path, grid)
    return path, grid


def _write_unlock_bundle(
    directory: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    evaluation_unlocked: bool = True,
    checkpoint_contracts: dict[str, str] | None = None,
    guard_claims: dict[str, bool] | None = None,
    result_bytes: bytes | None = None,
    score_threshold: float = 0.9,
    assignment_sha256: str | None = None,
    checkpoint_epoch: int = 100,
    development_result_schema_version: int = 3,
    policy_schema_version: int = 3,
) -> tuple[Path, str, dict[str, object]]:
    contracts = _development_contracts()
    current_rows = _valid_development_rows() if rows is None else rows
    checkpoint = directory / "division.pt"
    model_config = DivisionModelConfig("scalar")
    model = DivisionModel(model_config)
    torch.save(
        {
            "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
            "contract_sha256": (
                EXPECTED_DIVISION_CONTRACT_SHA256
                if checkpoint_contracts is None
                else checkpoint_contracts
            ),
            "dataset_manifest_sha256": "d" * 64,
            "division_checkpoint_schema_version": 2,
            "epoch": checkpoint_epoch,
            "feature_schema_sha256": "b" * 64,
            "head_state_dict": model.head.state_dict(),
            "metrics": {"development_mean_full_delta": 0.002},
            "model_config": asdict(model_config),
            "model_inputs_sha256": "e" * 64,
            "trainer_source_sha256": "c" * 64,
        },
        checkpoint,
    )
    checkpoint_record = {
        "path": checkpoint.relative_to(directory).as_posix(),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    grid_path, grid = _unlock_grid(
        directory,
        checkpoint_record=checkpoint_record,
        model_config=model_config,
    )
    rows_hash = canonical_json_sha256(current_rows)
    assignment_hash = (
        EXPECTED_DIVISION_CONTRACT_SHA256["assignment"]
        if assignment_sha256 is None
        else assignment_sha256
    )
    development_result = {
        "assignment_sha256": assignment_hash,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        "contract_sha256": EXPECTED_DIVISION_CONTRACT_SHA256,
        "dataset_manifest_sha256": "d" * 64,
        "development_result_schema_version": development_result_schema_version,
        "division_checkpoint": checkpoint_record,
        "feature_schema_sha256": "b" * 64,
        "model_config": asdict(model_config),
        "model_inputs_sha256": "e" * 64,
        "role": "development",
        "rows": current_rows,
        "rows_sha256": rows_hash,
        "score_threshold": score_threshold,
        "trainer_source_sha256": "c" * 64,
    }
    result_path = directory / "development.json"
    result_path.write_bytes(
        canonical_json_bytes(development_result) if result_bytes is None else result_bytes
    )
    guard_results = {name: True for name in contracts["acceptance"]["guards"]}
    if guard_claims is not None:
        guard_results.update(guard_claims)
    policy = {
        "assignment_sha256": assignment_hash,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        "contract_sha256": EXPECTED_DIVISION_CONTRACT_SHA256,
        "dataset_manifest_sha256": "d" * 64,
        "development_grid": {
            "path": grid_path.relative_to(directory).as_posix(),
            "rows_sha256": grid["rows_sha256"],
            "sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        },
        "development_guard_results": guard_results,
        "development_result": {
            "path": result_path.relative_to(directory).as_posix(),
            "rows_sha256": rows_hash,
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
        "division_checkpoint": checkpoint_record,
        "division_policy_schema_version": policy_schema_version,
        "evaluation_unlocked": evaluation_unlocked,
        "feature_schema_sha256": "b" * 64,
        "model_config": asdict(model_config),
        "model_inputs_sha256": "e" * 64,
        "policy_source_sha256": build_division_cache.source_bundle_sha256(
            (
                ROOT / "src" / "biohub" / "division_data.py",
                ROOT / "src" / "biohub" / "division_inference.py",
                ROOT / "scripts" / "freeze_division_policy.py",
                ROOT / "scripts" / "build_division_cache.py",
                ROOT / "scripts" / "validate_division_model.py",
            )
        ),
        "score_threshold": score_threshold,
        "selection_role": "development",
        "trainer_source_sha256": "c" * 64,
    }
    path = directory / "policy.json"
    content = canonical_json_bytes(policy)
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest(), policy


def _verify_unlock(
    path: Path,
    digest: str,
    root: Path,
) -> build_division_cache._EvaluationAuthorization:
    return build_division_cache.verify_evaluation_unlock_policy(
        path,
        expected_policy_sha256=digest,
        config_dir=ROOT / "configs",
        manifest_rows=_manifest_rows(),
        project_root=root,
    )


def _allow_synthetic_grid_base_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    original = build_division_cache._require_policy_artifact

    def resolve(record, *, field, project_root):
        if field == "grid base checkpoint":
            return (
                ROOT
                / "data"
                / "working"
                / "appearance_checkpoints"
                / "cellness_cnn_phase10_full.pt"
            )
        return original(record, field=field, project_root=project_root)

    monkeypatch.setattr(build_division_cache, "_require_policy_artifact", resolve)


def test_epoch_100_full_development_cohort_unlock_bundle_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    expected_datasets = _development_contracts()["split"]["role_datasets"]["development"]
    improved = [
        row
        for row in rows
        if float(row["candidate_full_score"]) > float(row["baseline_full_score"])
    ]
    assert len(improved) == 1
    assert [row["dataset"] for row in rows] == expected_datasets
    assert len(rows) == 20
    assert improved[0]["accepted_forks"] == 1
    assert improved[0]["candidate_graph_sha256"] != improved[0]["fallback_graph_sha256"]
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)
    _allow_synthetic_grid_base_checkpoint(monkeypatch)
    replayed: list[dict[str, object]] = []

    def replay(result, **kwargs):
        replayed.extend(result["rows"])
        assert kwargs["checkpoint_snapshot"]
        return tuple(result["rows"])

    monkeypatch.setattr("scripts.validate_division_model.replay_selected_policy", replay)

    observed = _verify_unlock(policy, policy_hash, tmp_path)

    assert observed.evaluation_datasets == tuple(
        _development_contracts()["split"]["role_datasets"]["evaluation"]
    )
    assert replayed == rows
    assert build_division_cache.DEVELOPMENT_RESULT_SCHEMA_VERSION == 3
    assert build_division_cache.EVALUATION_UNLOCK_POLICY_SCHEMA_VERSION == 3


def test_self_consistent_rows_cannot_unlock_when_strict_replay_disagrees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path)
    _allow_synthetic_grid_base_checkpoint(monkeypatch)

    def reject_fabricated_rows(*args, **kwargs):
        raise ValueError("development result replay disagrees for 6bba_synthetic")

    monkeypatch.setattr(
        "scripts.validate_division_model.replay_selected_policy",
        reject_fabricated_rows,
    )

    with pytest.raises(ValueError, match="replay disagrees"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize("checkpoint_epoch", [0, 99, 101])
def test_evaluation_policy_rejects_nonproduction_checkpoint_epoch(
    checkpoint_epoch: int,
    tmp_path: Path,
) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(
        tmp_path,
        checkpoint_epoch=checkpoint_epoch,
    )

    with pytest.raises(ValueError, match="epoch 100"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize("artifact", ["development result", "policy"])
def test_schema_v2_artifacts_cannot_unlock_evaluation(
    artifact: str,
    tmp_path: Path,
) -> None:
    kwargs = (
        {"development_result_schema_version": 2}
        if artifact == "development result"
        else {"policy_schema_version": 2}
    )
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, **kwargs)

    with pytest.raises(ValueError, match="unsupported .*schema"):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_evaluation_policy_rejects_arbitrary_result_bytes(tmp_path: Path) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(
        tmp_path,
        result_bytes=b"arbitrary bytes with a matching policy hash",
    )

    with pytest.raises(ValueError, match="canonical JSON|development result"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize("cohort_error", ["missing", "duplicate", "wrong"])
def test_evaluation_policy_rejects_inexact_development_dataset_cohort(
    cohort_error: str,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    if cohort_error == "missing":
        rows = rows[:-1]
    elif cohort_error == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows[-1] = {**rows[-1], "dataset": "6bba_wrong"}
        rows.sort(key=lambda row: str(row["dataset"]))
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="exact development dataset|duplicate|cohort"):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_development_row_truth_count_must_match_the_exact_manifest_movie(
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    rows[0]["truth_division_count"] = int(rows[0]["truth_division_count"]) + 1
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="manifest truth_division_count"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    ("truth_hash_error", "message"),
    [
        ("missing", "incomplete or unexpected fields"),
        ("wrong", "truth_graph_sha256 is wrong"),
        ("not-lowercase", "lowercase SHA-256"),
    ],
)
def test_development_row_requires_exact_manifest_truth_graph_hash(
    truth_hash_error: str,
    message: str,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    if truth_hash_error == "missing":
        rows[0].pop("truth_graph_sha256")
    elif truth_hash_error == "wrong":
        rows[0]["truth_graph_sha256"] = "f" * 64
    else:
        rows[0]["truth_graph_sha256"] = "A" * 64
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=message):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_control_movie_cannot_claim_a_strict_true_recovery(tmp_path: Path) -> None:
    rows = _valid_development_rows()
    control = next(row for row in rows if row["truth_division_count"] == 0)
    _mark_row_accepted(control)
    control["strict_true_recoveries"] = 1
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="strict_true_recoveries.*truth_division_count"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("restored_node_count", -1, "non-negative integer"),
        ("edge_removal_count", 0.5, "non-negative integer"),
        ("added_edge_count", -1, "non-negative integer"),
    ],
)
def test_development_rows_reject_noninteger_or_negative_counts(
    field: str,
    value: float,
    message: str,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    next(row for row in rows if row["accepted_forks"])[field] = value
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=message):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_accepted_forks_must_cover_true_and_wrong_fork_counts(tmp_path: Path) -> None:
    rows = _valid_development_rows()
    accepted = next(row for row in rows if row["accepted_forks"])
    accepted["fully_matched_wrong_forks"] = 1
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="accepted_forks.*strict_true_recoveries"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("added_edge_count", 0, "added_edge_count"),
        ("added_edge_count", 3, "added_edge_count"),
        ("restored_node_count", 4, "mutation counts"),
        ("edge_removal_count", 4, "mutation counts"),
        ("candidate_predicted_nodes", 101, "fallback nodes plus restored_node_count"),
    ],
)
def test_accepted_fork_mutation_counts_must_be_internally_possible(
    field: str,
    value: int,
    message: str,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    accepted = next(row for row in rows if row["accepted_forks"])
    accepted[field] = value
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=message):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_same_graph_hash_cannot_carry_fabricated_candidate_metrics(tmp_path: Path) -> None:
    rows = _valid_development_rows()
    accepted = next(row for row in rows if row["accepted_forks"])
    accepted["candidate_graph_sha256"] = accepted["fallback_graph_sha256"]
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="same graph hash|identical graph hashes"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_full_score", 0.51),
        ("candidate_edge_jaccard", 0.51),
        ("candidate_predicted_nodes", 101),
    ],
)
def test_no_fork_rows_require_exact_fallback_candidate_metrics(
    field: str,
    value: float,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    no_fork = next(row for row in rows if row["accepted_forks"] == 0)
    no_fork[field] = value
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="no-fork.*metrics|same graph hash"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    "field",
    ["restored_node_count", "edge_removal_count", "added_edge_count"],
)
def test_no_fork_rows_require_zero_mutation_counts(field: str, tmp_path: Path) -> None:
    rows = _valid_development_rows()
    no_fork = next(row for row in rows if row["accepted_forks"] == 0)
    no_fork[field] = 1
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="no-fork.*mutation"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize(
    "failed_guard",
    [
        "mean_full_score_delta",
        "mean_edge_jaccard_delta",
        "aggregate_node_ratio_delta",
        "matched_division_false_positives_per_no_division_movie",
        "fully_matched_wrong_forks",
        "strict_true_recoveries",
        "full_score_wins_vs_losses",
        "no_fork_fallback_identity",
    ],
)
def test_evaluation_policy_rejects_each_failed_development_guard(
    failed_guard: str,
    tmp_path: Path,
) -> None:
    rows = _valid_development_rows()
    if failed_guard == "mean_full_score_delta":
        next(row for row in rows if row["accepted_forks"])["candidate_full_score"] = 0.5
    elif failed_guard == "mean_edge_jaccard_delta":
        next(row for row in rows if row["accepted_forks"])["candidate_edge_jaccard"] = 0.47
    elif failed_guard == "aggregate_node_ratio_delta":
        for row in rows:
            _mark_row_accepted(row)
            row["restored_node_count"] = 3
            row["candidate_predicted_nodes"] = int(row["baseline_predicted_nodes"]) + 3
    elif failed_guard == "matched_division_false_positives_per_no_division_movie":
        row = next(row for row in rows if row["truth_division_count"] == 0)
        _mark_row_accepted(row)
        row["matched_division_false_positives"] = 1
    elif failed_guard == "fully_matched_wrong_forks":
        row = next(
            row
            for row in rows
            if row["accepted_forks"] == 0 and int(row["truth_division_count"]) > 0
        )
        _mark_row_accepted(row)
        row["fully_matched_wrong_forks"] = 1
    elif failed_guard == "strict_true_recoveries":
        for row in rows:
            row["strict_true_recoveries"] = 0
    elif failed_guard == "full_score_wins_vs_losses":
        strict_dataset = next(row["dataset"] for row in rows if row["strict_true_recoveries"] == 1)
        for index, row in enumerate(rows):
            _mark_row_accepted(row)
            row["strict_true_recoveries"] = int(row["dataset"] == strict_dataset)
            row["candidate_full_score"] = 0.51 if index < 9 else 0.499
    else:
        row = next(row for row in rows if row["accepted_forks"] == 0)
        row["candidate_graph_sha256"] = "f" * 64
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=failed_guard):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_evaluation_policy_rejects_forged_guard_booleans(tmp_path: Path) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(
        tmp_path,
        guard_claims={"mean_full_score_delta": False},
    )

    with pytest.raises(ValueError, match="guard claims"):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_evaluation_policy_requires_an_explicit_true_unlock(tmp_path: Path) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(tmp_path, evaluation_unlocked=False)

    with pytest.raises(ValueError, match="did not unlock"):
        _verify_unlock(policy, policy_hash, tmp_path)


def test_evaluation_policy_validates_the_internal_checkpoint_contract_chain(
    tmp_path: Path,
) -> None:
    policy, policy_hash, _ = _write_unlock_bundle(
        tmp_path,
        checkpoint_contracts={
            **EXPECTED_DIVISION_CONTRACT_SHA256,
            "sampling": "f" * 64,
        },
    )

    with pytest.raises(ValueError, match="checkpoint contract chain"):
        _verify_unlock(policy, policy_hash, tmp_path)


@pytest.mark.parametrize("identity", ["threshold", "assignment", "model_config"])
def test_evaluation_policy_rejects_nonfrozen_selected_identity(
    identity: str,
    tmp_path: Path,
) -> None:
    kwargs: dict[str, object] = {}
    if identity == "threshold":
        kwargs["score_threshold"] = 0.91
    elif identity == "assignment":
        kwargs["assignment_sha256"] = "f" * 64
    policy, policy_hash, payload = _write_unlock_bundle(tmp_path, **kwargs)
    if identity == "model_config":
        payload["model_config"]["dropout"] = 0.5
        content = canonical_json_bytes(payload)
        policy.write_bytes(content)
        policy_hash = hashlib.sha256(content).hexdigest()

    with pytest.raises(ValueError, match="threshold|assignment|model config"):
        _verify_unlock(policy, policy_hash, tmp_path)
