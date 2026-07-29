"""Frozen appearance embeddings, symmetric division heads, and node caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from biohub.division_data import (
    ROLE_NAMES,
    canonical_json_bytes,
    file_sha256,
    read_division_contracts,
)
from biohub.metric import TrackingGraph

if TYPE_CHECKING:
    from biohub.appearance_model import CellnessCNN3D

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised in a subprocess test
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_TorchModule = object if nn is None else nn.Module
_OPTIONAL_ML_ERROR = (
    "Division model features require PyTorch. "
    "Install it with `python -m pip install -e '.[ml]'`."
)

DIVISION_CHECKPOINT_SCHEMA_VERSION = 2
DIVISION_NODE_CACHE_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 1

EXPECTED_BASE_CHECKPOINT_SHA256 = "bd35d082c985641728bc340817d08d4750a3c83c34d25f99d1507cbc8b7bb8d4"
EXPECTED_SCORING_SOURCE_SHA256 = "426c09f04c99af55fd5457921fbf435de2f71bdf7ce08dbdf0e0fde6c02443c9"
EXPECTED_DIVISION_CONTRACT_SHA256 = {
    "split": "974cda6c2cc907c0ac71af93ee969bcd151bb42b0ad56343a307aa2942ba6ecb",
    "proposal": "83ec5cb8818cab4470cc00c90e3d34aa23d01bcd90e2f4619031bc17a7630f80",
    "sampling": "35cc862afb4c87bcdd975e6662fe3e0716a768bc906eebd3c815fb2c630a3806",
    "acceptance": "6f3678436d9b11f61a1f17b068f8543452d9b3089a9d258360934a59696a9c5c",
    "assignment": "22bbd5cef2864769ac2b17f7fe9f4275a7ea53d8d4715103cfffce03b4459a35",
}
DIVISION_CONTRACT_KINDS = tuple(EXPECTED_DIVISION_CONTRACT_SHA256)

DIVISION_MODEL_NAMES = ("scalar", "embedding_linear", "embedding_mlp32")
NODE_CACHE_EMBEDDING_DIM = 64
NODE_CACHE_DETECTOR_THRESHOLD_K = 3.0
NODE_CACHE_PATCH_SHAPE = (9, 33, 33)
NODE_CACHE_NORMALIZATION = "per_patch_mean_std_clip_6"
CELLNESS_MATCH_ATOL = 2e-7
CACHE_BACKEND_POLICY = "phase11_local_mps_float32_v1"
CACHE_DEVICE = "mps"
CACHE_BACKEND = "pytorch_mps"
CACHE_INFERENCE_DTYPE = "float32"
CACHE_TORCH_VERSION = "2.13.0" if torch is None else str(torch.__version__)
EXPECTED_MODEL_STATE_SHA256 = "194aa53956888e5aa85d5a196f1876acaf439cfe309a990f63c1f461598de573"
MODEL_STATE_HASH_SCHEMA = "biohub-cellness-model-state-v1"
INTENSITY_SUMMARY_NAMES = (
    "mean",
    "std",
    "center_background_contrast",
    "min",
    "max",
)
INTENSITY_CONTRAST_DEFINITION = "center_voxel_minus_noncenter_mean"
NODE_CACHE_CONTENT_HASH_SCHEMA = "biohub-phase11-division-node-cache-content-v1"

_CANDIDATE_FLAG_NAMES = (
    "present_in_linked_graph",
    "is_current_child",
    "is_parentless",
    "has_wrong_parent",
    "has_future",
    "survives_pruning",
)
_MOTHER_FLAG_NAMES = (
    "mother_has_history",
    "mother_has_current_child",
    "mother_survives_pruning",
)
_CONTEXT_FEATURE_SPEC = (
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
    *(
        (feature_name, kind)
        for flag_name in _CANDIDATE_FLAG_NAMES
        for feature_name, kind in (
            (f"{flag_name}_count", "count"),
            (f"{flag_name}_any", "binary"),
            (f"{flag_name}_both", "binary"),
        )
    ),
    *((name, "binary") for name in _MOTHER_FLAG_NAMES),
    ("restored_node_count", "count"),
    ("edge_removal_count", "count"),
    *(
        (feature_name, "continuous")
        for summary_name in INTENSITY_SUMMARY_NAMES
        for feature_name in (
            f"mother_intensity_{summary_name}",
            f"daughter_intensity_{summary_name}_min",
            f"daughter_intensity_{summary_name}_max",
            f"daughter_intensity_{summary_name}_mean",
            f"daughter_intensity_{summary_name}_abs_diff",
        )
    ),
)
CONTEXT_FEATURE_NAMES = tuple(name for name, _ in _CONTEXT_FEATURE_SPEC)
CONTEXT_FEATURE_KINDS = tuple(kind for _, kind in _CONTEXT_FEATURE_SPEC)
CONTEXT_CONTINUOUS_MASK = tuple(kind == "continuous" for kind in CONTEXT_FEATURE_KINDS)

_CHECKPOINT_FIELDS = {
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
_NODE_CACHE_ARRAY_FIELDS = {
    "metadata_json",
    "node_ids",
    "tzyx",
    "cellness",
    "embeddings",
    "intensity_summaries",
}
_NODE_CACHE_METADATA_FIELDS = {
    "backend",
    "backend_policy",
    "cache_source_sha256",
    "candidate_count",
    "cellness_match_atol",
    "checkpoint_sha256",
    "content_sha256",
    "contract_sha256",
    "data_split",
    "dataset",
    "detector_method",
    "detector_threshold_k",
    "device",
    "division_node_cache_schema_version",
    "embedding_dim",
    "image_sha256",
    "inference_dtype",
    "intensity_contrast_definition",
    "intensity_summary_names",
    "upstream_manifest_sha256",
    "model_state_sha256",
    "normalization",
    "patch_shape",
    "phase10_max_cellness_abs_diff",
    "phase10_score_cache_sha256",
    "role",
    "scoring_source_sha256",
    "torch_version",
}


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(_OPTIONAL_ML_ERROR)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def canonical_model_state_sha256(model_state: Mapping[str, torch.Tensor]) -> str:
    """Hash logical tensor state independently of the checkpoint container."""
    _require_torch()
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("model state must be a nonempty tensor mapping")
    digest = hashlib.sha256()
    digest.update(MODEL_STATE_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for name in sorted(model_state):
        tensor = model_state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("model state must map string names to tensors")
        values = tensor.detach().cpu().contiguous().numpy()
        shape = ",".join(str(value) for value in values.shape)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0dtype\0")
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(b"\0shape\0")
        digest.update(shape.encode("ascii"))
        digest.update(b"\0data\0")
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} has wrong fields; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


class FrozenCellnessEncoder(_TorchModule):
    """Expose the accepted CNN backbone while permanently freezing its state."""

    def __init__(self, backbone: CellnessCNN3D) -> None:
        _require_torch()
        from biohub.appearance_model import CellnessCNN3D

        super().__init__()
        if not isinstance(backbone, CellnessCNN3D):
            raise TypeError("FrozenCellnessEncoder requires a CellnessCNN3D backbone")
        self.backbone = backbone
        self.output_dim = int(backbone.config.base_channels) * 4
        final_linear = backbone.head[-1]
        if not isinstance(final_linear, nn.Linear) or final_linear.in_features != self.output_dim:
            raise ValueError("CellnessCNN3D head does not match its configured backbone width")
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenCellnessEncoder:
        super().train(mode)
        self._freeze_backbone()
        return self

    @staticmethod
    def _validate_inputs(inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("encoder inputs must be a torch.Tensor")
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError(
                f"FrozenCellnessEncoder expects [batch, 1, z, y, x], got {tuple(inputs.shape)}"
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        self._freeze_backbone()
        with torch.no_grad():
            features = self.backbone.stem(inputs)
            features = self.backbone.features(features)
            features = self.backbone.pool(features)
            return torch.flatten(features, start_dim=1)

    def cellness_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.output_dim:
            raise ValueError(
                f"embeddings must have shape [batch, {self.output_dim}], "
                f"got {tuple(embeddings.shape)}"
            )
        self._freeze_backbone()
        with torch.no_grad():
            return torch.sigmoid(self.backbone.head(embeddings).squeeze(1))

    def forward_with_cellness(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings and accepted cellness from one backbone execution."""
        embeddings = self(inputs)
        return embeddings, self.cellness_from_embeddings(embeddings)


def daughter_symmetric_features(
    mother: torch.Tensor,
    daughter_1: torch.Tensor,
    daughter_2: torch.Tensor,
) -> torch.Tensor:
    """Return exactly ``[mother, daughter mean, abs diff, product]``."""
    _require_torch()
    if not all(isinstance(value, torch.Tensor) for value in (mother, daughter_1, daughter_2)):
        raise TypeError("mother and daughter embeddings must be torch tensors")
    if mother.ndim != 2 or daughter_1.shape != mother.shape or daughter_2.shape != mother.shape:
        raise ValueError("mother and daughter embeddings must share shape [batch, embedding_dim]")
    return torch.cat(
        [
            mother,
            (daughter_1 + daughter_2) / 2.0,
            torch.abs(daughter_1 - daughter_2),
            daughter_1 * daughter_2,
        ],
        dim=1,
    )


def _batch_vector(values: Any, *, name: str, count: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 1 or (count is not None and array.shape != (count,)):
        expected = "[batch]" if count is None else str((count,))
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _batch_pairs(values: Any, *, name: str, count: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.shape != (count, 2):
        raise ValueError(f"{name} must have shape {(count, 2)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_values(values: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary values") from exc
    if array.shape != shape or not np.all(np.isin(array, (0.0, 1.0))):
        raise ValueError(f"{name} must contain binary values with shape {shape}")
    return array


def _continuous_pair_features(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = values[:, 0]
    second = values[:, 1]
    return (
        np.minimum(first, second),
        np.maximum(first, second),
        (first + second) / 2.0,
        np.abs(first - second),
    )


def build_symmetric_context_features(
    *,
    mother_cellness: Any,
    daughter_cellness: Any,
    daughter_distance_um: Any,
    daughter_angle_deg: Any,
    candidate_flags: Mapping[str, Any],
    mother_flags: Mapping[str, Any],
    restored_node_count: Any,
    edge_removal_count: Any,
    mother_intensity_summaries: Any,
    daughter_intensity_summaries: Any,
) -> np.ndarray:
    """Assemble the fixed identifier-free, daughter-symmetric scalar context."""
    mother_scores = _batch_vector(mother_cellness, name="mother_cellness")
    count = len(mother_scores)
    daughter_scores = _batch_pairs(
        daughter_cellness,
        name="daughter_cellness",
        count=count,
    )
    distances = _batch_pairs(
        daughter_distance_um,
        name="daughter_distance_um",
        count=count,
    )
    angles = _batch_vector(daughter_angle_deg, name="daughter_angle_deg", count=count)
    restored = _batch_vector(restored_node_count, name="restored_node_count", count=count)
    removals = _batch_vector(edge_removal_count, name="edge_removal_count", count=count)
    if (
        np.any(mother_scores < 0.0)
        or np.any(mother_scores > 1.0)
        or np.any(daughter_scores < 0.0)
        or np.any(daughter_scores > 1.0)
    ):
        raise ValueError("frozen cellness context must be between zero and one")
    if np.any(distances < 0.0) or np.any(angles < 0.0) or np.any(angles > 180.0):
        raise ValueError("division distances and angles are outside their valid ranges")
    if (
        np.any(restored < 0.0)
        or np.any(removals < 0.0)
        or not np.all(restored == np.floor(restored))
        or not np.all(removals == np.floor(removals))
    ):
        raise ValueError("restored and removal counts must be non-negative integers")

    if not isinstance(candidate_flags, Mapping) or set(candidate_flags) != set(
        _CANDIDATE_FLAG_NAMES
    ):
        raise ValueError("candidate_flags has incomplete or unexpected fields")
    if not isinstance(mother_flags, Mapping) or set(mother_flags) != set(_MOTHER_FLAG_NAMES):
        raise ValueError("mother_flags has incomplete or unexpected fields")
    candidate_values = {
        name: _binary_values(candidate_flags[name], name=name, shape=(count, 2))
        for name in _CANDIDATE_FLAG_NAMES
    }
    mother_values = {
        name: _binary_values(mother_flags[name], name=name, shape=(count,))
        for name in _MOTHER_FLAG_NAMES
    }

    try:
        mother_intensity = np.asarray(mother_intensity_summaries, dtype=np.float64)
        daughter_intensity = np.asarray(daughter_intensity_summaries, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("intensity summaries must contain numeric values") from exc
    expected_mother_shape = (count, len(INTENSITY_SUMMARY_NAMES))
    expected_daughter_shape = (count, 2, len(INTENSITY_SUMMARY_NAMES))
    if mother_intensity.shape != expected_mother_shape:
        raise ValueError(
            f"mother intensity summaries must have shape {expected_mother_shape}, "
            f"got {mother_intensity.shape}"
        )
    if daughter_intensity.shape != expected_daughter_shape:
        raise ValueError(
            f"daughter intensity summaries must have shape {expected_daughter_shape}, "
            f"got {daughter_intensity.shape}"
        )
    if not np.all(np.isfinite(mother_intensity)) or not np.all(np.isfinite(daughter_intensity)):
        raise ValueError("intensity summaries must contain only finite values")

    columns: list[np.ndarray] = [mother_scores]
    columns.extend(_continuous_pair_features(daughter_scores))
    columns.extend(_continuous_pair_features(distances))
    columns.append(angles)
    for name in _CANDIDATE_FLAG_NAMES:
        values = candidate_values[name]
        columns.extend((values.sum(axis=1), values.max(axis=1), values.min(axis=1)))
    columns.extend(mother_values[name] for name in _MOTHER_FLAG_NAMES)
    columns.extend((restored, removals))
    for summary_index in range(len(INTENSITY_SUMMARY_NAMES)):
        columns.append(mother_intensity[:, summary_index])
        columns.extend(_continuous_pair_features(daughter_intensity[:, :, summary_index]))
    features = np.column_stack(columns).astype(np.float32, copy=False)
    if features.shape != (count, len(CONTEXT_FEATURE_NAMES)):
        raise RuntimeError("symmetric context assembly disagrees with its frozen feature names")
    return features


@dataclass(frozen=True)
class FeatureSchema:
    """Fit-only typed normalization for the exact symmetric scalar context."""

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        kinds = tuple(self.kinds)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.means):
            raise ValueError("FeatureSchema means must be numeric")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.scales):
            raise ValueError("FeatureSchema scales must be numeric")
        means = tuple(float(value) for value in self.means)
        scales = tuple(float(value) for value in self.scales)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        if names != CONTEXT_FEATURE_NAMES or kinds != CONTEXT_FEATURE_KINDS:
            raise ValueError("FeatureSchema must use the exact frozen context names and kinds")
        if len(means) != len(names) or len(scales) != len(names):
            raise ValueError(
                "FeatureSchema names, kinds, means, and scales must have equal lengths"
            )
        if not np.all(np.isfinite(means)):
            raise ValueError("FeatureSchema means must be finite")
        if not np.all(np.isfinite(scales)) or any(value <= 0.0 for value in scales):
            raise ValueError("FeatureSchema scales must be positive and finite")
        for kind, mean, scale in zip(kinds, means, scales, strict=True):
            if kind != "continuous" and (mean != 0.0 or scale != 1.0):
                raise ValueError("binary and count features must remain unstandardized")

    @property
    def continuous_mask(self) -> tuple[bool, ...]:
        return tuple(kind == "continuous" for kind in self.kinds)

    @staticmethod
    def _validate_typed_values(array: np.ndarray) -> None:
        if not np.all(np.isfinite(array)):
            raise ValueError("context values must be finite")
        for index, kind in enumerate(CONTEXT_FEATURE_KINDS):
            values = array[:, index]
            if kind == "binary" and not np.all(np.isin(values, (0.0, 1.0))):
                raise ValueError(
                    f"binary context feature {CONTEXT_FEATURE_NAMES[index]} is invalid"
                )
            if kind == "count" and (np.any(values < 0.0) or not np.all(values == np.floor(values))):
                raise ValueError(f"count context feature {CONTEXT_FEATURE_NAMES[index]} is invalid")

    @classmethod
    def fit(
        cls,
        names: Sequence[str],
        values: Any,
        *,
        kinds: Sequence[str],
        source_role: str,
    ) -> FeatureSchema:
        if source_role != "fit":
            raise ValueError("FeatureSchema statistics may be fitted only from the fit role")
        ordered_names = tuple(names)
        ordered_kinds = tuple(kinds)
        if ordered_names != CONTEXT_FEATURE_NAMES or ordered_kinds != CONTEXT_FEATURE_KINDS:
            raise ValueError("FeatureSchema must use the exact frozen context names and kinds")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("FeatureSchema fit values must be numeric") from exc
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != len(ordered_names):
            raise ValueError(
                "FeatureSchema fit values must have shape "
                f"[rows, {len(ordered_names)}], got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("FeatureSchema fit values must be finite")
        cls._validate_typed_values(array)
        means = np.zeros(len(ordered_names), dtype=np.float64)
        scales = np.ones(len(ordered_names), dtype=np.float64)
        continuous = np.asarray(CONTEXT_CONTINUOUS_MASK, dtype=bool)
        means[continuous] = array[:, continuous].mean(axis=0)
        scales[continuous] = array[:, continuous].std(axis=0)
        scales[continuous & (scales == 0.0)] = 1.0
        return cls(
            names=ordered_names,
            kinds=ordered_kinds,
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "fit_role": "fit",
            "continuous_mask": list(self.continuous_mask),
            "kinds": list(self.kinds),
            "means": list(self.means),
            "names": list(self.names),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSchema:
        if not isinstance(payload, Mapping):
            raise TypeError("FeatureSchema payload must be a mapping")
        expected = {
            "continuous_mask",
            "feature_schema_version",
            "fit_role",
            "kinds",
            "means",
            "names",
            "scales",
        }
        _require_exact_fields(payload, expected, label="FeatureSchema payload")
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported FeatureSchema version")
        if payload["fit_role"] != "fit":
            raise ValueError("FeatureSchema payload was not fitted on the fit role")
        mask = payload["continuous_mask"]
        if not isinstance(mask, list) or any(not isinstance(value, bool) for value in mask):
            raise ValueError("FeatureSchema continuous mask must contain only booleans")
        schema = cls(
            names=tuple(payload["names"]),
            kinds=tuple(payload["kinds"]),
            means=tuple(payload["means"]),
            scales=tuple(payload["scales"]),
        )
        if mask != list(schema.continuous_mask):
            raise ValueError("FeatureSchema continuous mask disagrees with its feature kinds")
        return schema

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def standardize(self, values: Any) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("context values must be numeric") from exc
        if array.ndim != 2 or array.shape[1] != len(self.names):
            raise ValueError(
                f"context values must have shape [rows, {len(self.names)}], got {array.shape}"
            )
        self._validate_typed_values(array)
        standardized = (array - np.asarray(self.means)) / np.asarray(self.scales)
        return standardized.astype(np.float32, copy=False)

    def standardize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        _require_torch()
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError(
                f"context tensor must have shape [rows, {len(self.names)}], "
                f"got {tuple(values.shape)}"
            )
        if not torch.is_floating_point(values) or not bool(torch.isfinite(values).all()):
            raise ValueError("context tensor must contain finite floating-point values")
        for index, kind in enumerate(self.kinds):
            column = values[:, index]
            if kind == "binary" and not bool(((column == 0.0) | (column == 1.0)).all()):
                raise ValueError(f"binary context feature {self.names[index]} is invalid")
            if kind == "count" and not bool(
                ((column >= 0.0) & (column == torch.floor(column))).all()
            ):
                raise ValueError(f"count context feature {self.names[index]} is invalid")
        means = values.new_tensor(self.means)
        scales = values.new_tensor(self.scales)
        return (values - means) / scales


@dataclass(frozen=True)
class DivisionModelConfig:
    """One of the three fixed daughter-invariant division-head configurations."""

    model_name: str
    embedding_dim: int = NODE_CACHE_EMBEDDING_DIM
    context_dim: int = len(CONTEXT_FEATURE_NAMES)
    hidden_dim: int | None = None
    dropout: float | None = None

    def __post_init__(self) -> None:
        if self.model_name not in DIVISION_MODEL_NAMES:
            raise ValueError(
                f"model_name must be one of {DIVISION_MODEL_NAMES}, got {self.model_name!r}"
            )
        if (
            isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, Integral)
            or int(self.embedding_dim) != NODE_CACHE_EMBEDDING_DIM
        ):
            raise ValueError(f"embedding_dim must be exactly {NODE_CACHE_EMBEDDING_DIM}")
        if (
            isinstance(self.context_dim, bool)
            or not isinstance(self.context_dim, Integral)
            or int(self.context_dim) != len(CONTEXT_FEATURE_NAMES)
        ):
            raise ValueError(f"context_dim must be exactly {len(CONTEXT_FEATURE_NAMES)}")
        object.__setattr__(self, "embedding_dim", int(self.embedding_dim))
        object.__setattr__(self, "context_dim", int(self.context_dim))
        expected_hidden = 32 if self.model_name == "embedding_mlp32" else 0
        expected_dropout = 0.1 if self.model_name == "embedding_mlp32" else 0.0
        hidden_dim = expected_hidden if self.hidden_dim is None else self.hidden_dim
        dropout = expected_dropout if self.dropout is None else self.dropout
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral):
            raise ValueError("hidden_dim must be an integer")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise ValueError("dropout must be numeric")
        if int(hidden_dim) != expected_hidden or float(dropout) != expected_dropout:
            raise ValueError(
                f"{self.model_name} requires hidden_dim={expected_hidden} "
                f"and dropout={expected_dropout}"
            )
        object.__setattr__(self, "hidden_dim", int(hidden_dim))
        object.__setattr__(self, "dropout", float(dropout))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DivisionModelConfig:
        if not isinstance(payload, Mapping):
            raise ValueError("model config must be a mapping")
        _require_exact_fields(
            payload,
            {"model_name", "embedding_dim", "context_dim", "hidden_dim", "dropout"},
            label="model config",
        )
        try:
            return cls(**payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid model config: {exc}") from exc


FIXED_DIVISION_MODEL_CONFIGS = {
    name: DivisionModelConfig(model_name=name) for name in DIVISION_MODEL_NAMES
}


class DivisionModel(_TorchModule):
    """A fixed head over standardized context and optional invariant embeddings."""

    def __init__(self, config: DivisionModelConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, DivisionModelConfig):
            raise TypeError("DivisionModel requires a DivisionModelConfig")
        self.config = config
        if config.model_name == "scalar":
            self.head: nn.Module = nn.Linear(config.context_dim, 1)
        else:
            input_dim = config.embedding_dim * 4 + config.context_dim
            if config.model_name == "embedding_linear":
                self.head = nn.Linear(input_dim, 1)
            else:
                hidden_dim = int(config.hidden_dim)
                dropout = float(config.dropout)
                self.head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

    def forward(
        self,
        mother_embedding: torch.Tensor,
        daughter_1_embedding: torch.Tensor,
        daughter_2_embedding: torch.Tensor,
        standardized_context: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(standardized_context, torch.Tensor):
            raise TypeError("standardized context must be a torch tensor")
        if (
            standardized_context.ndim != 2
            or standardized_context.shape[1] != self.config.context_dim
        ):
            raise ValueError(
                "standardized context must have shape "
                f"[batch, {self.config.context_dim}], got {tuple(standardized_context.shape)}"
            )
        if not torch.is_floating_point(standardized_context) or not bool(
            torch.isfinite(standardized_context).all()
        ):
            raise ValueError("standardized context must contain finite floating-point values")
        batch = standardized_context.shape[0]
        expected_embedding_shape = (batch, self.config.embedding_dim)
        embeddings = (mother_embedding, daughter_1_embedding, daughter_2_embedding)
        if any(not isinstance(value, torch.Tensor) for value in embeddings):
            raise TypeError("all embeddings must be torch tensors")
        if any(value.shape != expected_embedding_shape for value in embeddings):
            raise ValueError(f"all embeddings must have shape {expected_embedding_shape}")
        if any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in embeddings
        ):
            raise ValueError("all embeddings must contain finite floating-point values")
        if any(
            value.dtype != standardized_context.dtype
            or value.device != standardized_context.device
            for value in embeddings
        ):
            raise ValueError("embeddings and standardized context must share dtype and device")
        image_features = daughter_symmetric_features(*embeddings)
        return self.forward_cached_features(image_features, standardized_context)

    def forward_cached_features(
        self,
        image_features: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Score model-ready dataset rows without consulting node caches or private heads."""
        _require_torch()
        if not isinstance(image_features, torch.Tensor) or not isinstance(
            normalized_context, torch.Tensor
        ):
            raise TypeError("cached image features and normalized context must be torch tensors")
        batch = normalized_context.shape[0] if normalized_context.ndim == 2 else -1
        expected_context_shape = (batch, self.config.context_dim)
        expected_image_shape = (batch, self.config.embedding_dim * 4)
        if normalized_context.shape != expected_context_shape:
            raise ValueError(
                "normalized context must have shape "
                f"[batch, {self.config.context_dim}], got {tuple(normalized_context.shape)}"
            )
        if image_features.shape != expected_image_shape:
            raise ValueError(
                "cached image features must have shape "
                f"[batch, {self.config.embedding_dim * 4}], got {tuple(image_features.shape)}"
            )
        if any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in (image_features, normalized_context)
        ):
            raise ValueError("cached model inputs must contain finite floating-point values")
        if (
            image_features.dtype != normalized_context.dtype
            or image_features.device != normalized_context.device
        ):
            raise ValueError("cached model inputs must share dtype and device")
        inputs = normalized_context
        if self.config.model_name != "scalar":
            inputs = torch.cat([image_features, normalized_context], dim=1)
        return self.head(inputs).squeeze(1)


def _validate_fixed_model_architecture(model: DivisionModel) -> None:
    config = DivisionModelConfig.from_dict(asdict(model.config))
    if config.model_name == "scalar":
        valid = (
            type(model.head) is nn.Linear
            and model.head.in_features == config.context_dim
            and model.head.out_features == 1
        )
    elif config.model_name == "embedding_linear":
        valid = (
            type(model.head) is nn.Linear
            and model.head.in_features == config.embedding_dim * 4 + config.context_dim
            and model.head.out_features == 1
        )
    else:
        input_dim = config.embedding_dim * 4 + config.context_dim
        layers = tuple(model.head.children()) if type(model.head) is nn.Sequential else ()
        valid = (
            len(layers) == 5
            and type(layers[0]) is nn.LayerNorm
            and layers[0].normalized_shape == (input_dim,)
            and type(layers[1]) is nn.Linear
            and layers[1].in_features == input_dim
            and layers[1].out_features == config.hidden_dim
            and type(layers[2]) is nn.SiLU
            and type(layers[3]) is nn.Dropout
            and layers[3].p == config.dropout
            and type(layers[4]) is nn.Linear
            and layers[4].in_features == config.hidden_dim
            and layers[4].out_features == 1
        )
    if not valid:
        raise ValueError(f"{config.model_name} head does not match its fixed architecture")


def _validate_contract_hashes(values: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != set(DIVISION_CONTRACT_KINDS):
        raise ValueError("contract SHA-256 mapping must contain all five frozen contracts")
    normalized = {
        kind: _require_sha256(values[kind], field=f"{kind} contract SHA-256")
        for kind in DIVISION_CONTRACT_KINDS
    }
    if normalized != EXPECTED_DIVISION_CONTRACT_SHA256:
        raise ValueError("contract SHA-256 mapping does not match the frozen Task 1 configs")
    return normalized


def current_division_contract_hashes(config_dir: str | Path) -> dict[str, str]:
    """Strictly read and fingerprint the five immutable Task 1 config files."""
    directory = Path(config_dir)
    read_division_contracts(directory)
    observed = {
        kind: file_sha256(directory / f"phase11_division_{kind}.json")
        for kind in DIVISION_CONTRACT_KINDS
    }
    return _validate_contract_hashes(observed)


def _validate_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    return int(value)


def _validate_metrics(values: Any) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("checkpoint metrics must be a mapping")
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint metrics require non-empty string names")
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
            raise ValueError("checkpoint metrics must contain only finite numeric values")
        metrics[name] = float(value)
    return metrics


def _verify_external_base_checkpoint(path: str | Path) -> str:
    observed = file_sha256(path)
    if observed != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError(
            "external base checkpoint SHA-256 does not match the accepted Phase 10 checkpoint"
        )
    return observed


def _atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_finite_checkpoint_tensors(value: Any, *, field: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{field} contains non-finite tensor values")
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            _validate_finite_checkpoint_tensors(child, field=f"{field}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_checkpoint_tensors(child, field=f"{field}[{index}]")


def save_division_checkpoint(
    path: str | Path,
    model: DivisionModel,
    *,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    feature_schema: FeatureSchema,
    dataset_manifest_sha256: str,
    model_inputs_sha256: str,
    trainer_source_sha256: str,
    epoch: int,
    metrics: Mapping[str, Real],
) -> str:
    """Atomically save only a division head and its complete external contract."""
    _require_torch()
    if not isinstance(model, DivisionModel):
        raise TypeError("division checkpoints require a DivisionModel")
    if not isinstance(feature_schema, FeatureSchema):
        raise TypeError("division checkpoints require a FeatureSchema")
    if (
        feature_schema.names != CONTEXT_FEATURE_NAMES
        or feature_schema.kinds != CONTEXT_FEATURE_KINDS
    ):
        raise ValueError("division checkpoint feature schema is not the exact frozen context")
    _validate_fixed_model_architecture(model)
    base_sha256 = _verify_external_base_checkpoint(base_checkpoint_path)
    contracts = current_division_contract_hashes(config_dir)
    checkpoint = {
        "base_checkpoint_sha256": base_sha256,
        "contract_sha256": contracts,
        "dataset_manifest_sha256": _require_sha256(
            dataset_manifest_sha256,
            field="dataset manifest SHA-256",
        ),
        "division_checkpoint_schema_version": DIVISION_CHECKPOINT_SCHEMA_VERSION,
        "epoch": _validate_epoch(epoch),
        "feature_schema_sha256": feature_schema.sha256,
        "head_state_dict": model.head.state_dict(),
        "metrics": _validate_metrics(metrics),
        "model_config": asdict(model.config),
        "model_inputs_sha256": _require_sha256(
            model_inputs_sha256,
            field="model inputs SHA-256",
        ),
        "trainer_source_sha256": _require_sha256(
            trainer_source_sha256,
            field="trainer source SHA-256",
        ),
    }
    _validate_finite_checkpoint_tensors(checkpoint)
    _atomic_torch_save(path, checkpoint)
    return file_sha256(path)


def load_division_checkpoint(
    path: str | Path,
    *,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    feature_schema: FeatureSchema,
    expected_dataset_manifest_sha256: str,
    expected_model_inputs_sha256: str,
    expected_trainer_source_sha256: str,
    expected_model_config: DivisionModelConfig | None = None,
    device: str | torch.device = "cpu",
) -> tuple[DivisionModel, dict[str, Any]]:
    """Load a head only after every external identity matches current frozen files."""
    _require_torch()
    if not isinstance(feature_schema, FeatureSchema):
        raise TypeError("division checkpoint loading requires a FeatureSchema")
    if (
        feature_schema.names != CONTEXT_FEATURE_NAMES
        or feature_schema.kinds != CONTEXT_FEATURE_KINDS
    ):
        raise ValueError("division checkpoint feature schema is not the exact frozen context")
    base_hash = _verify_external_base_checkpoint(base_checkpoint_path)
    contracts = current_division_contract_hashes(config_dir)
    dataset_manifest_hash = _require_sha256(
        expected_dataset_manifest_sha256,
        field="expected dataset manifest SHA-256",
    )
    model_inputs_hash = _require_sha256(
        expected_model_inputs_sha256,
        field="expected model inputs SHA-256",
    )
    trainer_source_hash = _require_sha256(
        expected_trainer_source_sha256,
        field="expected trainer source SHA-256",
    )
    target_device = torch.device(device)
    payload = torch.load(path, map_location=target_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("division checkpoint must contain one mapping")
    _require_exact_fields(payload, _CHECKPOINT_FIELDS, label="division checkpoint")
    if payload["division_checkpoint_schema_version"] != DIVISION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported division checkpoint schema")
    if payload["base_checkpoint_sha256"] != base_hash:
        raise ValueError("division checkpoint base checkpoint SHA-256 mismatch")
    if _validate_contract_hashes(payload["contract_sha256"]) != contracts:
        raise ValueError("division checkpoint contract SHA-256 mismatch")
    if payload["dataset_manifest_sha256"] != dataset_manifest_hash:
        raise ValueError("division checkpoint dataset manifest SHA-256 mismatch")
    if payload["feature_schema_sha256"] != feature_schema.sha256:
        raise ValueError("division checkpoint feature schema SHA-256 mismatch")
    if payload["model_inputs_sha256"] != model_inputs_hash:
        raise ValueError("division checkpoint model inputs SHA-256 mismatch")
    if payload["trainer_source_sha256"] != trainer_source_hash:
        raise ValueError("division checkpoint trainer source SHA-256 mismatch")
    config = DivisionModelConfig.from_dict(payload["model_config"])
    if config.context_dim != len(feature_schema.names):
        raise ValueError("division checkpoint model config disagrees with the feature schema")
    if expected_model_config is not None and config != expected_model_config:
        raise ValueError("division checkpoint model config does not match the expected config")
    epoch = _validate_epoch(payload["epoch"])
    metrics = _validate_metrics(payload["metrics"])
    state = payload["head_state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("division checkpoint head state is invalid")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("division checkpoint head state contains non-finite values")
    model = DivisionModel(config).to(target_device)
    try:
        model.head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"division checkpoint head state does not match model config: {exc}"
        ) from exc
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "head_state_dict"}
    metadata["epoch"] = epoch
    metadata["metrics"] = metrics
    return model, metadata


def raw_patch_intensity_summaries(patches: Any) -> np.ndarray:
    """Return raw ``mean,std,center-background,min,max`` rows as float32."""
    array = np.asarray(patches)
    if array.ndim != 4:
        raise ValueError(f"patches must have shape [batch,z,y,x], got {array.shape}")
    if any(size < 1 or size % 2 == 0 for size in array.shape[1:]):
        raise ValueError("patch dimensions must be positive odd values")
    if int(np.prod(array.shape[1:])) < 2:
        raise ValueError("patches need at least two voxels for a background estimate")
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("patches must contain numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("patches must contain only finite values")
    flat = values.reshape(len(values), -1)
    center_indices = tuple(size // 2 for size in values.shape[1:])
    centers = values[(slice(None), *center_indices)]
    backgrounds = (flat.sum(axis=1) - centers) / (flat.shape[1] - 1)
    summaries = np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            centers - backgrounds,
            flat.min(axis=1),
            flat.max(axis=1),
        ]
    )
    return summaries.astype(np.float32, copy=False)


def _canonical_graph_arrays(graph: TrackingGraph) -> tuple[np.ndarray, np.ndarray]:
    raw_ids = np.asarray(graph.node_ids)
    if raw_ids.ndim != 1 or np.issubdtype(raw_ids.dtype, np.bool_):
        raise ValueError("node-cache node identifiers must be a one-dimensional integer array")
    try:
        numeric_ids = np.asarray(raw_ids, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("node-cache node identifiers must be integral") from exc
    int64 = np.iinfo(np.int64)
    if (
        not np.all(np.isfinite(numeric_ids))
        or not np.all(numeric_ids == np.floor(numeric_ids))
        or np.any(numeric_ids < int64.min)
        or np.any(numeric_ids >= 2**63)
    ):
        raise ValueError("node-cache node identifiers must be signed 64-bit integers")
    node_ids = np.ascontiguousarray(raw_ids, dtype="<i8")
    count = len(node_ids)
    if count < 1:
        raise ValueError("node cache must contain at least one candidate")
    if len(np.unique(node_ids)) != count:
        raise ValueError("node cache contains duplicate node identifiers")

    columns: list[np.ndarray] = []
    for name in ("t", "z", "y", "x"):
        try:
            values = np.asarray(getattr(graph, name), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node-cache {name} coordinates must be numeric") from exc
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"node-cache {name} coordinates must be finite with shape {(count,)}")
        if np.any(values < 0.0):
            raise ValueError(f"node-cache {name} coordinates must be non-negative")
        columns.append(values)
    if not np.all(columns[0] == np.floor(columns[0])):
        raise ValueError("node-cache time coordinates must be integral")
    edges = np.asarray(graph.edges)
    if edges.size != 0:
        raise ValueError("raw division node caches cannot contain graph edges")
    tzyx = np.ascontiguousarray(np.column_stack(columns), dtype="<f8")
    return node_ids, tzyx


def _validate_node_cache_arrays(
    node_ids: np.ndarray,
    tzyx: np.ndarray,
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
) -> None:
    if node_ids.dtype != np.dtype("<i8") or node_ids.ndim != 1:
        raise ValueError("node-cache node_ids must have exact dtype int64 and shape [N]")
    count = len(node_ids)
    if count < 1 or len(np.unique(node_ids)) != count:
        raise ValueError("node-cache node identifiers must be non-empty and unique")
    if tzyx.dtype != np.dtype("<f8") or tzyx.shape != (count, 4):
        raise ValueError("node-cache tzyx must have exact dtype float64 and shape [N,4]")
    if not np.all(np.isfinite(tzyx)) or np.any(tzyx < 0.0):
        raise ValueError("node-cache tzyx values must be finite and non-negative")
    if not np.all(tzyx[:, 0] == np.floor(tzyx[:, 0])):
        raise ValueError("node-cache time coordinates must be integral")
    if cellness.dtype != np.dtype("<f4") or cellness.shape != (count,):
        raise ValueError("node-cache cellness must have exact dtype float32 and shape [N]")
    if not np.all(np.isfinite(cellness)) or np.any(cellness < 0.0) or np.any(cellness > 1.0):
        raise ValueError("node-cache cellness must be finite and between zero and one")
    if embeddings.dtype != np.dtype("<f4") or embeddings.shape != (
        count,
        NODE_CACHE_EMBEDDING_DIM,
    ):
        raise ValueError(
            "node-cache embeddings must have exact dtype float32 and shape "
            f"[N,{NODE_CACHE_EMBEDDING_DIM}]"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("node-cache embeddings must contain only finite values")
    if intensity_summaries.dtype != np.dtype("<f4") or intensity_summaries.shape != (
        count,
        len(INTENSITY_SUMMARY_NAMES),
    ):
        raise ValueError(
            "node-cache intensity summaries must have exact dtype float32 and shape "
            f"[N,{len(INTENSITY_SUMMARY_NAMES)}]"
        )
    if not np.all(np.isfinite(intensity_summaries)):
        raise ValueError("node-cache intensity summaries must contain only finite values")
    means = intensity_summaries[:, 0]
    standard_deviations = intensity_summaries[:, 1]
    minimums = intensity_summaries[:, 3]
    maximums = intensity_summaries[:, 4]
    if (
        np.any(standard_deviations < 0.0)
        or np.any(minimums > maximums)
        or np.any(means < minimums)
        or np.any(means > maximums)
    ):
        raise ValueError("node-cache intensity summaries are internally inconsistent")


def _read_npy_member_header(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with archive.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"node-cache array uses unsupported NPY version {version}")
        data_offset = handle.tell()
    expected_size = data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if member.file_size != expected_size:
        raise ValueError("node-cache array declared shape does not match its member size")
    return tuple(int(value) for value in shape), bool(fortran_order), np.dtype(dtype)


def _read_node_cache_headers(
    source: Path,
) -> tuple[dict[str, tuple[tuple[int, ...], bool, np.dtype[Any]]], np.ndarray]:
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            expected_names = {f"{name}.npy" for name in _NODE_CACHE_ARRAY_FIELDS}
            if (
                len(names) != len(set(names))
                or set(names) != expected_names
                or any(member.is_dir() for member in members)
            ):
                raise ValueError("node cache has incomplete, duplicate, or unexpected members")
            by_name = {member.filename: member for member in members}
            if any(member.compress_type != zipfile.ZIP_DEFLATED for member in members):
                raise ValueError("node-cache arrays must use compressed NPZ members")
            headers = {
                name: _read_npy_member_header(archive, by_name[f"{name}.npy"])
                for name in _NODE_CACHE_ARRAY_FIELDS
            }
            metadata_member = by_name["metadata_json.npy"]
            if metadata_member.file_size > 1_000_000:
                raise ValueError("node-cache metadata member is unreasonably large")
            with archive.open(metadata_member, "r") as handle:
                metadata_array = np.lib.format.read_array(handle, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("node cache"):
            raise
        raise ValueError(f"invalid division node cache {source}: {exc}") from exc
    return headers, np.asarray(metadata_array)


def _validate_node_cache_headers(
    headers: Mapping[str, tuple[tuple[int, ...], bool, np.dtype[Any]]],
) -> int:
    metadata_shape, metadata_fortran, metadata_dtype = headers["metadata_json"]
    if metadata_shape != () or metadata_fortran or metadata_dtype.kind != "U":
        raise ValueError("node-cache metadata_json must be one C-order Unicode scalar")
    node_shape, _, _ = headers["node_ids"]
    if len(node_shape) != 1 or node_shape[0] < 1:
        raise ValueError("node-cache node_ids header must declare non-empty shape [N]")
    count = node_shape[0]
    expected = {
        "node_ids": (np.dtype("<i8"), (count,)),
        "tzyx": (np.dtype("<f8"), (count, 4)),
        "cellness": (np.dtype("<f4"), (count,)),
        "embeddings": (np.dtype("<f4"), (count, NODE_CACHE_EMBEDDING_DIM)),
        "intensity_summaries": (
            np.dtype("<f4"),
            (count, len(INTENSITY_SUMMARY_NAMES)),
        ),
    }
    for name, (expected_dtype, expected_shape) in expected.items():
        shape, fortran_order, dtype = headers[name]
        if dtype != expected_dtype or shape != expected_shape or fortran_order:
            raise ValueError(
                f"node-cache {name} header must declare C-order {expected_dtype} "
                f"with shape {expected_shape}"
            )
    return count


def _content_sha256(
    metadata_without_content_hash: Mapping[str, Any],
    *,
    node_ids: np.ndarray,
    tzyx: np.ndarray,
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(NODE_CACHE_CONTENT_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0metadata\0")
    digest.update(canonical_json_bytes(dict(metadata_without_content_hash)))
    digest.update(b"\0")
    arrays = {
        "node_ids": node_ids,
        "tzyx": tzyx,
        "cellness": cellness,
        "embeddings": embeddings,
        "intensity_summaries": intensity_summaries,
    }
    for name, values in arrays.items():
        shape = ",".join(str(value) for value in values.shape)
        digest.update(name.encode("ascii"))
        digest.update(b"\0dtype\0")
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(b"\0shape\0")
        digest.update(shape.encode("ascii"))
        digest.update(b"\0data\0")
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_node_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    candidate_count: int,
    expected_dataset: str | None,
    expected_role: str | None,
    expected_cache_source_sha256: str | None,
    expected_contract_sha256: Mapping[str, str] | None,
    expected_upstream_manifest_sha256: str | None,
    expected_image_sha256: str | None,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("node-cache metadata must be a mapping")
    _require_exact_fields(metadata, _NODE_CACHE_METADATA_FIELDS, label="node-cache metadata")
    if metadata["division_node_cache_schema_version"] != DIVISION_NODE_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported division node-cache schema")
    dataset = metadata["dataset"]
    if not isinstance(dataset, str) or not dataset.startswith("6bba_"):
        raise ValueError("division node caches accept only 6bba datasets")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError("node-cache dataset does not match the expected dataset")
    role = metadata["role"]
    if role not in ROLE_NAMES:
        raise ValueError("node-cache role is invalid")
    if expected_role is not None and role != expected_role:
        raise ValueError("node-cache role does not match the frozen split")
    if metadata["data_split"] != "train":
        raise ValueError("division node caches must come from the labelled train split")
    if (
        isinstance(metadata["candidate_count"], bool)
        or not isinstance(metadata["candidate_count"], Integral)
        or int(metadata["candidate_count"]) != candidate_count
    ):
        raise ValueError("node-cache candidate count is inconsistent")
    if metadata["detector_method"] != "dog" or metadata["detector_threshold_k"] != 3.0:
        raise ValueError("node cache must use exact DoG k=3 candidate generation")
    if metadata["patch_shape"] != list(NODE_CACHE_PATCH_SHAPE):
        raise ValueError("node cache must use exact patch shape 9x33x33")
    if metadata["normalization"] != NODE_CACHE_NORMALIZATION:
        raise ValueError("node cache uses an unsupported patch normalization")
    if metadata["embedding_dim"] != NODE_CACHE_EMBEDDING_DIM:
        raise ValueError("node cache must contain exact 64-D embeddings")
    if metadata["intensity_summary_names"] != list(INTENSITY_SUMMARY_NAMES):
        raise ValueError("node cache uses an unsupported intensity-summary order")
    if metadata["intensity_contrast_definition"] != INTENSITY_CONTRAST_DEFINITION:
        raise ValueError("node cache uses an unsupported intensity-contrast definition")
    if metadata["backend_policy"] != CACHE_BACKEND_POLICY:
        raise ValueError("node-cache backend policy is not the approved Phase 11 policy")
    if metadata["device"] != CACHE_DEVICE:
        raise ValueError("node-cache device is not the approved MPS device")
    if metadata["backend"] != CACHE_BACKEND:
        raise ValueError("node-cache backend is not the approved PyTorch MPS backend")
    if metadata["torch_version"] != CACHE_TORCH_VERSION:
        raise ValueError("node-cache torch version does not match the approved cache runtime")
    if metadata["inference_dtype"] != CACHE_INFERENCE_DTYPE:
        raise ValueError("node-cache inference dtype must be float32")
    if metadata["checkpoint_sha256"] != EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError("node-cache checkpoint SHA-256 is not the accepted Phase 10 checkpoint")
    if metadata["model_state_sha256"] != EXPECTED_MODEL_STATE_SHA256:
        raise ValueError("node-cache model-state fingerprint is not the accepted Phase 10 state")
    image_hash = _require_sha256(metadata["image_sha256"], field="image SHA-256")
    if expected_image_sha256 is not None and image_hash != _require_sha256(
        expected_image_sha256,
        field="expected image SHA-256",
    ):
        raise ValueError("node-cache image SHA-256 does not match the expected image")
    if metadata["scoring_source_sha256"] != EXPECTED_SCORING_SOURCE_SHA256:
        raise ValueError("node-cache scoring source SHA-256 is not the frozen Phase 10 source")
    cache_source = _require_sha256(
        metadata["cache_source_sha256"],
        field="node-cache source SHA-256",
    )
    if expected_cache_source_sha256 is not None and cache_source != expected_cache_source_sha256:
        raise ValueError("node-cache source SHA-256 does not match current source")
    contracts = _validate_contract_hashes(metadata["contract_sha256"])
    if expected_contract_sha256 is not None and contracts != _validate_contract_hashes(
        expected_contract_sha256
    ):
        raise ValueError("node-cache contract SHA-256 mapping does not match current configs")
    upstream_manifest_hash = _require_sha256(
        metadata["upstream_manifest_sha256"],
        field="upstream manifest SHA-256",
    )
    if (
        expected_upstream_manifest_sha256 is not None
        and upstream_manifest_hash != expected_upstream_manifest_sha256
    ):
        raise ValueError("node-cache upstream manifest SHA-256 does not match the frozen manifest")
    if metadata["cellness_match_atol"] != CELLNESS_MATCH_ATOL:
        raise ValueError("node-cache cellness tolerance is not the frozen tolerance")
    phase10_hash = metadata["phase10_score_cache_sha256"]
    maximum_difference = metadata["phase10_max_cellness_abs_diff"]
    if phase10_hash is None:
        if maximum_difference is not None:
            raise ValueError("node cache records a Phase 10 difference without a cache identity")
    else:
        _require_sha256(phase10_hash, field="Phase 10 score-cache SHA-256")
        if (
            isinstance(maximum_difference, bool)
            or not isinstance(maximum_difference, Real)
            or not np.isfinite(float(maximum_difference))
            or float(maximum_difference) < 0.0
            or float(maximum_difference) > CELLNESS_MATCH_ATOL
        ):
            raise ValueError("node cache exceeds the frozen Phase 10 cellness tolerance")
    _require_sha256(metadata["content_sha256"], field="node-cache content SHA-256")


def _atomic_npz_write(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_division_node_cache(
    path: str | Path,
    graph: TrackingGraph,
    cellness: Any,
    embeddings: Any,
    intensity_summaries: Any,
    *,
    dataset: str,
    role: str,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
    image_sha256: str,
    model_state_sha256: str,
    phase10_score_cache_sha256: str | None,
    phase10_max_cellness_abs_diff: float | None,
) -> str:
    """Atomically write one canonical, provenance-bound movie node cache."""
    node_ids, tzyx = _canonical_graph_arrays(graph)
    score_array = np.ascontiguousarray(cellness, dtype="<f4")
    embedding_array = np.ascontiguousarray(embeddings, dtype="<f4")
    intensity_array = np.ascontiguousarray(intensity_summaries, dtype="<f4")
    _validate_node_cache_arrays(
        node_ids,
        tzyx,
        score_array,
        embedding_array,
        intensity_array,
    )
    metadata_without_hash = {
        "backend": CACHE_BACKEND,
        "backend_policy": CACHE_BACKEND_POLICY,
        "cache_source_sha256": _require_sha256(
            cache_source_sha256,
            field="node-cache source SHA-256",
        ),
        "candidate_count": len(node_ids),
        "cellness_match_atol": CELLNESS_MATCH_ATOL,
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": _validate_contract_hashes(contract_sha256),
        "data_split": "train",
        "dataset": dataset,
        "detector_method": "dog",
        "detector_threshold_k": NODE_CACHE_DETECTOR_THRESHOLD_K,
        "device": CACHE_DEVICE,
        "division_node_cache_schema_version": DIVISION_NODE_CACHE_SCHEMA_VERSION,
        "embedding_dim": NODE_CACHE_EMBEDDING_DIM,
        "image_sha256": _require_sha256(image_sha256, field="image SHA-256"),
        "inference_dtype": CACHE_INFERENCE_DTYPE,
        "intensity_contrast_definition": INTENSITY_CONTRAST_DEFINITION,
        "intensity_summary_names": list(INTENSITY_SUMMARY_NAMES),
        "upstream_manifest_sha256": _require_sha256(
            upstream_manifest_sha256,
            field="upstream manifest SHA-256",
        ),
        "model_state_sha256": _require_sha256(
            model_state_sha256,
            field="model-state SHA-256",
        ),
        "normalization": NODE_CACHE_NORMALIZATION,
        "patch_shape": list(NODE_CACHE_PATCH_SHAPE),
        "phase10_max_cellness_abs_diff": (
            None if phase10_max_cellness_abs_diff is None else float(phase10_max_cellness_abs_diff)
        ),
        "phase10_score_cache_sha256": phase10_score_cache_sha256,
        "role": role,
        "scoring_source_sha256": scoring_source_sha256,
        "torch_version": CACHE_TORCH_VERSION,
    }
    content_hash = _content_sha256(
        metadata_without_hash,
        node_ids=node_ids,
        tzyx=tzyx,
        cellness=score_array,
        embeddings=embedding_array,
        intensity_summaries=intensity_array,
    )
    metadata = {**metadata_without_hash, "content_sha256": content_hash}
    _validate_node_cache_metadata(
        metadata,
        candidate_count=len(node_ids),
        expected_dataset=dataset,
        expected_role=role,
        expected_cache_source_sha256=cache_source_sha256,
        expected_contract_sha256=contract_sha256,
        expected_upstream_manifest_sha256=upstream_manifest_sha256,
        expected_image_sha256=image_sha256,
    )
    metadata_json = canonical_json_bytes(metadata).decode("ascii")
    _atomic_npz_write(
        path,
        {
            "metadata_json": np.array(metadata_json),
            "node_ids": node_ids,
            "tzyx": tzyx,
            "cellness": score_array,
            "embeddings": embedding_array,
            "intensity_summaries": intensity_array,
        },
    )
    return content_hash


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"node-cache metadata contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"node-cache metadata contains non-finite value {value}")


def _parse_node_cache_metadata_array(metadata_array: np.ndarray) -> dict[str, Any]:
    if metadata_array.shape != () or metadata_array.dtype.kind != "U":
        raise ValueError("node-cache metadata_json must be one Unicode scalar")
    metadata_json = str(metadata_array.item())
    try:
        metadata = json.loads(
            metadata_json,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid node-cache metadata JSON: {exc}") from exc
    if (
        not isinstance(metadata, dict)
        or canonical_json_bytes(metadata).decode("ascii") != metadata_json
    ):
        raise ValueError("node-cache metadata JSON is not canonical")
    return metadata


def read_division_node_cache_metadata(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_role: str | None = None,
    expected_cache_source_sha256: str | None = None,
    expected_contract_sha256: Mapping[str, str] | None = None,
    expected_upstream_manifest_sha256: str | None = None,
    expected_image_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate cache provenance and array headers without materializing payload arrays."""
    source = Path(path)
    headers, metadata_array = _read_node_cache_headers(source)
    candidate_count = _validate_node_cache_headers(headers)
    metadata = _parse_node_cache_metadata_array(metadata_array)
    _validate_node_cache_metadata(
        metadata,
        candidate_count=candidate_count,
        expected_dataset=expected_dataset,
        expected_role=expected_role,
        expected_cache_source_sha256=expected_cache_source_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_upstream_manifest_sha256=expected_upstream_manifest_sha256,
        expected_image_sha256=expected_image_sha256,
    )
    observed_file_sha256 = file_sha256(source)
    if expected_file_sha256 is not None and observed_file_sha256 != _require_sha256(
        expected_file_sha256,
        field="expected node-cache file SHA-256",
    ):
        raise ValueError("node-cache file SHA-256 does not match the expected snapshot")
    return metadata, observed_file_sha256


def read_division_node_cache(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_role: str | None = None,
    expected_cache_source_sha256: str | None = None,
    expected_contract_sha256: Mapping[str, str] | None = None,
    expected_upstream_manifest_sha256: str | None = None,
    expected_image_sha256: str | None = None,
) -> tuple[TrackingGraph, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a cache only after strict array, provenance, and content validation."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != _NODE_CACHE_ARRAY_FIELDS:
                raise ValueError(f"node cache {source} has wrong arrays: {sorted(payload.files)}")
            metadata_array = np.asarray(payload["metadata_json"])
            node_ids = np.array(payload["node_ids"], copy=True)
            tzyx = np.array(payload["tzyx"], copy=True)
            cellness = np.array(payload["cellness"], copy=True)
            embeddings = np.array(payload["embeddings"], copy=True)
            intensity_summaries = np.array(payload["intensity_summaries"], copy=True)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("node cache"):
            raise
        raise ValueError(f"invalid division node cache {source}: {exc}") from exc
    metadata = _parse_node_cache_metadata_array(metadata_array)
    _validate_node_cache_arrays(
        node_ids,
        tzyx,
        cellness,
        embeddings,
        intensity_summaries,
    )
    _validate_node_cache_metadata(
        metadata,
        candidate_count=len(node_ids),
        expected_dataset=expected_dataset,
        expected_role=expected_role,
        expected_cache_source_sha256=expected_cache_source_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_upstream_manifest_sha256=expected_upstream_manifest_sha256,
        expected_image_sha256=expected_image_sha256,
    )
    metadata_without_hash = dict(metadata)
    recorded_content_hash = metadata_without_hash.pop("content_sha256")
    observed_content_hash = _content_sha256(
        metadata_without_hash,
        node_ids=node_ids,
        tzyx=tzyx,
        cellness=cellness,
        embeddings=embeddings,
        intensity_summaries=intensity_summaries,
    )
    if observed_content_hash != recorded_content_hash:
        raise ValueError("node-cache content SHA-256 does not match its arrays and metadata")
    graph = TrackingGraph(
        node_ids=node_ids,
        t=tzyx[:, 0].astype(np.int64),
        z=tzyx[:, 1].copy(),
        y=tzyx[:, 2].copy(),
        x=tzyx[:, 3].copy(),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    return graph, cellness, embeddings, intensity_summaries, metadata


def validate_division_node_cache_cohort(
    paths: Sequence[str | Path],
) -> tuple[dict[str, Any], ...]:
    """Validate caches and require one inference backend provenance cohort."""
    if not paths:
        raise ValueError("division node-cache cohort must not be empty")
    metadata_rows = tuple(read_division_node_cache_metadata(path)[0] for path in paths)
    backend_identities = {
        (
            row["backend_policy"],
            row["device"],
            row["backend"],
            row["torch_version"],
            row["inference_dtype"],
        )
        for row in metadata_rows
    }
    if len(backend_identities) != 1:
        raise ValueError("division node-cache cohort mixes backend provenance")
    return metadata_rows
