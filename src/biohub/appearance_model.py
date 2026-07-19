"""PyTorch model and inference helpers for 3D candidate cellness scoring."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biohub.appearance import (
    APPEARANCE_HOLDOUT_EMBRYO,
    APPEARANCE_SCHEMA_VERSION,
    APPEARANCE_TRAIN_EMBRYO,
    SAMPLE_KNOWN_POSITIVE,
    SAMPLE_UNLABELED,
    discover_appearance_shards,
    extract_patches,
    normalize_patch,
    normalize_patches,
    read_appearance_shard,
    validate_patch_shape,
)
from biohub.metric import TrackingGraph

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without the ML extra
    raise ModuleNotFoundError(
        "The appearance model requires PyTorch. Install it with `python -m pip install -e '.[ml]'`."
    ) from exc

CHECKPOINT_SCHEMA_VERSION = 2
PATCH_MEMMAP_CACHE_SIZE = 16


@dataclass(frozen=True)
class CellnessModelConfig:
    """Architecture parameters saved inside every model checkpoint."""

    base_channels: int = 16
    dropout: float = 0.2


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock3D(nn.Module):
    """A compact 3D residual block with batch-size-independent normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip: nn.Module
        if in_channels == out_channels and stride == (1, 1, 1):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        features = F.silu(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return F.silu(features + residual)


class CellnessCNN3D(nn.Module):
    """Small anisotropy-aware CNN that emits one cellness logit per 3D patch.

    The first kernel sees more context in ``y`` and ``x`` than in ``z``.  The first
    downsampling step also preserves ``z`` resolution because the microscopy slices are
    four times farther apart physically than adjacent pixels in the image plane.
    """

    def __init__(self, config: CellnessModelConfig | None = None) -> None:
        super().__init__()
        config = config or CellnessModelConfig()
        if config.base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        channels = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                channels,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.features = nn.Sequential(
            ResidualBlock3D(channels, channels),
            ResidualBlock3D(channels, channels * 2, stride=(1, 2, 2)),
            ResidualBlock3D(channels * 2, channels * 4, stride=(2, 2, 2)),
            ResidualBlock3D(channels * 4, channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(channels * 4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError(
                f"CellnessCNN3D expects [batch, 1, z, y, x], got {tuple(inputs.shape)}"
            )
        return self.head(self.pool(self.features(self.stem(inputs)))).squeeze(1)


def augment_patch(patch: torch.Tensor) -> torch.Tensor:
    """Apply orientation-preserving random augmentation to one ``[1,z,y,x]`` patch."""
    if torch.rand(()) < 0.5:
        patch = torch.flip(patch, dims=(1,))
    if torch.rand(()) < 0.5:
        patch = torch.flip(patch, dims=(2,))
    if torch.rand(()) < 0.5:
        patch = torch.flip(patch, dims=(3,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        patch = torch.rot90(patch, rotations, dims=(2, 3))
    return patch


class CandidatePatchDataset(Dataset):
    """Memory-mapped patches drawn from one or more complete movie shards."""

    def __init__(self, shards: list[str | Path], *, augment: bool = False) -> None:
        if not shards:
            raise ValueError("CandidatePatchDataset requires at least one shard")
        self.shards = [Path(path) for path in shards]
        self.summaries = [read_appearance_shard(path) for path in self.shards]
        embryos = {summary.embryo for summary in self.summaries}
        if embryos != {APPEARANCE_TRAIN_EMBRYO}:
            raise ValueError(
                f"appearance fitting accepts only embryo {APPEARANCE_TRAIN_EMBRYO}, "
                f"found {sorted(embryos)}"
            )
        shapes = {summary.patch_shape for summary in self.summaries}
        if len(shapes) != 1:
            raise ValueError(f"all shards must use the same patch shape, found {sorted(shapes)}")
        detector_thresholds = {summary.detector_threshold_k for summary in self.summaries}
        if len(detector_thresholds) != 1:
            raise ValueError(
                f"all shards must use the same DoG threshold, found {sorted(detector_thresholds)}"
            )

        self.patch_shape = next(iter(shapes))
        self.detector_threshold_k = next(iter(detector_thresholds))
        self.augment = augment
        self._offsets = [0]
        self._roles: list[np.ndarray] = []
        positive_indices: list[np.ndarray] = []
        unlabeled_indices: list[np.ndarray] = []
        for shard, summary in zip(self.shards, self.summaries, strict=True):
            roles = np.load(shard / "sample_roles.npy", allow_pickle=False)
            if roles.ndim != 1:
                raise ValueError(f"sample roles must be one-dimensional in {shard}")
            if not np.all(np.isin(roles, [SAMPLE_UNLABELED, SAMPLE_KNOWN_POSITIVE])):
                raise ValueError(f"unknown sample role in {shard}")
            if int(np.sum(roles == SAMPLE_KNOWN_POSITIVE)) != summary.known_positive_samples:
                raise ValueError(f"known-positive count does not match metadata in {shard}")
            if int(np.sum(roles == SAMPLE_UNLABELED)) != summary.unlabeled_samples:
                raise ValueError(f"unlabeled count does not match metadata in {shard}")
            self._roles.append(np.asarray(roles, dtype=np.uint8))
            offset = self._offsets[-1]
            positive_indices.append(np.flatnonzero(roles == SAMPLE_KNOWN_POSITIVE) + offset)
            unlabeled_indices.append(np.flatnonzero(roles == SAMPLE_UNLABELED) + offset)
            self._offsets.append(offset + len(roles))
        self.positive_indices = np.concatenate(positive_indices).astype(np.int64)
        self.unlabeled_indices = np.concatenate(unlabeled_indices).astype(np.int64)
        self._patch_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    @classmethod
    def from_root(cls, root: str | Path, *, augment: bool = False) -> CandidatePatchDataset:
        return cls(discover_appearance_shards(root), augment=augment)

    @property
    def class_prior(self) -> float:
        """PU class prior weighted by each shard's sampled unlabeled population."""
        weights = np.array([summary.unlabeled_samples for summary in self.summaries], dtype=float)
        priors = np.array(
            [summary.estimated_candidate_prior for summary in self.summaries], dtype=float
        )
        return float(np.average(priors, weights=weights))

    def __len__(self) -> int:
        return self._offsets[-1]

    def _patch_array(self, shard_index: int) -> np.ndarray:
        shard = self.shards[shard_index]
        if shard in self._patch_cache:
            self._patch_cache.move_to_end(shard)
            return self._patch_cache[shard]
        patches = np.load(shard / "patches.npy", mmap_mode="r", allow_pickle=False)
        self._patch_cache[shard] = patches
        if len(self._patch_cache) > PATCH_MEMMAP_CACHE_SIZE:
            self._patch_cache.popitem(last=False)
        return patches

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[shard_index]
        patches = self._patch_array(shard_index)
        patch = torch.from_numpy(normalize_patch(patches[local_index])).unsqueeze(0)
        if self.augment:
            patch = augment_patch(patch)
        role = self._roles[shard_index][local_index]
        return patch, torch.tensor(float(role), dtype=torch.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_patch_cache"] = OrderedDict()
        return state


def non_negative_pu_loss(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    class_prior: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the non-negative positive-unlabeled logistic risk.

    ``class_prior`` is the estimated fraction of DoG candidates that are genuine cells.
    The unlabeled batch remains a real candidate-population mixture.  Clamping only the
    estimated negative risk prevents the model from exploiting sparse labels by driving
    the empirical risk below zero.
    """
    if positive_logits.numel() == 0 or unlabeled_logits.numel() == 0:
        raise ValueError("positive and unlabeled batches must both be non-empty")
    if not 0.0 < class_prior < 1.0:
        raise ValueError("class_prior must be strictly between 0 and 1")
    positive_as_positive = F.softplus(-positive_logits).mean()
    positive_as_negative = F.softplus(positive_logits).mean()
    unlabeled_as_negative = F.softplus(unlabeled_logits).mean()
    positive_risk = class_prior * positive_as_positive
    raw_negative_risk = unlabeled_as_negative - class_prior * positive_as_negative
    corrected_negative_risk = torch.clamp(raw_negative_risk, min=0.0)
    loss = positive_risk + corrected_negative_risk
    return loss, {
        "positive_risk": positive_risk,
        "raw_negative_risk": raw_negative_risk,
        "corrected_negative_risk": corrected_negative_risk,
    }


def nnpu_training_objective(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    class_prior: float,
    *,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return the large-scale nnPU update objective and non-negative risk estimate.

    The non-negative risk is the quantity used for development evaluation. During
    stochastic training, a minibatch with raw negative risk below ``-beta`` instead
    follows the reversed negative-risk gradient, discounted by ``gamma``. This is the
    correction step from Algorithm 1 of Kiryo et al. (NeurIPS 2017).
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be greater than 0 and at most 1")
    non_negative_risk, risks = non_negative_pu_loss(
        positive_logits,
        unlabeled_logits,
        class_prior,
    )
    raw_negative_risk = risks["raw_negative_risk"]
    correction_active = raw_negative_risk < -beta
    unbiased_risk = risks["positive_risk"] + raw_negative_risk
    correction_objective = -gamma * raw_negative_risk
    objective = torch.where(correction_active, correction_objective, unbiased_risk)
    return objective, non_negative_risk, {
        **risks,
        "correction_active": correction_active.to(dtype=positive_logits.dtype),
    }


def positive_unlabeled_auc(
    positive_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
) -> float:
    """Measure how often a known positive ranks above an unlabeled candidate.

    The unlabeled population contains real cells, so this is a conservative PU ranking
    diagnostic rather than an ordinary positive-vs-negative ROC AUC.
    """
    from scipy.stats import rankdata

    positive = np.asarray(positive_scores, dtype=float).reshape(-1)
    unlabeled = np.asarray(unlabeled_scores, dtype=float).reshape(-1)
    if len(positive) == 0 or len(unlabeled) == 0:
        raise ValueError("positive and unlabeled score arrays must both be non-empty")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(unlabeled)):
        raise ValueError("positive and unlabeled scores must all be finite")
    combined = np.concatenate([positive, unlabeled])
    ranks = rankdata(combined, method="average")
    positive_rank_sum = float(ranks[: len(positive)].sum())
    baseline = len(positive) * (len(positive) + 1) / 2.0
    return (positive_rank_sum - baseline) / (len(positive) * len(unlabeled))


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in that order."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def score_candidates(
    model: CellnessCNN3D,
    image_array,
    candidates: TrackingGraph,
    *,
    patch_shape: tuple[int, int, int],
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Score every candidate in graph order while reading each image frame once."""
    shape = validate_patch_shape(patch_shape)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    scores = np.empty(len(candidates.node_ids), dtype=np.float32)
    candidate_t = np.asarray(candidates.t).astype(int)
    zyx = np.column_stack([candidates.z, candidates.y, candidates.x])
    for timepoint in np.unique(candidate_t):
        frame_indices = np.flatnonzero(candidate_t == timepoint)
        volume = np.asarray(image_array[int(timepoint)])
        patches = normalize_patches(extract_patches(volume, zyx[frame_indices], shape))
        frame_scores: list[torch.Tensor] = []
        for start in range(0, len(frame_indices), batch_size):
            inputs = torch.from_numpy(patches[start : start + batch_size]).unsqueeze(1)
            inputs = inputs.to(target_device)
            frame_scores.append(torch.sigmoid(model(inputs)))
        scores[frame_indices] = torch.cat(frame_scores).cpu().numpy()
    return scores


def save_cellness_checkpoint(
    path: str | Path,
    model: CellnessCNN3D,
    *,
    patch_shape: tuple[int, int, int],
    class_prior: float,
    detector_threshold_k: float,
    epoch: int,
    train_shards: list[str],
    dev_shards: list[str],
    metrics: dict[str, float],
    training_config: dict[str, Any],
) -> None:
    """Atomically save model weights plus the complete train and split contract."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "appearance_schema_version": APPEARANCE_SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "patch_shape": list(validate_patch_shape(patch_shape)),
        "normalization": "per_patch_mean_std_clip_6",
        "class_prior": float(class_prior),
        "detector_threshold_k": float(detector_threshold_k),
        "train_embryo": APPEARANCE_TRAIN_EMBRYO,
        "holdout_embryo": APPEARANCE_HOLDOUT_EMBRYO,
        "epoch": int(epoch),
        "train_shards": list(train_shards),
        "dev_shards": list(dev_shards),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "training_config": dict(training_config),
        "model_state_dict": model.state_dict(),
    }
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def load_cellness_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CellnessCNN3D, dict[str, Any]]:
    """Load a trusted project checkpoint and validate its leakage guardrails."""
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location=target_device, weights_only=True)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported cellness checkpoint schema")
    if checkpoint.get("appearance_schema_version") != APPEARANCE_SCHEMA_VERSION:
        raise ValueError("checkpoint uses an incompatible appearance-data schema")
    if checkpoint.get("train_embryo") != APPEARANCE_TRAIN_EMBRYO:
        raise ValueError("checkpoint was not fitted exclusively on the training embryo")
    if checkpoint.get("holdout_embryo") != APPEARANCE_HOLDOUT_EMBRYO:
        raise ValueError("checkpoint does not preserve the configured embryo holdout")
    if checkpoint.get("normalization") != "per_patch_mean_std_clip_6":
        raise ValueError("checkpoint uses an unsupported patch normalization")
    if float(checkpoint.get("detector_threshold_k", 0.0)) <= 0.0:
        raise ValueError("checkpoint does not record a valid DoG candidate threshold")
    validate_patch_shape(tuple(checkpoint["patch_shape"]))
    model = CellnessCNN3D(CellnessModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint
