#!/usr/bin/env python3
"""Train the fixed Phase 11 division-head ladder from cached fit features."""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by the clean-clone help test
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from biohub import division_model as division_model_module  # noqa: E402
from biohub.division_data import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    read_canonical_json,
)
from biohub.division_dataset import (  # noqa: E402
    BUILD_MODE_FIT_COMPLETE,
    BUILD_MODE_SMOKE,
    DIAGNOSTIC_UNKNOWN,
    EXPECTED_FIT_MOVIES,
    SUPERVISED_STRATA,
    LoadedDivisionDataset,
    ProposalRecord,
    balanced_epoch_indices,
    read_division_dataset,
    read_smoke_division_dataset,
)
from biohub.division_model import (  # noqa: E402
    DIVISION_CHECKPOINT_SCHEMA_VERSION,
    DIVISION_MODEL_NAMES,
    FIXED_DIVISION_MODEL_CONFIGS,
    DivisionModel,
    current_division_contract_hashes,
    load_division_checkpoint,
    save_division_checkpoint,
)

SEED = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BATCH_SIZE = 64
GRADIENT_CLIP_NORM = 1.0
PRODUCTION_EPOCHS = 100
NUM_WORKERS = 0
MODEL_NAMES = DIVISION_MODEL_NAMES

TRAINING_STATE_SCHEMA_VERSION = 1
TRAINING_HISTORY_SCHEMA_VERSION = 1

DEFAULT_DATASET = ROOT / "data" / "working" / "phase11_division_dataset" / "fit_complete"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "working" / "phase11_division_checkpoints"
DEFAULT_SMOKE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "smoke"
DEFAULT_BASE_CHECKPOINT = (
    ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn_phase10_full.pt"
)
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"
DEFAULT_CACHE_DIR = ROOT / "data" / "working" / "phase11_division_cache"
DEFAULT_UPSTREAM_MANIFEST = ROOT / "data" / "working" / "phase11_division_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "working" / "phase11_division_manifest_summary.json"

_STATE_FIELDS = {
    "elapsed_seconds",
    "head_state_dict",
    "history",
    "last_epoch",
    "nonproduction",
    "numpy_generator_state",
    "optimizer_state_dict",
    "torch_cpu_rng_state",
    "torch_mps_rng_state",
    "training_signature",
    "training_signature_sha256",
    "training_state_schema_version",
}
_HISTORY_FIELDS = {
    "completed_epochs",
    "elapsed_seconds",
    "epochs_requested",
    "history",
    "model_name",
    "nonproduction",
    "training_history_schema_version",
    "training_signature",
    "training_signature_sha256",
}


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "Division training requires PyTorch. Install it with "
            "`.venv/bin/python -m pip install -e '.[ml]'`."
        )


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


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    seed: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip_norm: float
    epochs: int
    loss: str
    num_workers: int
    nonproduction: bool

    @classmethod
    def for_mode(
        cls,
        *,
        nonproduction: bool,
        epochs: int | None,
    ) -> TrainingSettings:
        if not isinstance(nonproduction, bool):
            raise TypeError("nonproduction must be boolean")
        resolved_epochs = PRODUCTION_EPOCHS if epochs is None else epochs
        if (
            isinstance(resolved_epochs, bool)
            or not isinstance(resolved_epochs, Integral)
            or int(resolved_epochs) < 1
        ):
            raise ValueError("training epochs must be a positive integer")
        resolved_epochs = int(resolved_epochs)
        if not nonproduction and resolved_epochs != PRODUCTION_EPOCHS:
            raise ValueError("production training requires exactly 100 epochs")
        if nonproduction and resolved_epochs > PRODUCTION_EPOCHS:
            raise ValueError("nonproduction smoke training may not exceed 100 epochs")
        return cls(
            seed=SEED,
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            epochs=resolved_epochs,
            loss="BCEWithLogitsLoss",
            num_workers=NUM_WORKERS,
            nonproduction=nonproduction,
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactPaths:
    checkpoint: Path
    history: Path
    state: Path
    lock: Path


@dataclass(frozen=True, slots=True)
class ReaderInputs:
    dataset_path: Path
    project_root: Path
    data_root: Path
    config_dir: Path
    upstream_manifest_path: Path
    summary_path: Path
    cache_dir: Path
    checkpoint_path: Path

    @classmethod
    def synthetic(cls, root: str | Path) -> ReaderInputs:
        directory = Path(root)
        return cls(
            dataset_path=directory / "dataset",
            project_root=directory,
            data_root=directory / "raw",
            config_dir=directory / "configs",
            upstream_manifest_path=directory / "manifest.csv",
            summary_path=directory / "summary.json",
            cache_dir=directory / "cache",
            checkpoint_path=directory / "base.pt",
        )

    def reader_kwargs(self) -> dict[str, Path]:
        return {
            "project_root": self.project_root,
            "data_root": self.data_root,
            "config_dir": self.config_dir,
            "upstream_manifest_path": self.upstream_manifest_path,
            "summary_path": self.summary_path,
            "cache_dir": self.cache_dir,
            "checkpoint_path": self.checkpoint_path,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_name: str
    paths: ModelArtifactPaths
    completed_epochs: int
    checkpoint_sha256: str
    trainer_source_sha256: str
    metrics: Mapping[str, float]
    resumed: bool


def model_artifact_paths(output_dir: str | Path, model_name: str) -> ModelArtifactPaths:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model name must be one of {MODEL_NAMES}")
    output = Path(output_dir)
    return ModelArtifactPaths(
        checkpoint=output / f"{model_name}.pt",
        history=output / f"{model_name}.history.json",
        state=output / f"{model_name}.state.pt",
        lock=output / f"{model_name}.lock",
    )


def validate_artifact_start(paths: ModelArtifactPaths, *, resume: bool) -> None:
    artifacts = (paths.checkpoint, paths.history, paths.state)
    if not resume:
        existing = [path for path in artifacts if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"training artifact already exists: {names}; pass --resume")
        return
    if not paths.state.is_file():
        raise FileNotFoundError(f"resume state does not exist: {paths.state}")


@contextmanager
def exclusive_model_lock(paths: ModelArtifactPaths) -> Iterator[None]:
    paths.lock.parent.mkdir(parents=True, exist_ok=True)
    handle = paths.lock.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(
                f"division training is already active for {paths.checkpoint.stem}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, writer: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json_write(path: str | Path, payload: Mapping[str, Any]) -> str:
    content = canonical_json_bytes(dict(payload))
    _atomic_write(Path(path), lambda handle: handle.write(content))
    return hashlib.sha256(content).hexdigest()


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> str:
    _require_torch()
    destination = Path(path)
    _atomic_write(destination, lambda handle: torch.save(dict(payload), handle))
    return file_sha256(destination)


def read_training_dataset(
    inputs: ReaderInputs,
    *,
    nonproduction: bool,
) -> LoadedDivisionDataset:
    if not isinstance(inputs, ReaderInputs):
        raise TypeError("division training reader requires ReaderInputs")
    reader = read_smoke_division_dataset if nonproduction else read_division_dataset
    loaded = reader(inputs.dataset_path, **inputs.reader_kwargs())
    expected_mode = BUILD_MODE_SMOKE if nonproduction else BUILD_MODE_FIT_COMPLETE
    if loaded.manifest.get("build_mode") != expected_mode:
        if nonproduction:
            raise ValueError("nonproduction training requires an explicit smoke dataset")
        raise ValueError("production training requires the complete fit_complete dataset")
    datasets = loaded.manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("division training dataset does not declare its movie tuple")
    if nonproduction:
        if len(datasets) != 1:
            raise ValueError("smoke training requires exactly one fit movie")
    elif len(datasets) != EXPECTED_FIT_MOVIES:
        raise ValueError(
            f"production fit_complete training requires exactly {EXPECTED_FIT_MOVIES} movies"
        )
    return loaded


def _validate_epoch_indices(
    records: Sequence[ProposalRecord],
    indices: np.ndarray,
) -> dict[str, int]:
    if indices.dtype != np.dtype("int64") or indices.ndim != 1 or len(indices) == 0:
        raise ValueError("balanced epoch indices must be a nonempty int64 vector")
    if np.any(indices < 0) or np.any(indices >= len(records)):
        raise ValueError("balanced epoch index is outside the dataset")
    selected = [records[int(index)] for index in indices]
    if any(record.sampling_stratum == DIAGNOSTIC_UNKNOWN for record in selected):
        raise RuntimeError("diagnostic rows entered a supervised training epoch")
    if any(record.label not in (0, 1) or record.loss_weight != 1.0 for record in selected):
        raise RuntimeError("training epoch contains an unsupervised label or weight")
    counts = Counter(record.sampling_stratum for record in selected)
    if set(counts) != set(SUPERVISED_STRATA) or len(set(counts.values())) != 1:
        raise RuntimeError("training epoch is not balanced 1:1:1 across supervised strata")
    expected_target = max(
        sum(record.sampling_stratum == stratum for record in records)
        for stratum in SUPERVISED_STRATA
    )
    expected_length = expected_target * len(SUPERVISED_STRATA)
    if len(indices) != expected_length or set(counts.values()) != {expected_target}:
        raise RuntimeError("training epoch length disagrees with the balanced sampler")
    return {stratum: counts[stratum] for stratum in SUPERVISED_STRATA}


def epoch_batches(
    records: Sequence[ProposalRecord],
    *,
    epoch: int,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    indices = balanced_epoch_indices(records, seed=int(seed), epoch=int(epoch))
    _validate_epoch_indices(records, indices)
    size = int(batch_size)
    return tuple(indices[start : start + size].copy() for start in range(0, len(indices), size))


def build_optimizer(model: DivisionModel) -> torch.optim.AdamW:
    _require_torch()
    if not isinstance(model, DivisionModel):
        raise TypeError("division optimizer requires a DivisionModel")
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    validate_head_only_optimizer(model, optimizer)
    return optimizer


def validate_head_only_optimizer(
    model: DivisionModel,
    optimizer: torch.optim.Optimizer,
) -> None:
    _require_torch()
    expected = [parameter for parameter in model.head.parameters()]
    observed = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    if (
        len(observed) != len(expected)
        or len({id(parameter) for parameter in observed}) != len(observed)
        or {id(parameter) for parameter in observed} != {id(parameter) for parameter in expected}
    ):
        raise ValueError("division optimizer must contain head parameters only")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    if model_parameters != {id(parameter) for parameter in expected}:
        raise ValueError("division training model unexpectedly contains non-head parameters")
    fixed_group = {
        "lr": LEARNING_RATE,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1 or any(
        optimizer.param_groups[0].get(name) != value for name, value in fixed_group.items()
    ):
        raise ValueError("division optimizer must use the exact fixed AdamW settings")


def _require_finite_tensor(value: torch.Tensor, *, field: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite {field} detected")


def _require_finite_tensors(value: Any, *, field: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            _require_finite_tensor(value, field=field)
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            _require_finite_tensors(child, field=f"{field}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tensors(child, field=f"{field}[{index}]")


def _validate_model_weights(model: DivisionModel) -> None:
    for name, parameter in model.head.named_parameters():
        _require_finite_tensor(parameter, field=f"model weights {name}")


def _validate_gradients(model: DivisionModel) -> None:
    for name, parameter in model.head.named_parameters():
        if parameter.grad is None:
            raise FloatingPointError(f"missing gradients for head parameter {name}")
        _require_finite_tensor(parameter.grad, field=f"gradients {name}")


def run_epoch(
    model: DivisionModel,
    optimizer: torch.optim.Optimizer,
    loaded: LoadedDivisionDataset,
    *,
    epoch: int,
    settings: TrainingSettings,
    device: torch.device,
) -> dict[str, Any]:
    _require_torch()
    if model.config != FIXED_DIVISION_MODEL_CONFIGS.get(model.config.model_name):
        raise ValueError("division model does not use a fixed ladder config")
    validate_head_only_optimizer(model, optimizer)
    _validate_model_weights(model)
    batches = epoch_batches(
        loaded.records,
        epoch=epoch,
        batch_size=settings.batch_size,
        seed=settings.seed,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    totals = {
        stratum: {"count": 0, "logit_sum": 0.0, "loss_sum": 0.0}
        for stratum in SUPERVISED_STRATA
    }
    total_loss = 0.0
    total_count = 0
    maximum_gradient_norm = 0.0
    model.train()
    for indices in batches:
        image_features = torch.from_numpy(
            np.array(loaded.image_features[indices], dtype=np.float32, copy=True)
        ).to(device)
        normalized_context = torch.from_numpy(
            np.array(loaded.normalized_context[indices], dtype=np.float32, copy=True)
        ).to(device)
        labels = torch.tensor(
            [loaded.records[int(index)].label for index in indices],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_cached_features(image_features, normalized_context)
        _require_finite_tensor(logits, field="logits")
        losses = criterion(logits, labels)
        _require_finite_tensor(losses, field="loss")
        loss = losses.mean()
        _require_finite_tensor(loss, field="loss")
        loss.backward()
        _validate_gradients(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.head.parameters(),
            max_norm=settings.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        _require_finite_tensor(gradient_norm, field="gradient norm")
        _validate_gradients(model)
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.detach().cpu()))
        optimizer.step()
        _validate_model_weights(model)
        _require_finite_tensors(optimizer.state_dict(), field="optimizer state")

        detached_logits = logits.detach().cpu()
        detached_losses = losses.detach().cpu()
        total_loss += float(detached_losses.sum())
        total_count += len(indices)
        for offset, raw_index in enumerate(indices):
            stratum = loaded.records[int(raw_index)].sampling_stratum
            if stratum not in totals:
                raise RuntimeError("a non-supervised stratum entered training diagnostics")
            totals[stratum]["count"] += 1
            totals[stratum]["logit_sum"] += float(detached_logits[offset])
            totals[stratum]["loss_sum"] += float(detached_losses[offset])

    diagnostics: dict[str, dict[str, float | int]] = {}
    for stratum in SUPERVISED_STRATA:
        count = int(totals[stratum]["count"])
        if count < 1:
            raise RuntimeError(f"training diagnostics are missing stratum {stratum}")
        diagnostics[stratum] = {
            "count": count,
            "mean_logit": float(totals[stratum]["logit_sum"]) / count,
            "loss": float(totals[stratum]["loss_sum"]) / count,
        }
    if len({int(values["count"]) for values in diagnostics.values()}) != 1:
        raise RuntimeError("training diagnostics are not balanced 1:1:1")
    return {
        "batch_count": len(batches),
        "gradient_norm_max": maximum_gradient_norm,
        "loss": total_loss / total_count,
        "sample_count": total_count,
        "strata": diagnostics,
    }


def _flat_checkpoint_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    flat = {
        "loss": float(metrics["loss"]),
        "batch_count": float(metrics["batch_count"]),
        "sample_count": float(metrics["sample_count"]),
        "gradient_norm_max": float(metrics["gradient_norm_max"]),
    }
    strata = metrics["strata"]
    for stratum in SUPERVISED_STRATA:
        values = strata[stratum]
        flat[f"{stratum}_count"] = float(values["count"])
        flat[f"{stratum}_mean_logit"] = float(values["mean_logit"])
        flat[f"{stratum}_loss"] = float(values["loss"])
    if any(not np.isfinite(value) for value in flat.values()):
        raise FloatingPointError("non-finite checkpoint metrics detected")
    return flat


def trainer_source_sha256(path: str | Path | None = None) -> str:
    return file_sha256(Path(__file__).resolve() if path is None else path)


def _validate_loaded_mode(loaded: LoadedDivisionDataset, settings: TrainingSettings) -> str:
    observed = loaded.manifest.get("build_mode")
    expected = BUILD_MODE_SMOKE if settings.nonproduction else BUILD_MODE_FIT_COMPLETE
    if observed != expected:
        raise ValueError(f"training settings require a {expected} dataset, got {observed!r}")
    return str(observed)


def build_training_signature(
    loaded: LoadedDivisionDataset,
    *,
    model_name: str,
    settings: TrainingSettings,
    config_dir: str | Path,
    base_checkpoint_path: str | Path,
    trainer_source_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    _require_torch()
    build_mode = _validate_loaded_mode(loaded, settings)
    config = FIXED_DIVISION_MODEL_CONFIGS.get(model_name)
    if config is None:
        raise ValueError(f"model name must be one of {MODEL_NAMES}")
    contracts = current_division_contract_hashes(config_dir)
    recorded_contracts = loaded.manifest.get("contract_sha256")
    if recorded_contracts is not None and dict(recorded_contracts) != contracts:
        raise ValueError("training dataset contract hashes changed")
    base_hash = file_sha256(base_checkpoint_path)
    expected_base_hash = division_model_module.EXPECTED_BASE_CHECKPOINT_SHA256
    if base_hash != expected_base_hash:
        raise ValueError("base checkpoint SHA-256 does not match the accepted checkpoint")
    sample_hash = _require_sha256(
        loaded.manifest.get("sample_tuple_sha256"), field="sample tuple SHA-256"
    )
    return {
        "base_checkpoint_sha256": base_hash,
        "batch_size": settings.batch_size,
        "contract_sha256": contracts,
        "dataset_build_mode": build_mode,
        "dataset_manifest_sha256": _require_sha256(
            loaded.dataset_manifest_sha256,
            field="dataset manifest SHA-256",
        ),
        "device": str(device),
        "epochs": settings.epochs,
        "feature_schema_sha256": loaded.feature_schema.sha256,
        "gradient_clip_norm": settings.gradient_clip_norm,
        "loss": settings.loss,
        "model_config": asdict(config),
        "model_inputs_sha256": _require_sha256(
            loaded.model_inputs_sha256,
            field="model inputs SHA-256",
        ),
        "nonproduction": settings.nonproduction,
        "num_workers": settings.num_workers,
        "numpy_version": np.__version__,
        "optimizer": {
            "name": settings.optimizer,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
        },
        "sample_tuple_sha256": sample_hash,
        "seed": settings.seed,
        "stateless_batching": True,
        "torch_version": str(torch.__version__),
        "trainer_source_sha256": _require_sha256(
            trainer_source_hash,
            field="trainer source SHA-256",
        ),
    }


def _history_payload(
    *,
    model_name: str,
    settings: TrainingSettings,
    signature: Mapping[str, Any],
    signature_hash: str,
    history: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "completed_epochs": len(history),
        "elapsed_seconds": float(elapsed_seconds),
        "epochs_requested": settings.epochs,
        "history": [dict(entry) for entry in history],
        "model_name": model_name,
        "nonproduction": settings.nonproduction,
        "training_history_schema_version": TRAINING_HISTORY_SCHEMA_VERSION,
        "training_signature": dict(signature),
        "training_signature_sha256": signature_hash,
    }


def _mps_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type != "mps":
        return None
    return torch.mps.get_rng_state().cpu()


def _training_state_payload(
    *,
    model: DivisionModel,
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
    signature: Mapping[str, Any],
    signature_hash: str,
    history: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    numpy_generator: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    payload = {
        "elapsed_seconds": float(elapsed_seconds),
        "head_state_dict": model.head.state_dict(),
        "history": [dict(entry) for entry in history],
        "last_epoch": len(history),
        "nonproduction": settings.nonproduction,
        "numpy_generator_state": copy.deepcopy(numpy_generator.bit_generator.state),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_mps_rng_state": _mps_rng_state(device),
        "training_signature": dict(signature),
        "training_signature_sha256": signature_hash,
        "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
    }
    _require_finite_tensors(payload, field="training state")
    return payload


def _read_history(
    path: Path,
    *,
    expected_signature: Mapping[str, Any],
    expected_signature_hash: str,
    model_name: str,
    settings: TrainingSettings,
) -> dict[str, Any]:
    payload = read_canonical_json(path)
    if set(payload) != _HISTORY_FIELDS:
        raise ValueError("training history has incomplete or unexpected fields")
    if payload["training_history_schema_version"] != TRAINING_HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported training history schema")
    if (
        payload["training_signature"] != expected_signature
        or payload["training_signature_sha256"] != expected_signature_hash
    ):
        raise ValueError("training history resume signature mismatch")
    if (
        payload["model_name"] != model_name
        or payload["nonproduction"] is not settings.nonproduction
        or payload["epochs_requested"] != settings.epochs
    ):
        raise ValueError("training history does not match the requested run")
    history = payload["history"]
    if not isinstance(history, list) or payload["completed_epochs"] != len(history):
        raise ValueError("training history epoch count is invalid")
    return payload


def _load_training_state(
    path: Path,
    *,
    expected_signature: Mapping[str, Any],
    expected_signature_hash: str,
    settings: TrainingSettings,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid resume state {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _STATE_FIELDS:
        raise ValueError("resume state has incomplete or unexpected fields")
    if payload["training_state_schema_version"] != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported resume state schema")
    if (
        payload["training_signature"] != expected_signature
        or payload["training_signature_sha256"] != expected_signature_hash
    ):
        raise ValueError("resume signature does not match current training inputs")
    if payload["nonproduction"] is not settings.nonproduction:
        raise ValueError("resume state production marker mismatch")
    last_epoch = payload["last_epoch"]
    history = payload["history"]
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, Integral)
        or not 1 <= int(last_epoch) <= settings.epochs
        or not isinstance(history, list)
        or len(history) != int(last_epoch)
    ):
        raise ValueError("resume state epoch history is invalid")
    elapsed = payload["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, Real)
        or not np.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("resume state elapsed time is invalid")
    if not isinstance(payload["torch_cpu_rng_state"], torch.Tensor):
        raise ValueError("resume state CPU RNG is invalid")
    _require_finite_tensors(payload, field="resume state")
    return payload


def _restore_training_state(
    payload: Mapping[str, Any],
    *,
    model: DivisionModel,
    optimizer: torch.optim.Optimizer,
    numpy_generator: np.random.Generator,
    device: torch.device,
) -> None:
    try:
        model.head.load_state_dict(payload["head_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"resume state model or optimizer is invalid: {exc}") from exc
    validate_head_only_optimizer(model, optimizer)
    _validate_model_weights(model)
    _require_finite_tensors(optimizer.state_dict(), field="optimizer state")
    try:
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        numpy_generator.bit_generator.state = copy.deepcopy(payload["numpy_generator_state"])
        if device.type == "mps":
            mps_state = payload["torch_mps_rng_state"]
            if not isinstance(mps_state, torch.Tensor):
                raise ValueError("resume state is missing the MPS RNG")
            torch.mps.set_rng_state(mps_state)
        elif payload["torch_mps_rng_state"] is not None:
            raise ValueError("CPU resume state unexpectedly contains an MPS RNG")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"resume RNG state is invalid: {exc}") from exc


def _validate_metrics(metrics: Mapping[str, Real]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
        ):
            raise ValueError("checkpoint metrics must be finite named numbers")
        result[name] = float(value)
    return result


def _save_smoke_checkpoint(
    path: Path,
    model: DivisionModel,
    loaded: LoadedDivisionDataset,
    *,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    epoch: int,
    trainer_source_hash: str,
    metrics: Mapping[str, Real],
) -> str:
    if loaded.manifest.get("build_mode") != BUILD_MODE_SMOKE:
        raise ValueError("nonproduction checkpoint requires a smoke dataset")
    base_hash = file_sha256(base_checkpoint_path)
    if base_hash != division_model_module.EXPECTED_BASE_CHECKPOINT_SHA256:
        raise ValueError("base checkpoint SHA-256 does not match the accepted checkpoint")
    payload = {
        "base_checkpoint_sha256": base_hash,
        "build_mode": BUILD_MODE_SMOKE,
        "contract_sha256": current_division_contract_hashes(config_dir),
        "dataset_manifest_sha256": _require_sha256(
            loaded.dataset_manifest_sha256,
            field="dataset manifest SHA-256",
        ),
        "division_checkpoint_schema_version": DIVISION_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "feature_schema_sha256": loaded.feature_schema.sha256,
        "head_state_dict": model.head.state_dict(),
        "metrics": _validate_metrics(metrics),
        "model_config": asdict(model.config),
        "model_inputs_sha256": _require_sha256(
            loaded.model_inputs_sha256,
            field="model inputs SHA-256",
        ),
        "nonproduction": True,
        "production_consumable": False,
        "sample_tuple_sha256": _require_sha256(
            loaded.manifest.get("sample_tuple_sha256"),
            field="sample tuple SHA-256",
        ),
        "trainer_source_sha256": _require_sha256(
            trainer_source_hash,
            field="trainer source SHA-256",
        ),
    }
    _require_finite_tensors(payload, field="smoke checkpoint")
    return atomic_torch_save(path, payload)


def save_final_checkpoint(
    paths: ModelArtifactPaths,
    model: DivisionModel,
    loaded: LoadedDivisionDataset,
    *,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    settings: TrainingSettings,
    trainer_source_hash: str,
    metrics: Mapping[str, Real],
) -> str:
    if settings.nonproduction:
        return _save_smoke_checkpoint(
            paths.checkpoint,
            model,
            loaded,
            base_checkpoint_path=base_checkpoint_path,
            config_dir=config_dir,
            epoch=settings.epochs,
            trainer_source_hash=trainer_source_hash,
            metrics=metrics,
        )
    if settings.epochs != PRODUCTION_EPOCHS:
        raise ValueError("production checkpoint requires exact epoch 100 completion")
    if loaded.manifest.get("build_mode") != BUILD_MODE_FIT_COMPLETE:
        raise ValueError("production checkpoint requires the fit_complete dataset")
    return save_division_checkpoint(
        paths.checkpoint,
        model,
        base_checkpoint_path=base_checkpoint_path,
        config_dir=config_dir,
        feature_schema=loaded.feature_schema,
        dataset_manifest_sha256=loaded.dataset_manifest_sha256,
        model_inputs_sha256=loaded.model_inputs_sha256,
        trainer_source_sha256=trainer_source_hash,
        epoch=PRODUCTION_EPOCHS,
        metrics=metrics,
    )


def _validate_existing_smoke_checkpoint(
    path: Path,
    *,
    model_name: str,
    loaded: LoadedDivisionDataset,
    settings: TrainingSettings,
    trainer_source_hash: str,
) -> dict[str, float]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "build_mode",
        "dataset_manifest_sha256",
        "epoch",
        "model_config",
        "model_inputs_sha256",
        "nonproduction",
        "production_consumable",
        "trainer_source_sha256",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("existing smoke checkpoint is invalid")
    if (
        payload["build_mode"] != BUILD_MODE_SMOKE
        or payload["nonproduction"] is not True
        or payload["production_consumable"] is not False
        or payload["epoch"] != settings.epochs
        or payload["model_config"] != asdict(FIXED_DIVISION_MODEL_CONFIGS[model_name])
        or payload["dataset_manifest_sha256"] != loaded.dataset_manifest_sha256
        or payload["model_inputs_sha256"] != loaded.model_inputs_sha256
        or payload["trainer_source_sha256"] != trainer_source_hash
    ):
        raise ValueError("existing smoke checkpoint identity mismatch")
    _require_finite_tensors(payload, field="existing smoke checkpoint")
    return _validate_metrics(payload["metrics"])


def _validate_existing_production_checkpoint(
    path: Path,
    *,
    model_name: str,
    loaded: LoadedDivisionDataset,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    trainer_source_hash: str,
) -> dict[str, float]:
    _, metadata = load_division_checkpoint(
        path,
        base_checkpoint_path=base_checkpoint_path,
        config_dir=config_dir,
        feature_schema=loaded.feature_schema,
        expected_dataset_manifest_sha256=loaded.dataset_manifest_sha256,
        expected_model_inputs_sha256=loaded.model_inputs_sha256,
        expected_trainer_source_sha256=trainer_source_hash,
        expected_model_config=FIXED_DIVISION_MODEL_CONFIGS[model_name],
        device="cpu",
    )
    if metadata["epoch"] != PRODUCTION_EPOCHS:
        raise ValueError("existing production checkpoint is not epoch 100")
    return dict(metadata["metrics"])


def _resolve_device(value: str | torch.device) -> torch.device:
    _require_torch()
    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type not in {"cpu", "mps"}:
        raise ValueError("division training supports only CPU or MPS")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def train_one_model(
    loaded: LoadedDivisionDataset,
    *,
    model_name: str,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    config_dir: str | Path,
    settings: TrainingSettings,
    device: str | torch.device,
    resume: bool,
    after_epoch: Callable[[int], None] | None = None,
) -> TrainingResult:
    _require_torch()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model name must be one of {MODEL_NAMES}")
    if not isinstance(settings, TrainingSettings):
        raise TypeError("division training requires TrainingSettings")
    _validate_loaded_mode(loaded, settings)
    resolved_device = _resolve_device(device)
    paths = model_artifact_paths(output_dir, model_name)
    with exclusive_model_lock(paths):
        validate_artifact_start(paths, resume=resume)
        source_hash = trainer_source_sha256(Path(__file__).resolve())
        signature = build_training_signature(
            loaded,
            model_name=model_name,
            settings=settings,
            config_dir=config_dir,
            base_checkpoint_path=base_checkpoint_path,
            trainer_source_hash=source_hash,
            device=resolved_device,
        )
        signature_hash = canonical_json_sha256(signature)

        torch.manual_seed(settings.seed)
        if resolved_device.type == "mps":
            torch.mps.manual_seed(settings.seed)
        numpy_generator = np.random.default_rng(settings.seed)
        model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS[model_name]).to(resolved_device)
        optimizer = build_optimizer(model)
        validate_head_only_optimizer(model, optimizer)

        history: list[dict[str, Any]] = []
        elapsed_before_resume = 0.0
        start_epoch = 1
        if resume:
            state = _load_training_state(
                paths.state,
                expected_signature=signature,
                expected_signature_hash=signature_hash,
                settings=settings,
            )
            expected_history_payload = _history_payload(
                model_name=model_name,
                settings=settings,
                signature=signature,
                signature_hash=signature_hash,
                history=state["history"],
                elapsed_seconds=float(state["elapsed_seconds"]),
            )
            history_payload = None
            if paths.history.is_file():
                history_payload = _read_history(
                    paths.history,
                    expected_signature=signature,
                    expected_signature_hash=signature_hash,
                    model_name=model_name,
                    settings=settings,
                )
                recorded_history = history_payload["history"]
                state_history = state["history"]
                recoverable_prefix = (
                    len(recorded_history) < len(state_history)
                    and recorded_history == state_history[: len(recorded_history)]
                )
                if not recoverable_prefix and history_payload != expected_history_payload:
                    raise ValueError("resume history and state disagree")
            if history_payload != expected_history_payload:
                atomic_json_write(paths.history, expected_history_payload)
            _restore_training_state(
                state,
                model=model,
                optimizer=optimizer,
                numpy_generator=numpy_generator,
                device=resolved_device,
            )
            history = [dict(entry) for entry in state["history"]]
            elapsed_before_resume = float(state["elapsed_seconds"])
            start_epoch = int(state["last_epoch"]) + 1
            if paths.checkpoint.exists():
                if start_epoch != settings.epochs + 1:
                    raise ValueError("checkpoint exists before fixed-epoch training completed")
                if settings.nonproduction:
                    metrics = _validate_existing_smoke_checkpoint(
                        paths.checkpoint,
                        model_name=model_name,
                        loaded=loaded,
                        settings=settings,
                        trainer_source_hash=source_hash,
                    )
                else:
                    metrics = _validate_existing_production_checkpoint(
                        paths.checkpoint,
                        model_name=model_name,
                        loaded=loaded,
                        base_checkpoint_path=base_checkpoint_path,
                        config_dir=config_dir,
                        trainer_source_hash=source_hash,
                    )
                return TrainingResult(
                    model_name=model_name,
                    paths=paths,
                    completed_epochs=settings.epochs,
                    checkpoint_sha256=file_sha256(paths.checkpoint),
                    trainer_source_sha256=source_hash,
                    metrics=metrics,
                    resumed=True,
                )

        run_start = time.perf_counter()
        for epoch in range(start_epoch, settings.epochs + 1):
            metrics = run_epoch(
                model,
                optimizer,
                loaded,
                epoch=epoch,
                settings=settings,
                device=resolved_device,
            )
            history.append({"epoch": epoch, "metrics": metrics})
            elapsed_seconds = elapsed_before_resume + time.perf_counter() - run_start
            history_payload = _history_payload(
                model_name=model_name,
                settings=settings,
                signature=signature,
                signature_hash=signature_hash,
                history=history,
                elapsed_seconds=elapsed_seconds,
            )
            state_payload = _training_state_payload(
                model=model,
                optimizer=optimizer,
                settings=settings,
                signature=signature,
                signature_hash=signature_hash,
                history=history,
                elapsed_seconds=elapsed_seconds,
                numpy_generator=numpy_generator,
                device=resolved_device,
            )
            atomic_torch_save(paths.state, state_payload)
            atomic_json_write(paths.history, history_payload)
            if after_epoch is not None:
                after_epoch(epoch)

        if not history or len(history) != settings.epochs:
            raise RuntimeError("fixed-epoch division training did not complete")
        final_metrics = _flat_checkpoint_metrics(history[-1]["metrics"])
        checkpoint_hash = save_final_checkpoint(
            paths,
            model,
            loaded,
            base_checkpoint_path=base_checkpoint_path,
            config_dir=config_dir,
            settings=settings,
            trainer_source_hash=source_hash,
            metrics=final_metrics,
        )
        return TrainingResult(
            model_name=model_name,
            paths=paths,
            completed_epochs=settings.epochs,
            checkpoint_sha256=checkpoint_hash,
            trainer_source_sha256=source_hash,
            metrics=final_metrics,
            resumed=resume,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    ladder = parser.add_mutually_exclusive_group(required=True)
    ladder.add_argument("--model", choices=MODEL_NAMES)
    ladder.add_argument("--all", action="store_true", help="train the exact three-model ladder")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="fit_complete artifact; defaults to the canonical production dataset",
    )
    parser.add_argument(
        "--smoke-dataset",
        type=Path,
        help="explicit smoke artifact; produces nonproduction checkpoints only",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--epochs",
        type=int,
        help="smoke-only short epoch count; production is fixed at 100",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--upstream-manifest", type=Path, default=DEFAULT_UPSTREAM_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    nonproduction = args.smoke_dataset is not None
    if nonproduction and args.dataset is not None:
        parser.error("--dataset cannot be combined with --smoke-dataset")
    try:
        settings = TrainingSettings.for_mode(
            nonproduction=nonproduction,
            epochs=args.epochs,
        )
        output_dir = args.output_dir or (
            DEFAULT_SMOKE_OUTPUT_DIR if nonproduction else DEFAULT_OUTPUT_DIR
        )
        if nonproduction and output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
            raise ValueError("smoke checkpoints must use an isolated output directory")
        if not nonproduction and output_dir.resolve() == DEFAULT_SMOKE_OUTPUT_DIR.resolve():
            raise ValueError("production checkpoints cannot use the smoke output directory")
        inputs = ReaderInputs(
            dataset_path=args.smoke_dataset or args.dataset or DEFAULT_DATASET,
            project_root=ROOT,
            data_root=args.data_root,
            config_dir=args.config_dir,
            upstream_manifest_path=args.upstream_manifest,
            summary_path=args.summary,
            cache_dir=args.cache_dir,
            checkpoint_path=args.base_checkpoint,
        )
        loaded = read_training_dataset(inputs, nonproduction=nonproduction)
        model_names = MODEL_NAMES if args.all else (args.model,)
        for model_name in model_names:
            result = train_one_model(
                loaded,
                model_name=model_name,
                output_dir=output_dir,
                base_checkpoint_path=args.base_checkpoint,
                config_dir=args.config_dir,
                settings=settings,
                device=args.device,
                resume=args.resume,
            )
            print(
                f"model={result.model_name} epochs={result.completed_epochs} "
                f"loss={result.metrics['loss']:.8f} "
                f"checkpoint_sha256={result.checkpoint_sha256} "
                f"trainer_source_sha256={result.trainer_source_sha256} "
                f"checkpoint={result.paths.checkpoint}",
                flush=True,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        FloatingPointError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
