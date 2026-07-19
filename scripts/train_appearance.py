#!/usr/bin/env python3
"""Train the first 3D DoG-candidate cellness CNN with positive-unlabeled risk.

The shard split happens at the movie level and accepts embryo 6bba only.  Embryo 44b6
is never loaded by this script.  Usage:

    python scripts/train_appearance.py
    python scripts/train_appearance.py --epochs 20 --batch-size 64
    python scripts/train_appearance.py --epochs 30 --steps-per-epoch 1000
    python scripts/train_appearance.py --epochs 1 --max-steps-per-epoch 2
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    from torch.utils.data import DataLoader, Subset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Install the ML extra with "
        "`.venv/bin/python -m pip install -e '.[ml]'`."
    ) from exc

from biohub.appearance import appearance_shards_sha256, discover_appearance_shards  # noqa: E402
from biohub.appearance_evaluation import files_sha256  # noqa: E402
from biohub.appearance_model import (  # noqa: E402
    CandidatePatchDataset,
    CellnessCNN3D,
    CellnessModelConfig,
    nnpu_training_objective,
    non_negative_pu_loss,
    positive_unlabeled_auc,
    resolve_device,
    save_cellness_checkpoint,
)

TRAINING_STATE_SCHEMA_VERSION = 3


def _sidecar_path(checkpoint: Path, suffix: str) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}{suffix}")


def write_training_history(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the auditable epoch-by-epoch training record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_training_state(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the latest model and optimizer state for safe resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_state(path: Path) -> dict[str, Any]:
    """Load a trusted local resume state and validate its schema."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("training_state_schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported training state schema in {path}")
    return state


def split_shards(
    shards: list[Path],
    dev_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Make a deterministic movie-level internal split within embryo 6bba."""
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in [0, 1)")
    if dev_fraction == 0.0:
        return list(shards), []
    if len(shards) < 2:
        raise ValueError("at least two shards are required when dev_fraction is non-zero")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(shards))
    n_dev = min(len(shards) - 1, max(1, round(len(shards) * dev_fraction)))
    dev_indices = set(int(index) for index in order[:n_dev])
    train = [shard for index, shard in enumerate(shards) if index not in dev_indices]
    dev = [shard for index, shard in enumerate(shards) if index in dev_indices]
    return train, dev


def selection_contract(name: str) -> tuple[str, str]:
    """Map a user-facing checkpoint criterion to its metric key and direction."""
    if name == "pu-auc":
        return "positive_unlabeled_auc", "max"
    if name == "nnpu-risk":
        return "loss", "min"
    raise ValueError(f"unsupported selection metric: {name}")


def selection_improved(
    value: float,
    best: float,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    """Apply one explicit meaningful-improvement rule for max or min metrics."""
    if mode == "max":
        return value > best + min_delta
    if mode == "min":
        return value < best - min_delta
    raise ValueError(f"unsupported selection mode: {mode}")


def make_loader(
    dataset: CandidatePatchDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    if len(indices) == 0:
        raise ValueError("cannot make a loader from an empty sample role")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def next_or_restart(
    iterator: Iterator,
    loader: DataLoader,
) -> tuple[tuple[torch.Tensor, torch.Tensor], Iterator]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def run_epoch(
    model: CellnessCNN3D,
    positive_loader: DataLoader,
    unlabeled_loader: DataLoader,
    *,
    class_prior: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None,
    nnpu_beta: float = 0.0,
    nnpu_gamma: float = 1.0,
) -> dict[str, float]:
    """Run one paired positive-unlabeled epoch for training or evaluation."""
    training = optimizer is not None
    model.train(training)
    positive_iterator = iter(positive_loader)
    unlabeled_iterator = iter(unlabeled_loader)
    steps = max(len(positive_loader), len(unlabeled_loader))
    if max_steps is not None:
        steps = min(steps, max_steps)
    metric_names = (
        "loss",
        "optimization_objective",
        "correction_fraction",
        "positive_risk",
        "raw_negative_risk",
        "corrected_negative_risk",
        "positive_score",
        "unlabeled_score",
        "positive_unlabeled_auc",
        "positive_recall_at_0_5",
        "unlabeled_fraction_at_0_5",
    )
    totals = {name: torch.zeros((), device=device) for name in metric_names}

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for _ in range(steps):
            (positive_inputs, _), positive_iterator = next_or_restart(
                positive_iterator, positive_loader
            )
            (unlabeled_inputs, _), unlabeled_iterator = next_or_restart(
                unlabeled_iterator, unlabeled_loader
            )
            positive_inputs = positive_inputs.to(device)
            unlabeled_inputs = unlabeled_inputs.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            positive_logits = model(positive_inputs)
            unlabeled_logits = model(unlabeled_inputs)
            if training:
                objective, loss, risks = nnpu_training_objective(
                    positive_logits,
                    unlabeled_logits,
                    class_prior,
                    beta=nnpu_beta,
                    gamma=nnpu_gamma,
                )
            else:
                loss, risks = non_negative_pu_loss(
                    positive_logits,
                    unlabeled_logits,
                    class_prior,
                )
                objective = loss
                risks["correction_active"] = torch.zeros_like(loss)
            if optimizer is not None:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            positive_scores = torch.sigmoid(positive_logits).detach()
            unlabeled_scores = torch.sigmoid(unlabeled_logits).detach()
            totals["loss"] += loss.detach()
            totals["optimization_objective"] += objective.detach()
            totals["correction_fraction"] += risks["correction_active"].detach()
            totals["positive_risk"] += risks["positive_risk"].detach()
            totals["raw_negative_risk"] += risks["raw_negative_risk"].detach()
            totals["corrected_negative_risk"] += risks["corrected_negative_risk"].detach()
            totals["positive_score"] += positive_scores.mean()
            totals["unlabeled_score"] += unlabeled_scores.mean()
            pairwise = positive_scores[:, None] - unlabeled_scores[None, :]
            batch_auc = (pairwise > 0).float() + 0.5 * (pairwise == 0).float()
            totals["positive_unlabeled_auc"] += batch_auc.mean()
            totals["positive_recall_at_0_5"] += (positive_scores >= 0.5).float().mean()
            totals["unlabeled_fraction_at_0_5"] += (unlabeled_scores >= 0.5).float().mean()
    averaged = torch.stack([totals[name] for name in metric_names]).div(steps).cpu().tolist()
    return dict(zip(metric_names, averaged, strict=True))


def evaluate_model(
    model: CellnessCNN3D,
    positive_loader: DataLoader,
    unlabeled_loader: DataLoader,
    *,
    class_prior: float,
    device: torch.device,
    max_steps: int | None,
) -> dict[str, float]:
    """Evaluate each development patch exactly once unless a smoke limit is set."""

    @torch.inference_mode()
    def collect(loader: DataLoader) -> torch.Tensor:
        logits: list[torch.Tensor] = []
        for step, (inputs, _) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            logits.append(model(inputs.to(device)))
        return torch.cat(logits).cpu()

    model.eval()
    positive_logits = collect(positive_loader)
    unlabeled_logits = collect(unlabeled_loader)
    loss, risks = non_negative_pu_loss(positive_logits, unlabeled_logits, class_prior)
    positive_scores = torch.sigmoid(positive_logits).numpy()
    unlabeled_scores = torch.sigmoid(unlabeled_logits).numpy()
    return {
        "loss": float(loss),
        "positive_risk": float(risks["positive_risk"]),
        "raw_negative_risk": float(risks["raw_negative_risk"]),
        "corrected_negative_risk": float(risks["corrected_negative_risk"]),
        "positive_score": float(positive_scores.mean()),
        "unlabeled_score": float(unlabeled_scores.mean()),
        "positive_unlabeled_auc": positive_unlabeled_auc(positive_scores, unlabeled_scores),
        "positive_recall_at_0_5": float((positive_scores >= 0.5).mean()),
        "unlabeled_fraction_at_0_5": float((unlabeled_scores >= 0.5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_shards",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dev-batch-size", type=int, default=256)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--class-prior", type=float, help="override the training estimate")
    parser.add_argument("--dev-class-prior", type=float, help="override the dev estimate")
    parser.add_argument("--nnpu-beta", type=float, default=0.0)
    parser.add_argument("--nnpu-gamma", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-movies",
        type=int,
        help="refuse to train unless this exact number of complete movie shards exists",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        help="epoch metrics output; defaults beside the checkpoint",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="latest resumable training state; defaults beside the checkpoint",
    )
    parser.add_argument("--resume", action="store_true", help="resume from --state")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="stop after this many epochs without improvement; 0 disables early stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="minimum full-dev metric change counted as a real improvement",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("pu-auc", "nnpu-risk"),
        default="pu-auc",
        help="full-dev criterion for checkpoints, scheduling, and early stopping",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        help="training updates between full development evaluations",
    )
    parser.add_argument(
        "--max-dev-steps",
        type=int,
        help="optional development-loader smoke limit; omit for complete evaluation",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="legacy smoke limit applied to both training and development",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.dev_batch_size < 1:
        parser.error("--dev-batch-size must be positive")
    if not 0.0 < args.positive_fraction < 1.0:
        parser.error("--positive-fraction must be between 0 and 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.nnpu_beta < 0.0:
        parser.error("--nnpu-beta must be non-negative")
    if not 0.0 < args.nnpu_gamma <= 1.0:
        parser.error("--nnpu-gamma must be greater than 0 and at most 1")
    if not 0.0 < args.lr_factor < 1.0:
        parser.error("--lr-factor must be between 0 and 1")
    if args.lr_patience < 0:
        parser.error("--lr-patience must be non-negative")
    if not 0.0 < args.min_learning_rate <= args.learning_rate:
        parser.error("--min-learning-rate must be positive and at most --learning-rate")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.require_movies is not None and args.require_movies < 1:
        parser.error("--require-movies must be positive")
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    if args.min_delta < 0.0:
        parser.error("--min-delta must be non-negative")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        parser.error("--steps-per-epoch must be positive")
    if args.max_dev_steps is not None and args.max_dev_steps < 1:
        parser.error("--max-dev-steps must be positive")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch < 1:
        parser.error("--max-steps-per-epoch must be positive")
    if args.max_steps_per_epoch is not None and (
        args.steps_per_epoch is not None or args.max_dev_steps is not None
    ):
        parser.error(
            "--max-steps-per-epoch cannot be combined with --steps-per-epoch or --max-dev-steps"
        )
    train_step_limit = args.steps_per_epoch or args.max_steps_per_epoch
    dev_step_limit = args.max_dev_steps or args.max_steps_per_epoch
    selection_key, selection_mode = selection_contract(args.selection_metric)

    history_path = args.history_json or _sidecar_path(args.checkpoint, ".history.json")
    state_path = args.state or _sidecar_path(args.checkpoint, ".state.pt")
    if args.resume and not state_path.is_file():
        parser.error(f"resume state does not exist: {state_path}")
    if not args.resume and state_path.exists():
        parser.error(f"training state already exists: {state_path}; pass --resume to continue")
    if not args.resume and args.checkpoint.exists():
        parser.error(f"checkpoint already exists: {args.checkpoint}; choose a new path")
    if not args.resume and history_path.exists():
        parser.error(f"training history already exists: {history_path}; choose a new path")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    shards = discover_appearance_shards(args.shards)
    if args.require_movies is not None and len(shards) != args.require_movies:
        parser.error(
            f"--require-movies expected {args.require_movies} complete shards, found {len(shards)}"
        )
    try:
        train_shards, dev_shards = split_shards(shards, args.dev_fraction, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    train_dataset = CandidatePatchDataset(train_shards, augment=True)
    dev_dataset = CandidatePatchDataset(dev_shards) if dev_shards else None
    class_prior = args.class_prior if args.class_prior is not None else train_dataset.class_prior
    if not 0.0 < class_prior < 1.0:
        parser.error("the resolved class prior must be strictly between 0 and 1")
    dev_class_prior = None
    if dev_dataset is not None:
        dev_class_prior = (
            args.dev_class_prior if args.dev_class_prior is not None else dev_dataset.class_prior
        )
        if not 0.0 < dev_class_prior < 1.0:
            parser.error("the resolved dev class prior must be strictly between 0 and 1")
    elif args.dev_class_prior is not None:
        parser.error("--dev-class-prior requires a non-empty dev split")

    positive_batch = max(1, round(args.batch_size * args.positive_fraction))
    unlabeled_batch = args.batch_size - positive_batch
    if unlabeled_batch < 1:
        parser.error("the batch split must include at least one unlabeled sample")
    train_positive_loader = make_loader(
        train_dataset,
        train_dataset.positive_indices,
        batch_size=positive_batch,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    train_unlabeled_loader = make_loader(
        train_dataset,
        train_dataset.unlabeled_indices,
        batch_size=unlabeled_batch,
        shuffle=True,
        seed=args.seed + 1,
        num_workers=args.num_workers,
    )

    dev_loaders = None
    if dev_dataset is not None:
        dev_loaders = (
            make_loader(
                dev_dataset,
                dev_dataset.positive_indices,
                batch_size=args.dev_batch_size,
                shuffle=False,
                seed=args.seed + 2,
                num_workers=args.num_workers,
            ),
            make_loader(
                dev_dataset,
                dev_dataset.unlabeled_indices,
                batch_size=args.dev_batch_size,
                shuffle=False,
                seed=args.seed + 3,
                num_workers=args.num_workers,
            ),
        )

    device = resolve_device(args.device)
    shard_contents_sha256 = appearance_shards_sha256(shards)
    training_source_sha256 = files_sha256(
        [
            SRC / "biohub" / "appearance.py",
            SRC / "biohub" / "appearance_model.py",
            Path(__file__).resolve(),
        ]
    )
    model_config = CellnessModelConfig(
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    model = CellnessCNN3D(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=selection_mode,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_steps_per_epoch = max(len(train_positive_loader), len(train_unlabeled_loader))
    if train_step_limit is not None:
        train_steps_per_epoch = min(train_steps_per_epoch, train_step_limit)
    data_summary = {
        "train_movies": len(train_shards),
        "dev_movies": len(dev_shards),
        "train_positive_samples": len(train_dataset.positive_indices),
        "train_unlabeled_samples": len(train_dataset.unlabeled_indices),
        "dev_positive_samples": (
            len(dev_dataset.positive_indices) if dev_dataset is not None else 0
        ),
        "dev_unlabeled_samples": (
            len(dev_dataset.unlabeled_indices) if dev_dataset is not None else 0
        ),
        "train_steps_per_epoch": train_steps_per_epoch,
    }
    training_signature = {
        "model_config": {
            "base_channels": model_config.base_channels,
            "dropout": model_config.dropout,
        },
        "patch_shape": list(train_dataset.patch_shape),
        "detector_threshold_k": train_dataset.detector_threshold_k,
        "train_shards": [path.name for path in train_shards],
        "dev_shards": [path.name for path in dev_shards],
        "batch_size": args.batch_size,
        "dev_batch_size": args.dev_batch_size,
        "positive_fraction": args.positive_fraction,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_learning_rate": args.min_learning_rate,
        "dev_fraction": args.dev_fraction,
        "class_prior": class_prior,
        "dev_class_prior": dev_class_prior,
        "nnpu_beta": args.nnpu_beta,
        "nnpu_gamma": args.nnpu_gamma,
        "seed": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "selection_metric": args.selection_metric,
        "steps_per_epoch": train_step_limit,
        "max_dev_steps": dev_step_limit,
        "shard_contents_sha256": shard_contents_sha256,
        "training_source_sha256": training_source_sha256,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    checkpoint_training_config = {
        **training_signature,
        "data_summary": data_summary,
        "epochs_requested": args.epochs,
        "early_stopping_patience": args.patience,
    }

    best_selection_value = float("-inf") if selection_mode == "max" else float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    elapsed_before_resume = 0.0
    history: list[dict[str, Any]] = []
    if args.resume:
        try:
            state = load_training_state(state_path)
        except ValueError as exc:
            parser.error(str(exc))
        if state.get("training_signature") != training_signature:
            parser.error(
                "resume state does not match the requested model, data split, or training config"
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best_selection_value = float(state["best_selection_value"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        history = list(state["history"])
        elapsed_before_resume = float(state["elapsed_seconds"])
        start_epoch = int(state["last_epoch"]) + 1
        torch.set_rng_state(state["torch_rng_state"])
        train_positive_loader.generator.set_state(state["positive_loader_rng_state"])
        train_unlabeled_loader.generator.set_state(state["unlabeled_loader_rng_state"])
        if device.type == "mps" and state.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(state["mps_rng_state"])
        if start_epoch > args.epochs:
            parser.error(
                f"resume state already completed epoch {start_epoch - 1}; "
                f"increase --epochs above that value"
            )

    print(f"Device: {device}", flush=True)
    print(f"Model parameters: {parameter_count:,}", flush=True)
    print(f"Patch shape: {train_dataset.patch_shape}", flush=True)
    print(f"Shard contents SHA-256: {shard_contents_sha256}", flush=True)
    print(f"Training source SHA-256: {training_source_sha256}", flush=True)
    print(
        f"Training samples: {data_summary['train_positive_samples']:,} positive + "
        f"{data_summary['train_unlabeled_samples']:,} unlabeled; "
        f"steps/epoch={train_steps_per_epoch:,}",
        flush=True,
    )
    if dev_dataset is not None:
        print(
            f"Dev samples: {data_summary['dev_positive_samples']:,} positive + "
            f"{data_summary['dev_unlabeled_samples']:,} unlabeled; "
            f"batch={args.dev_batch_size}",
            flush=True,
        )
    print(f"Training candidate class prior: {class_prior:.4f}", flush=True)
    if dev_class_prior is not None:
        print(f"Dev candidate class prior: {dev_class_prior:.4f}", flush=True)
    print(
        f"Checkpoint selection: {args.selection_metric} ({selection_mode}); "
        f"min_delta={args.min_delta}",
        flush=True,
    )
    print(
        f"Train movies ({len(train_shards)}): {', '.join(path.name for path in train_shards)}",
        flush=True,
    )
    dev_names = ", ".join(path.name for path in dev_shards) or "none"
    print(f"Dev movies ({len(dev_shards)}): {dev_names}", flush=True)
    if args.resume:
        print(
            f"Resuming at epoch {start_epoch}; best epoch so far is {best_epoch}",
            flush=True,
        )

    run_start = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model,
            train_positive_loader,
            train_unlabeled_loader,
            class_prior=class_prior,
            device=device,
            optimizer=optimizer,
            max_steps=train_step_limit,
            nnpu_beta=args.nnpu_beta,
            nnpu_gamma=args.nnpu_gamma,
        )
        dev_metrics = None
        if dev_loaders is not None:
            dev_metrics = evaluate_model(
                model,
                dev_loaders[0],
                dev_loaders[1],
                class_prior=dev_class_prior,
                device=device,
                max_steps=dev_step_limit,
            )
        selection_metrics = dev_metrics or train_metrics
        selection_value = selection_metrics[selection_key]
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = elapsed_before_resume + time.perf_counter() - run_start
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_pos_score={train_metrics['positive_score']:.3f} "
            f"train_u_score={train_metrics['unlabeled_score']:.3f} "
            f"correction={train_metrics['correction_fraction']:.2%} "
            f"lr={learning_rate:.2e}",
            end="",
        )
        if dev_metrics is not None:
            print(
                f" dev_loss={dev_metrics['loss']:.4f} "
                f"dev_pu_auc={dev_metrics['positive_unlabeled_auc']:.3f} "
                f"dev_pos_score={dev_metrics['positive_score']:.3f} "
                f"dev_u_score={dev_metrics['unlabeled_score']:.3f}",
                end="",
            )

        improved = selection_improved(
            selection_value,
            best_selection_value,
            mode=selection_mode,
            min_delta=args.min_delta,
        )
        if improved:
            best_selection_value = selection_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            if dev_metrics is not None:
                checkpoint_metrics.update(
                    {f"dev_{key}": value for key, value in dev_metrics.items()}
                )
                checkpoint_metrics["dev_class_prior"] = float(dev_class_prior)
            checkpoint_metrics["epoch_seconds"] = epoch_seconds
            checkpoint_metrics["elapsed_seconds"] = elapsed_seconds
            checkpoint_metrics["learning_rate"] = learning_rate
            save_cellness_checkpoint(
                args.checkpoint,
                model,
                patch_shape=train_dataset.patch_shape,
                class_prior=class_prior,
                detector_threshold_k=train_dataset.detector_threshold_k,
                epoch=epoch,
                train_shards=[path.name for path in train_shards],
                dev_shards=[path.name for path in dev_shards],
                metrics=checkpoint_metrics,
                training_config=checkpoint_training_config,
            )
            print(f" time={epoch_seconds:.1f}s checkpoint=updated", flush=True)
        else:
            epochs_without_improvement += 1
            print(f" time={epoch_seconds:.1f}s", flush=True)

        scheduler.step(selection_value)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        if next_learning_rate < learning_rate:
            print(f"Learning rate reduced to {next_learning_rate:.2e}.", flush=True)

        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "selection_metric": args.selection_metric,
                "selection_value": selection_value,
                "improved": improved,
                "learning_rate": learning_rate,
                "next_learning_rate": next_learning_rate,
                "train": train_metrics,
                "dev": dev_metrics,
            }
        )
        stopped_early = args.patience > 0 and epochs_without_improvement >= args.patience
        history_payload = {
            "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "training_signature": training_signature,
            "data_summary": data_summary,
            "device": str(device),
            "model_parameters": parameter_count,
            "epochs_requested": args.epochs,
            "early_stopping_patience": args.patience,
            "selection_metric": args.selection_metric,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_selection_value": best_selection_value,
            "epochs_without_improvement": epochs_without_improvement,
            "stopped_early": stopped_early,
            "history": history,
        }
        write_training_history(history_path, history_payload)
        mps_rng_state = torch.mps.get_rng_state() if device.type == "mps" else None
        save_training_state(
            state_path,
            {
                **history_payload,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "last_epoch": epoch,
                "elapsed_seconds": elapsed_seconds,
                "torch_rng_state": torch.get_rng_state(),
                "mps_rng_state": mps_rng_state,
                "positive_loader_rng_state": train_positive_loader.generator.get_state(),
                "unlabeled_loader_rng_state": train_unlabeled_loader.generator.get_state(),
            },
        )
        if stopped_early:
            print(
                f"Early stopping after {args.patience} epochs without improvement.",
                flush=True,
            )
            break
    print(f"Best checkpoint: {args.checkpoint} (epoch {best_epoch})", flush=True)
    print(f"Training history: {history_path}", flush=True)
    print(f"Resume state: {state_path}", flush=True)


if __name__ == "__main__":
    main()
