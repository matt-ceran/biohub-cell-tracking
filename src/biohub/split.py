"""Embryo-disjoint validation splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from biohub.io import embryo_id


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    train_embryos: tuple[str, ...]
    validation_embryos: tuple[str, ...]


def make_embryo_holdout(
    datasets: Iterable[str],
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """Split dataset names so no embryo appears in both train and validation."""
    names = sorted(set(datasets))
    if not names:
        raise ValueError("At least one dataset name is required.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    by_embryo: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_embryo[embryo_id(name)].append(name)

    embryos = sorted(by_embryo)
    rng = random.Random(seed)
    rng.shuffle(embryos)
    n_val = max(1, round(len(embryos) * validation_fraction))
    if len(embryos) > 1:
        n_val = min(n_val, len(embryos) - 1)

    validation_embryos = tuple(sorted(embryos[:n_val]))
    train_embryos = tuple(sorted(embryos[n_val:]))
    validation_set = set(validation_embryos)

    train = tuple(name for name in names if embryo_id(name) not in validation_set)
    validation = tuple(name for name in names if embryo_id(name) in validation_set)
    return DatasetSplit(train, validation, train_embryos, validation_embryos)
