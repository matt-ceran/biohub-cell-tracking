"""3D appearance-data utilities for scoring DoG cell candidates.

The competition annotations are sparse: a candidate that does not match a GEFF
coordinate is unknown, not a confirmed non-cell.  This module therefore builds two
separate sample roles for positive-unlabeled learning:

* known-positive patches are DoG candidates matched one-to-one to GEFF nodes;
* unlabeled patches are a uniform sample of the complete DoG candidate population.

The unlabeled pool intentionally remains a mixture of real cells and non-cells.  It is
never written or exposed as a negative target.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biohub.constants import MATCH_RADIUS_UM
from biohub.detect import DetectorConfig, detect_movie
from biohub.io import SamplePaths, open_image_array, read_geff_graph
from biohub.metric import TrackingGraph, match_nodes

APPEARANCE_SCHEMA_VERSION = 1
APPEARANCE_TRAIN_EMBRYO = "6bba"
APPEARANCE_HOLDOUT_EMBRYO = "44b6"
DEFAULT_PATCH_SHAPE = (9, 33, 33)

# These values describe how a patch participates in positive-unlabeled training.
# UNLABELED is deliberately not called NEGATIVE because it contains unknown real cells.
SAMPLE_UNLABELED = np.uint8(0)
SAMPLE_KNOWN_POSITIVE = np.uint8(1)
APPEARANCE_SHARD_FILES = (
    "metadata.json",
    "patches.npy",
    "sample_roles.npy",
    "candidate_tzyx.npy",
)


@dataclass(frozen=True)
class AppearanceShardSummary:
    """Metadata required to audit one candidate-patch shard."""

    schema_version: int
    dataset: str
    embryo: str
    patch_shape: tuple[int, int, int]
    detector_threshold_k: float
    unlabeled_ratio: float
    seed: int
    raw_candidate_count: int
    truth_node_count: int
    matched_candidate_count: int
    sparse_label_recall: float
    estimated_node_count: int
    estimated_candidate_prior: float
    known_positive_samples: int
    unlabeled_samples: int

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AppearanceShardSummary:
        """Parse JSON metadata while restoring the patch shape tuple."""
        parsed = dict(values)
        parsed["patch_shape"] = tuple(int(value) for value in parsed["patch_shape"])
        return cls(**parsed)


def validate_patch_shape(patch_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Require an odd, positive ``(z, y, x)`` shape with one exact center voxel."""
    shape = tuple(int(value) for value in patch_shape)
    if len(shape) != 3:
        raise ValueError(f"patch_shape must have three dimensions, got {shape!r}")
    if any(value < 1 or value % 2 == 0 for value in shape):
        raise ValueError(f"patch_shape values must be positive odd integers, got {shape!r}")
    return shape


def extract_patch(
    volume: np.ndarray,
    center_zyx: tuple[float, float, float] | np.ndarray,
    patch_shape: tuple[int, int, int] = DEFAULT_PATCH_SHAPE,
) -> np.ndarray:
    """Extract one centered patch, reflecting image content at volume boundaries."""
    shape = validate_patch_shape(patch_shape)
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {array.shape!r}")

    center = np.rint(np.asarray(center_zyx, dtype=float)).astype(int)
    if center.shape != (3,):
        raise ValueError(f"center_zyx must contain three coordinates, got {center_zyx!r}")
    if any(value < 0 or value >= size for value, size in zip(center, array.shape, strict=True)):
        raise ValueError(f"center {tuple(center)} is outside volume shape {array.shape}")

    radii = np.asarray(shape) // 2
    starts = center - radii
    stops = center + radii + 1
    source_slices = tuple(
        slice(max(0, int(start)), min(size, int(stop)))
        for start, stop, size in zip(starts, stops, array.shape, strict=True)
    )
    cropped = array[source_slices]
    padding = tuple(
        (max(0, -int(start)), max(0, int(stop) - size))
        for start, stop, size in zip(starts, stops, array.shape, strict=True)
    )
    if any(before or after for before, after in padding):
        mode = "reflect" if all(size > 1 for size in cropped.shape) else "edge"
        cropped = np.pad(cropped, padding, mode=mode)
    if cropped.shape != shape:
        raise RuntimeError(f"patch extraction produced {cropped.shape}, expected {shape}")
    return np.asarray(cropped)


def extract_patches(
    volume: np.ndarray,
    centers_zyx: np.ndarray,
    patch_shape: tuple[int, int, int] = DEFAULT_PATCH_SHAPE,
) -> np.ndarray:
    """Vectorize exact reflected patch extraction for an ``[n,3]`` center array."""
    shape = validate_patch_shape(patch_shape)
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {array.shape!r}")
    centers = np.asarray(centers_zyx, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError(f"centers_zyx must have shape [n,3], got {centers.shape!r}")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_zyx must all be finite")
    rounded = np.rint(centers).astype(np.int64)
    if any(
        np.any(rounded[:, axis] < 0) or np.any(rounded[:, axis] >= size)
        for axis, size in enumerate(array.shape)
    ):
        raise ValueError(f"at least one center is outside volume shape {array.shape}")
    if len(rounded) == 0:
        return np.empty((0, *shape), dtype=array.dtype)
    if any(size == 1 for size in shape):
        return np.stack([extract_patch(array, center, shape) for center in centers])

    radii = np.asarray(shape) // 2
    padding = tuple((int(radius), int(radius)) for radius in radii)
    mode = "reflect" if all(size > 1 for size in array.shape) else "edge"
    padded = np.pad(array, padding, mode=mode)
    z_indices = rounded[:, 0, None, None, None] + np.arange(shape[0])[None, :, None, None]
    y_indices = rounded[:, 1, None, None, None] + np.arange(shape[1])[None, None, :, None]
    x_indices = rounded[:, 2, None, None, None] + np.arange(shape[2])[None, None, None, :]
    return np.asarray(padded[z_indices, y_indices, x_indices])


def known_positive_mask(candidates: TrackingGraph, truth: TrackingGraph) -> np.ndarray:
    """Mark DoG candidates matched one-to-one to sparse GEFF nodes."""
    matches = match_nodes(candidates, truth, max_distance=MATCH_RADIUS_UM)
    matched_ids = np.fromiter(matches, dtype=np.int64, count=len(matches))
    return np.isin(np.asarray(candidates.node_ids, dtype=np.int64), matched_ids)


def normalize_patch(patch: np.ndarray, clip_sigma: float = 6.0) -> np.ndarray:
    """Standardize one patch while preserving its local 3D intensity structure."""
    array = np.asarray(patch, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if not np.isfinite(std) or std < 1.0:
        std = 1.0
    normalized = (array - mean) / std
    return np.clip(normalized, -clip_sigma, clip_sigma).astype(np.float32, copy=False)


def normalize_patches(patches: np.ndarray, clip_sigma: float = 6.0) -> np.ndarray:
    """Vectorize per-patch normalization for a ``[n,z,y,x]`` patch batch."""
    array = np.asarray(patches, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"patches must have shape [n,z,y,x], got {array.shape!r}")
    if not np.isfinite(clip_sigma) or clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be positive and finite")
    axes = (1, 2, 3)
    means = array.mean(axis=axes, keepdims=True)
    standard_deviations = array.std(axis=axes, keepdims=True)
    standard_deviations = np.where(
        np.isfinite(standard_deviations) & (standard_deviations >= 1.0),
        standard_deviations,
        1.0,
    )
    normalized = (array - means) / standard_deviations
    return np.clip(normalized, -clip_sigma, clip_sigma).astype(np.float32, copy=False)


def filter_candidates(
    graph: TrackingGraph,
    scores: np.ndarray,
    min_score: float = 0.5,
) -> TrackingGraph:
    """Keep candidates at or above a learned cellness threshold.

    Any existing edge incident to a removed node is also removed.  The original node
    identifiers are retained so score arrays and downstream graph diagnostics remain
    auditable.
    """
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(graph.node_ids),):
        raise ValueError(f"scores must have shape {(len(graph.node_ids),)}, got {values.shape}")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must all be finite")

    keep = values >= min_score
    kept_ids = np.asarray(graph.node_ids)[keep]
    kept_set = set(int(node_id) for node_id in kept_ids)
    edges = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    kept_edges = np.array(
        [edge for edge in edges if int(edge[0]) in kept_set and int(edge[1]) in kept_set],
        dtype=np.int64,
    ).reshape(-1, 2)
    return TrackingGraph(
        node_ids=kept_ids,
        t=np.asarray(graph.t)[keep],
        z=np.asarray(graph.z)[keep],
        y=np.asarray(graph.y)[keep],
        x=np.asarray(graph.x)[keep],
        edges=kept_edges,
    )


def read_appearance_shard(path: str | Path) -> AppearanceShardSummary:
    """Read and validate one completed shard's metadata and array contract."""
    shard = Path(path)
    missing = [name for name in APPEARANCE_SHARD_FILES if not (shard / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete appearance shard {shard}: missing {missing}")
    summary = AppearanceShardSummary.from_dict(json.loads((shard / "metadata.json").read_text()))
    if summary.schema_version != APPEARANCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported appearance shard schema {summary.schema_version}; "
            f"expected {APPEARANCE_SCHEMA_VERSION}"
        )
    if summary.dataset != shard.name:
        raise ValueError(
            f"Appearance shard {shard} records dataset {summary.dataset!r}; "
            "the directory name must match"
        )
    if summary.embryo != APPEARANCE_TRAIN_EMBRYO:
        raise ValueError(
            f"Appearance shard {shard} is outside training embryo {APPEARANCE_TRAIN_EMBRYO}"
        )
    validate_patch_shape(summary.patch_shape)
    if not np.isfinite(summary.detector_threshold_k) or summary.detector_threshold_k <= 0.0:
        raise ValueError(f"Appearance shard {shard} has an invalid DoG threshold")
    if not np.isfinite(summary.unlabeled_ratio) or summary.unlabeled_ratio <= 0.0:
        raise ValueError(f"Appearance shard {shard} has an invalid unlabeled ratio")
    count_fields = {
        "raw candidates": summary.raw_candidate_count,
        "truth nodes": summary.truth_node_count,
        "matched candidates": summary.matched_candidate_count,
        "estimated nodes": summary.estimated_node_count,
        "known-positive samples": summary.known_positive_samples,
        "unlabeled samples": summary.unlabeled_samples,
    }
    invalid_counts = [name for name, value in count_fields.items() if value < 1]
    if invalid_counts:
        raise ValueError(f"Appearance shard {shard} has non-positive counts: {invalid_counts}")
    expected_samples = summary.known_positive_samples + summary.unlabeled_samples
    patches = np.load(shard / "patches.npy", mmap_mode="r", allow_pickle=False)
    roles = np.load(shard / "sample_roles.npy", mmap_mode="r", allow_pickle=False)
    candidate_tzyx = np.load(shard / "candidate_tzyx.npy", mmap_mode="r", allow_pickle=False)
    expected_patch_shape = (expected_samples, *summary.patch_shape)
    if patches.shape != expected_patch_shape:
        raise ValueError(
            f"Appearance shard {shard} patches have shape {patches.shape}, "
            f"expected {expected_patch_shape}"
        )
    if roles.shape != (expected_samples,):
        raise ValueError(
            f"Appearance shard {shard} roles have shape {roles.shape}, "
            f"expected {(expected_samples,)}"
        )
    if candidate_tzyx.shape != (expected_samples, 4):
        raise ValueError(
            f"Appearance shard {shard} coordinates have shape {candidate_tzyx.shape}, "
            f"expected {(expected_samples, 4)}"
        )
    if not np.all(np.isin(roles, [SAMPLE_UNLABELED, SAMPLE_KNOWN_POSITIVE])):
        raise ValueError(f"Appearance shard {shard} contains an unknown sample role")
    if int(np.sum(roles == SAMPLE_KNOWN_POSITIVE)) != summary.known_positive_samples:
        raise ValueError(f"Appearance shard {shard} known-positive count is inconsistent")
    if int(np.sum(roles == SAMPLE_UNLABELED)) != summary.unlabeled_samples:
        raise ValueError(f"Appearance shard {shard} unlabeled count is inconsistent")
    if not np.all(np.isfinite(candidate_tzyx)):
        raise ValueError(f"Appearance shard {shard} contains non-finite coordinates")
    if np.any(candidate_tzyx < 0):
        raise ValueError(f"Appearance shard {shard} contains negative coordinates")
    if summary.matched_candidate_count != summary.known_positive_samples:
        raise ValueError(f"Appearance shard {shard} matched-candidate count is inconsistent")
    if not 0.0 <= summary.sparse_label_recall <= 1.0:
        raise ValueError(f"Appearance shard {shard} has invalid sparse-label recall")
    expected_recall = summary.matched_candidate_count / summary.truth_node_count
    if not np.isclose(summary.sparse_label_recall, expected_recall):
        raise ValueError(f"Appearance shard {shard} sparse-label recall is inconsistent")
    if not 0.0 < summary.estimated_candidate_prior < 1.0:
        raise ValueError(f"Appearance shard {shard} has invalid candidate prior")
    expected_prior = np.clip(
        summary.estimated_node_count * expected_recall / summary.raw_candidate_count,
        0.01,
        0.99,
    )
    if not np.isclose(summary.estimated_candidate_prior, expected_prior):
        raise ValueError(f"Appearance shard {shard} candidate prior is inconsistent")
    if summary.raw_candidate_count < max(
        summary.known_positive_samples,
        summary.unlabeled_samples,
    ):
        raise ValueError(f"Appearance shard {shard} sample counts exceed raw candidates")
    return summary


def discover_appearance_shards(root: str | Path) -> list[Path]:
    """Return complete appearance shards in deterministic dataset order."""
    directory = Path(root)
    if not directory.exists():
        raise FileNotFoundError(f"Appearance shard directory does not exist: {directory}")
    shards: list[Path] = []
    for path in sorted(
        item for item in directory.iterdir() if item.is_dir() and not item.name.startswith(".")
    ):
        read_appearance_shard(path)
        shards.append(path)
    if not shards:
        raise FileNotFoundError(f"No complete appearance shards found under {directory}")
    return shards


def appearance_shards_sha256(shards: list[str | Path]) -> str:
    """Hash the complete ordered shard contents for training and resume provenance."""
    if not shards:
        raise ValueError("at least one appearance shard is required for a fingerprint")
    digest = hashlib.sha256()
    paths = [Path(value) for value in shards]
    for shard in paths:
        read_appearance_shard(shard)
        digest.update(shard.name.encode())
        digest.update(b"\0")
        for name in APPEARANCE_SHARD_FILES:
            digest.update(name.encode())
            digest.update(b"\0")
            with (shard / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _stable_dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _estimated_node_count(labels: Path) -> int:
    metadata = json.loads((labels / "zarr.json").read_text())
    return int(metadata["attributes"]["geff"]["extra"]["estimated_number_of_nodes"])


def _extract_selected_patches(
    image_array,
    candidates: TrackingGraph,
    selected_indices: np.ndarray,
    patch_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Read each needed timepoint once and materialize selected candidate patches."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    patches = np.empty((len(selected), *patch_shape), dtype=np.dtype(image_array.dtype))
    tzyx = np.column_stack(
        [
            np.asarray(candidates.t)[selected],
            np.asarray(candidates.z)[selected],
            np.asarray(candidates.y)[selected],
            np.asarray(candidates.x)[selected],
        ]
    ).astype(np.float32)
    for timepoint in np.unique(tzyx[:, 0].astype(int)):
        output_indices = np.flatnonzero(tzyx[:, 0].astype(int) == timepoint)
        volume = np.asarray(image_array[int(timepoint)])
        for output_index in output_indices:
            patches[output_index] = extract_patch(
                volume,
                tzyx[output_index, 1:],
                patch_shape=patch_shape,
            )
    return patches, tzyx


def build_appearance_shard(
    sample: SamplePaths,
    output_root: str | Path,
    *,
    detector_threshold_k: float = 3.0,
    patch_shape: tuple[int, int, int] = DEFAULT_PATCH_SHAPE,
    unlabeled_ratio: float = 4.0,
    seed: int = 0,
    overwrite: bool = False,
) -> AppearanceShardSummary:
    """Build one deterministic, auditable positive-unlabeled patch shard.

    Only the designated training embryo is accepted.  The held-out embryo is rejected
    here rather than relying on calling code to remember the split contract.
    """
    if sample.embryo != APPEARANCE_TRAIN_EMBRYO:
        raise ValueError(
            f"appearance training shards must come from embryo {APPEARANCE_TRAIN_EMBRYO}, "
            f"got {sample.dataset} from {sample.embryo}"
        )
    if sample.labels is None:
        raise ValueError(f"appearance training requires GEFF labels: {sample.dataset}")
    if detector_threshold_k <= 0:
        raise ValueError("detector_threshold_k must be positive")
    if unlabeled_ratio <= 0:
        raise ValueError("unlabeled_ratio must be positive")
    shape = validate_patch_shape(patch_shape)

    output = Path(output_root) / sample.dataset
    if output.exists() and not overwrite:
        existing = read_appearance_shard(output)
        requested = (shape, float(detector_threshold_k), float(unlabeled_ratio), int(seed))
        recorded = (
            existing.patch_shape,
            existing.detector_threshold_k,
            existing.unlabeled_ratio,
            existing.seed,
        )
        if recorded != requested:
            raise FileExistsError(
                f"existing shard {output} uses {recorded}, requested {requested}; "
                "pass overwrite=True to rebuild it"
            )
        return existing

    image = open_image_array(sample.image)
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    detector = DetectorConfig(method="dog", threshold_k=detector_threshold_k)
    candidates = detect_movie(image, detector)
    positive_mask = known_positive_mask(candidates, truth)
    positive_indices = np.flatnonzero(positive_mask)
    if len(positive_indices) == 0:
        raise RuntimeError(f"DoG produced no GEFF-matched candidates for {sample.dataset}")

    rng = np.random.default_rng(_stable_dataset_seed(seed, sample.dataset))
    unlabeled_count = min(
        len(candidates.node_ids),
        max(1, int(np.ceil(len(positive_indices) * unlabeled_ratio))),
    )
    # Draw the PU population from every DoG candidate, including possible positives.
    # This preserves the candidate population mixture required by the PU risk estimator.
    unlabeled_indices = rng.choice(
        len(candidates.node_ids), size=unlabeled_count, replace=False
    ).astype(np.int64)
    selected_indices = np.concatenate([positive_indices, unlabeled_indices])
    sample_roles = np.concatenate(
        [
            np.full(len(positive_indices), SAMPLE_KNOWN_POSITIVE, dtype=np.uint8),
            np.full(len(unlabeled_indices), SAMPLE_UNLABELED, dtype=np.uint8),
        ]
    )

    patches, tzyx = _extract_selected_patches(image, candidates, selected_indices, shape)
    estimated_nodes = _estimated_node_count(sample.labels)
    recall = len(positive_indices) / len(truth.node_ids) if len(truth.node_ids) else 0.0
    prior = min(
        0.99,
        max(0.01, estimated_nodes * recall / max(1, len(candidates.node_ids))),
    )
    summary = AppearanceShardSummary(
        schema_version=APPEARANCE_SCHEMA_VERSION,
        dataset=sample.dataset,
        embryo=sample.embryo,
        patch_shape=shape,
        detector_threshold_k=float(detector_threshold_k),
        unlabeled_ratio=float(unlabeled_ratio),
        seed=int(seed),
        raw_candidate_count=len(candidates.node_ids),
        truth_node_count=len(truth.node_ids),
        matched_candidate_count=len(positive_indices),
        sparse_label_recall=recall,
        estimated_node_count=estimated_nodes,
        estimated_candidate_prior=prior,
        known_positive_samples=len(positive_indices),
        unlabeled_samples=len(unlabeled_indices),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        np.save(temporary / "patches.npy", patches, allow_pickle=False)
        np.save(temporary / "sample_roles.npy", sample_roles, allow_pickle=False)
        np.save(temporary / "candidate_tzyx.npy", tzyx, allow_pickle=False)
        (temporary / "metadata.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            if not overwrite:
                raise FileExistsError(f"appearance shard already exists: {output}")
            shutil.rmtree(output)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary
