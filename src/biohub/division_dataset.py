"""Strict sampled records for the frozen Phase 11 division proposal contract."""

from __future__ import annotations

import csv
import errno
import fcntl
import hashlib
import io
import os
import shutil
from bisect import insort
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from biohub.appearance import filter_candidates
from biohub.division import (
    DivisionPairProposal,
    generate_division_pair_neighborhoods,
    generate_division_pair_proposals,
)
from biohub.division_data import (
    PROPOSAL_RULE_CONTRACT,
    SAMPLING_STATIC_CONTRACT,
    canonical_json_bytes,
    file_sha256,
    logical_graph_sha256,
    read_canonical_json,
    read_division_manifest,
    source_bundle_sha256,
    truth_divisions,
    verify_division_contract_bundle,
)
from biohub.division_model import (
    CONTEXT_FEATURE_KINDS,
    CONTEXT_FEATURE_NAMES,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    NODE_CACHE_EMBEDDING_DIM,
    FeatureSchema,
    build_symmetric_context_features,
    current_division_contract_hashes,
    read_division_node_cache,
    read_division_node_cache_metadata,
)
from biohub.io import SamplePaths, discover_split, read_geff_graph
from biohub.link import link_graph_flow, prune_to_tracks
from biohub.metric import TrackingGraph, match_nodes

DIVISION_DATASET_SCHEMA_VERSION = 2
DIVISION_FUNNEL_SCHEMA_VERSION = 1
LABEL_UNKNOWN = -1
POSITIVE = "positive"
SAME_MOTHER_NEGATIVE = "same_mother_negative"
MATCHED_NONDIVISION_CONTROL = "matched_nondivision_control"
DIAGNOSTIC_UNKNOWN = "diagnostic_unknown"
SUPERVISED_STRATA = (POSITIVE, SAME_MOTHER_NEGATIVE, MATCHED_NONDIVISION_CONTROL)
SAMPLING_STRATA = (*SUPERVISED_STRATA, DIAGNOSTIC_UNKNOWN)
STRATUM_CODES = {name: index for index, name in enumerate(SAMPLING_STRATA)}
CODE_STRATA = {value: key for key, value in STRATUM_CODES.items()}
IMAGE_FEATURE_DIM = NODE_CACHE_EMBEDDING_DIM * 4
FROZEN_CELLNESS_THRESHOLD = 0.20
FROZEN_MIN_TRACK_LENGTH = 6
DATASET_SOURCE_FILES = (
    "src/biohub/division_dataset.py",
    "scripts/build_division_dataset.py",
)
CACHE_SOURCE_FILES = (
    "src/biohub/appearance.py",
    "src/biohub/appearance_evaluation.py",
    "src/biohub/appearance_model.py",
    "src/biohub/detect.py",
    "src/biohub/division_data.py",
    "src/biohub/division_model.py",
    "src/biohub/io.py",
    "scripts/build_division_cache.py",
)
MODEL_INPUTS_HASH_SCHEMA = "biohub-phase11-division-model-inputs-v1"
SAMPLE_TUPLE_HASH_SCHEMA = "biohub-phase11-division-sample-tuple-v1"
EPOCH_INDEX_HASH_SALT = "biohub-phase11-division-epoch-index-v1"
RECORD_RANK_SALT = "biohub-phase11-division-record-v1"
BUILD_MODE_SMOKE = "smoke"
BUILD_MODE_FIT_COMPLETE = "fit_complete"
BUILD_MODE_DEVELOPMENT_FUNNEL = "development_funnel"
DATASET_BUILD_MODES = (BUILD_MODE_SMOKE, BUILD_MODE_FIT_COMPLETE)
EXPECTED_FIT_MOVIES = 88
EXPECTED_DEVELOPMENT_MOVIES = 20
DATASET_TRANSACTION_SCHEMA_VERSION = 2
TRANSACTION_STAGING = "staging"
TRANSACTION_PROMOTING = "promoting"
TRANSACTION_PROMOTED = "promoted"

DATASET_FILES = (
    "records.npz",
    "records.csv",
    "feature_schema.json",
    "funnel.json",
    "manifest.json",
)
DATASET_ARRAY_FIELDS = frozenset(
    {
        "candidate_ids",
        "dataset_indices",
        "deterministic_ranks",
        "edge_removal_counts",
        "image_features",
        "labels",
        "loss_weights",
        "minimum_cellness",
        "mother_ids",
        "node_references",
        "raw_context",
        "restored_node_counts",
        "stratum_codes",
    }
)
RECORD_CSV_FIELDS = (
    "record_index",
    "dataset",
    "mother_id",
    "candidate_1_id",
    "candidate_2_id",
    "mother_node_index",
    "candidate_1_node_index",
    "candidate_2_node_index",
    "sampling_stratum",
    "source_role",
    "label",
    "loss_weight",
    "minimum_cellness",
    "restored_node_count",
    "edge_removal_count",
    "deterministic_rank",
)
SAMPLE_TUPLE_FIELDS = RECORD_CSV_FIELDS[1:]
_CACHE_IDENTITY_FIELDS = {
    "content_sha256",
    "file_sha256",
    "image_sha256",
    "role",
}
_MANIFEST_FIELDS = {
    "build_mode",
    "cache_files",
    "cache_source_sha256",
    "contract_sha256",
    "dataset_source_files",
    "dataset_source_sha256",
    "datasets",
    "division_dataset_schema_version",
    "feature_schema",
    "files",
    "model_inputs_sha256",
    "proposal_rule",
    "record_count",
    "sample_tuple_fields",
    "sample_tuple_sha256",
    "source_role",
    "stratum_counts",
    "supervised_record_count",
    "unknown_record_count",
    "upstream_manifest_sha256",
}


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


def _require_int(value: Any, *, field: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{field} is outside its valid range")
    return result


def _require_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
        raise ValueError(f"{field} must be finite and numeric")
    return float(value)


def _typed_context(values: Sequence[Real]) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_context must contain numeric values") from exc
    if array.shape != (len(CONTEXT_FEATURE_NAMES),) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"raw_context must contain {len(CONTEXT_FEATURE_NAMES)} finite values"
        )
    for index, kind in enumerate(CONTEXT_FEATURE_KINDS):
        value = float(array[index])
        if kind == "binary" and value not in (0.0, 1.0):
            raise ValueError(
                f"raw_context binary feature {CONTEXT_FEATURE_NAMES[index]} is invalid"
            )
        if kind == "count" and (value < 0.0 or value != np.floor(value)):
            raise ValueError(f"raw_context count feature {CONTEXT_FEATURE_NAMES[index]} is invalid")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """One immutable identifier-bearing proposal and identifier-free model inputs."""

    dataset: str
    mother_id: int
    candidate_1_id: int
    candidate_2_id: int
    mother_node_index: int
    candidate_1_node_index: int
    candidate_2_node_index: int
    sampling_stratum: str
    source_role: str
    label: int
    loss_weight: float
    minimum_cellness: float
    restored_node_count: int
    edge_removal_count: int
    deterministic_rank: str
    raw_context: tuple[float, ...]
    image_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.startswith("6bba_"):
            raise ValueError("ProposalRecord accepts only 6bba datasets")
        identifiers = {
            name: _require_int(getattr(self, name), field=name)
            for name in ("mother_id", "candidate_1_id", "candidate_2_id")
        }
        if not identifiers["candidate_1_id"] < identifiers["candidate_2_id"]:
            raise ValueError("ProposalRecord candidate identifiers must be canonical and distinct")
        references = {
            name: _require_int(getattr(self, name), field=name)
            for name in (
                "mother_node_index",
                "candidate_1_node_index",
                "candidate_2_node_index",
            )
        }
        if len(set(references.values())) != 3:
            raise ValueError("ProposalRecord node references must be distinct")
        if self.sampling_stratum not in SAMPLING_STRATA:
            raise ValueError("ProposalRecord has an unsupported sampling stratum")
        if self.source_role != "fit":
            raise ValueError("ProposalRecord training records must come only from fit")
        label = _require_int(self.label, field="label", minimum=LABEL_UNKNOWN, maximum=1)
        expected_label = 1 if self.sampling_stratum == POSITIVE else 0
        expected_weight = 1.0
        if self.sampling_stratum == DIAGNOSTIC_UNKNOWN:
            expected_label = LABEL_UNKNOWN
            expected_weight = 0.0
        weight = _require_finite(self.loss_weight, field="loss_weight")
        if label != expected_label or weight != expected_weight:
            raise ValueError("ProposalRecord label or loss weight disagrees with its stratum")
        minimum_cellness = _require_finite(
            self.minimum_cellness,
            field="minimum_cellness",
        )
        if not 0.0 <= minimum_cellness <= 1.0:
            raise ValueError("ProposalRecord minimum cellness must be between zero and one")
        restored = _require_int(
            self.restored_node_count,
            field="restored_node_count",
            maximum=255,
        )
        removals = _require_int(
            self.edge_removal_count,
            field="edge_removal_count",
            maximum=255,
        )
        rank = _require_sha256(self.deterministic_rank, field="deterministic_rank")
        context = _typed_context(self.raw_context)
        try:
            image = np.asarray(self.image_features, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("image_features must contain numeric values") from exc
        if image.shape != (IMAGE_FEATURE_DIM,) or not np.all(np.isfinite(image)):
            raise ValueError(f"image_features must contain {IMAGE_FEATURE_DIM} finite values")

        for name, value in {**identifiers, **references}.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "loss_weight", float(np.float32(weight)))
        object.__setattr__(
            self,
            "minimum_cellness",
            float(np.float32(minimum_cellness)),
        )
        object.__setattr__(self, "restored_node_count", restored)
        object.__setattr__(self, "edge_removal_count", removals)
        object.__setattr__(self, "deterministic_rank", rank)
        object.__setattr__(self, "raw_context", context)
        object.__setattr__(
            self,
            "image_features",
            tuple(float(value) for value in image),
        )


@dataclass(frozen=True, slots=True)
class DatasetBuildIdentity:
    """Complete upstream identity required to serialize or resume one fit dataset."""

    cache_files: Mapping[str, Mapping[str, Any]]
    cache_source_sha256: str
    build_mode: str
    contract_sha256: Mapping[str, str]
    dataset_source_sha256: str
    datasets: tuple[str, ...]
    upstream_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.build_mode not in DATASET_BUILD_MODES:
            raise ValueError("dataset identity requires smoke or fit_complete build mode")
        datasets = tuple(self.datasets)
        if (
            not datasets
            or datasets != tuple(sorted(set(datasets)))
            or any(
                not isinstance(value, str) or not value.startswith("6bba_")
                for value in datasets
            )
        ):
            raise ValueError("dataset identity requires unique sorted 6bba datasets")
        if self.build_mode == BUILD_MODE_SMOKE and len(datasets) != 1:
            raise ValueError("smoke dataset identity requires exactly one fit movie")
        if self.build_mode == BUILD_MODE_FIT_COMPLETE and len(datasets) != EXPECTED_FIT_MOVIES:
            raise ValueError(
                f"fit_complete dataset identity requires exactly {EXPECTED_FIT_MOVIES} movies"
            )
        contracts = dict(self.contract_sha256)
        if set(contracts) != set(EXPECTED_DIVISION_CONTRACT_SHA256):
            raise ValueError("dataset identity requires all five contract SHA-256 values")
        contracts = {
            name: _require_sha256(value, field=f"{name} contract SHA-256")
            for name, value in contracts.items()
        }
        if contracts != EXPECTED_DIVISION_CONTRACT_SHA256:
            raise ValueError("dataset identity contract hashes are not the frozen Task 1 values")
        cache_files = dict(self.cache_files)
        if set(cache_files) != set(datasets):
            raise ValueError("dataset identity cache tuple does not match its datasets")
        normalized_caches: dict[str, Mapping[str, Any]] = {}
        for dataset in datasets:
            record = dict(cache_files[dataset])
            if set(record) != _CACHE_IDENTITY_FIELDS:
                raise ValueError("dataset cache identity has incomplete or unexpected fields")
            if record["role"] != "fit":
                raise ValueError("dataset cache identity accepts fit caches only")
            normalized_caches[dataset] = MappingProxyType(
                {
                    "content_sha256": _require_sha256(
                        record["content_sha256"], field="cache content SHA-256"
                    ),
                    "file_sha256": _require_sha256(
                        record["file_sha256"], field="cache file SHA-256"
                    ),
                    "image_sha256": _require_sha256(
                        record["image_sha256"], field="cache image SHA-256"
                    ),
                    "role": "fit",
                }
            )
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "contract_sha256", MappingProxyType(contracts))
        object.__setattr__(self, "cache_files", MappingProxyType(normalized_caches))
        object.__setattr__(
            self,
            "cache_source_sha256",
            _require_sha256(self.cache_source_sha256, field="cache source SHA-256"),
        )
        object.__setattr__(
            self,
            "dataset_source_sha256",
            _require_sha256(self.dataset_source_sha256, field="dataset source SHA-256"),
        )
        object.__setattr__(
            self,
            "upstream_manifest_sha256",
            _require_sha256(
                self.upstream_manifest_sha256,
                field="upstream manifest SHA-256",
            ),
        )

    def cache_payload(self) -> dict[str, dict[str, Any]]:
        return {dataset: dict(self.cache_files[dataset]) for dataset in self.datasets}

    def contract_payload(self) -> dict[str, str]:
        return dict(self.contract_sha256)


@dataclass(frozen=True, slots=True)
class DatasetArtifactResult:
    path: Path
    build_mode: str
    record_count: int
    stratum_counts: Mapping[str, int]
    feature_schema_sha256: str
    sample_tuple_sha256: str
    dataset_manifest_sha256: str
    model_inputs_sha256: str
    skipped: bool


@dataclass(frozen=True, slots=True)
class LoadedDivisionDataset:
    records: tuple[ProposalRecord, ...]
    feature_schema: FeatureSchema
    image_features: np.ndarray
    manifest: Mapping[str, Any]
    dataset_manifest_sha256: str
    model_inputs_sha256: str
    normalized_context: np.ndarray


def _readonly_array(
    values: Any,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    field: str,
    finite: bool = True,
) -> np.ndarray:
    try:
        array = np.array(values, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if array.shape != shape or (finite and not np.all(np.isfinite(array))):
        raise ValueError(f"{field} has an invalid shape or value")
    array.setflags(write=False)
    return array


def _readonly_graph_copy(graph: TrackingGraph, *, field: str) -> TrackingGraph:
    if not isinstance(graph, TrackingGraph):
        raise TypeError(f"{field} must be a TrackingGraph")
    count = len(np.asarray(graph.node_ids))
    node_ids = _readonly_array(
        graph.node_ids,
        dtype=np.dtype(np.int64),
        shape=(count,),
        field=f"{field} node identifiers",
    )
    if len(np.unique(node_ids)) != count:
        raise ValueError(f"{field} contains duplicate node identifiers")
    t = _readonly_array(
        graph.t,
        dtype=np.dtype(np.int64),
        shape=(count,),
        field=f"{field} time coordinates",
    )
    coordinates = {
        name: _readonly_array(
            getattr(graph, name),
            dtype=np.dtype(np.float64),
            shape=(count,),
            field=f"{field} {name} coordinates",
        )
        for name in ("z", "y", "x")
    }
    raw_edges = np.asarray(graph.edges)
    if raw_edges.size % 2:
        raise ValueError(f"{field} edges have an invalid shape")
    edges = _readonly_array(
        raw_edges.reshape(-1, 2),
        dtype=np.dtype(np.int64),
        shape=(raw_edges.size // 2, 2),
        field=f"{field} edges",
    )
    known = set(int(value) for value in node_ids)
    if any(int(source) not in known or int(target) not in known for source, target in edges):
        raise ValueError(f"{field} contains an edge with a missing endpoint")
    return TrackingGraph(
        node_ids=node_ids,
        t=t,
        z=coordinates["z"],
        y=coordinates["y"],
        x=coordinates["x"],
        edges=edges,
    )


@dataclass(frozen=True, slots=True)
class InferenceMovie:
    """One label-free movie prepared for bounded proposal inference."""

    dataset: str
    source_role: str
    raw: TrackingGraph
    filtered: TrackingGraph
    linked: TrackingGraph
    fallback: TrackingGraph
    cellness: np.ndarray
    embeddings: np.ndarray
    intensity_summaries: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.startswith("6bba_"):
            raise ValueError("inference movies accept only 6bba datasets")
        if self.source_role not in {"development", "evaluation"}:
            raise ValueError("inference movies require the development or evaluation role")
        raw = _readonly_graph_copy(self.raw, field="raw inference graph")
        filtered = _readonly_graph_copy(self.filtered, field="filtered inference graph")
        linked = _readonly_graph_copy(self.linked, field="linked inference graph")
        fallback = _readonly_graph_copy(self.fallback, field="fallback inference graph")
        count = len(raw.node_ids)
        cellness = _readonly_array(
            self.cellness,
            dtype=np.dtype(np.float32),
            shape=(count,),
            field="inference cellness",
        )
        embeddings = _readonly_array(
            self.embeddings,
            dtype=np.dtype(np.float32),
            shape=(count, NODE_CACHE_EMBEDDING_DIM),
            field="inference embeddings",
        )
        intensity = _readonly_array(
            self.intensity_summaries,
            dtype=np.dtype(np.float32),
            shape=(count, 5),
            field="inference intensity summaries",
        )
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "filtered", filtered)
        object.__setattr__(self, "linked", linked)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "cellness", cellness)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "intensity_summaries", intensity)


@dataclass(frozen=True, slots=True)
class InferenceProposalBatch:
    """An immutable, label-free, model-ready group of complete mother neighborhoods."""

    dataset: str
    source_role: str
    proposals: tuple[DivisionPairProposal, ...]
    raw_context: np.ndarray
    normalized_context: np.ndarray
    image_features: np.ndarray
    minimum_cellness: np.ndarray
    restored_node_counts: np.ndarray
    edge_removal_counts: np.ndarray
    linked_graph: TrackingGraph
    fallback_graph: TrackingGraph

    def __post_init__(self) -> None:
        proposals = tuple(self.proposals)
        count = len(proposals)
        if not proposals:
            raise ValueError("inference proposal batches cannot be empty")
        if not isinstance(self.dataset, str) or not self.dataset.startswith("6bba_"):
            raise ValueError("inference proposal batches accept only 6bba datasets")
        if self.source_role not in {"development", "evaluation"}:
            raise ValueError("inference proposal batches require development or evaluation")
        if any(not isinstance(value, DivisionPairProposal) for value in proposals):
            raise TypeError("inference proposal batches require DivisionPairProposal values")
        raw_context = _readonly_array(
            self.raw_context,
            dtype=np.dtype(np.float32),
            shape=(count, len(CONTEXT_FEATURE_NAMES)),
            field="inference raw context",
        )
        normalized_context = _readonly_array(
            self.normalized_context,
            dtype=np.dtype(np.float32),
            shape=(count, len(CONTEXT_FEATURE_NAMES)),
            field="inference normalized context",
        )
        image_features = _readonly_array(
            self.image_features,
            dtype=np.dtype(np.float32),
            shape=(count, IMAGE_FEATURE_DIM),
            field="inference image features",
        )
        minimum_cellness = _readonly_array(
            self.minimum_cellness,
            dtype=np.dtype(np.float32),
            shape=(count,),
            field="inference minimum cellness",
        )
        if np.any((minimum_cellness < 0.0) | (minimum_cellness > 1.0)):
            raise ValueError("inference minimum cellness must be between zero and one")
        restored = _readonly_array(
            self.restored_node_counts,
            dtype=np.dtype(np.uint8),
            shape=(count,),
            field="inference restored node counts",
            finite=False,
        )
        removals = _readonly_array(
            self.edge_removal_counts,
            dtype=np.dtype(np.uint8),
            shape=(count,),
            field="inference edge removal counts",
            finite=False,
        )
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "raw_context", raw_context)
        object.__setattr__(self, "normalized_context", normalized_context)
        object.__setattr__(self, "image_features", image_features)
        object.__setattr__(self, "minimum_cellness", minimum_cellness)
        object.__setattr__(self, "restored_node_counts", restored)
        object.__setattr__(self, "edge_removal_counts", removals)

    @property
    def mother_count(self) -> int:
        return len({proposal.mother_id for proposal in self.proposals})


def invariant_image_features(
    mother: Any,
    daughter_1: Any,
    daughter_2: Any,
) -> np.ndarray:
    """Precompute exact ``[mother, daughter mean, absdiff, product]`` features."""
    arrays: list[np.ndarray] = []
    for name, value in (
        ("mother", mother),
        ("daughter_1", daughter_1),
        ("daughter_2", daughter_2),
    ):
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} embedding must be numeric") from exc
        if array.shape != (NODE_CACHE_EMBEDDING_DIM,) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} embedding must contain {NODE_CACHE_EMBEDDING_DIM} finite values"
            )
        arrays.append(array)
    mother_values, first, second = arrays
    return np.ascontiguousarray(
        np.concatenate(
            [
                mother_values,
                (first + second) / np.float32(2.0),
                np.abs(first - second),
                first * second,
            ]
        ),
        dtype=np.float32,
    )


def proposal_hardness_key(record: ProposalRecord) -> tuple[float, int, int, int, int]:
    """Return the exact frozen descending-cellness hardness ordering."""
    if not isinstance(record, ProposalRecord):
        raise TypeError("proposal hardness requires ProposalRecord values")
    return (
        -record.minimum_cellness,
        record.restored_node_count,
        record.edge_removal_count,
        record.candidate_1_id,
        record.candidate_2_id,
    )


def _rank_bytes(salt: str, dataset: str, *identifiers: int) -> bytes:
    if not isinstance(salt, str) or not salt:
        raise ValueError("sampling rank requires a non-empty salt")
    encoded = [salt.encode("ascii"), dataset.encode("ascii")]
    encoded.extend(
        str(_require_int(value, field="rank identifier")).encode("ascii")
        for value in identifiers
    )
    return hashlib.sha256(b"\0".join(encoded)).digest()


def _record_rank(record: ProposalRecord, *, salt: str) -> str:
    return _rank_bytes(
        salt,
        record.dataset,
        record.mother_id,
        record.candidate_1_id,
        record.candidate_2_id,
    ).hex()


def select_same_mother_negatives(
    records: Sequence[ProposalRecord],
    *,
    max_pairs_per_mother: int = 12,
) -> list[ProposalRecord]:
    """Keep each truth mother's hardest non-true pairs under the frozen cap."""
    cap = _require_int(max_pairs_per_mother, field="max_pairs_per_mother", minimum=1)
    grouped: dict[tuple[str, int], list[ProposalRecord]] = defaultdict(list)
    for record in records:
        if (
            not isinstance(record, ProposalRecord)
            or record.sampling_stratum != SAME_MOTHER_NEGATIVE
        ):
            raise ValueError("same-mother selection accepts only same-mother negative records")
        grouped[(record.dataset, record.mother_id)].append(record)
    selected: list[ProposalRecord] = []
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key], key=proposal_hardness_key)[:cap])
    return selected


def _require_sampling_subcontract(
    contract: Mapping[str, Any],
    *,
    expected_fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or set(contract) != expected_fields:
        raise ValueError(f"{label} sampling contract has incomplete or unexpected fields")
    return dict(contract)


def select_matched_nondivision_controls(
    records: Sequence[ProposalRecord],
    *,
    dataset: str,
    contract: Mapping[str, Any],
) -> list[ProposalRecord]:
    """SHA-rank control mothers and keep one hardest pair for each selected mother."""
    policy = _require_sampling_subcontract(
        contract,
        expected_fields={
            "hardest_pair_order",
            "max_mothers_per_movie",
            "mother_rank",
            "pairs_per_mother",
            "source_role",
            "supervised",
        },
        label="matched non-division",
    )
    rank = policy["mother_rank"]
    if (
        not isinstance(rank, Mapping)
        or set(rank) != {"algorithm", "encoding", "salt"}
        or rank["algorithm"] != "sha256"
        or policy["pairs_per_mother"] != 1
        or policy["source_role"] != "fit"
        or policy["supervised"] is not True
    ):
        raise ValueError("matched non-division sampling contract is invalid")
    grouped: dict[int, list[ProposalRecord]] = defaultdict(list)
    for record in records:
        if (
            not isinstance(record, ProposalRecord)
            or record.dataset != dataset
            or record.sampling_stratum != MATCHED_NONDIVISION_CONTROL
        ):
            raise ValueError("control selection received a record outside its movie or stratum")
        grouped[record.mother_id].append(record)
    ranked_mothers = sorted(
        grouped,
        key=lambda mother: (_rank_bytes(str(rank["salt"]), dataset, mother), mother),
    )[: _require_int(policy["max_mothers_per_movie"], field="control mother cap", minimum=1)]
    result: list[ProposalRecord] = []
    for mother in ranked_mothers:
        hardest = min(grouped[mother], key=proposal_hardness_key)
        result.append(
            replace(
                hardest,
                deterministic_rank=_rank_bytes(str(rank["salt"]), dataset, mother).hex(),
            )
        )
    return result


def select_unknown_diagnostics(
    records: Sequence[ProposalRecord],
    *,
    dataset: str,
    contract: Mapping[str, Any],
) -> list[ProposalRecord]:
    """Keep the exact smallest proposal hashes for zero-weight unknown diagnostics."""
    policy = _require_sampling_subcontract(
        contract,
        expected_fields={
            "loss_weight",
            "max_proposals_per_movie",
            "proposal_rank",
            "source_role",
            "supervised",
        },
        label="unmatched diagnostic",
    )
    rank = policy["proposal_rank"]
    if (
        not isinstance(rank, Mapping)
        or set(rank) != {"algorithm", "encoding", "salt"}
        or rank["algorithm"] != "sha256"
        or policy["loss_weight"] != 0.0
        or policy["source_role"] != "fit"
        or policy["supervised"] is not False
    ):
        raise ValueError("unmatched diagnostic sampling contract is invalid")
    ranked: list[tuple[bytes, int, int, int, ProposalRecord]] = []
    for record in records:
        if (
            not isinstance(record, ProposalRecord)
            or record.dataset != dataset
            or record.sampling_stratum != DIAGNOSTIC_UNKNOWN
        ):
            raise ValueError("unknown selection received a record outside its movie or stratum")
        digest = _rank_bytes(
            str(rank["salt"]),
            dataset,
            record.mother_id,
            record.candidate_1_id,
            record.candidate_2_id,
        )
        ranked.append(
            (digest, record.mother_id, record.candidate_1_id, record.candidate_2_id, record)
        )
    cap = _require_int(policy["max_proposals_per_movie"], field="diagnostic cap", minimum=1)
    return [
        replace(record, deterministic_rank=digest.hex())
        for digest, _, _, _, record in sorted(ranked)[:cap]
    ]


def fit_feature_schema(records: Sequence[ProposalRecord]) -> FeatureSchema:
    """Fit continuous context statistics from supervised fit records only."""
    materialized = tuple(records)
    supervised = [
        record.raw_context
        for record in materialized
        if isinstance(record, ProposalRecord)
        and record.sampling_stratum in SUPERVISED_STRATA
        and record.label in (0, 1)
        and record.loss_weight == 1.0
    ]
    if not supervised:
        raise ValueError("feature normalization requires at least one supervised fit record")
    return FeatureSchema.fit(
        CONTEXT_FEATURE_NAMES,
        np.asarray(supervised, dtype=np.float32),
        kinds=CONTEXT_FEATURE_KINDS,
        source_role="fit",
    )


def _epoch_order(
    indices: Sequence[int],
    *,
    seed: int,
    epoch: int,
    stratum: str,
    cycle: int,
) -> list[int]:
    prefix = f"{EPOCH_INDEX_HASH_SALT}\0{seed}\0{epoch}\0{stratum}\0{cycle}\0".encode(
        "ascii"
    )
    return sorted(
        indices,
        key=lambda index: (hashlib.sha256(prefix + str(index).encode("ascii")).digest(), index),
    )


def balanced_epoch_indices(
    records: Sequence[ProposalRecord],
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Generate deterministic interleaved 1:1:1 supervised epoch indices."""
    seed_value = _require_int(seed, field="seed")
    epoch_value = _require_int(epoch, field="epoch")
    grouped = {
        stratum: [
            index
            for index, record in enumerate(records)
            if isinstance(record, ProposalRecord) and record.sampling_stratum == stratum
        ]
        for stratum in SUPERVISED_STRATA
    }
    missing = [stratum for stratum, values in grouped.items() if not values]
    if missing:
        raise ValueError("balanced epochs require every supervised stratum: " + ", ".join(missing))
    target = max(len(values) for values in grouped.values())
    sequences: dict[str, list[int]] = {}
    for stratum, values in grouped.items():
        sequence: list[int] = []
        cycle = 0
        while len(sequence) < target:
            sequence.extend(
                _epoch_order(
                    values,
                    seed=seed_value,
                    epoch=epoch_value,
                    stratum=stratum,
                    cycle=cycle,
                )
            )
            cycle += 1
        sequences[stratum] = sequence[:target]
    interleaved = [
        sequences[stratum][position]
        for position in range(target)
        for stratum in SUPERVISED_STRATA
    ]
    if any(records[index].sampling_stratum == DIAGNOSTIC_UNKNOWN for index in interleaved):
        raise RuntimeError("unknown diagnostics entered supervised epoch indices")
    return np.asarray(interleaved, dtype=np.int64)


def reconstruct_upstream(
    raw: TrackingGraph,
    cellness: Any,
) -> tuple[TrackingGraph, TrackingGraph, TrackingGraph]:
    """Reconstruct exact threshold-0.20, flow-link, minimum-length-6 upstream graphs."""
    scores = np.asarray(cellness)
    if scores.shape != (len(raw.node_ids),):
        raise ValueError("cellness does not align with the raw node cache")
    filtered = filter_candidates(raw, scores, min_score=FROZEN_CELLNESS_THRESHOLD)
    linked = link_graph_flow(filtered)
    pruned = prune_to_tracks(linked, min_track_length=FROZEN_MIN_TRACK_LENGTH)
    return filtered, linked, pruned


def prepare_inference_movie(
    dataset: str,
    *,
    raw: TrackingGraph,
    cellness: Any,
    embeddings: Any,
    intensity_summaries: Any,
    proposal_contract: Mapping[str, Any],
    source_role: str,
) -> InferenceMovie:
    """Reconstruct the frozen fallback without accepting labels or truth graphs."""
    if not isinstance(proposal_contract, Mapping) or dict(proposal_contract) != dict(
        PROPOSAL_RULE_CONTRACT
    ):
        raise ValueError("inference requires the exact frozen proposal rule")
    filtered, linked, fallback = reconstruct_upstream(raw, cellness)
    return InferenceMovie(
        dataset=dataset,
        source_role=source_role,
        raw=raw,
        filtered=filtered,
        linked=linked,
        fallback=fallback,
        cellness=cellness,
        embeddings=embeddings,
        intensity_summaries=intensity_summaries,
    )


def _oriented_edges(graph: TrackingGraph) -> set[tuple[int, int]]:
    times = {
        int(node_id): int(time)
        for node_id, time in zip(graph.node_ids, graph.t, strict=True)
    }
    oriented: set[tuple[int, int]] = set()
    for raw_source, raw_target in np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2):
        source = int(raw_source)
        target = int(raw_target)
        if source not in times or target not in times:
            raise ValueError("proposal context graph contains an edge with a missing endpoint")
        if times[target] < times[source]:
            source, target = target, source
        if times[target] <= times[source]:
            raise ValueError("proposal context graph edge is not strictly forward in time")
        oriented.add((source, target))
    return oriented


@dataclass(frozen=True, slots=True)
class _CandidateSample:
    proposal: DivisionPairProposal
    minimum_cellness: float
    restored_node_count: int
    edge_removal_count: int
    deterministic_rank: str | None = None

    @property
    def hardness(self) -> tuple[float, int, int, int, int]:
        return (
            -self.minimum_cellness,
            self.restored_node_count,
            self.edge_removal_count,
            self.proposal.candidate_1.node_id,
            self.proposal.candidate_2.node_id,
        )


def _candidate_sample(
    proposal: DivisionPairProposal,
    *,
    score_by_id: Mapping[int, float],
    pruned_ids: set[int],
    pruned_successors: Mapping[int, set[int]],
    pruned_predecessors: Mapping[int, set[int]],
) -> _CandidateSample:
    first = proposal.candidate_1.node_id
    second = proposal.candidate_2.node_id
    members = (proposal.mother_id, first, second)
    restored = sum(node_id not in pruned_ids for node_id in members)
    pair = {first, second}
    removals = {
        (proposal.mother_id, child)
        for child in pruned_successors.get(proposal.mother_id, set())
        if child not in pair
    }
    for daughter in pair:
        removals.update(
            (parent, daughter)
            for parent in pruned_predecessors.get(daughter, set())
            if parent != proposal.mother_id
        )
    try:
        minimum = min(score_by_id[node_id] for node_id in members)
    except KeyError as exc:
        raise ValueError("proposal references a node absent from the score cache") from exc
    return _CandidateSample(
        proposal=proposal,
        minimum_cellness=float(np.float32(minimum)),
        restored_node_count=restored,
        edge_removal_count=len(removals),
    )


def _proposal_model_inputs(
    candidate: _CandidateSample,
    *,
    node_row: Mapping[int, int],
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = candidate.proposal
    mother_id = proposal.mother_id
    first_id = proposal.candidate_1.node_id
    second_id = proposal.candidate_2.node_id
    try:
        mother_row = node_row[mother_id]
        first_row = node_row[first_id]
        second_row = node_row[second_id]
    except KeyError as exc:
        raise ValueError("selected proposal has an invalid node reference") from exc
    first = proposal.candidate_1
    second = proposal.candidate_2
    context = build_symmetric_context_features(
        mother_cellness=np.array([cellness[mother_row]], dtype=np.float32),
        daughter_cellness=np.array(
            [[cellness[first_row], cellness[second_row]]], dtype=np.float32
        ),
        daughter_distance_um=np.array(
            [[first.distance_um, second.distance_um]], dtype=np.float32
        ),
        daughter_angle_deg=np.array([proposal.daughter_angle_deg], dtype=np.float32),
        candidate_flags={
            "present_in_linked_graph": np.array(
                [[first.present_in_linked_graph, second.present_in_linked_graph]],
                dtype=np.uint8,
            ),
            "is_current_child": np.array(
                [[first.is_current_child, second.is_current_child]], dtype=np.uint8
            ),
            "is_parentless": np.array(
                [[first.is_parentless, second.is_parentless]], dtype=np.uint8
            ),
            "has_wrong_parent": np.array(
                [[first.has_wrong_parent, second.has_wrong_parent]], dtype=np.uint8
            ),
            "has_future": np.array([[first.has_future, second.has_future]], dtype=np.uint8),
            "survives_pruning": np.array(
                [[first.survives_pruning, second.survives_pruning]], dtype=np.uint8
            ),
        },
        mother_flags={
            "mother_has_history": np.array([proposal.mother_has_history], dtype=np.uint8),
            "mother_has_current_child": np.array(
                [proposal.mother_has_current_child], dtype=np.uint8
            ),
            "mother_survives_pruning": np.array(
                [proposal.mother_survives_pruning], dtype=np.uint8
            ),
        },
        restored_node_count=np.array([candidate.restored_node_count], dtype=np.uint8),
        edge_removal_count=np.array([candidate.edge_removal_count], dtype=np.uint8),
        mother_intensity_summaries=intensity_summaries[mother_row][None, :],
        daughter_intensity_summaries=np.stack(
            [intensity_summaries[first_row], intensity_summaries[second_row]]
        )[None, :, :],
    )[0]
    image = invariant_image_features(
        embeddings[mother_row],
        embeddings[first_row],
        embeddings[second_row],
    )
    return context, image


def _retain_smallest_ranked(
    retained: list[tuple[bytes, int, int, int, _CandidateSample]],
    item: tuple[bytes, int, int, int, _CandidateSample],
    *,
    cap: int,
) -> None:
    insort(retained, item, key=lambda value: value[:4])
    if len(retained) > cap:
        retained.pop()


def _validate_proposal_and_sampling_contracts(
    proposal_contract: Mapping[str, Any],
    sampling_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposal = dict(proposal_contract)
    if proposal != PROPOSAL_RULE_CONTRACT:
        raise ValueError("dataset construction requires the exact frozen proposal rule")
    if not isinstance(sampling_contract, Mapping):
        raise ValueError("dataset construction requires the frozen sampling contract")
    required = {
        "matched_nondivision_controls",
        "positive_records",
        "record_identity_fields",
        "same_mother_alternatives",
        "supervised_batches",
        "unmatched_diagnostics",
    }
    sampling = {
        key: sampling_contract[key]
        for key in required
        if key in sampling_contract
    }
    if set(sampling) != required:
        raise ValueError("dataset sampling contract is incomplete")
    if sampling != SAMPLING_STATIC_CONTRACT:
        raise ValueError("dataset construction requires the exact frozen sampling policy")
    for name in (
        "positive_records",
        "same_mother_alternatives",
        "matched_nondivision_controls",
        "unmatched_diagnostics",
    ):
        if sampling[name].get("source_role") != "fit":
            raise ValueError("frozen sampling records must all declare fit as their source")
    return proposal, sampling


def _proposal_record(
    dataset: str,
    candidate: _CandidateSample,
    *,
    sampling_stratum: str,
    node_row: Mapping[int, int],
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
    deterministic_rank: str,
) -> ProposalRecord:
    proposal = candidate.proposal
    mother_id = proposal.mother_id
    first_id = proposal.candidate_1.node_id
    second_id = proposal.candidate_2.node_id
    try:
        mother_row = node_row[mother_id]
        first_row = node_row[first_id]
        second_row = node_row[second_id]
    except KeyError as exc:
        raise ValueError("selected proposal has an invalid node reference") from exc
    context, image = _proposal_model_inputs(
        candidate,
        node_row=node_row,
        cellness=cellness,
        embeddings=embeddings,
        intensity_summaries=intensity_summaries,
    )
    label = 1 if sampling_stratum == POSITIVE else 0
    weight = 1.0
    if sampling_stratum == DIAGNOSTIC_UNKNOWN:
        label = LABEL_UNKNOWN
        weight = 0.0
    return ProposalRecord(
        dataset=dataset,
        mother_id=mother_id,
        candidate_1_id=first_id,
        candidate_2_id=second_id,
        mother_node_index=mother_row,
        candidate_1_node_index=first_row,
        candidate_2_node_index=second_row,
        sampling_stratum=sampling_stratum,
        source_role="fit",
        label=label,
        loss_weight=weight,
        minimum_cellness=candidate.minimum_cellness,
        restored_node_count=candidate.restored_node_count,
        edge_removal_count=candidate.edge_removal_count,
        deterministic_rank=deterministic_rank,
        raw_context=tuple(float(value) for value in context),
        image_features=tuple(float(value) for value in image),
    )


def iter_inference_proposal_batches(
    movie: InferenceMovie,
    *,
    feature_schema: FeatureSchema,
    proposal_contract: Mapping[str, Any],
    max_mothers_per_batch: int = 128,
) -> Iterator[InferenceProposalBatch]:
    """Yield complete mother neighborhoods without materializing all proposal pairs."""
    if not isinstance(movie, InferenceMovie):
        raise TypeError("inference streaming requires an InferenceMovie")
    if not isinstance(feature_schema, FeatureSchema):
        raise TypeError("inference streaming requires a FeatureSchema")
    if not isinstance(proposal_contract, Mapping) or dict(proposal_contract) != dict(
        PROPOSAL_RULE_CONTRACT
    ):
        raise ValueError("inference requires the exact frozen proposal rule")
    if (
        isinstance(max_mothers_per_batch, bool)
        or not isinstance(max_mothers_per_batch, Integral)
        or int(max_mothers_per_batch) < 1
    ):
        raise ValueError("max_mothers_per_batch must be a positive integer")
    mother_bound = int(max_mothers_per_batch)
    raw = movie.raw
    node_row = {int(node_id): index for index, node_id in enumerate(raw.node_ids)}
    if len(node_row) != len(raw.node_ids):
        raise ValueError("movie raw candidates contain duplicate node identifiers")
    score_by_id = {
        int(node_id): float(score)
        for node_id, score in zip(raw.node_ids, movie.cellness, strict=True)
    }
    fallback_ids = {int(node_id) for node_id in movie.fallback.node_ids}
    fallback_edges = _oriented_edges(movie.fallback)
    fallback_successors: dict[int, set[int]] = defaultdict(set)
    fallback_predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in fallback_edges:
        fallback_successors[source].add(target)
        fallback_predecessors[target].add(source)
    proposal_rule = dict(proposal_contract)
    neighborhoods = generate_division_pair_neighborhoods(
        movie.linked,
        raw,
        pruned_node_ids=fallback_ids,
        max_distance_um=float(proposal_rule["max_distance_um"]),
    )
    for start in range(0, len(neighborhoods), mother_bound):
        neighborhood_batch = neighborhoods[start : start + mother_bound]
        proposals = tuple(
            generate_division_pair_proposals(
                neighborhood_batch,
                min_distance_um=float(proposal_rule["min_distance_um"]),
                max_distance_um=float(proposal_rule["max_distance_um"]),
                min_angle_deg=float(proposal_rule["min_angle_deg"]),
                max_candidates_per_mother=int(
                    proposal_rule["max_candidates_per_mother"]
                ),
            )
        )
        if not proposals:
            continue
        candidates = tuple(
            _candidate_sample(
                proposal,
                score_by_id=score_by_id,
                pruned_ids=fallback_ids,
                pruned_successors=fallback_successors,
                pruned_predecessors=fallback_predecessors,
            )
            for proposal in proposals
        )
        model_inputs = tuple(
            _proposal_model_inputs(
                candidate,
                node_row=node_row,
                cellness=movie.cellness,
                embeddings=movie.embeddings,
                intensity_summaries=movie.intensity_summaries,
            )
            for candidate in candidates
        )
        raw_context = np.stack([item[0] for item in model_inputs]).astype(
            np.float32,
            copy=False,
        )
        image_features = np.stack([item[1] for item in model_inputs]).astype(
            np.float32,
            copy=False,
        )
        yield InferenceProposalBatch(
            dataset=movie.dataset,
            source_role=movie.source_role,
            proposals=proposals,
            raw_context=raw_context,
            normalized_context=feature_schema.standardize(raw_context),
            image_features=image_features,
            minimum_cellness=np.asarray(
                [candidate.minimum_cellness for candidate in candidates],
                dtype=np.float32,
            ),
            restored_node_counts=np.asarray(
                [candidate.restored_node_count for candidate in candidates],
                dtype=np.uint8,
            ),
            edge_removal_counts=np.asarray(
                [candidate.edge_removal_count for candidate in candidates],
                dtype=np.uint8,
            ),
            linked_graph=movie.linked,
            fallback_graph=movie.fallback,
        )


def sample_movie_proposals(
    dataset: str,
    *,
    raw: TrackingGraph,
    cellness: Any,
    embeddings: Any,
    intensity_summaries: Any,
    filtered: TrackingGraph,
    linked: TrackingGraph,
    pruned: TrackingGraph,
    truth: TrackingGraph,
    proposal_contract: Mapping[str, Any],
    sampling_contract: Mapping[str, Any],
    source_role: str,
    materialize_records: bool = True,
) -> tuple[list[ProposalRecord], dict[str, Any]]:
    """Classify and sample one movie without materializing its unknown proposal table."""
    if not isinstance(dataset, str) or not dataset.startswith("6bba_"):
        raise ValueError("division proposal sampling accepts only 6bba datasets")
    if source_role not in {"fit", "development"}:
        raise ValueError("proposal funnels accept only fit or development")
    if source_role != "fit" and materialize_records:
        raise ValueError("development funnel mode cannot materialize training records")
    proposal_rule, sampling = _validate_proposal_and_sampling_contracts(
        proposal_contract,
        sampling_contract,
    )
    scores = np.asarray(cellness, dtype=np.float32)
    embedding_values = np.asarray(embeddings, dtype=np.float32)
    intensity_values = np.asarray(intensity_summaries, dtype=np.float32)
    count = len(raw.node_ids)
    if scores.shape != (count,) or not np.all(np.isfinite(scores)):
        raise ValueError("movie cellness must align with raw candidates and be finite")
    if embedding_values.shape != (count, NODE_CACHE_EMBEDDING_DIM) or not np.all(
        np.isfinite(embedding_values)
    ):
        raise ValueError("movie embeddings have an invalid shape or value")
    if intensity_values.shape != (count, 5) or not np.all(np.isfinite(intensity_values)):
        raise ValueError("movie intensity summaries have an invalid shape or value")
    node_row = {int(node_id): index for index, node_id in enumerate(raw.node_ids)}
    if len(node_row) != count:
        raise ValueError("movie raw candidates contain duplicate node identifiers")
    score_by_id = {
        int(node_id): float(score)
        for node_id, score in zip(raw.node_ids, scores, strict=True)
    }
    mother_matches = match_nodes(filtered, truth)
    candidate_matches = match_nodes(raw, truth)
    if len(set(mother_matches.values())) != len(mother_matches) or len(
        set(candidate_matches.values())
    ) != len(candidate_matches):
        raise RuntimeError("public node matching produced duplicate truth assignments")
    truth_to_mother = {truth_id: predicted_id for predicted_id, truth_id in mother_matches.items()}
    truth_to_candidate = {
        truth_id: predicted_id for predicted_id, truth_id in candidate_matches.items()
    }
    divisions = truth_divisions(truth)
    pruned_ids = {int(node_id) for node_id in pruned.node_ids}
    neighborhoods = generate_division_pair_neighborhoods(
        linked,
        raw,
        pruned_node_ids=pruned_ids,
        max_distance_um=float(proposal_rule["max_distance_um"]),
    )
    neighborhood_by_mother = {item.mother_id: item for item in neighborhoods}
    if len(neighborhood_by_mother) != len(neighborhoods):
        raise RuntimeError("proposal generation produced duplicate mother neighborhoods")

    events: list[dict[str, Any]] = []
    event_by_truth_mother: dict[int, dict[str, Any]] = {}
    for truth_mother, truth_daughters in sorted(divisions.items()):
        predicted_mother = truth_to_mother.get(truth_mother)
        predicted_daughters = [truth_to_candidate.get(value) for value in truth_daughters]
        ranks: list[int | None] = []
        neighborhood = neighborhood_by_mother.get(predicted_mother)
        rank_by_candidate = (
            {
                candidate.node_id: index + 1
                for index, candidate in enumerate(neighborhood.candidates)
            }
            if neighborhood is not None
            else {}
        )
        for predicted in predicted_daughters:
            ranks.append(None if predicted is None else rank_by_candidate.get(predicted))
        event = {
            "candidate_ranks": ranks,
            "missing_daughter_ids": [
                truth_id
                for truth_id, predicted in zip(truth_daughters, predicted_daughters, strict=True)
                if predicted is None
            ],
            "predicted_daughter_ids": predicted_daughters,
            "predicted_mother_id": predicted_mother,
            "recoverable_positive": False,
            "truth_daughter_ids": list(truth_daughters),
            "truth_mother_id": truth_mother,
        }
        events.append(event)
        event_by_truth_mother[truth_mother] = event

    positive_candidates: list[_CandidateSample] = []
    alternatives: dict[int, list[_CandidateSample]] = defaultdict(list)
    controls: dict[int, _CandidateSample] = {}
    unknown_policy = sampling["unmatched_diagnostics"]
    unknown_rank = unknown_policy["proposal_rank"]
    unknown_cap = _require_int(
        unknown_policy["max_proposals_per_movie"],
        field="diagnostic proposal cap",
        minimum=1,
    )
    unknown: list[tuple[bytes, int, int, int, _CandidateSample]] = []
    role_counts: Counter[str] = Counter()
    pruned_edges = _oriented_edges(pruned)
    pruned_successors: dict[int, set[int]] = defaultdict(set)
    pruned_predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in pruned_edges:
        pruned_successors[source].add(target)
        pruned_predecessors[target].add(source)
    for neighborhood in neighborhoods:
        proposals = generate_division_pair_proposals(
            [neighborhood],
            min_distance_um=float(proposal_rule["min_distance_um"]),
            max_distance_um=float(proposal_rule["max_distance_um"]),
            min_angle_deg=float(proposal_rule["min_angle_deg"]),
            max_candidates_per_mother=int(proposal_rule["max_candidates_per_mother"]),
        )
        for proposal in proposals:
            candidate = _candidate_sample(
                proposal,
                score_by_id=score_by_id,
                pruned_ids=pruned_ids,
                pruned_successors=pruned_successors,
                pruned_predecessors=pruned_predecessors,
            )
            truth_mother = mother_matches.get(proposal.mother_id)
            if truth_mother is None:
                role_counts[DIAGNOSTIC_UNKNOWN] += 1
                digest = _rank_bytes(
                    str(unknown_rank["salt"]),
                    dataset,
                    proposal.mother_id,
                    proposal.candidate_1.node_id,
                    proposal.candidate_2.node_id,
                )
                candidate = replace(candidate, deterministic_rank=digest.hex())
                _retain_smallest_ranked(
                    unknown,
                    (
                        digest,
                        proposal.mother_id,
                        proposal.candidate_1.node_id,
                        proposal.candidate_2.node_id,
                        candidate,
                    ),
                    cap=unknown_cap,
                )
                continue
            if truth_mother not in divisions:
                role_counts[MATCHED_NONDIVISION_CONTROL] += 1
                current = controls.get(proposal.mother_id)
                if current is None or candidate.hardness < current.hardness:
                    controls[proposal.mother_id] = candidate
                continue
            first_truth = candidate_matches.get(proposal.candidate_1.node_id)
            second_truth = candidate_matches.get(proposal.candidate_2.node_id)
            daughters = set(divisions[truth_mother])
            is_true = (
                first_truth is not None
                and second_truth is not None
                and first_truth != second_truth
                and {first_truth, second_truth} == daughters
            )
            if is_true:
                role_counts[POSITIVE] += 1
                positive_candidates.append(candidate)
                event_by_truth_mother[truth_mother]["recoverable_positive"] = True
            else:
                role_counts[SAME_MOTHER_NEGATIVE] += 1
                alternatives[proposal.mother_id].append(candidate)

    positive_candidates.sort(
        key=lambda value: (
            value.proposal.mother_id,
            value.proposal.candidate_1.node_id,
            value.proposal.candidate_2.node_id,
        )
    )
    recoverable_mothers = {value.proposal.mother_id for value in positive_candidates}
    alternative_cap = _require_int(
        sampling["same_mother_alternatives"]["max_pairs_per_mother"],
        field="same-mother alternative cap",
        minimum=1,
    )
    selected_alternatives = [
        candidate
        for mother in sorted(recoverable_mothers)
        for candidate in sorted(alternatives.get(mother, ()), key=lambda value: value.hardness)[
            :alternative_cap
        ]
    ]
    control_policy = sampling["matched_nondivision_controls"]
    control_rank = control_policy["mother_rank"]
    control_cap = _require_int(
        control_policy["max_mothers_per_movie"],
        field="matched control mother cap",
        minimum=1,
    )
    selected_control_mothers = sorted(
        controls,
        key=lambda mother: (
            _rank_bytes(str(control_rank["salt"]), dataset, mother),
            mother,
        ),
    )[:control_cap]
    selected_controls = [
        replace(
            controls[mother],
            deterministic_rank=_rank_bytes(str(control_rank["salt"]), dataset, mother).hex(),
        )
        for mother in selected_control_mothers
    ]
    selected_unknown = [item[4] for item in unknown]
    selected_counts = {
        POSITIVE: len(positive_candidates),
        SAME_MOTHER_NEGATIVE: len(selected_alternatives),
        MATCHED_NONDIVISION_CONTROL: len(selected_controls),
        DIAGNOSTIC_UNKNOWN: len(selected_unknown),
    }
    records: list[ProposalRecord] = []
    if materialize_records:
        selections = (
            (POSITIVE, positive_candidates),
            (SAME_MOTHER_NEGATIVE, selected_alternatives),
            (MATCHED_NONDIVISION_CONTROL, selected_controls),
            (DIAGNOSTIC_UNKNOWN, selected_unknown),
        )
        for stratum, candidates in selections:
            for candidate in candidates:
                rank = candidate.deterministic_rank
                if rank is None:
                    proposal = candidate.proposal
                    rank = _rank_bytes(
                        RECORD_RANK_SALT,
                        dataset,
                        proposal.mother_id,
                        proposal.candidate_1.node_id,
                        proposal.candidate_2.node_id,
                        STRATUM_CODES[stratum],
                    ).hex()
                records.append(
                    _proposal_record(
                        dataset,
                        candidate,
                        sampling_stratum=stratum,
                        node_row=node_row,
                        cellness=scores,
                        embeddings=embedding_values,
                        intensity_summaries=intensity_values,
                        deterministic_rank=rank,
                    )
                )
    funnel = {
        "candidate_ranks": [
            rank for event in events for rank in event["candidate_ranks"] if rank is not None
        ],
        "dataset": dataset,
        "events": events,
        "filtered_candidates": len(filtered.node_ids),
        "linked_edges": len(linked.edges),
        "missing_daughters": sum(len(event["missing_daughter_ids"]) for event in events),
        "missing_mothers": sum(event["predicted_mother_id"] is None for event in events),
        "proposal_role_counts": {
            stratum: int(role_counts[stratum]) for stratum in SAMPLING_STRATA
        },
        "pruned_nodes": len(pruned.node_ids),
        "raw_candidates": len(raw.node_ids),
        "recoverable_positives": sum(event["recoverable_positive"] for event in events),
        "selected_role_counts": {
            stratum: selected_counts[stratum] for stratum in SAMPLING_STRATA
        },
        "source_role": source_role,
        "truth_events": len(events),
    }
    return records, funnel


def _record_csv_payload(index: int, record: ProposalRecord) -> dict[str, str | int]:
    return {
        "record_index": index,
        "dataset": record.dataset,
        "mother_id": record.mother_id,
        "candidate_1_id": record.candidate_1_id,
        "candidate_2_id": record.candidate_2_id,
        "mother_node_index": record.mother_node_index,
        "candidate_1_node_index": record.candidate_1_node_index,
        "candidate_2_node_index": record.candidate_2_node_index,
        "sampling_stratum": record.sampling_stratum,
        "source_role": record.source_role,
        "label": record.label,
        "loss_weight": repr(record.loss_weight),
        "minimum_cellness": repr(record.minimum_cellness),
        "restored_node_count": record.restored_node_count,
        "edge_removal_count": record.edge_removal_count,
        "deterministic_rank": record.deterministic_rank,
    }


def _records_csv_bytes(records: Sequence[ProposalRecord]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=RECORD_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, record in enumerate(records):
        writer.writerow(_record_csv_payload(index, record))
    return stream.getvalue().encode("ascii")


def _sample_tuple_payload(record: ProposalRecord) -> dict[str, Any]:
    payload = _record_csv_payload(0, record)
    payload.pop("record_index")
    payload["loss_weight"] = record.loss_weight
    payload["minimum_cellness"] = record.minimum_cellness
    return payload


def _sample_tuple_sha256(records: Sequence[ProposalRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(SAMPLE_TUPLE_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for record in records:
        digest.update(canonical_json_bytes(_sample_tuple_payload(record)))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_model_inputs_sha256(raw_context: Any, image_features: Any) -> str:
    """Hash every model-input byte with fixed field, dtype, shape, and C-order framing."""
    arrays: list[tuple[str, np.ndarray]] = []
    for name, values in (
        ("raw_context", raw_context),
        ("image_features", image_features),
    ):
        try:
            array = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{name} must be a two-dimensional floating-point array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        arrays.append((name, array))
    if arrays[0][1].shape[0] != arrays[1][1].shape[0]:
        raise ValueError("raw context and image features must contain the same row count")

    digest = hashlib.sha256()
    digest.update(MODEL_INPUTS_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for name, array in arrays:
        digest.update(name.encode("ascii"))
        digest.update(b"\0dtype\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0shape\0")
        digest.update(",".join(str(value) for value in array.shape).encode("ascii"))
        digest.update(b"\0order\0C\0data\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _records_arrays(
    records: Sequence[ProposalRecord],
    *,
    datasets: tuple[str, ...],
) -> dict[str, np.ndarray]:
    dataset_index = {dataset: index for index, dataset in enumerate(datasets)}
    return {
        "dataset_indices": np.asarray(
            [dataset_index[record.dataset] for record in records], dtype="<i4"
        ),
        "mother_ids": np.asarray([record.mother_id for record in records], dtype="<i8"),
        "candidate_ids": np.asarray(
            [[record.candidate_1_id, record.candidate_2_id] for record in records],
            dtype="<i8",
        ),
        "node_references": np.asarray(
            [
                [
                    record.mother_node_index,
                    record.candidate_1_node_index,
                    record.candidate_2_node_index,
                ]
                for record in records
            ],
            dtype="<i8",
        ),
        "stratum_codes": np.asarray(
            [STRATUM_CODES[record.sampling_stratum] for record in records], dtype=np.uint8
        ),
        "labels": np.asarray([record.label for record in records], dtype=np.int8),
        "loss_weights": np.asarray(
            [record.loss_weight for record in records], dtype="<f4"
        ),
        "minimum_cellness": np.asarray(
            [record.minimum_cellness for record in records], dtype="<f4"
        ),
        "restored_node_counts": np.asarray(
            [record.restored_node_count for record in records], dtype=np.uint8
        ),
        "edge_removal_counts": np.asarray(
            [record.edge_removal_count for record in records], dtype=np.uint8
        ),
        "deterministic_ranks": np.asarray(
            [record.deterministic_rank.encode("ascii") for record in records], dtype="S64"
        ),
        "raw_context": np.asarray([record.raw_context for record in records], dtype="<f4"),
        "image_features": np.asarray(
            [record.image_features for record in records], dtype="<f4"
        ),
    }


def _write_records_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def dataset_lock_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(f"{path.name}.lock")


def dataset_transaction_path(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(f".{path.name}.transaction")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _dataset_file_lock(
    output: str | Path,
    *,
    exclusive: bool,
    blocking: bool,
) -> Iterator[None]:
    lock = dataset_lock_path(output)
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open("a+")
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        try:
            fcntl.flock(handle.fileno(), operation)
        except OSError as exc:
            if blocking or exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(f"dataset builder is already active for {Path(output)}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@contextmanager
def exclusive_dataset_builder_lock(output: str | Path) -> Iterator[None]:
    """Hold a nonblocking process lock across resume validation and promotion."""
    with _dataset_file_lock(output, exclusive=True, blocking=False):
        yield


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _transaction_marker_payload(
    *,
    state: str,
    had_destination: bool,
    staged_dataset_manifest_sha256: str | None,
) -> dict[str, Any]:
    if state not in {TRANSACTION_STAGING, TRANSACTION_PROMOTING, TRANSACTION_PROMOTED}:
        raise ValueError("dataset transaction marker has an invalid state")
    if not isinstance(had_destination, bool):
        raise ValueError("dataset transaction marker destination flag must be boolean")
    if staged_dataset_manifest_sha256 is not None:
        _require_sha256(
            staged_dataset_manifest_sha256,
            field="staged dataset manifest SHA-256",
        )
    return {
        "dataset_transaction_schema_version": DATASET_TRANSACTION_SCHEMA_VERSION,
        "had_destination": had_destination,
        "staged_dataset_manifest_sha256": staged_dataset_manifest_sha256,
        "state": state,
    }


def _write_transaction_marker(
    transaction: Path,
    *,
    state: str,
    had_destination: bool,
    staged_dataset_manifest_sha256: str | None,
) -> None:
    payload = _transaction_marker_payload(
        state=state,
        had_destination=had_destination,
        staged_dataset_manifest_sha256=staged_dataset_manifest_sha256,
    )
    marker = transaction / "marker.json"
    temporary = transaction / ".marker.json.tmp"
    _write_bytes(temporary, canonical_json_bytes(payload))
    os.replace(temporary, marker)
    _fsync_directory(transaction)


def _read_transaction_marker(transaction: Path) -> dict[str, Any]:
    marker = read_canonical_json(transaction / "marker.json")
    expected = {
        "dataset_transaction_schema_version",
        "had_destination",
        "staged_dataset_manifest_sha256",
        "state",
    }
    if set(marker) != expected:
        raise ValueError("dataset transaction marker has incomplete or unexpected fields")
    normalized = _transaction_marker_payload(
        state=marker["state"],
        had_destination=marker["had_destination"],
        staged_dataset_manifest_sha256=marker["staged_dataset_manifest_sha256"],
    )
    if marker["dataset_transaction_schema_version"] != DATASET_TRANSACTION_SCHEMA_VERSION:
        raise ValueError("unsupported dataset transaction schema")
    return normalized


def _recover_dataset_transaction(output: str | Path) -> None:
    """Recover an interrupted promotion while the caller holds the exclusive lock."""
    destination = Path(output)
    transaction = dataset_transaction_path(destination)
    if not transaction.exists():
        return
    if not transaction.is_dir():
        raise ValueError("dataset transaction path is not a directory")
    backup = transaction / "backup"
    staged = transaction / "staged"
    marker_path = transaction / "marker.json"
    if not marker_path.is_file():
        if backup.exists():
            if destination.exists():
                _remove_path(destination)
            os.replace(backup, destination)
        _remove_path(transaction)
        _fsync_directory(destination.parent)
        return
    marker = _read_transaction_marker(transaction)
    state = marker["state"]
    had_destination = marker["had_destination"]
    if state == TRANSACTION_PROMOTED:
        if not destination.exists() and backup.exists():
            os.replace(backup, destination)
        if staged.exists():
            _remove_path(staged)
        if backup.exists():
            _remove_path(backup)
        _remove_path(transaction)
        _fsync_directory(destination.parent)
        return

    if backup.exists():
        if destination.exists():
            _remove_path(destination)
        os.replace(backup, destination)
    elif state == TRANSACTION_PROMOTING and not had_destination and destination.exists():
        _remove_path(destination)
    _remove_path(transaction)
    _fsync_directory(destination.parent)


@contextmanager
def shared_dataset_reader_lock(output: str | Path) -> Iterator[None]:
    """Block behind writers and recover abandoned transactions before strict reads."""
    destination = Path(output)
    if not destination.parent.exists():
        raise FileNotFoundError(f"division dataset parent does not exist: {destination.parent}")
    while True:
        with _dataset_file_lock(destination, exclusive=False, blocking=True):
            if not dataset_transaction_path(destination).exists():
                yield
                return
        with _dataset_file_lock(destination, exclusive=True, blocking=True):
            _recover_dataset_transaction(destination)


def _require_trainable_records(records: Sequence[ProposalRecord]) -> tuple[ProposalRecord, ...]:
    materialized = tuple(records)
    if not materialized or any(not isinstance(record, ProposalRecord) for record in materialized):
        raise ValueError("division dataset requires immutable ProposalRecord values")
    observed = Counter(record.sampling_stratum for record in materialized)
    missing = [stratum for stratum in SUPERVISED_STRATA if observed[stratum] == 0]
    if missing:
        raise ValueError("division dataset is missing supervised strata: " + ", ".join(missing))
    identities = [
        (
            record.dataset,
            record.mother_id,
            record.candidate_1_id,
            record.candidate_2_id,
            record.sampling_stratum,
            record.source_role,
        )
        for record in materialized
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("division dataset contains a duplicate realized sample tuple")
    return materialized


def _funnel_payload(funnels: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(funnels, Mapping) or set(funnels) != {"fit"}:
        raise ValueError("training dataset artifacts require exactly one fit funnel")
    return {
        "division_funnel_schema_version": DIVISION_FUNNEL_SCHEMA_VERSION,
        "roles": {"fit": funnels["fit"]},
    }


def _manifest_payload(
    records: Sequence[ProposalRecord],
    *,
    identity: DatasetBuildIdentity,
    schema: FeatureSchema,
    output: Path,
    model_inputs_hash: str,
    sample_hash: str,
) -> dict[str, Any]:
    counts = Counter(record.sampling_stratum for record in records)
    files = {
        name: {"path": name, "sha256": file_sha256(output / name)}
        for name in DATASET_FILES[:-1]
    }
    return {
        "build_mode": identity.build_mode,
        "cache_files": identity.cache_payload(),
        "cache_source_sha256": identity.cache_source_sha256,
        "contract_sha256": identity.contract_payload(),
        "dataset_source_files": list(DATASET_SOURCE_FILES),
        "dataset_source_sha256": identity.dataset_source_sha256,
        "datasets": list(identity.datasets),
        "division_dataset_schema_version": DIVISION_DATASET_SCHEMA_VERSION,
        "feature_schema": {
            "path": "feature_schema.json",
            "sha256": schema.sha256,
        },
        "files": files,
        "model_inputs_sha256": model_inputs_hash,
        "proposal_rule": PROPOSAL_RULE_CONTRACT,
        "record_count": len(records),
        "sample_tuple_fields": list(SAMPLE_TUPLE_FIELDS),
        "sample_tuple_sha256": sample_hash,
        "source_role": "fit",
        "stratum_counts": {stratum: counts[stratum] for stratum in SAMPLING_STRATA},
        "supervised_record_count": sum(counts[stratum] for stratum in SUPERVISED_STRATA),
        "unknown_record_count": counts[DIAGNOSTIC_UNKNOWN],
        "upstream_manifest_sha256": identity.upstream_manifest_sha256,
    }


def _artifact_result(
    path: Path,
    loaded: LoadedDivisionDataset,
    *,
    skipped: bool,
) -> DatasetArtifactResult:
    return DatasetArtifactResult(
        path=path,
        build_mode=str(loaded.manifest["build_mode"]),
        record_count=len(loaded.records),
        stratum_counts=MappingProxyType(dict(loaded.manifest["stratum_counts"])),
        feature_schema_sha256=loaded.feature_schema.sha256,
        sample_tuple_sha256=str(loaded.manifest["sample_tuple_sha256"]),
        dataset_manifest_sha256=loaded.dataset_manifest_sha256,
        model_inputs_sha256=loaded.model_inputs_sha256,
        skipped=skipped,
    )


def _promote_dataset_transaction(
    output: Path,
    transaction: Path,
    *,
    identity: DatasetBuildIdentity,
) -> LoadedDivisionDataset:
    staging = transaction / "staged"
    backup = transaction / "backup"
    dataset_manifest_hash = file_sha256(staging / "manifest.json")
    had_destination = output.exists()
    _write_transaction_marker(
        transaction,
        state=TRANSACTION_PROMOTING,
        had_destination=had_destination,
        staged_dataset_manifest_sha256=dataset_manifest_hash,
    )
    if had_destination:
        os.replace(output, backup)
        _fsync_directory(transaction)
        _fsync_directory(output.parent)
    os.replace(staging, output)
    _fsync_directory(transaction)
    _fsync_directory(output.parent)
    loaded = _read_dataset_artifacts_unlocked(
        output,
        expected_identity=identity,
        expected_dataset_source_sha256=identity.dataset_source_sha256,
    )
    if loaded.dataset_manifest_sha256 != dataset_manifest_hash:
        raise ValueError("promoted dataset manifest hash changed during transaction")
    _write_transaction_marker(
        transaction,
        state=TRANSACTION_PROMOTED,
        had_destination=had_destination,
        staged_dataset_manifest_sha256=dataset_manifest_hash,
    )
    if backup.exists():
        _remove_path(backup)
    _remove_path(transaction)
    _fsync_directory(output.parent)
    return loaded


def _build_dataset_artifacts_locked(
    destination: Path,
    records: Sequence[ProposalRecord],
    *,
    identity: DatasetBuildIdentity,
    funnels: Mapping[str, Any],
    overwrite: bool = False,
    before_promotion: Callable[[], None] | None = None,
) -> DatasetArtifactResult:
    """Serialize or resume while the caller holds the destination builder lock."""
    materialized = _require_trainable_records(records)
    if tuple(sorted({record.dataset for record in materialized})) != identity.datasets:
        raise ValueError("realized sample datasets do not match the dataset identity")
    schema = fit_feature_schema(materialized)
    sample_hash = _sample_tuple_sha256(materialized)
    arrays = _records_arrays(materialized, datasets=identity.datasets)
    model_inputs_hash = canonical_model_inputs_sha256(
        arrays["raw_context"],
        arrays["image_features"],
    )
    funnel_payload = _funnel_payload(funnels)
    expected_funnel_hash = hashlib.sha256(canonical_json_bytes(funnel_payload)).hexdigest()
    if destination.exists() and not overwrite:
        loaded = _read_dataset_artifacts_unlocked(
            destination,
            expected_identity=identity,
            expected_dataset_source_sha256=identity.dataset_source_sha256,
        )
        if (
            loaded.manifest["sample_tuple_sha256"] != sample_hash
            or loaded.model_inputs_sha256 != model_inputs_hash
            or loaded.feature_schema != schema
            or loaded.manifest["files"]["funnel.json"]["sha256"] != expected_funnel_hash
        ):
            raise FileExistsError("existing division dataset does not match requested inputs")
        return _artifact_result(destination, loaded, skipped=True)

    transaction = dataset_transaction_path(destination)
    if transaction.exists():
        raise RuntimeError("dataset transaction recovery did not clear the transaction")
    transaction.mkdir()
    _fsync_directory(destination.parent)
    staging = transaction / "staged"
    staging.mkdir()
    _write_transaction_marker(
        transaction,
        state=TRANSACTION_STAGING,
        had_destination=destination.exists(),
        staged_dataset_manifest_sha256=None,
    )
    try:
        _write_records_npz(staging / "records.npz", arrays)
        _write_bytes(staging / "records.csv", _records_csv_bytes(materialized))
        _write_bytes(staging / "feature_schema.json", schema.canonical_bytes())
        _write_bytes(staging / "funnel.json", canonical_json_bytes(funnel_payload))
        manifest = _manifest_payload(
            materialized,
            identity=identity,
            schema=schema,
            output=staging,
            model_inputs_hash=model_inputs_hash,
            sample_hash=sample_hash,
        )
        _write_bytes(staging / "manifest.json", canonical_json_bytes(manifest))
        staged_dataset_manifest_hash = file_sha256(staging / "manifest.json")
        _write_transaction_marker(
            transaction,
            state=TRANSACTION_STAGING,
            had_destination=destination.exists(),
            staged_dataset_manifest_sha256=staged_dataset_manifest_hash,
        )
        _read_dataset_artifacts_unlocked(
            staging,
            expected_identity=identity,
            expected_dataset_source_sha256=identity.dataset_source_sha256,
        )
        _fsync_directory(staging)
        if before_promotion is not None:
            before_promotion()
        loaded = _promote_dataset_transaction(
            destination,
            transaction,
            identity=identity,
        )
        return _artifact_result(destination, loaded, skipped=False)
    except BaseException:
        _recover_dataset_transaction(destination)
        raise


def build_dataset_artifacts(
    output: str | Path,
    records: Sequence[ProposalRecord],
    *,
    identity: DatasetBuildIdentity,
    funnels: Mapping[str, Any],
    overwrite: bool = False,
    before_promotion: Callable[[], None] | None = None,
) -> DatasetArtifactResult:
    """Atomically serialize one fit table, or strictly validate an exact resume."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_dataset_builder_lock(destination):
        _recover_dataset_transaction(destination)
        return _build_dataset_artifacts_locked(
            destination,
            records,
            identity=identity,
            funnels=funnels,
            overwrite=overwrite,
            before_promotion=before_promotion,
        )


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != DATASET_ARRAY_FIELDS:
                raise ValueError("division dataset NPZ has incomplete or unexpected arrays")
            return {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("division dataset"):
            raise
        raise ValueError(f"invalid division dataset NPZ {path}: {exc}") from exc


def _validate_array_layout(arrays: Mapping[str, np.ndarray], *, count: int) -> None:
    expected = {
        "dataset_indices": (np.dtype("<i4"), (count,)),
        "mother_ids": (np.dtype("<i8"), (count,)),
        "candidate_ids": (np.dtype("<i8"), (count, 2)),
        "node_references": (np.dtype("<i8"), (count, 3)),
        "stratum_codes": (np.dtype("uint8"), (count,)),
        "labels": (np.dtype("int8"), (count,)),
        "loss_weights": (np.dtype("<f4"), (count,)),
        "minimum_cellness": (np.dtype("<f4"), (count,)),
        "restored_node_counts": (np.dtype("uint8"), (count,)),
        "edge_removal_counts": (np.dtype("uint8"), (count,)),
        "deterministic_ranks": (np.dtype("S64"), (count,)),
        "raw_context": (np.dtype("<f4"), (count, len(CONTEXT_FEATURE_NAMES))),
        "image_features": (np.dtype("<f4"), (count, IMAGE_FEATURE_DIM)),
    }
    for name, (dtype, shape) in expected.items():
        values = arrays[name]
        if values.dtype != dtype or values.shape != shape:
            raise ValueError(
                f"division dataset array {name} must have dtype {dtype} and shape {shape}"
            )
    for name in ("loss_weights", "minimum_cellness", "raw_context", "image_features"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"division dataset array {name} contains non-finite values")


def _read_csv_records(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    datasets: tuple[str, ...],
) -> tuple[ProposalRecord, ...]:
    content = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(content.decode("ascii"), newline=""))
    except UnicodeDecodeError as exc:
        raise ValueError("division dataset CSV must be ASCII") from exc
    if tuple(reader.fieldnames or ()) != RECORD_CSV_FIELDS:
        raise ValueError("division dataset CSV has an unsupported schema")
    payloads = list(reader)
    count = len(arrays["labels"])
    if len(payloads) != count:
        raise ValueError("division dataset CSV and NPZ record counts disagree")
    records: list[ProposalRecord] = []
    try:
        for index, payload in enumerate(payloads):
            dataset_index = int(arrays["dataset_indices"][index])
            if not 0 <= dataset_index < len(datasets):
                raise ValueError("dataset index is outside the manifest tuple")
            code = int(arrays["stratum_codes"][index])
            if code not in CODE_STRATA:
                raise ValueError("dataset array contains an unknown stratum code")
            rank = bytes(arrays["deterministic_ranks"][index]).decode("ascii")
            record = ProposalRecord(
                dataset=datasets[dataset_index],
                mother_id=int(arrays["mother_ids"][index]),
                candidate_1_id=int(arrays["candidate_ids"][index, 0]),
                candidate_2_id=int(arrays["candidate_ids"][index, 1]),
                mother_node_index=int(arrays["node_references"][index, 0]),
                candidate_1_node_index=int(arrays["node_references"][index, 1]),
                candidate_2_node_index=int(arrays["node_references"][index, 2]),
                sampling_stratum=CODE_STRATA[code],
                source_role="fit",
                label=int(arrays["labels"][index]),
                loss_weight=float(arrays["loss_weights"][index]),
                minimum_cellness=float(arrays["minimum_cellness"][index]),
                restored_node_count=int(arrays["restored_node_counts"][index]),
                edge_removal_count=int(arrays["edge_removal_counts"][index]),
                deterministic_rank=rank,
                raw_context=tuple(float(value) for value in arrays["raw_context"][index]),
                image_features=tuple(
                    float(value) for value in arrays["image_features"][index]
                ),
            )
            if payload != {
                name: str(value)
                for name, value in _record_csv_payload(index, record).items()
            }:
                raise ValueError("division dataset CSV sample tuple disagrees with NPZ arrays")
            records.append(record)
    except (KeyError, TypeError, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("division dataset"):
            raise
        raise ValueError(f"invalid division dataset CSV record: {exc}") from exc
    if _records_csv_bytes(records) != content:
        raise ValueError("division dataset CSV is not canonical")
    return tuple(records)


def _identity_from_manifest(manifest: Mapping[str, Any]) -> DatasetBuildIdentity:
    return DatasetBuildIdentity(
        build_mode=manifest["build_mode"],
        cache_files=manifest["cache_files"],
        cache_source_sha256=manifest["cache_source_sha256"],
        contract_sha256=manifest["contract_sha256"],
        dataset_source_sha256=manifest["dataset_source_sha256"],
        datasets=tuple(manifest["datasets"]),
        upstream_manifest_sha256=manifest["upstream_manifest_sha256"],
    )


def _read_dataset_artifacts_unlocked(
    path: str | Path,
    *,
    expected_identity: DatasetBuildIdentity,
    expected_dataset_source_sha256: str,
) -> LoadedDivisionDataset:
    """Strictly read arrays, CSV, schema, hashes, labels, roles, and sample tuples."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"division dataset directory does not exist: {directory}")
    observed_entries = tuple(item.name for item in directory.iterdir())
    if set(observed_entries) != set(DATASET_FILES) or any(
        not (directory / name).is_file() for name in DATASET_FILES
    ):
        raise ValueError("division dataset directory is incomplete or has unexpected files")
    manifest = read_canonical_json(directory / "manifest.json")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("division dataset manifest has incomplete or unexpected fields")
    if manifest["division_dataset_schema_version"] != DIVISION_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported division dataset schema")
    if manifest["build_mode"] not in DATASET_BUILD_MODES:
        raise ValueError("division dataset manifest has an unsupported build mode")
    if manifest["source_role"] != "fit":
        raise ValueError("division dataset manifest may contain only fit records")
    if manifest["proposal_rule"] != PROPOSAL_RULE_CONTRACT:
        raise ValueError("division dataset manifest changed the frozen proposal rule")
    if manifest["dataset_source_files"] != list(DATASET_SOURCE_FILES):
        raise ValueError("division dataset source file tuple is invalid")
    expected_source = _require_sha256(
        expected_dataset_source_sha256,
        field="expected dataset source SHA-256",
    )
    if manifest["dataset_source_sha256"] != expected_source:
        raise ValueError("division dataset source SHA-256 mismatch")
    observed_identity = _identity_from_manifest(manifest)
    if observed_identity != expected_identity:
        raise ValueError("division dataset upstream identity mismatch")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != set(DATASET_FILES[:-1]):
        raise ValueError("division dataset manifest has an incomplete file table")
    for name in DATASET_FILES[:-1]:
        record = files[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or record["path"] != name
            or not _is_sha256(record["sha256"])
            or file_sha256(directory / name) != record["sha256"]
        ):
            raise ValueError(f"division dataset file identity mismatch for {name}")
    schema_payload = read_canonical_json(directory / "feature_schema.json")
    schema = FeatureSchema.from_dict(schema_payload)
    if manifest["feature_schema"] != {
        "path": "feature_schema.json",
        "sha256": schema.sha256,
    }:
        raise ValueError("division dataset feature schema identity mismatch")
    funnel = read_canonical_json(directory / "funnel.json")
    if (
        set(funnel) != {"division_funnel_schema_version", "roles"}
        or funnel["division_funnel_schema_version"] != DIVISION_FUNNEL_SCHEMA_VERSION
        or not isinstance(funnel["roles"], Mapping)
        or set(funnel["roles"]) != {"fit"}
    ):
        raise ValueError("division dataset funnel artifact is invalid")
    record_count = _require_int(manifest["record_count"], field="record count", minimum=1)
    arrays = _read_npz(directory / "records.npz")
    _validate_array_layout(arrays, count=record_count)
    model_inputs_hash = canonical_model_inputs_sha256(
        arrays["raw_context"],
        arrays["image_features"],
    )
    if manifest["model_inputs_sha256"] != model_inputs_hash:
        raise ValueError("division dataset model inputs SHA-256 mismatch")
    records = _read_csv_records(
        directory / "records.csv",
        arrays=arrays,
        datasets=observed_identity.datasets,
    )
    _require_trainable_records(records)
    if tuple(sorted({record.dataset for record in records})) != observed_identity.datasets:
        raise ValueError("division dataset records do not cover the complete declared cohort")
    if fit_feature_schema(records) != schema:
        raise ValueError("division dataset feature schema was not fitted from supervised fit rows")
    if manifest["sample_tuple_fields"] != list(SAMPLE_TUPLE_FIELDS):
        raise ValueError("division dataset sample tuple fields are invalid")
    sample_hash = _sample_tuple_sha256(records)
    if manifest["sample_tuple_sha256"] != sample_hash:
        raise ValueError("division dataset sample tuple SHA-256 mismatch")
    counts = Counter(record.sampling_stratum for record in records)
    expected_counts = {stratum: counts[stratum] for stratum in SAMPLING_STRATA}
    if manifest["stratum_counts"] != expected_counts:
        raise ValueError("division dataset stratum counts are inconsistent")
    supervised_count = sum(counts[stratum] for stratum in SUPERVISED_STRATA)
    if (
        manifest["supervised_record_count"] != supervised_count
        or manifest["unknown_record_count"] != counts[DIAGNOSTIC_UNKNOWN]
        or record_count != supervised_count + counts[DIAGNOSTIC_UNKNOWN]
    ):
        raise ValueError("division dataset supervised or unknown counts are inconsistent")
    normalized_context = np.ascontiguousarray(
        schema.standardize(arrays["raw_context"]),
        dtype=np.float32,
    )
    image_features = np.ascontiguousarray(arrays["image_features"], dtype=np.float32)
    normalized_context.setflags(write=False)
    image_features.setflags(write=False)
    return LoadedDivisionDataset(
        records=records,
        feature_schema=schema,
        image_features=image_features,
        manifest=MappingProxyType(dict(manifest)),
        dataset_manifest_sha256=file_sha256(directory / "manifest.json"),
        model_inputs_sha256=model_inputs_hash,
        normalized_context=normalized_context,
    )


def read_dataset_artifacts(
    path: str | Path,
    *,
    expected_identity: DatasetBuildIdentity,
    expected_dataset_source_sha256: str,
) -> LoadedDivisionDataset:
    """Read one smoke or complete artifact under the shared recoverable reader lock."""
    directory = Path(path)
    with shared_dataset_reader_lock(directory):
        return _read_dataset_artifacts_unlocked(
            directory,
            expected_identity=expected_identity,
            expected_dataset_source_sha256=expected_dataset_source_sha256,
        )


def division_dataset_source_sha256(project_root: str | Path) -> str:
    """Hash the exact new Task 3 library and CLI source tuple."""
    root = Path(project_root)
    return source_bundle_sha256(root / path for path in DATASET_SOURCE_FILES)


def division_cache_source_sha256(project_root: str | Path) -> str:
    """Hash the immutable Task 2 node-cache implementation tuple."""
    root = Path(project_root)
    return source_bundle_sha256(root / path for path in CACHE_SOURCE_FILES)


def validate_node_cache_identity_cohort(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    expected_role: str,
) -> None:
    """Reject mixed backend, source, manifest, role, checkpoint, or model identities."""
    if expected_role not in {"fit", "development"} or not metadata_rows:
        raise ValueError("node-cache cohort requires a nonempty fit or development role")
    common_fields = (
        "backend",
        "backend_policy",
        "cache_source_sha256",
        "checkpoint_sha256",
        "contract_sha256",
        "data_split",
        "detector_method",
        "detector_threshold_k",
        "device",
        "embedding_dim",
        "inference_dtype",
        "upstream_manifest_sha256",
        "model_state_sha256",
        "normalization",
        "patch_shape",
        "role",
        "scoring_source_sha256",
        "torch_version",
    )
    first = metadata_rows[0]
    for row in metadata_rows:
        if row.get("role") != expected_role:
            raise ValueError("node-cache cohort mixes frozen movie roles")
        for field in common_fields:
            if row.get(field) != first.get(field):
                raise ValueError(f"node-cache cohort mixes {field} identities")


@dataclass(frozen=True, slots=True)
class _NodeCacheIdentity:
    path: Path
    metadata: Mapping[str, Any]
    file_sha256: str


@dataclass(frozen=True, slots=True)
class _CohortContext:
    caches: Mapping[str, _NodeCacheIdentity]
    contracts: Mapping[str, Mapping[str, Any]]
    contract_sha256: Mapping[str, str]
    rows: tuple[Any, ...]
    samples: Mapping[str, SamplePaths]
    cache_source_sha256: str
    upstream_manifest_sha256: str


def _selected_manifest_rows(
    rows: Sequence[Any],
    *,
    role: str,
    datasets: Sequence[str] | None,
) -> tuple[Any, ...]:
    if role not in {"fit", "development"}:
        raise ValueError("division dataset access permits only fit or development")
    by_dataset = {row.dataset: row for row in rows}
    if datasets is None:
        selected = tuple(row for row in rows if row.role == role)
    else:
        requested = tuple(sorted(datasets))
        if len(requested) != len(set(requested)):
            raise ValueError("requested dataset values must be unique")
        missing = sorted(set(requested) - set(by_dataset))
        if missing:
            raise ValueError(
                "requested datasets are outside the frozen manifest: " + ", ".join(missing)
            )
        selected = tuple(by_dataset[dataset] for dataset in requested)
        if any(row.role != role for row in selected):
            raise ValueError(f"requested dataset is not assigned to the frozen {role} role")
    if not selected:
        raise ValueError(f"frozen manifest contains no selected {role} datasets")
    return tuple(sorted(selected, key=lambda row: row.dataset))


def validate_complete_role_tuple(
    observed: Sequence[str],
    expected: Sequence[str],
    *,
    role: str,
    expected_count: int,
) -> tuple[str, ...]:
    """Require one exact sorted frozen production role tuple."""
    observed_tuple = tuple(observed)
    expected_tuple = tuple(expected)
    if (
        expected_tuple != tuple(sorted(set(expected_tuple)))
        or len(expected_tuple) != expected_count
    ):
        raise ValueError(f"frozen {role} role tuple is invalid")
    if observed_tuple != expected_tuple:
        raise ValueError(f"{role} role tuple does not match the exact frozen cohort")
    return observed_tuple


def _prepare_cohort(
    *,
    project_root: Path,
    data_root: Path,
    config_dir: Path,
    upstream_manifest_path: Path,
    summary_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    role: str,
    datasets: Sequence[str] | None,
) -> _CohortContext:
    contracts = verify_division_contract_bundle(
        config_dir,
        manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        project_root=project_root,
        checkpoint_path=checkpoint_path,
    )
    contract_hashes = current_division_contract_hashes(config_dir)
    if contract_hashes != EXPECTED_DIVISION_CONTRACT_SHA256:
        raise ValueError("current contract hashes are not the frozen Task 1 identities")
    manifest_rows = read_division_manifest(upstream_manifest_path)
    selected_rows = _selected_manifest_rows(
        manifest_rows,
        role=role,
        datasets=datasets,
    )
    frozen_role = tuple(contracts["split"]["role_datasets"][role])
    expected_count = EXPECTED_FIT_MOVIES if role == "fit" else EXPECTED_DEVELOPMENT_MOVIES
    if datasets is None:
        validate_complete_role_tuple(
            (row.dataset for row in selected_rows),
            frozen_role,
            role=role,
            expected_count=expected_count,
        )
    upstream_manifest_hash = file_sha256(upstream_manifest_path)
    cache_source = division_cache_source_sha256(project_root)
    cache_paths = {row.dataset: cache_dir / f"{row.dataset}.npz" for row in selected_rows}
    missing_caches = [str(path) for path in cache_paths.values() if not path.is_file()]
    if missing_caches:
        raise FileNotFoundError("missing Task 2 node caches: " + ", ".join(missing_caches))

    cache_identities: dict[str, _NodeCacheIdentity] = {}
    for row in selected_rows:
        metadata, cache_file_hash = read_division_node_cache_metadata(
            cache_paths[row.dataset],
            expected_dataset=row.dataset,
            expected_role=role,
            expected_cache_source_sha256=cache_source,
            expected_contract_sha256=contract_hashes,
            expected_upstream_manifest_sha256=upstream_manifest_hash,
        )
        cache_identities[row.dataset] = _NodeCacheIdentity(
            path=cache_paths[row.dataset],
            metadata=MappingProxyType(metadata),
            file_sha256=cache_file_hash,
        )
    validate_node_cache_identity_cohort(
        tuple(identity.metadata for identity in cache_identities.values()),
        expected_role=role,
    )

    discovered = discover_split(data_root, "train")
    samples = {
        sample.dataset: sample
        for sample in discovered
        if sample.embryo == "6bba" and sample.dataset in cache_paths
    }
    missing_samples = [row.dataset for row in selected_rows if row.dataset not in samples]
    if missing_samples:
        raise FileNotFoundError("selected frozen movies are missing: " + ", ".join(missing_samples))
    for row in selected_rows:
        sample = samples[row.dataset]
        if sample.labels is None:
            raise FileNotFoundError(f"selected movie is missing labels: {row.dataset}")
    return _CohortContext(
        caches=MappingProxyType(cache_identities),
        contracts=contracts,
        contract_sha256=contract_hashes,
        rows=selected_rows,
        samples=MappingProxyType(samples),
        cache_source_sha256=cache_source,
        upstream_manifest_sha256=upstream_manifest_hash,
    )


def _load_truth(sample: SamplePaths, *, expected_sha256: str) -> TrackingGraph:
    if sample.labels is None:
        raise FileNotFoundError(f"selected movie is missing labels: {sample.dataset}")
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    if logical_graph_sha256(truth) != expected_sha256:
        raise ValueError(f"truth graph identity mismatch for {sample.dataset}")
    return truth


def _aggregate_funnels(role: str, funnels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate_names = (
        "filtered_candidates",
        "linked_edges",
        "missing_daughters",
        "missing_mothers",
        "pruned_nodes",
        "raw_candidates",
        "recoverable_positives",
        "truth_events",
    )
    return {
        "aggregate": {
            **{
                name: sum(int(funnel[name]) for funnel in funnels)
                for name in aggregate_names
            },
            "proposal_role_counts": {
                stratum: sum(
                    int(funnel["proposal_role_counts"][stratum]) for funnel in funnels
                )
                for stratum in SAMPLING_STRATA
            },
            "selected_role_counts": {
                stratum: sum(
                    int(funnel["selected_role_counts"][stratum]) for funnel in funnels
                )
                for stratum in SAMPLING_STRATA
            },
        },
        "movies": list(funnels),
        "source_role": role,
    }


def _cache_identity(context: _CohortContext) -> dict[str, dict[str, Any]]:
    return {
        row.dataset: {
            "content_sha256": context.caches[row.dataset].metadata["content_sha256"],
            "file_sha256": context.caches[row.dataset].file_sha256,
            "image_sha256": context.caches[row.dataset].metadata["image_sha256"],
            "role": "fit",
        }
        for row in context.rows
    }


def _revalidate_input_cache_files(
    context: _CohortContext,
) -> None:
    """Detect cache drift by rehashing snapshots without loading compressed arrays again."""
    for row in context.rows:
        identity = context.caches[row.dataset]
        if file_sha256(identity.path) != identity.file_sha256:
            raise ValueError(f"input cache file changed after sampling: {row.dataset}")


def _sample_cached_movie(
    row: Any,
    *,
    context: _CohortContext,
    source_role: str,
) -> tuple[list[ProposalRecord], dict[str, Any]]:
    """Strict-load and process one cache inside a short-lived per-movie frame."""
    identity = context.caches[row.dataset]
    raw, scores, embeddings, intensity, metadata = read_division_node_cache(
        identity.path,
        expected_dataset=row.dataset,
        expected_role=source_role,
        expected_cache_source_sha256=context.cache_source_sha256,
        expected_contract_sha256=context.contract_sha256,
        expected_upstream_manifest_sha256=context.upstream_manifest_sha256,
        expected_image_sha256=identity.metadata["image_sha256"],
    )
    if file_sha256(identity.path) != identity.file_sha256:
        raise ValueError(f"node cache changed during strict sampling load: {row.dataset}")
    if dict(metadata) != dict(identity.metadata):
        raise ValueError(f"node cache metadata changed after strict preflight: {row.dataset}")
    truth = _load_truth(
        context.samples[row.dataset],
        expected_sha256=row.truth_graph_sha256,
    )
    filtered, linked, pruned = reconstruct_upstream(raw, scores)
    records, funnel = sample_movie_proposals(
        row.dataset,
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        filtered=filtered,
        linked=linked,
        pruned=pruned,
        truth=truth,
        proposal_contract=context.contracts["proposal"]["proposal_rule"],
        sampling_contract=context.contracts["sampling"],
        source_role=source_role,
        materialize_records=source_role == "fit",
    )
    return records, funnel


def _build_sampled_fit_dataset(
    output: str | Path,
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    build_mode: str,
    datasets: Sequence[str] | None,
    overwrite: bool = False,
) -> DatasetArtifactResult:
    if build_mode == BUILD_MODE_FIT_COMPLETE:
        if datasets is not None:
            raise ValueError("fit_complete construction does not accept a dataset subset")
    elif build_mode == BUILD_MODE_SMOKE:
        if datasets is None or len(tuple(datasets)) != 1:
            raise ValueError("smoke construction requires exactly one explicit fit dataset")
    else:
        raise ValueError("sampled fit construction has an unsupported build mode")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_dataset_builder_lock(destination):
        _recover_dataset_transaction(destination)
        return _build_sampled_fit_dataset_locked(
            destination,
            project_root=Path(project_root),
            data_root=Path(data_root),
            config_dir=Path(config_dir),
            upstream_manifest_path=Path(upstream_manifest_path),
            summary_path=Path(summary_path),
            cache_dir=Path(cache_dir),
            checkpoint_path=Path(checkpoint_path),
            build_mode=build_mode,
            datasets=datasets,
            overwrite=overwrite,
        )


def _build_sampled_fit_dataset_locked(
    destination: Path,
    *,
    project_root: Path,
    data_root: Path,
    config_dir: Path,
    upstream_manifest_path: Path,
    summary_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    build_mode: str,
    datasets: Sequence[str] | None,
    overwrite: bool,
) -> DatasetArtifactResult:
    """Construct one sampled artifact while its recovered output lock is held."""
    root = project_root
    cache_root = cache_dir
    configs = config_dir
    upstream_manifest = upstream_manifest_path
    summary = summary_path
    checkpoint = checkpoint_path
    if destination.exists() and not overwrite:
        loaded = _read_bound_division_dataset_unlocked(
            destination,
            project_root=root,
            config_dir=configs,
            upstream_manifest_path=upstream_manifest,
            summary_path=summary,
            checkpoint_path=checkpoint,
            expected_build_mode=build_mode,
        )
        if datasets is not None and tuple(loaded.manifest["datasets"]) != tuple(datasets):
            raise FileExistsError("existing smoke artifact belongs to a different fit movie")
        return _artifact_result(destination, loaded, skipped=True)
    dataset_source = division_dataset_source_sha256(root)
    context = _prepare_cohort(
        project_root=root,
        data_root=data_root,
        config_dir=configs,
        upstream_manifest_path=upstream_manifest,
        summary_path=summary,
        cache_dir=cache_root,
        checkpoint_path=checkpoint,
        role="fit",
        datasets=datasets,
    )
    identity = DatasetBuildIdentity(
        build_mode=build_mode,
        cache_files=_cache_identity(context),
        cache_source_sha256=context.cache_source_sha256,
        contract_sha256=context.contract_sha256,
        dataset_source_sha256=dataset_source,
        datasets=tuple(row.dataset for row in context.rows),
        upstream_manifest_sha256=context.upstream_manifest_sha256,
    )
    records: list[ProposalRecord] = []
    funnels: list[dict[str, Any]] = []
    for row in context.rows:
        movie_records, funnel = _sample_cached_movie(
            row,
            context=context,
            source_role="fit",
        )
        records.extend(movie_records)
        funnels.append(funnel)

    def verify_inputs_before_promotion() -> None:
        verify_division_contract_bundle(
            configs,
            manifest_path=upstream_manifest,
            summary_path=summary,
            project_root=root,
            checkpoint_path=checkpoint,
        )
        if (
            current_division_contract_hashes(configs) != context.contract_sha256
            or file_sha256(upstream_manifest) != context.upstream_manifest_sha256
            or division_cache_source_sha256(root) != context.cache_source_sha256
            or division_dataset_source_sha256(root) != dataset_source
        ):
            raise RuntimeError("division dataset provenance changed during construction")
        _revalidate_input_cache_files(context)

    return _build_dataset_artifacts_locked(
        destination,
        records,
        identity=identity,
        funnels={"fit": _aggregate_funnels("fit", funnels)},
        overwrite=overwrite,
        before_promotion=verify_inputs_before_promotion,
    )


def build_fit_dataset(
    output: str | Path,
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    overwrite: bool = False,
) -> DatasetArtifactResult:
    """Build only the exact frozen 88-movie canonical fit dataset."""
    return _build_sampled_fit_dataset(
        output,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        build_mode=BUILD_MODE_FIT_COMPLETE,
        datasets=None,
        overwrite=overwrite,
    )


def build_smoke_dataset(
    output: str | Path,
    *,
    dataset: str,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    overwrite: bool = False,
) -> DatasetArtifactResult:
    """Build one explicitly named fit movie as a non-production smoke artifact."""
    return _build_sampled_fit_dataset(
        output,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        build_mode=BUILD_MODE_SMOKE,
        datasets=(dataset,),
        overwrite=overwrite,
    )


def development_proposal_funnel(
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Inspect development labels in memory without records, statistics, or file writes."""
    root = Path(project_root)
    cache_root = Path(cache_dir)
    configs = Path(config_dir)
    upstream_manifest = Path(upstream_manifest_path)
    summary = Path(summary_path)
    checkpoint = Path(checkpoint_path)
    dataset_source = division_dataset_source_sha256(root)
    context = _prepare_cohort(
        project_root=root,
        data_root=Path(data_root),
        config_dir=configs,
        upstream_manifest_path=upstream_manifest,
        summary_path=summary,
        cache_dir=cache_root,
        checkpoint_path=checkpoint,
        role="development",
        datasets=None,
    )
    funnels: list[dict[str, Any]] = []
    for row in context.rows:
        records, funnel = _sample_cached_movie(
            row,
            context=context,
            source_role="development",
        )
        if records:
            raise RuntimeError("development funnel unexpectedly materialized training records")
        funnels.append(funnel)
    verify_division_contract_bundle(
        configs,
        manifest_path=upstream_manifest,
        summary_path=summary,
        project_root=root,
        checkpoint_path=checkpoint,
    )
    if (
        current_division_contract_hashes(configs) != context.contract_sha256
        or file_sha256(upstream_manifest) != context.upstream_manifest_sha256
        or division_cache_source_sha256(root) != context.cache_source_sha256
        or division_dataset_source_sha256(root) != dataset_source
    ):
        raise RuntimeError("development funnel provenance changed during inspection")
    _revalidate_input_cache_files(context)
    return {
        "build_mode": BUILD_MODE_DEVELOPMENT_FUNNEL,
        "contract_sha256": dict(context.contract_sha256),
        "dataset_source_sha256": dataset_source,
        "datasets": [row.dataset for row in context.rows],
        "division_funnel_schema_version": DIVISION_FUNNEL_SCHEMA_VERSION,
        "funnel": _aggregate_funnels("development", funnels),
        "upstream_manifest_sha256": context.upstream_manifest_sha256,
        "proposal_rule": PROPOSAL_RULE_CONTRACT,
        "read_only": True,
        "source_role": "development",
    }


def _read_bound_division_dataset_unlocked(
    directory: Path,
    *,
    project_root: Path,
    config_dir: Path,
    upstream_manifest_path: Path,
    summary_path: Path,
    checkpoint_path: Path,
    expected_build_mode: str,
) -> LoadedDivisionDataset:
    preliminary = read_canonical_json(directory / "manifest.json")
    if preliminary.get("build_mode") != expected_build_mode:
        if expected_build_mode == BUILD_MODE_FIT_COMPLETE:
            raise ValueError("production training reader rejects smoke or subset artifacts")
        raise ValueError("smoke reader requires an explicit smoke artifact")
    datasets = preliminary.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("division dataset manifest does not declare a dataset tuple")
    contracts = verify_division_contract_bundle(
        config_dir,
        manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        project_root=project_root,
        checkpoint_path=checkpoint_path,
    )
    contract_hashes = current_division_contract_hashes(config_dir)
    manifest_rows = read_division_manifest(upstream_manifest_path)
    selected = None if expected_build_mode == BUILD_MODE_FIT_COMPLETE else datasets
    selected_rows = _selected_manifest_rows(
        manifest_rows,
        role="fit",
        datasets=selected,
    )
    expected_datasets = tuple(row.dataset for row in selected_rows)
    frozen_fit = tuple(contracts["split"]["role_datasets"]["fit"])
    validate_complete_role_tuple(
        frozen_fit,
        frozen_fit,
        role="fit",
        expected_count=EXPECTED_FIT_MOVIES,
    )
    if expected_build_mode == BUILD_MODE_FIT_COMPLETE:
        validate_complete_role_tuple(
            expected_datasets,
            frozen_fit,
            role="fit",
            expected_count=EXPECTED_FIT_MOVIES,
        )
        validate_complete_role_tuple(
            datasets,
            frozen_fit,
            role="fit",
            expected_count=EXPECTED_FIT_MOVIES,
        )
    elif tuple(datasets) != expected_datasets:
        raise ValueError("smoke dataset tuple does not match its frozen fit movie")
    identity = DatasetBuildIdentity(
        build_mode=expected_build_mode,
        cache_files=preliminary.get("cache_files", {}),
        cache_source_sha256=division_cache_source_sha256(project_root),
        contract_sha256=contract_hashes,
        dataset_source_sha256=division_dataset_source_sha256(project_root),
        datasets=expected_datasets,
        upstream_manifest_sha256=file_sha256(upstream_manifest_path),
    )
    return _read_dataset_artifacts_unlocked(
        directory,
        expected_identity=identity,
        expected_dataset_source_sha256=identity.dataset_source_sha256,
    )


def _read_bound_division_dataset(
    path: str | Path,
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    expected_build_mode: str,
) -> LoadedDivisionDataset:
    directory = Path(path)
    with shared_dataset_reader_lock(directory):
        return _read_bound_division_dataset_unlocked(
            directory,
            project_root=Path(project_root),
            config_dir=Path(config_dir),
            upstream_manifest_path=Path(upstream_manifest_path),
            summary_path=Path(summary_path),
            checkpoint_path=Path(checkpoint_path),
            expected_build_mode=expected_build_mode,
        )


def read_division_dataset(
    path: str | Path,
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
) -> LoadedDivisionDataset:
    """Read only the exact 88-movie canonical fit artifact for production training."""
    return _read_bound_division_dataset(
        path,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        expected_build_mode=BUILD_MODE_FIT_COMPLETE,
    )


def read_smoke_division_dataset(
    path: str | Path,
    *,
    project_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
    upstream_manifest_path: str | Path,
    summary_path: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
) -> LoadedDivisionDataset:
    """Strictly validate one explicitly non-production smoke artifact."""
    return _read_bound_division_dataset(
        path,
        project_root=project_root,
        data_root=data_root,
        config_dir=config_dir,
        upstream_manifest_path=upstream_manifest_path,
        summary_path=summary_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        expected_build_mode=BUILD_MODE_SMOKE,
    )
