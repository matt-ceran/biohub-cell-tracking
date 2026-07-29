"""Frozen truth, split, and pre-model contracts for Phase 11 divisions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from biohub.io import SamplePaths, embryo_id, open_image_array, read_geff_graph
from biohub.metric import TrackingGraph

GRAPH_HASH_SCHEMA = "biohub-logical-tracking-graph-v1"
MANIFEST_CSV_SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
SPLIT_RANK_PREFIX = b"biohub-phase11-division-split-v1\0"
SPLIT_RANK_SALT = "biohub-phase11-division-split-v1"
SPLIT_QUOTAS: dict[int, tuple[int, int, int]] = {
    0: (42, 10, 10),
    1: (20, 5, 5),
    2: (14, 3, 3),
    3: (8, 1, 1),
    4: (3, 1, 1),
    5: (1, 0, 0),
}
EXPECTED_DIVISION_HISTOGRAM = {
    division_count: sum(quotas) for division_count, quotas in SPLIT_QUOTAS.items()
}
ROLE_NAMES = ("fit", "development", "evaluation")
EXPECTED_ROLE_MOVIES = {"fit": 88, "development": 20, "evaluation": 20}
EXPECTED_ROLE_DIVISIONS = {"fit": 89, "development": 18, "evaluation": 18}
EXPECTED_POLICY_SHA256 = "1a333aa0cf3c1c5b92cdfc468fdbc483567a8bde2f4d2e73bd2332069144f3fc"
EXPECTED_CHECKPOINT_SHA256 = "bd35d082c985641728bc340817d08d4750a3c83c34d25f99d1507cbc8b7bb8d4"
EXPECTED_SCORING_SOURCE_SHA256 = (
    "426c09f04c99af55fd5457921fbf435de2f71bdf7ce08dbdf0e0fde6c02443c9"
)
EXPECTED_EVALUATION_SOURCE_SHA256 = (
    "be227a8d94434ba401209ba0ada2864e5ad4a49b8ace941b86c5c5f109d571f9"
)
EXPECTED_PAIR_AUDIT_SOURCE_SHA256 = (
    "01533bf5362e05aa2cf7147cfb825e82dae79ef992d4196373710219288372a0"
)
EXPECTED_PAIR_AUDIT_SUMMARY_SHA256 = (
    "af12eef7d4621daf0ce877beccd77497ccf664f1974a49b1725182f8c2b3b010"
)
EXPECTED_EXPOSED_DATASETS = (
    "6bba_474be664",
    "6bba_4f99ce20",
    "6bba_6ca87370",
    "6bba_789f8168",
    "6bba_7af54fde",
    "6bba_825bd1c6",
    "6bba_907271db",
    "6bba_afb141ff",
    "6bba_d1acb6ff",
    "6bba_ebdf3b34",
    "6bba_ef7b4f7e",
    "6bba_f4ae811c",
    "6bba_fc5f39dc",
)
PHASE10_CHECKPOINT_PATH = (
    "data/working/appearance_checkpoints/cellness_cnn_phase10_full.pt"
)
PHASE10_POLICY_PATH = "data/working/phase10_frozen_candidate_policy.json"
PAIR_AUDIT_SUMMARY_PATH = (
    "data/working/phase11_division_audit/full13_pairs/summary.json"
)
SCORING_SOURCE_PATHS = (
    "src/biohub/appearance.py",
    "src/biohub/appearance_model.py",
    "src/biohub/detect.py",
    "src/biohub/io.py",
    "src/biohub/constants.py",
)
BASELINE_EVALUATION_SOURCE_PATHS = (
    *SCORING_SOURCE_PATHS,
    "src/biohub/appearance_evaluation.py",
    "src/biohub/link.py",
    "src/biohub/metric.py",
    "scripts/freeze_appearance_policy.py",
    "scripts/validate_appearance.py",
)
PAIR_AUDIT_SOURCE_PATHS = (
    "src/biohub/division.py",
    "scripts/audit_division_proposals.py",
    "scripts/audit_division_pair_proposals.py",
)
CONTRACT_SOURCE_PATHS = {
    "division_data_sha256": "src/biohub/division_data.py",
    "freeze_division_contracts_sha256": "scripts/freeze_division_contracts.py",
    "io_sha256": "src/biohub/io.py",
    "metric_sha256": "src/biohub/metric.py",
}

MANIFEST_FIELDS = (
    "dataset",
    "embryo",
    "division_count",
    "no_division",
    "proposal_exposed",
    "role",
    "truth_graph_sha256",
)
SOURCE_FINGERPRINT_FIELDS = {
    "division_data_sha256",
    "freeze_division_contracts_sha256",
    "io_sha256",
    "metric_sha256",
}


@dataclass(frozen=True)
class DivisionManifestRow:
    """One path-free logical truth identity and its frozen Phase 11 role."""

    dataset: str
    embryo: str
    division_count: int
    no_division: bool
    truth_graph_sha256: str
    proposal_exposed: bool = False
    role: str | None = None


@dataclass(frozen=True)
class _CanonicalGraph:
    node_ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _as_int64(values: Any, *, field: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if shape is not None and array.shape != shape:
        raise ValueError(f"logical graph {field} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"logical graph {field} must contain integral values")
    int64 = np.iinfo(np.int64)
    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.unsignedinteger) and np.any(array > int64.max):
            raise ValueError(f"logical graph {field} exceed signed 64-bit range")
        return np.ascontiguousarray(array, dtype="<i8")
    if np.issubdtype(array.dtype, np.floating):
        if (
            not np.all(np.isfinite(array))
            or not np.all(array == np.floor(array))
            or np.any(array < int64.min)
            or np.any(array >= 2**63)
        ):
            raise ValueError(f"logical graph {field} must contain integral values")
        return np.ascontiguousarray(array, dtype="<i8")

    converted: list[int] = []
    for value in array.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, Real)):
            raise ValueError(f"logical graph {field} must contain integral values")
        numeric = float(value)
        integer = int(value)
        if (
            not np.isfinite(numeric)
            or numeric != integer
            or not int64.min <= integer <= int64.max
        ):
            raise ValueError(f"logical graph {field} must contain integral values")
        converted.append(integer)
    return np.asarray(converted, dtype="<i8").reshape(array.shape)


def _as_coordinate(values: Any, *, axis: str, count: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"logical graph must contain finite {axis} coordinates") from exc
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"logical graph must contain finite {axis} coordinates with shape {(count,)}"
        )
    array = np.array(array, dtype="<f8", order="C", copy=True)
    array[array == 0.0] = 0.0  # Give positive and negative zero one logical identity.
    return array


def _canonical_graph(
    graph: TrackingGraph,
    *,
    require_unique_edges: bool,
) -> _CanonicalGraph:
    raw_node_ids = np.asarray(graph.node_ids)
    if raw_node_ids.ndim != 1:
        raise ValueError("logical graph node identifiers must be one-dimensional")
    node_ids = _as_int64(raw_node_ids, field="node identifiers")
    count = len(node_ids)
    if len(np.unique(node_ids)) != count:
        raise ValueError("logical graph contains a duplicate node identifier")
    t = _as_int64(graph.t, field="time values", shape=(count,))
    z = _as_coordinate(graph.z, axis="z", count=count)
    y = _as_coordinate(graph.y, axis="y", count=count)
    x = _as_coordinate(graph.x, axis="x", count=count)

    raw_edges = np.asarray(graph.edges)
    if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
        raise ValueError(
            f"logical graph edges must have shape (M, 2), got {raw_edges.shape}"
        )
    edges = _as_int64(raw_edges, field="edge identifiers")
    known_ids = {int(node_id) for node_id in node_ids}
    if any(
        int(source) not in known_ids or int(target) not in known_ids
        for source, target in edges
    ):
        raise ValueError("logical graph edge has a missing endpoint")
    if any(int(source) == int(target) for source, target in edges):
        raise ValueError("logical graph contains a self-loop edge")

    time_by_id = {
        int(node_id): int(time) for node_id, time in zip(node_ids, t, strict=True)
    }
    forward_edges: list[tuple[int, int]] = []
    for raw_source, raw_target in edges:
        source = int(raw_source)
        target = int(raw_target)
        if time_by_id[target] <= time_by_id[source]:
            raise ValueError(
                "logical graph edges must be strictly forward in time "
                f"({source}@{time_by_id[source]} -> {target}@{time_by_id[target]})"
            )
        forward_edges.append((source, target))
    unique_edges = set(forward_edges)
    if require_unique_edges and len(unique_edges) != len(forward_edges):
        raise ValueError("logical graph contains a duplicate edge")
    canonical_edges = np.asarray(sorted(unique_edges), dtype="<i8").reshape(-1, 2)

    order = np.lexsort((x, y, z, t, node_ids))
    return _CanonicalGraph(
        node_ids=np.ascontiguousarray(node_ids[order], dtype="<i8"),
        t=np.ascontiguousarray(t[order], dtype="<i8"),
        z=np.ascontiguousarray(z[order], dtype="<f8"),
        y=np.ascontiguousarray(y[order], dtype="<f8"),
        x=np.ascontiguousarray(x[order], dtype="<f8"),
        edges=np.ascontiguousarray(canonical_edges, dtype="<i8"),
    )


def _update_tagged_array(digest: Any, field: str, values: np.ndarray) -> None:
    shape = ",".join(str(value) for value in values.shape)
    digest.update(b"field\0")
    digest.update(field.encode("ascii"))
    digest.update(b"\0dtype\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0shape\0")
    digest.update(shape.encode("ascii"))
    digest.update(b"\0data\0")
    digest.update(values.tobytes(order="C"))
    digest.update(b"\0")


def logical_graph_sha256(
    graph: TrackingGraph,
    *,
    require_unique_edges: bool = True,
) -> str:
    """Hash a canonical, path-free logical ``TrackingGraph`` representation."""
    canonical = _canonical_graph(graph, require_unique_edges=require_unique_edges)
    digest = hashlib.sha256()
    digest.update(GRAPH_HASH_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for field in ("node_ids", "t", "z", "y", "x", "edges"):
        tag = "node_id" if field == "node_ids" else field
        _update_tagged_array(digest, tag, getattr(canonical, field))
    return digest.hexdigest()


def truth_divisions(graph: TrackingGraph) -> dict[int, tuple[int, int]]:
    """Return strict truth mother-to-daughter pairs in stable identifier order."""
    canonical = _canonical_graph(graph, require_unique_edges=True)
    time_by_id = {
        int(node_id): int(time)
        for node_id, time in zip(canonical.node_ids, canonical.t, strict=True)
    }
    successors: dict[int, list[int]] = {}
    for source, target in canonical.edges:
        successors.setdefault(int(source), []).append(int(target))

    divisions: dict[int, tuple[int, int]] = {}
    for mother, children in sorted(successors.items()):
        unique_children = tuple(sorted(children))
        if len(unique_children) < 2:
            continue
        if len(unique_children) != 2 or any(
            time_by_id[daughter] != time_by_id[mother] + 1 for daughter in unique_children
        ):
            raise ValueError(
                f"truth division mother {mother} must have exactly two daughters at t+1"
            )
        divisions[mother] = unique_children
    return divisions


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically with sorted keys and one trailing newline."""
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not valid canonical JSON: {exc}") from exc
    return (text + "\n").encode("ascii")


def canonical_json_sha256(payload: Any) -> str:
    """Return the SHA-256 of the exact bytes written by ``write_canonical_json``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one file without incorporating its local path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(paths: Iterable[str | Path]) -> str:
    """Hash an ordered source bundle using file names rather than absolute paths."""
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_bytes(path: str | Path, content: bytes) -> None:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_canonical_json(path: str | Path, payload: Any) -> str:
    """Atomically write canonical JSON and return its exact file SHA-256."""
    content = canonical_json_bytes(payload)
    _atomic_write_bytes(path, content)
    return hashlib.sha256(content).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite value {value}")


def read_canonical_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one canonical JSON object, rejecting duplicate keys and byte drift."""
    source = Path(path)
    content = source.read_bytes()
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        observed = hashlib.sha256(content).hexdigest()
        if observed != expected_sha256:
            raise ValueError(
                f"canonical JSON {source} has SHA-256 {observed}, expected {expected_sha256}"
            )
    try:
        payload = json.loads(
            content.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid canonical JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"canonical JSON {source} must contain one object")
    if canonical_json_bytes(payload) != content:
        raise ValueError(f"canonical JSON {source} is not canonical")
    return payload


def _validate_manifest_row(row: DivisionManifestRow, *, require_role: bool = False) -> None:
    if not isinstance(row.dataset, str) or not row.dataset.startswith("6bba_"):
        raise ValueError("division manifest accepts only 6bba datasets")
    if row.embryo != "6bba" or embryo_id(row.dataset) != row.embryo:
        raise ValueError("division manifest dataset and embryo identifiers disagree")
    if isinstance(row.division_count, bool) or not isinstance(row.division_count, int):
        raise ValueError("division manifest counts must be integers")
    if row.division_count < 0:
        raise ValueError("division manifest counts must be non-negative")
    if not isinstance(row.no_division, bool) or row.no_division != (row.division_count == 0):
        raise ValueError("division manifest no-division flag is inconsistent")
    if not _is_sha256(row.truth_graph_sha256):
        raise ValueError("division manifest requires a logical truth-graph SHA-256")
    if not isinstance(row.proposal_exposed, bool):
        raise ValueError("division manifest proposal-exposed flag must be boolean")
    if row.role is not None and row.role not in ROLE_NAMES:
        raise ValueError("division manifest has an unsupported role")
    if require_role and row.role not in ROLE_NAMES:
        raise ValueError("division manifest row has not been assigned a frozen role")


def _materialize_manifest_rows(
    rows: Iterable[DivisionManifestRow],
    *,
    require_role: bool = False,
) -> tuple[DivisionManifestRow, ...]:
    materialized = tuple(rows)
    for row in materialized:
        if not isinstance(row, DivisionManifestRow):
            raise TypeError("manifest rows must be DivisionManifestRow instances")
        _validate_manifest_row(row, require_role=require_role)
    datasets = [row.dataset for row in materialized]
    if len(set(datasets)) != len(datasets):
        raise ValueError("division manifest contains a duplicate dataset")
    return tuple(sorted(materialized, key=lambda row: row.dataset))


def validate_6bba_image_inputs(samples: Iterable[SamplePaths]) -> None:
    """Require every sample to expose one readable four-dimensional Zarr array."""
    for sample in samples:
        image_path = Path(sample.image)
        if not image_path.is_dir() or not os.access(image_path, os.R_OK):
            raise FileNotFoundError(
                f"missing or unreadable 6bba image for {sample.dataset}: {image_path}"
            )
        try:
            image = open_image_array(image_path)
            shape = tuple(int(value) for value in image.shape)
            np.dtype(image.dtype)
        except Exception as exc:
            raise ValueError(
                f"6bba image for {sample.dataset} is not a readable Zarr array: {image_path}"
            ) from exc
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise ValueError(
                f"6bba image for {sample.dataset} must be a non-empty tzyx Zarr array"
            )


def division_count_manifest(samples: Iterable[SamplePaths]) -> tuple[DivisionManifestRow, ...]:
    """Load a preflighted 6bba cohort and construct path-free truth identities."""
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("division manifest requires at least one 6bba sample")

    # Complete cohort preflight deliberately happens before the first label is opened.
    for sample in materialized:
        if not isinstance(sample, SamplePaths):
            raise TypeError("division manifest samples must be SamplePaths instances")
        if not sample.dataset.startswith("6bba_") or sample.embryo != "6bba":
            raise ValueError("division manifest accepts only 6bba datasets")
        if embryo_id(sample.dataset) != sample.embryo:
            raise ValueError("division manifest dataset and embryo identifiers disagree")
    datasets = [sample.dataset for sample in materialized]
    if len(set(datasets)) != len(datasets):
        raise ValueError("division manifest contains a duplicate dataset")
    validate_6bba_image_inputs(materialized)
    for sample in materialized:
        if sample.labels is None or not Path(sample.labels).is_dir():
            raise FileNotFoundError(f"missing 6bba labels for {sample.dataset}: {sample.labels}")

    rows: list[DivisionManifestRow] = []
    for sample in sorted(materialized, key=lambda item: item.dataset):
        graph = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        divisions = truth_divisions(graph)
        rows.append(
            DivisionManifestRow(
                dataset=sample.dataset,
                embryo=sample.embryo,
                division_count=len(divisions),
                no_division=not divisions,
                truth_graph_sha256=logical_graph_sha256(graph),
            )
        )
    return _materialize_manifest_rows(rows)


def assign_division_roles(
    rows: Iterable[DivisionManifestRow],
    exposed_datasets: Iterable[str],
) -> tuple[DivisionManifestRow, ...]:
    """Assign the exact deterministic 88/20/20 Phase 11 movie split."""
    materialized = _materialize_manifest_rows(rows)
    histogram = Counter(row.division_count for row in materialized)
    if dict(sorted(histogram.items())) != EXPECTED_DIVISION_HISTOGRAM:
        raise ValueError(
            "division split requires the exact division-count histogram "
            f"{EXPECTED_DIVISION_HISTOGRAM}, got {dict(sorted(histogram.items()))}"
        )

    exposed = tuple(sorted(exposed_datasets))
    if len(exposed) != 13 or len(set(exposed)) != 13:
        raise ValueError("division split requires exactly 13 unique proposal-exposed movies")
    if not all(isinstance(dataset, str) and dataset.startswith("6bba_") for dataset in exposed):
        raise ValueError("division split proposal-exposed movies must be only 6bba datasets")
    row_datasets = {row.dataset for row in materialized}
    missing = sorted(set(exposed) - row_datasets)
    if missing:
        raise ValueError(f"proposal-exposed movies are missing from the manifest: {missing}")
    exposed_set = set(exposed)

    role_by_dataset: dict[str, str] = {}
    for division_count, (fit_quota, development_quota, evaluation_quota) in SPLIT_QUOTAS.items():
        stratum = [row.dataset for row in materialized if row.division_count == division_count]
        forced = sorted(dataset for dataset in stratum if dataset in exposed_set)
        if len(forced) > fit_quota:
            raise ValueError(
                f"division-count {division_count} has more forced-fit movies than its fit quota"
            )
        remaining = sorted(
            (dataset for dataset in stratum if dataset not in exposed_set),
            key=lambda dataset: (
                hashlib.sha256(SPLIT_RANK_PREFIX + dataset.encode("utf-8")).digest(),
                dataset,
            ),
        )
        evaluation = remaining[:evaluation_quota]
        development = remaining[
            evaluation_quota : evaluation_quota + development_quota
        ]
        fit = [*forced, *remaining[evaluation_quota + development_quota :]]
        if len(fit) != fit_quota:
            raise ValueError(f"division-count {division_count} cannot satisfy the frozen quotas")
        role_by_dataset.update({dataset: "evaluation" for dataset in evaluation})
        role_by_dataset.update({dataset: "development" for dataset in development})
        role_by_dataset.update({dataset: "fit" for dataset in fit})

    assigned = tuple(
        replace(
            row,
            proposal_exposed=row.dataset in exposed_set,
            role=role_by_dataset[row.dataset],
        )
        for row in materialized
    )
    summary = division_role_summary(assigned)
    for role in ROLE_NAMES:
        if summary[role]["movies"] != EXPECTED_ROLE_MOVIES[role]:
            raise ValueError(f"division split produced the wrong {role} movie total")
        if summary[role]["labelled_divisions"] != EXPECTED_ROLE_DIVISIONS[role]:
            raise ValueError(f"division split produced the wrong {role} event total")
    for role in ("development", "evaluation"):
        if summary[role]["division_movies"] != 10 or summary[role]["no_division_movies"] != 10:
            raise ValueError(f"division split requires ten division and ten control {role} movies")
    return assigned


def division_role_summary(
    rows: Iterable[DivisionManifestRow],
) -> dict[str, dict[str, int]]:
    """Summarize assigned movie, event, division-movie, and control totals."""
    materialized = _materialize_manifest_rows(rows, require_role=True)
    return {
        role: {
            "movies": sum(row.role == role for row in materialized),
            "labelled_divisions": sum(
                row.division_count for row in materialized if row.role == role
            ),
            "division_movies": sum(
                row.role == role and not row.no_division for row in materialized
            ),
            "no_division_movies": sum(
                row.role == role and row.no_division for row in materialized
            ),
        }
        for role in ROLE_NAMES
    }


def canonical_manifest_bytes(rows: Iterable[DivisionManifestRow]) -> bytes:
    """Encode the exact sorted Phase 11 manifest CSV."""
    materialized = _materialize_manifest_rows(rows, require_role=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in materialized:
        payload = asdict(row)
        payload["no_division"] = str(row.no_division).lower()
        payload["proposal_exposed"] = str(row.proposal_exposed).lower()
        writer.writerow(payload)
    return stream.getvalue().encode("ascii")


def write_division_manifest(
    path: str | Path,
    rows: Iterable[DivisionManifestRow],
) -> str:
    """Atomically write the canonical manifest and return its SHA-256."""
    content = canonical_manifest_bytes(rows)
    _atomic_write_bytes(path, content)
    return hashlib.sha256(content).hexdigest()


def _parse_manifest_bool(value: str, *, field: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"manifest {field} must be true or false")
    return value == "true"


def read_division_manifest(path: str | Path) -> tuple[DivisionManifestRow, ...]:
    """Read and byte-validate a canonical assigned manifest CSV."""
    source = Path(path)
    content = source.read_bytes()
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("division manifest must be ASCII") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
        raise ValueError("division manifest has an unsupported CSV schema")
    rows: list[DivisionManifestRow] = []
    try:
        for payload in reader:
            rows.append(
                DivisionManifestRow(
                    dataset=payload["dataset"],
                    embryo=payload["embryo"],
                    division_count=int(payload["division_count"]),
                    no_division=_parse_manifest_bool(
                        payload["no_division"], field="no_division"
                    ),
                    proposal_exposed=_parse_manifest_bool(
                        payload["proposal_exposed"], field="proposal_exposed"
                    ),
                    role=payload["role"],
                    truth_graph_sha256=payload["truth_graph_sha256"],
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid division manifest row: {exc}") from exc
    materialized = _materialize_manifest_rows(rows, require_role=True)
    if canonical_manifest_bytes(materialized) != content:
        raise ValueError("division manifest is not canonical")
    return materialized


LOGICAL_GRAPH_HASH_CONTRACT = {
    "edge_dtype": "<i8",
    "edge_orientation": "source_time_strictly_less_than_target_time",
    "edge_order": ["source_id", "target_id"],
    "edge_shape": "(M,2)",
    "field_tags": ["node_id", "t", "z", "y", "x", "edges"],
    "node_dtypes": {"node_id": "<i8", "t": "<i8", "x": "<f8", "y": "<f8", "z": "<f8"},
    "node_row_order": ["node_id", "t", "z", "y", "x"],
    "node_shape": "(N,)",
    "paths_included": False,
    "require_unique_edges": True,
    "schema": GRAPH_HASH_SCHEMA,
    "truth_division_daughter_dt": 1,
}
PROPOSAL_RULE_CONTRACT = {
    "candidate_order": ["distance_um:ascending", "node_id:ascending"],
    "candidate_source": "raw_scored",
    "max_candidates_per_mother": 12,
    "max_distance_um": 16.0,
    "min_angle_deg": 0.0,
    "min_distance_um": 0.0,
    "mother_source": "cnn_filtered_preprune",
    "rule_id": "dmax16_cap12_dmin0_angle0",
}
PHASE10_INPUT_CONTRACT = {
    "checkpoint_path": PHASE10_CHECKPOINT_PATH,
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "evaluation_source_files": list(BASELINE_EVALUATION_SOURCE_PATHS),
    "evaluation_source_sha256": EXPECTED_EVALUATION_SOURCE_SHA256,
    "frozen_policy_path": PHASE10_POLICY_PATH,
    "frozen_policy_sha256": EXPECTED_POLICY_SHA256,
    "scoring_source_files": list(SCORING_SOURCE_PATHS),
    "scoring_source_sha256": EXPECTED_SCORING_SOURCE_SHA256,
}
PAIR_AUDIT_OUTPUTS = {
    "events_csv": {
        "path": "data/working/phase11_division_audit/full13_pairs/division_events.csv",
        "sha256": "deda36a3b75a08bf58c4862720765fd4622f10bef20362f5615a7baf72c19b88",
    },
    "proposals_csv": {
        "path": "data/working/phase11_division_audit/full13_pairs/selected_pair_proposals.csv",
        "sha256": "7617b3cc902e81b609d810c46f895d6d75633eb368ac07f0b1d3f279ad63a660",
    },
    "rule_movies_csv": {
        "path": "data/working/phase11_division_audit/full13_pairs/rule_movies.csv",
        "sha256": "bcf2dac0c3b0c13be6fd4172380b6252723a763eeae05383d1163a90485964d4",
    },
    "rule_sweep_csv": {
        "path": "data/working/phase11_division_audit/full13_pairs/rule_sweep.csv",
        "sha256": "04efc864348c4c9da4e8bc8e51a9396d4227cb51edcdea0137c183e57697848d",
    },
    "selected_movies_csv": {
        "path": "data/working/phase11_division_audit/full13_pairs/selected_movies.csv",
        "sha256": "c2bc6ff67b46de1fb8056bbad333311e8b9ea1466ca051b1dddff5ccb2d02f7a",
    },
}
PAIR_AUDIT_CONTRACT = {
    "cohort": "exact_phase10_internal_dev",
    "datasets": list(EXPECTED_EXPOSED_DATASETS),
    "labelled_event_coverage": 18,
    "outputs": PAIR_AUDIT_OUTPUTS,
    "pair_audit_schema_version": 1,
    "source_files": list(PAIR_AUDIT_SOURCE_PATHS),
    "source_sha256": EXPECTED_PAIR_AUDIT_SOURCE_SHA256,
    "summary_path": PAIR_AUDIT_SUMMARY_PATH,
    "summary_sha256": EXPECTED_PAIR_AUDIT_SUMMARY_SHA256,
}
HARDNESS_ORDER = [
    "minimum_cellness:descending",
    "restored_node_count:ascending",
    "edge_removal_count:ascending",
    "candidate_1_id:ascending",
    "candidate_2_id:ascending",
]
SAMPLING_STATIC_CONTRACT = {
    "matched_nondivision_controls": {
        "hardest_pair_order": HARDNESS_ORDER,
        "max_mothers_per_movie": 32,
        "mother_rank": {
            "algorithm": "sha256",
            "encoding": "salt_nul_dataset_nul_mother_id_ascii",
            "salt": "biohub-phase11-division-matched-control-v1",
        },
        "pairs_per_mother": 1,
        "source_role": "fit",
        "supervised": True,
    },
    "positive_records": {
        "include": "all_recoverable_fit_truth_triplets",
        "required_matches": ["filtered_mother", "raw_candidate_1", "raw_candidate_2"],
        "source_role": "fit",
        "supervised": True,
    },
    "record_identity_fields": [
        "dataset",
        "mother_id",
        "candidate_1_id",
        "candidate_2_id",
        "sampling_stratum",
        "source_role",
        "node_references",
        "minimum_cellness",
        "restored_node_count",
        "edge_removal_count",
        "deterministic_rank",
    ],
    "same_mother_alternatives": {
        "eligibility": "non_true_pair_at_truth_mother_with_exactly_two_known_daughters",
        "hardness_order": HARDNESS_ORDER,
        "max_pairs_per_mother": 12,
        "source_role": "fit",
        "supervised": True,
    },
    "supervised_batches": {
        "diagnostics_in_gradient_updates": False,
        "repeat_smaller_strata_deterministically": True,
        "stratum_ratio": {
            "matched_nondivision_control": 1,
            "positive": 1,
            "same_mother_negative": 1,
        },
    },
    "unmatched_diagnostics": {
        "loss_weight": 0.0,
        "max_proposals_per_movie": 32,
        "proposal_rank": {
            "algorithm": "sha256",
            "encoding": (
                "salt_nul_dataset_nul_mother_id_nul_candidate_1_id_nul_candidate_2_id_ascii"
            ),
            "salt": "biohub-phase11-division-unmatched-diagnostic-v1",
        },
        "source_role": "fit",
        "supervised": False,
    },
}
ACCEPTANCE_COHORT_CONTRACT = {
    "division_movies": 10,
    "labelled_divisions": 18,
    "movie_count": 20,
    "no_division_movies": 10,
}
ACCEPTANCE_GUARDS = {
    "aggregate_node_ratio_delta": {"operator": "<=", "value": 0.001},
    "complete_cohort_movies": {"operator": "==", "value": 20},
    "fully_matched_wrong_forks": {"operator": "==", "value": 0},
    "full_score_wins_vs_losses": {"operator": ">", "left": "wins", "right": "losses"},
    "matched_division_false_positives_per_no_division_movie": {
        "movie_count": 10,
        "operator": "==",
        "value": 0,
    },
    "mean_edge_jaccard_delta": {"operator": ">=", "value": -0.001},
    "mean_full_score_delta": {"operator": ">=", "value": 0.001},
    "no_fork_fallback_identity": {
        "operator": "logical_graph_sha256_equal",
        "value": True,
    },
    "strict_true_recoveries": {"operator": ">=", "value": 1},
}
ACCEPTANCE_DIAGNOSTICS = [
    "median_full_score_delta",
    "median_edge_jaccard_delta",
    "edge_wins_ties_losses",
    "full_score_wins_ties_losses",
    "worst_movie",
    "micro_division_counts",
    "unmatched_accepted_forks",
    "restored_nodes",
    "removed_edges",
    "added_edges",
]
THRESHOLD_GRID = [
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
]
ASSIGNMENT_STATIC_CONTRACT = {
    "conflict_rules": {
        "allow_daughter_as_later_frame_mother": True,
        "reject_daughter_already_assigned_to_accepted_fork": True,
        "reject_mother_already_has_fork": True,
    },
    "fallback_graph": "exact_phase10_divisions_off_pruned_graph",
    "max_forks_per_movie": 5,
    "movie_order": [
        "score:descending",
        "edge_removal_count:ascending",
        "restored_node_count:ascending",
        "mother_id:ascending",
        "candidate_1_id:ascending",
        "candidate_2_id:ascending",
    ],
    "per_mother_selection": {
        "keep": "highest_scoring_pair",
        "score_gate": "greater_than_or_equal_to_threshold",
        "tie_break": [
            "edge_removal_count:ascending",
            "restored_node_count:ascending",
            "candidate_1_id:ascending",
            "candidate_2_id:ascending",
        ],
    },
    "threshold_grid": THRESHOLD_GRID,
    "transaction": {
        "mutation_steps": [
            "restore_missing_members_from_raw_score_cache_graph",
            "remove_competing_mother_t_plus_1_child_outside_pair",
            "remove_daughter_incoming_edge_from_wrong_parent",
            "add_two_mother_to_daughter_edges",
            "preserve_all_unrelated_nodes_histories_futures_and_edges",
        ],
        "no_fork_result": "logical_graph_sha256_equal_to_fallback",
        "rollback": "whole_movie_returns_untouched_fallback_on_any_failure",
        "validation_invariants": [
            "distinct_daughters",
            "exact_t_to_t_plus_1_timing",
            "daughter_indegree_at_most_one",
            "outdegree_at_most_two",
            "unique_edges",
            "all_edge_endpoints_present",
            "acyclic",
        ],
    },
}


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} has wrong fields; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _validate_common_contract(
    payload: Mapping[str, Any],
    *,
    kind: str,
    schema_field: str,
    cross_keys: set[str],
) -> None:
    if payload.get("contract_name") != f"phase11_division_{kind}":
        raise ValueError(f"invalid Phase 11 {kind} contract name")
    if payload.get(schema_field) != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported Phase 11 {kind} contract schema")
    fingerprints = payload.get("source_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != SOURCE_FINGERPRINT_FIELDS:
        raise ValueError(f"Phase 11 {kind} contract has incomplete source fingerprints")
    if not all(_is_sha256(value) for value in fingerprints.values()):
        raise ValueError(f"Phase 11 {kind} contract has an invalid source fingerprint")
    cross = payload.get("cross_contract_sha256")
    if not isinstance(cross, dict) or set(cross) != cross_keys:
        raise ValueError(f"Phase 11 {kind} contract has incomplete cross-hashes")
    if not all(_is_sha256(value) for value in cross.values()):
        raise ValueError(f"Phase 11 {kind} contract has an invalid cross-hash")


def _expected_quota_json() -> dict[str, dict[str, int]]:
    return {
        str(division_count): {
            "development": quotas[1],
            "evaluation": quotas[2],
            "fit": quotas[0],
        }
        for division_count, quotas in SPLIT_QUOTAS.items()
    }


def _validate_dataset_tuple(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or value != sorted(set(value))
        or not all(isinstance(dataset, str) and dataset.startswith("6bba_") for dataset in value)
    ):
        raise ValueError(f"{label} must contain unique sorted 6bba datasets")
    return value


def _validate_split_contract(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "cohort",
        "contract_name",
        "cross_contract_sha256",
        "division_split_schema_version",
        "forced_fit",
        "logical_graph_hash",
        "manifest",
        "quotas_by_division_count",
        "ranking",
        "role_datasets",
        "role_summary",
        "source_fingerprints",
        "strata",
    }
    _require_exact_keys(payload, expected_keys, label="Phase 11 split contract")
    _validate_common_contract(
        payload,
        kind="split",
        schema_field="division_split_schema_version",
        cross_keys=set(),
    )
    expected_cohort = {
        "division_count_histogram": {
            str(key): value for key, value in EXPECTED_DIVISION_HISTOGRAM.items()
        },
        "division_movies": 66,
        "embryo": "6bba",
        "labelled_divisions": 125,
        "movies": 128,
        "no_division_movies": 62,
    }
    if payload["cohort"] != expected_cohort:
        raise ValueError("Phase 11 split contract has the wrong cohort")
    if payload["logical_graph_hash"] != LOGICAL_GRAPH_HASH_CONTRACT:
        raise ValueError("Phase 11 split contract has the wrong graph-hash protocol")
    if payload["quotas_by_division_count"] != _expected_quota_json():
        raise ValueError("Phase 11 split contract has the wrong quotas")
    expected_forced = {
        "policy_path": "data/working/phase10_frozen_candidate_policy.json",
        "policy_sha256": EXPECTED_POLICY_SHA256,
        "selection_datasets": list(EXPECTED_EXPOSED_DATASETS),
    }
    if payload["forced_fit"] != expected_forced:
        raise ValueError("Phase 11 split contract has the wrong forced-fit policy")
    expected_ranking = {
        "algorithm": "sha256",
        "assignment_order": ["evaluation", "development", "fit"],
        "encoding": "salt_nul_then_dataset_utf8",
        "salt": SPLIT_RANK_SALT,
    }
    if payload["ranking"] != expected_ranking:
        raise ValueError("Phase 11 split contract has the wrong deterministic ranking")
    manifest = payload["manifest"]
    if manifest != {
        "csv_schema_version": MANIFEST_CSV_SCHEMA_VERSION,
        "path": "data/working/phase11_division_manifest.csv",
        "sha256": manifest.get("sha256") if isinstance(manifest, dict) else None,
    } or not _is_sha256(manifest.get("sha256") if isinstance(manifest, dict) else None):
        raise ValueError("Phase 11 split contract has an invalid manifest identity")

    role_datasets = payload["role_datasets"]
    if not isinstance(role_datasets, dict) or set(role_datasets) != set(ROLE_NAMES):
        raise ValueError("Phase 11 split contract has incomplete role tuples")
    normalized_roles = {
        role: _validate_dataset_tuple(role_datasets[role], label=f"{role} role")
        for role in ROLE_NAMES
    }
    if any(len(normalized_roles[role]) != EXPECTED_ROLE_MOVIES[role] for role in ROLE_NAMES):
        raise ValueError("Phase 11 split contract has the wrong role movie totals")
    role_sets = [set(normalized_roles[role]) for role in ROLE_NAMES]
    if any(role_sets[left] & role_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Phase 11 split contract roles overlap")
    if len(set().union(*role_sets)) != 128:
        raise ValueError("Phase 11 split contract does not cover exactly 128 movies")
    if not set(EXPECTED_EXPOSED_DATASETS) <= role_sets[0]:
        raise ValueError("Phase 11 split contract does not force every exposed movie into fit")

    strata = payload["strata"]
    if not isinstance(strata, dict) or set(strata) != {str(value) for value in SPLIT_QUOTAS}:
        raise ValueError("Phase 11 split contract has incomplete strata")
    rebuilt_roles = {role: set() for role in ROLE_NAMES}
    exposed = set(EXPECTED_EXPOSED_DATASETS)
    for division_count, quotas in SPLIT_QUOTAS.items():
        stratum = strata[str(division_count)]
        if not isinstance(stratum, dict) or set(stratum) != set(ROLE_NAMES):
            raise ValueError("Phase 11 split stratum has incomplete roles")
        current = {
            role: _validate_dataset_tuple(
                stratum[role], label=f"division-count {division_count} {role} stratum"
            )
            for role in ROLE_NAMES
        }
        for role, quota in zip(ROLE_NAMES, quotas, strict=True):
            if len(current[role]) != quota:
                raise ValueError("Phase 11 split stratum violates its quota")
            rebuilt_roles[role].update(current[role])
        forced = sorted(dataset for dataset in current["fit"] if dataset in exposed)
        remaining = sorted(
            set().union(*(set(current[role]) for role in ROLE_NAMES)) - exposed,
            key=lambda dataset: (
                hashlib.sha256(SPLIT_RANK_PREFIX + dataset.encode("utf-8")).digest(),
                dataset,
            ),
        )
        evaluation_quota = quotas[2]
        development_quota = quotas[1]
        if set(current["evaluation"]) != set(remaining[:evaluation_quota]):
            raise ValueError("Phase 11 split stratum violates evaluation SHA ranking")
        if set(current["development"]) != set(
            remaining[evaluation_quota : evaluation_quota + development_quota]
        ):
            raise ValueError("Phase 11 split stratum violates development SHA ranking")
        if any(dataset in exposed for dataset in current["development"] + current["evaluation"]):
            raise ValueError("Phase 11 split stratum moved an exposed movie out of fit")
        if len(forced) > quotas[0]:
            raise ValueError("Phase 11 split stratum exceeds its forced-fit capacity")
    if any(rebuilt_roles[role] != set(normalized_roles[role]) for role in ROLE_NAMES):
        raise ValueError("Phase 11 split role tuples disagree with its strata")
    expected_summary = {
        "development": {
            "division_movies": 10,
            "labelled_divisions": 18,
            "movies": 20,
            "no_division_movies": 10,
        },
        "evaluation": {
            "division_movies": 10,
            "labelled_divisions": 18,
            "movies": 20,
            "no_division_movies": 10,
        },
        "fit": {
            "division_movies": 46,
            "labelled_divisions": 89,
            "movies": 88,
            "no_division_movies": 42,
        },
    }
    if payload["role_summary"] != expected_summary:
        raise ValueError("Phase 11 split contract has the wrong role summary")


def _validate_static_contract(
    payload: Mapping[str, Any],
    *,
    kind: str,
    schema_field: str,
    cross_keys: set[str],
    static_fields: Mapping[str, Any],
) -> None:
    expected_keys = {
        "contract_name",
        "cross_contract_sha256",
        schema_field,
        "source_fingerprints",
        *static_fields,
    }
    _require_exact_keys(payload, expected_keys, label=f"Phase 11 {kind} contract")
    _validate_common_contract(
        payload,
        kind=kind,
        schema_field=schema_field,
        cross_keys=cross_keys,
    )
    for field, expected in static_fields.items():
        if payload[field] != expected:
            raise ValueError(f"Phase 11 {kind} contract has the wrong {field}")


def validate_division_contract(contract_kind: str, payload: Mapping[str, Any]) -> None:
    """Strictly validate one of the five frozen pre-model Phase 11 contracts."""
    if not isinstance(payload, Mapping):
        raise TypeError("division contract must be a mapping")
    if contract_kind == "split":
        _validate_split_contract(payload)
        return
    if contract_kind == "proposal":
        _validate_static_contract(
            payload,
            kind="proposal",
            schema_field="division_proposal_schema_version",
            cross_keys={"split"},
            static_fields={
                "pair_audit": PAIR_AUDIT_CONTRACT,
                "phase10_inputs": PHASE10_INPUT_CONTRACT,
                "proposal_rule": PROPOSAL_RULE_CONTRACT,
            },
        )
        return
    if contract_kind == "sampling":
        _validate_static_contract(
            payload,
            kind="sampling",
            schema_field="division_sampling_schema_version",
            cross_keys={"proposal", "split"},
            static_fields=SAMPLING_STATIC_CONTRACT,
        )
        return
    if contract_kind == "acceptance":
        _validate_static_contract(
            payload,
            kind="acceptance",
            schema_field="division_acceptance_schema_version",
            cross_keys={"proposal", "sampling", "split"},
            static_fields={
                "applies_to": ["development_unlock", "locked_evaluation_acceptance"],
                "cohort": ACCEPTANCE_COHORT_CONTRACT,
                "diagnostics": ACCEPTANCE_DIAGNOSTICS,
                "evaluation_access": {
                    "development_must_pass_every_guard": True,
                    "evaluation_remains_unread_if_development_fails": True,
                    "evaluation_tuning_permitted": False,
                },
                "guards": ACCEPTANCE_GUARDS,
            },
        )
        return
    if contract_kind == "assignment":
        _validate_static_contract(
            payload,
            kind="assignment",
            schema_field="division_assignment_schema_version",
            cross_keys={"acceptance", "proposal", "sampling", "split"},
            static_fields=ASSIGNMENT_STATIC_CONTRACT,
        )
        return
    raise ValueError(f"unknown division contract kind {contract_kind!r}")


def read_division_contract(
    path: str | Path,
    *,
    contract_kind: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read canonical JSON and strictly validate one frozen division contract."""
    payload = read_canonical_json(path, expected_sha256=expected_sha256)
    if contract_kind is None:
        name = payload.get("contract_name")
        prefix = "phase11_division_"
        if not isinstance(name, str) or not name.startswith(prefix):
            raise ValueError("cannot infer division contract kind")
        contract_kind = name[len(prefix) :]
    validate_division_contract(contract_kind, payload)
    return payload


def read_division_contracts(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read all five contracts and verify every acyclic cross-hash."""
    directory = Path(config_dir)
    kinds = ("split", "proposal", "sampling", "acceptance", "assignment")
    contracts = {
        kind: read_division_contract(
            directory / f"phase11_division_{kind}.json",
            contract_kind=kind,
        )
        for kind in kinds
    }
    hashes = {kind: canonical_json_sha256(payload) for kind, payload in contracts.items()}
    for kind in kinds:
        for upstream, expected in contracts[kind]["cross_contract_sha256"].items():
            if expected != hashes[upstream]:
                raise ValueError(
                    f"Phase 11 {kind} contract cross-hash for {upstream} does not match"
                )
    fingerprints = contracts["split"]["source_fingerprints"]
    if any(contract["source_fingerprints"] != fingerprints for contract in contracts.values()):
        raise ValueError("Phase 11 contracts do not share one source fingerprint set")
    return contracts


def _current_contract_source_fingerprints(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root)
    return {
        field: file_sha256(root / logical_path)
        for field, logical_path in CONTRACT_SOURCE_PATHS.items()
    }


def _configured_source_paths(
    project_root: Path,
    configured_paths: list[str],
    supplied_paths: Iterable[str | Path] | None,
    *,
    label: str,
) -> tuple[Path, ...]:
    if supplied_paths is None:
        return tuple(project_root / path for path in configured_paths)
    paths = tuple(Path(path) for path in supplied_paths)
    try:
        logical_paths = [
            path.resolve().relative_to(project_root.resolve()).as_posix() for path in paths
        ]
    except ValueError as exc:
        raise ValueError(f"{label} source path is outside the project root") from exc
    if logical_paths != configured_paths:
        raise ValueError(f"{label} source file tuple does not match the frozen contract")
    return paths


def _verify_required_frozen_sources(
    contracts: Mapping[str, Mapping[str, Any]],
    project_root: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    pair_summary_path: str | Path | None = None,
    scoring_source_files: Iterable[str | Path] | None = None,
    evaluation_source_files: Iterable[str | Path] | None = None,
    pair_audit_source_files: Iterable[str | Path] | None = None,
) -> None:
    root = Path(project_root)
    proposal = contracts["proposal"]
    phase10 = proposal["phase10_inputs"]
    checkpoint = Path(checkpoint_path or root / phase10["checkpoint_path"])
    if file_sha256(checkpoint) != phase10["checkpoint_sha256"]:
        raise ValueError("current Phase 10 checkpoint does not match its frozen fingerprint")

    scoring_paths = _configured_source_paths(
        root,
        phase10["scoring_source_files"],
        scoring_source_files,
        label="Phase 10 scoring",
    )
    if source_bundle_sha256(scoring_paths) != phase10["scoring_source_sha256"]:
        raise ValueError("current Phase 10 scoring source bundle does not match its fingerprint")
    evaluation_paths = _configured_source_paths(
        root,
        phase10["evaluation_source_files"],
        evaluation_source_files,
        label="Phase 10 evaluation",
    )
    if source_bundle_sha256(evaluation_paths) != phase10["evaluation_source_sha256"]:
        raise ValueError("current Phase 10 evaluation source bundle does not match its fingerprint")

    policy = Path(policy_path or root / phase10["frozen_policy_path"])
    if file_sha256(policy) != phase10["frozen_policy_sha256"]:
        raise ValueError("current Phase 10 policy does not match its frozen fingerprint")
    pair_audit = proposal["pair_audit"]
    pair_summary = Path(pair_summary_path or root / pair_audit["summary_path"])
    if file_sha256(pair_summary) != pair_audit["summary_sha256"]:
        raise ValueError("current pair-audit summary does not match its frozen fingerprint")
    pair_source_paths = _configured_source_paths(
        root,
        pair_audit["source_files"],
        pair_audit_source_files,
        label="pair-audit",
    )
    if source_bundle_sha256(pair_source_paths) != pair_audit["source_sha256"]:
        raise ValueError("current pair-audit source bundle does not match its fingerprint")


def _verify_manifest_contract_binding(
    rows: tuple[DivisionManifestRow, ...],
    split: Mapping[str, Any],
    *,
    expected_datasets: Iterable[str] | None,
) -> None:
    datasets = tuple(row.dataset for row in rows)
    if len(rows) != 128 or len(set(datasets)) != 128:
        raise ValueError("division contract manifest must contain exactly 128 unique movies")
    if expected_datasets is not None and tuple(expected_datasets) != datasets:
        raise ValueError("division contract manifest does not match the exact dataset tuple")
    histogram = dict(sorted(Counter(row.division_count for row in rows).items()))
    if histogram != EXPECTED_DIVISION_HISTOGRAM:
        raise ValueError("division contract manifest has the wrong division histogram")
    if sum(row.division_count for row in rows) != 125:
        raise ValueError("division contract manifest must contain exactly 125 divisions")

    role_datasets = {
        role: [row.dataset for row in rows if row.role == role] for role in ROLE_NAMES
    }
    if split["role_datasets"] != role_datasets:
        raise ValueError("division contract role tuples do not match the actual manifest")
    if split["role_summary"] != division_role_summary(rows):
        raise ValueError("division contract role summary does not match the actual manifest")
    strata = {
        str(division_count): {
            role: [
                row.dataset
                for row in rows
                if row.division_count == division_count and row.role == role
            ]
            for role in ROLE_NAMES
        }
        for division_count in SPLIT_QUOTAS
    }
    if split["strata"] != strata:
        raise ValueError("division contract strata do not match the actual manifest")

    exposed = set(EXPECTED_EXPOSED_DATASETS)
    if any(row.proposal_exposed != (row.dataset in exposed) for row in rows):
        raise ValueError("division contract exposed flags do not match the frozen policy")
    reassigned = assign_division_roles(
        (
            replace(row, proposal_exposed=False, role=None)
            for row in rows
        ),
        EXPECTED_EXPOSED_DATASETS,
    )
    if any(
        observed.role != expected.role
        or observed.proposal_exposed != expected.proposal_exposed
        for observed, expected in zip(rows, reassigned, strict=True)
    ):
        raise ValueError("division contract manifest roles do not match deterministic assignment")


def _verify_division_summary(
    summary_path: str | Path,
    *,
    manifest_path: str | Path,
    config_dir: str | Path,
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    summary = read_canonical_json(summary_path)
    kinds = ("split", "proposal", "sampling", "acceptance", "assignment")
    expected = {
        "cohort": contracts["split"]["cohort"],
        "configs": {
            kind: {
                "path": f"configs/phase11_division_{kind}.json",
                "sha256": file_sha256(
                    Path(config_dir) / f"phase11_division_{kind}.json"
                ),
            }
            for kind in kinds
        },
        "division_manifest_summary_schema_version": 1,
        "outputs": {
            "manifest_csv": {
                "path": "data/working/phase11_division_manifest.csv",
                "sha256": file_sha256(manifest_path),
            }
        },
        "role_summary": contracts["split"]["role_summary"],
        "source_fingerprints": contracts["split"]["source_fingerprints"],
    }
    if summary != expected:
        raise ValueError("generated division manifest summary is stale or tampered")


def verify_division_contract_bundle(
    config_dir: str | Path,
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    summary_path: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    expected_datasets: Iterable[str] | None = None,
    checkpoint_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    pair_summary_path: str | Path | None = None,
    scoring_source_files: Iterable[str | Path] | None = None,
    evaluation_source_files: Iterable[str | Path] | None = None,
    pair_audit_source_files: Iterable[str | Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Bind all contracts to current manifest bytes and current source files."""
    contracts = read_division_contracts(config_dir)
    manifest_hash = file_sha256(manifest_path)
    declared_manifest_hash = contracts["split"]["manifest"]["sha256"]
    if manifest_hash != declared_manifest_hash:
        raise ValueError(
            f"actual manifest SHA-256 {manifest_hash} does not match {declared_manifest_hash}"
        )
    if expected_manifest_sha256 is not None:
        if not _is_sha256(expected_manifest_sha256):
            raise ValueError("expected manifest SHA-256 is invalid")
        if manifest_hash != expected_manifest_sha256:
            raise ValueError("actual manifest SHA-256 does not match the expected identity")
    rows = read_division_manifest(manifest_path)
    _verify_manifest_contract_binding(
        rows,
        contracts["split"],
        expected_datasets=expected_datasets,
    )

    actual_fingerprints = _current_contract_source_fingerprints(project_root)
    if contracts["split"]["source_fingerprints"] != actual_fingerprints:
        raise ValueError(
            "contract does not match the current contract-generation source fingerprint set"
        )
    _verify_required_frozen_sources(
        contracts,
        project_root,
        checkpoint_path=checkpoint_path,
        policy_path=policy_path,
        pair_summary_path=pair_summary_path,
        scoring_source_files=scoring_source_files,
        evaluation_source_files=evaluation_source_files,
        pair_audit_source_files=pair_audit_source_files,
    )
    if summary_path is not None:
        _verify_division_summary(
            summary_path,
            manifest_path=manifest_path,
            config_dir=config_dir,
            contracts=contracts,
        )
    return contracts


def read_split_contract(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    return read_division_contract(path, contract_kind="split", expected_sha256=expected_sha256)


def read_proposal_contract(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    return read_division_contract(path, contract_kind="proposal", expected_sha256=expected_sha256)


def read_sampling_contract(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    return read_division_contract(path, contract_kind="sampling", expected_sha256=expected_sha256)


def read_acceptance_contract(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    return read_division_contract(path, contract_kind="acceptance", expected_sha256=expected_sha256)


def read_assignment_contract(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    return read_division_contract(path, contract_kind="assignment", expected_sha256=expected_sha256)
