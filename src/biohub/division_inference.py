"""Deterministic assignment, mutation, and evaluation for division recovery."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from biohub.constants import DIVISION_WEIGHT, MATCH_RADIUS_UM, VOXEL_SCALE_UM
from biohub.division import DivisionPairProposal
from biohub.division_data import (
    ACCEPTANCE_COHORT_CONTRACT,
    ACCEPTANCE_GUARDS,
    ASSIGNMENT_STATIC_CONTRACT,
    THRESHOLD_GRID,
    logical_graph_sha256,
    truth_divisions,
)
from biohub.metric import MetricResult, TrackingGraph, match_nodes

PROBABILITY_THRESHOLDS = tuple(float(value) for value in THRESHOLD_GRID)
MAX_FORKS_PER_MOVIE = int(ASSIGNMENT_STATIC_CONTRACT["max_forks_per_movie"])

ForkClassification = Literal[
    "true_positive",
    "fully_matched_wrong_fork",
    "matched_division_false_positive",
    "unmatched_fork",
]
_FORK_CLASSIFICATIONS = frozenset(
    {
        "true_positive",
        "fully_matched_wrong_fork",
        "matched_division_false_positive",
        "unmatched_fork",
    }
)


def _require_identifier(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if not np.iinfo(np.int64).min <= result <= np.iinfo(np.int64).max:
        raise ValueError(f"{field} must be a signed 64-bit integer")
    return result


def _require_count(value: Any, *, field: str) -> int:
    result = _require_identifier(value, field=field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _require_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite and numeric")
    return float(value)


def _require_sha256(value: Any, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def logit_to_probability(logit: Real) -> float:
    """Convert one finite model logit to a probability with a stable sigmoid."""
    value = _require_finite(logit, field="logit")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class ScoredProposal:
    """One immutable identifier-bearing proposal scored by a model logit."""

    mother_id: int
    candidate_1_id: int
    candidate_2_id: int
    logit: float
    edge_removal_count: int
    restored_node_count: int
    fallback_graph_sha256: str

    def __post_init__(self) -> None:
        mother = _require_identifier(self.mother_id, field="mother_id")
        first = _require_identifier(self.candidate_1_id, field="candidate_1_id")
        second = _require_identifier(self.candidate_2_id, field="candidate_2_id")
        if first >= second:
            raise ValueError("candidate identifiers must be canonical and distinct")
        if mother in {first, second}:
            raise ValueError("a division mother and its daughters must be distinct")
        object.__setattr__(self, "mother_id", mother)
        object.__setattr__(self, "candidate_1_id", first)
        object.__setattr__(self, "candidate_2_id", second)
        object.__setattr__(self, "logit", _require_finite(self.logit, field="logit"))
        object.__setattr__(
            self,
            "edge_removal_count",
            _require_count(self.edge_removal_count, field="edge_removal_count"),
        )
        object.__setattr__(
            self,
            "restored_node_count",
            _require_count(self.restored_node_count, field="restored_node_count"),
        )
        object.__setattr__(
            self,
            "fallback_graph_sha256",
            _require_sha256(
                self.fallback_graph_sha256,
                field="fallback_graph_sha256",
            ),
        )

    @property
    def probability(self) -> float:
        return logit_to_probability(self.logit)

    @property
    def candidate_ids(self) -> tuple[int, int]:
        return (self.candidate_1_id, self.candidate_2_id)


@dataclass(frozen=True, slots=True)
class AcceptedFork:
    """One immutable fork retained by deterministic movie-level assignment."""

    mother_id: int
    candidate_1_id: int
    candidate_2_id: int
    logit: float
    edge_removal_count: int
    restored_node_count: int
    fallback_graph_sha256: str

    def __post_init__(self) -> None:
        validated = ScoredProposal(
            mother_id=self.mother_id,
            candidate_1_id=self.candidate_1_id,
            candidate_2_id=self.candidate_2_id,
            logit=self.logit,
            edge_removal_count=self.edge_removal_count,
            restored_node_count=self.restored_node_count,
            fallback_graph_sha256=self.fallback_graph_sha256,
        )
        for field in (
            "mother_id",
            "candidate_1_id",
            "candidate_2_id",
            "logit",
            "edge_removal_count",
            "restored_node_count",
            "fallback_graph_sha256",
        ):
            object.__setattr__(self, field, getattr(validated, field))

    @classmethod
    def from_scored(cls, proposal: ScoredProposal) -> AcceptedFork:
        if not isinstance(proposal, ScoredProposal):
            raise TypeError("accepted forks require ScoredProposal values")
        return cls(
            mother_id=proposal.mother_id,
            candidate_1_id=proposal.candidate_1_id,
            candidate_2_id=proposal.candidate_2_id,
            logit=proposal.logit,
            edge_removal_count=proposal.edge_removal_count,
            restored_node_count=proposal.restored_node_count,
            fallback_graph_sha256=proposal.fallback_graph_sha256,
        )

    @property
    def probability(self) -> float:
        return logit_to_probability(self.logit)

    @property
    def candidate_ids(self) -> tuple[int, int]:
        return (self.candidate_1_id, self.candidate_2_id)


@dataclass(frozen=True, slots=True)
class DivisionMutationResult:
    """Outcome and diagnostics for one whole-movie graph transaction."""

    graph: TrackingGraph
    fallback_graph_sha256: str | None
    candidate_graph_sha256: str | None
    attempted_forks: tuple[AcceptedFork, ...]
    applied_forks: tuple[AcceptedFork, ...]
    rolled_back: bool
    diagnostic: str | None
    restored_node_ids: tuple[int, ...]
    removed_edges: tuple[tuple[int, int], ...]
    added_edges: tuple[tuple[int, int], ...]

    @property
    def restored_node_count(self) -> int:
        return len(self.restored_node_ids)

    @property
    def removed_edge_count(self) -> int:
        return len(self.removed_edges)

    @property
    def added_edge_count(self) -> int:
        return len(self.added_edges)


@dataclass(frozen=True, slots=True)
class ForkMatch:
    """Node-matching detail for one strict predicted triplet."""

    predicted_triplet: tuple[int, int, int]
    mapped_triplet: tuple[int | None, int | None, int | None]
    truth_triplet: tuple[int, int, int] | None
    classification: ForkClassification

    def __post_init__(self) -> None:
        predicted = _validate_triplet(self.predicted_triplet, field="predicted_triplet")
        if not isinstance(self.mapped_triplet, tuple) or len(self.mapped_triplet) != 3:
            raise ValueError("mapped_triplet must contain exactly three node references")
        mapped = tuple(
            None if value is None else _require_identifier(value, field="mapped_triplet")
            for value in self.mapped_triplet
        )
        truth = (
            None
            if self.truth_triplet is None
            else _validate_triplet(self.truth_triplet, field="truth_triplet")
        )
        if self.classification not in _FORK_CLASSIFICATIONS:
            raise ValueError("fork match has an unsupported classification")
        object.__setattr__(self, "predicted_triplet", predicted)
        object.__setattr__(self, "mapped_triplet", mapped)
        object.__setattr__(self, "truth_triplet", truth)


@dataclass(frozen=True, slots=True)
class StrictTripletResult:
    """Strict daughter-aware fork counts and per-fork matching detail."""

    tp: int
    fp: int
    fn: int
    matched_division_false_positives: int
    fully_matched_wrong_forks: int
    unmatched_forks: int
    matched_truth_triplets: tuple[tuple[int, int, int], ...]
    details: tuple[ForkMatch, ...]

    def __post_init__(self) -> None:
        for field in (
            "tp",
            "fp",
            "fn",
            "matched_division_false_positives",
            "fully_matched_wrong_forks",
            "unmatched_forks",
        ):
            object.__setattr__(self, field, _require_count(getattr(self, field), field=field))
        details = tuple(self.details)
        if any(not isinstance(detail, ForkMatch) for detail in details):
            raise TypeError("strict triplet details must contain ForkMatch values")
        predicted = [detail.predicted_triplet for detail in details]
        if len(predicted) != len(set(predicted)):
            raise ValueError("strict triplet details contain duplicate predicted forks")
        matched = tuple(
            _validate_triplet(value, field="matched_truth_triplets")
            for value in self.matched_truth_triplets
        )
        if len(matched) != len(set(matched)):
            raise ValueError("matched truth triplets must be unique")

        by_classification = {
            classification: sum(detail.classification == classification for detail in details)
            for classification in _FORK_CLASSIFICATIONS
        }
        if self.tp != by_classification["true_positive"]:
            raise ValueError("strict TP count disagrees with detail classifications")
        if self.fp != len(details) - self.tp:
            raise ValueError("strict FP count disagrees with predicted fork details")
        if (
            self.matched_division_false_positives
            != by_classification["matched_division_false_positive"]
            or self.fully_matched_wrong_forks
            != by_classification["fully_matched_wrong_fork"]
            or self.unmatched_forks != by_classification["unmatched_fork"]
        ):
            raise ValueError("strict classification counts disagree with details")

        detail_matches: set[tuple[int, int, int]] = set()
        referenced_truth: list[tuple[int, int, int]] = []
        for detail in details:
            mapped_mother, mapped_first, mapped_second = detail.mapped_triplet
            truth = detail.truth_triplet
            mapped_ids = [value for value in detail.mapped_triplet if value is not None]
            if len(mapped_ids) != len(set(mapped_ids)):
                raise ValueError("fork detail maps multiple members to one truth node")
            if truth is not None:
                referenced_truth.append(truth)
            if detail.classification == "true_positive":
                if (
                    truth is None
                    or None in detail.mapped_triplet
                    or mapped_mother != truth[0]
                    or {mapped_first, mapped_second} != {truth[1], truth[2]}
                ):
                    raise ValueError("true-positive detail does not match its truth triplet")
                detail_matches.add(truth)
            elif detail.classification == "matched_division_false_positive":
                if mapped_mother is None or truth is not None:
                    raise ValueError("matched division false-positive detail is incoherent")
            elif detail.classification == "fully_matched_wrong_fork":
                if (
                    truth is None
                    or None in detail.mapped_triplet
                    or mapped_mother != truth[0]
                    or {mapped_first, mapped_second} == {truth[1], truth[2]}
                ):
                    raise ValueError("fully matched wrong-fork detail is incoherent")
            elif detail.classification == "unmatched_fork" and (
                (truth is None and mapped_mother is not None)
                or (truth is not None and None not in detail.mapped_triplet)
            ):
                raise ValueError("unmatched-fork detail is fully classifiable")
        if set(matched) != detail_matches or len(matched) != self.tp:
            raise ValueError("matched truth triplets disagree with true-positive details")
        if (
            len(referenced_truth) != len(set(referenced_truth))
            or len(referenced_truth) > self.tp + self.fn
        ):
            raise ValueError("strict details reference impossible truth triplets")
        object.__setattr__(self, "matched_truth_triplets", tuple(sorted(matched)))
        object.__setattr__(self, "details", details)


def _validate_triplet(value: Any, *, field: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three identifiers")
    mother, first, second = (
        _require_identifier(item, field=field) for item in value
    )
    if first >= second or mother in {first, second}:
        raise ValueError(f"{field} must contain one mother and two canonical distinct daughters")
    return (mother, first, second)


def _count_jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


def _validate_metric_result(result: MetricResult, *, label: str) -> None:
    if not isinstance(result, MetricResult):
        raise TypeError(f"{label} must be a MetricResult")
    floats = {
        field: _require_finite(getattr(result, field), field=f"{label} {field}")
        for field in ("score", "edge_jaccard", "division_jaccard")
    }
    if any(not 0.0 <= value <= 1.0 for value in floats.values()):
        raise ValueError(f"{label} score and Jaccards must be between zero and one")
    counts = {
        field: _require_count(getattr(result, field), field=f"{label} {field}")
        for field in (
            "edge_tp",
            "edge_fp",
            "edge_fn",
            "edge_ignored",
            "division_tp",
            "division_fp",
            "division_fn",
            "matched_nodes",
            "predicted_nodes",
            "truth_nodes",
        )
    }
    expected_edge = _count_jaccard(counts["edge_tp"], counts["edge_fp"], counts["edge_fn"])
    expected_division = _count_jaccard(
        counts["division_tp"],
        counts["division_fp"],
        counts["division_fn"],
    )
    expected_score = (
        (1.0 - DIVISION_WEIGHT) * expected_edge + DIVISION_WEIGHT * expected_division
    )
    if not math.isclose(floats["edge_jaccard"], expected_edge, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} edge Jaccard disagrees with TP/FP/FN")
    if not math.isclose(
        floats["division_jaccard"], expected_division, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label} division Jaccard disagrees with TP/FP/FN")
    if not math.isclose(floats["score"], expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} full score disagrees with frozen division weight")
    if counts["matched_nodes"] > min(counts["predicted_nodes"], counts["truth_nodes"]):
        raise ValueError(f"{label} matched node count is impossible")
    if counts["division_tp"] + counts["division_fp"] > counts["matched_nodes"]:
        raise ValueError(f"{label} matched division counts exceed matched nodes")
    if counts["division_tp"] + counts["division_fn"] > counts["truth_nodes"]:
        raise ValueError(f"{label} truth division count exceeds truth nodes")
    predicted_edges = (
        counts["edge_tp"] + counts["edge_fp"] + counts["edge_ignored"]
    )
    truth_edges = counts["edge_tp"] + counts["edge_fn"]
    if predicted_edges > counts["predicted_nodes"] * (counts["predicted_nodes"] - 1) // 2:
        raise ValueError(f"{label} predicted edge count is impossible")
    if truth_edges > counts["truth_nodes"] * (counts["truth_nodes"] - 1) // 2:
        raise ValueError(f"{label} truth edge count is impossible")
    if counts["edge_tp"] + counts["edge_fp"] and counts["matched_nodes"] < 2:
        raise ValueError(f"{label} scored edges require at least two matched nodes")


@dataclass(frozen=True, slots=True)
class MovieEvaluation:
    """All graph-derived values needed to aggregate one cohort movie."""

    dataset: str
    fallback_graph_sha256: str
    candidate_graph_sha256: str
    truth_graph_sha256: str
    baseline_metric: MetricResult
    candidate_metric: MetricResult
    baseline_triplets: StrictTripletResult
    candidate_triplets: StrictTripletResult
    accepted_forks: int
    estimated_nodes: int
    restored_node_count: int
    removed_edge_count: int
    added_edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("movie evaluation dataset must be a non-empty string")
        for field in (
            "fallback_graph_sha256",
            "candidate_graph_sha256",
            "truth_graph_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), field=field),
            )
        _validate_metric_result(self.baseline_metric, label="baseline metric")
        _validate_metric_result(self.candidate_metric, label="candidate metric")
        if not isinstance(self.baseline_triplets, StrictTripletResult) or not isinstance(
            self.candidate_triplets, StrictTripletResult
        ):
            raise TypeError("movie evaluation requires strict triplet results")
        truth_count = self.baseline_triplets.tp + self.baseline_triplets.fn
        candidate_truth_count = self.candidate_triplets.tp + self.candidate_triplets.fn
        if truth_count != candidate_truth_count:
            raise ValueError("baseline and candidate strict results disagree on truth divisions")
        accepted_count = _require_count(self.accepted_forks, field="accepted_forks")
        if accepted_count > MAX_FORKS_PER_MOVIE:
            raise ValueError("accepted_forks exceeds the frozen movie cap")
        estimated = _require_count(self.estimated_nodes, field="estimated_nodes")
        if estimated == 0:
            raise ValueError("estimated_nodes must be positive")
        object.__setattr__(self, "accepted_forks", accepted_count)
        object.__setattr__(self, "estimated_nodes", estimated)
        for field in ("restored_node_count", "removed_edge_count", "added_edge_count"):
            object.__setattr__(self, field, _require_count(getattr(self, field), field=field))

        baseline = self.baseline_metric
        candidate = self.candidate_metric
        baseline_truth_edges = baseline.edge_tp + baseline.edge_fn
        candidate_truth_edges = candidate.edge_tp + candidate.edge_fn
        baseline_truth_divisions = baseline.division_tp + baseline.division_fn
        candidate_truth_divisions = candidate.division_tp + candidate.division_fn
        if (
            baseline.truth_nodes != candidate.truth_nodes
            or baseline_truth_edges != candidate_truth_edges
            or baseline_truth_divisions != candidate_truth_divisions
        ):
            raise ValueError("baseline and candidate metrics disagree on truth totals")
        if truth_count != baseline_truth_divisions:
            raise ValueError("strict and metric truth division totals disagree")
        if (
            baseline.division_tp != 0
            or baseline.division_fp != 0
            or self.baseline_triplets.tp != 0
            or self.baseline_triplets.fp != 0
            or self.baseline_triplets.details
        ):
            raise ValueError("divisions-off fallback contains a predicted division")

        mutation_counts = (
            self.restored_node_count,
            self.removed_edge_count,
            self.added_edge_count,
        )
        if accepted_count == 0:
            if self.candidate_graph_sha256 != self.fallback_graph_sha256:
                raise ValueError("no-fork row requires an identical graph hash")
            if candidate != baseline:
                raise ValueError("no-fork row requires exact equal metrics")
            if self.candidate_triplets != self.baseline_triplets:
                raise ValueError("no-fork row requires exact equal strict triplets")
            if any(mutation_counts) or self.strict_true_recoveries:
                raise ValueError("no-fork row requires zero mutations and recoveries")
            return

        if self.candidate_graph_sha256 == self.fallback_graph_sha256:
            raise ValueError("accepted forks require a changed graph hash")
        if not accepted_count <= self.added_edge_count <= 2 * accepted_count:
            raise ValueError("accepted forks require one or two added edges per fork")
        if (
            self.restored_node_count > 3 * accepted_count
            or self.removed_edge_count > 3 * accepted_count
        ):
            raise ValueError("accepted-fork restored or removed counts exceed bounds")
        if candidate.predicted_nodes != baseline.predicted_nodes + self.restored_node_count:
            raise ValueError("candidate nodes must equal baseline nodes plus restored nodes")
        if len(self.candidate_triplets.details) != accepted_count:
            raise ValueError("accepted forks do not cover candidate strict classifications")
        if candidate.division_tp + candidate.division_fp > accepted_count:
            raise ValueError("candidate matched divisions exceed accepted forks")
        strict_metric_tp = sum(
            detail.truth_triplet is not None for detail in self.candidate_triplets.details
        )
        strict_metric_fp = sum(
            detail.mapped_triplet[0] is not None and detail.truth_triplet is None
            for detail in self.candidate_triplets.details
        )
        if candidate.division_tp != strict_metric_tp or candidate.division_fp != strict_metric_fp:
            raise ValueError("strict fork classifications disagree with metric divisions")

    @property
    def truth_division_count(self) -> int:
        return self.baseline_triplets.tp + self.baseline_triplets.fn

    @property
    def strict_true_recoveries(self) -> int:
        before = set(self.baseline_triplets.matched_truth_triplets)
        after = set(self.candidate_triplets.matched_truth_triplets)
        return len(after - before)


@dataclass(frozen=True, slots=True)
class AcceptanceSummary:
    """Frozen-guard outcomes plus complete cohort diagnostics."""

    passed: bool
    guards: Mapping[str, bool]
    metrics: Mapping[str, float | int | bool]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "guards", _deep_freeze(self.guards))
        object.__setattr__(self, "metrics", _deep_freeze(self.metrics))
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class _GraphArrays:
    node_ids: np.ndarray
    t: np.ndarray
    z: np.ndarray
    y: np.ndarray
    x: np.ndarray
    edges: np.ndarray


def _integral_array(values: Any, *, field: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{field} must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{field} must contain integral values")
    int64 = np.iinfo(np.int64)
    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.unsignedinteger) and np.any(array > int64.max):
            raise ValueError(f"{field} exceeds signed 64-bit range")
        return np.ascontiguousarray(array, dtype=np.int64)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{field} must contain integral values")
        if np.any(array < int64.min) or np.any(array >= 2**63):
            raise ValueError(f"{field} exceeds signed 64-bit range")
        return np.ascontiguousarray(array, dtype=np.int64)

    converted: list[int] = []
    for value in array.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{field} must contain integral values")
        if isinstance(value, Integral):
            integer = int(value)
            if not int64.min <= integer <= int64.max:
                raise ValueError(f"{field} exceeds signed 64-bit range")
        else:
            numeric = float(value)
            if not math.isfinite(numeric) or numeric != math.floor(numeric):
                raise ValueError(f"{field} must contain integral values")
            if numeric < int64.min or numeric >= 2**63:
                raise ValueError(f"{field} exceeds signed 64-bit range")
            integer = int(numeric)
        converted.append(integer)
    return np.asarray(converted, dtype=np.int64).reshape(array.shape)


def _graph_arrays(graph: TrackingGraph, *, label: str) -> _GraphArrays:
    if not isinstance(graph, TrackingGraph):
        raise TypeError(f"{label} must be a TrackingGraph")
    raw_ids = np.asarray(graph.node_ids)
    if raw_ids.ndim != 1:
        raise ValueError(f"{label} node identifiers must be one-dimensional")
    node_ids = _integral_array(raw_ids, field=f"{label} node identifiers")
    count = len(node_ids)
    if len(set(map(int, node_ids))) != count:
        raise ValueError(f"{label} contains duplicate node identifiers")
    times = _integral_array(graph.t, field=f"{label} times", shape=(count,))
    coordinates: list[np.ndarray] = []
    for axis in ("z", "y", "x"):
        try:
            values = np.asarray(getattr(graph, axis), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has invalid {axis} coordinates") from exc
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} has invalid {axis} coordinates")
        coordinates.append(np.ascontiguousarray(values))
    raw_edges = np.asarray(graph.edges)
    if raw_edges.ndim != 2 or raw_edges.shape[1:] != (2,):
        raise ValueError(f"{label} edges must have shape (M, 2)")
    edges = _integral_array(raw_edges, field=f"{label} edges")
    edge_tuples = [tuple(map(int, edge)) for edge in edges]
    if len(set(edge_tuples)) != len(edge_tuples):
        raise ValueError(f"{label} contains duplicate edges")
    known = set(map(int, node_ids))
    if any(source not in known or target not in known for source, target in edge_tuples):
        raise ValueError(f"{label} has an edge with a missing endpoint")
    time_by_id = {
        int(node_id): int(time) for node_id, time in zip(node_ids, times, strict=True)
    }
    if any(time_by_id[target] <= time_by_id[source] for source, target in edge_tuples):
        raise ValueError(f"{label} edges must be strictly forward in time")
    return _GraphArrays(node_ids, times, *coordinates, edges)


def _canonical_graph(graph: TrackingGraph, *, label: str) -> TrackingGraph:
    arrays = _graph_arrays(graph, label=label)
    order = np.argsort(arrays.node_ids, kind="stable")
    edges = np.asarray(sorted(map(tuple, arrays.edges.tolist())), dtype=np.int64).reshape(-1, 2)
    return TrackingGraph(
        node_ids=np.ascontiguousarray(arrays.node_ids[order]),
        t=np.ascontiguousarray(arrays.t[order]),
        z=np.ascontiguousarray(arrays.z[order]),
        y=np.ascontiguousarray(arrays.y[order]),
        x=np.ascontiguousarray(arrays.x[order]),
        edges=edges,
    )


def _validate_graph_invariants(
    graph: _GraphArrays,
    *,
    forks: Sequence[AcceptedFork],
    max_outdegree: int,
) -> None:
    node_ids = set(map(int, graph.node_ids))
    time_by_id = {
        int(node_id): int(time) for node_id, time in zip(graph.node_ids, graph.t, strict=True)
    }
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    successors: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for raw_source, raw_target in graph.edges:
        source = int(raw_source)
        target = int(raw_target)
        outgoing[source] = outgoing.get(source, 0) + 1
        incoming[target] = incoming.get(target, 0) + 1
        successors[source].append(target)
        indegree[target] += 1
    if any(value > 1 for value in incoming.values()):
        raise ValueError("mutated graph daughter indegree exceeds one")
    if any(value > max_outdegree for value in outgoing.values()):
        if max_outdegree == 1:
            raise ValueError("fallback graph outdegree exceeds one")
        raise ValueError("mutated graph outdegree exceeds two")

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(node_ids):
        raise ValueError("mutated graph contains a cycle")

    for fork in forks:
        first, second = fork.candidate_ids
        if first == second:
            raise ValueError("accepted fork daughters must be distinct")
        if any(member not in node_ids for member in (fork.mother_id, first, second)):
            raise ValueError("accepted fork has a missing endpoint")
        expected_time = time_by_id[fork.mother_id] + 1
        if time_by_id[first] != expected_time or time_by_id[second] != expected_time:
            raise ValueError("accepted fork daughters must occur at mother t+1")
        edge_set = set(map(tuple, graph.edges.tolist()))
        if (fork.mother_id, first) not in edge_set or (fork.mother_id, second) not in edge_set:
            raise ValueError("accepted fork edges are missing after mutation")


def _proposal_ids(proposal: DivisionPairProposal) -> tuple[int, int, int]:
    if not isinstance(proposal, DivisionPairProposal):
        raise TypeError("logit scoring requires DivisionPairProposal values")
    mother = _require_identifier(proposal.mother_id, field="mother_id")
    first = _require_identifier(proposal.candidate_1.node_id, field="candidate_1_id")
    second = _require_identifier(proposal.candidate_2.node_id, field="candidate_2_id")
    if first >= second or mother in {first, second}:
        raise ValueError("proposal identifiers must be canonical and distinct")
    return mother, first, second


def scored_proposals_from_logits(
    proposals: Sequence[DivisionPairProposal],
    logits: Sequence[Real] | np.ndarray,
    fallback: TrackingGraph,
) -> tuple[ScoredProposal, ...]:
    """Attach logits and exact fallback mutation costs to pair proposals."""
    materialized = tuple(proposals)
    try:
        values = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal logits must be numeric") from exc
    if values.shape != (len(materialized),) or not np.all(np.isfinite(values)):
        raise ValueError("proposal logits must be a finite vector aligned with proposals")
    graph = _graph_arrays(fallback, label="fallback graph")
    _validate_graph_invariants(graph, forks=(), max_outdegree=1)
    fallback_hash = logical_graph_sha256(fallback)
    fallback_ids = set(map(int, graph.node_ids))
    time_by_id = {
        int(node_id): int(time) for node_id, time in zip(graph.node_ids, graph.t, strict=True)
    }
    edges = [tuple(map(int, edge)) for edge in graph.edges]

    scored_values: list[ScoredProposal] = []
    for proposal, value in zip(materialized, values, strict=True):
        mother, first, second = _proposal_ids(proposal)
        pair_ids = {first, second}
        removals: set[tuple[int, int]] = set()
        mother_time = time_by_id.get(mother)
        for source, target in edges:
            if (
                mother_time is not None
                and source == mother
                and time_by_id[target] == mother_time + 1
                and target not in pair_ids
            ):
                removals.add((source, target))
            if target in pair_ids and source != mother:
                removals.add((source, target))
        restored = sum(member not in fallback_ids for member in (mother, first, second))
        scored_values.append(
            ScoredProposal(
                mother_id=mother,
                candidate_1_id=first,
                candidate_2_id=second,
                logit=float(value),
                edge_removal_count=len(removals),
                restored_node_count=restored,
                fallback_graph_sha256=fallback_hash,
            )
        )
    return tuple(scored_values)


def _per_mother_key(proposal: ScoredProposal) -> tuple[float, int, int, int, int]:
    return (
        -proposal.logit,
        proposal.edge_removal_count,
        proposal.restored_node_count,
        proposal.candidate_1_id,
        proposal.candidate_2_id,
    )


def _global_key(proposal: ScoredProposal) -> tuple[float, int, int, int, int, int]:
    return (
        -proposal.logit,
        proposal.edge_removal_count,
        proposal.restored_node_count,
        proposal.mother_id,
        proposal.candidate_1_id,
        proposal.candidate_2_id,
    )


def select_division_forks(
    proposals: Iterable[ScoredProposal],
    *,
    probability_threshold: float,
) -> tuple[AcceptedFork, ...]:
    """Apply the frozen per-mother, conflict, and five-fork assignment policy."""
    threshold = _require_finite(probability_threshold, field="probability_threshold")
    if threshold not in PROBABILITY_THRESHOLDS:
        raise ValueError("probability_threshold must belong to the frozen threshold grid")
    best_by_mother: dict[int, ScoredProposal] = {}
    fallback_hash: str | None = None
    for proposal in proposals:
        if not isinstance(proposal, ScoredProposal):
            raise TypeError("fork assignment requires ScoredProposal values")
        if fallback_hash is None:
            fallback_hash = proposal.fallback_graph_sha256
        elif proposal.fallback_graph_sha256 != fallback_hash:
            raise ValueError("fork assignment proposals must reference one fallback graph")
        if proposal.probability < threshold:
            continue
        current = best_by_mother.get(proposal.mother_id)
        if current is None or _per_mother_key(proposal) < _per_mother_key(current):
            best_by_mother[proposal.mother_id] = proposal

    accepted: list[AcceptedFork] = []
    assigned_mothers: set[int] = set()
    assigned_daughters: set[int] = set()
    for proposal in sorted(best_by_mother.values(), key=_global_key):
        if len(accepted) == MAX_FORKS_PER_MOVIE:
            break
        if proposal.mother_id in assigned_mothers:
            continue
        if assigned_daughters.intersection(proposal.candidate_ids):
            continue
        accepted.append(AcceptedFork.from_scored(proposal))
        assigned_mothers.add(proposal.mother_id)
        assigned_daughters.update(proposal.candidate_ids)
    return tuple(accepted)


def _validate_assignment(forks: Sequence[AcceptedFork]) -> None:
    if len(forks) > MAX_FORKS_PER_MOVIE:
        raise ValueError("accepted fork assignment exceeds the five-fork cap")
    mothers: set[int] = set()
    daughters: set[int] = set()
    for fork in forks:
        if not isinstance(fork, AcceptedFork):
            raise TypeError("graph mutation requires AcceptedFork values")
        if fork.mother_id in mothers:
            raise ValueError("accepted fork assignment reuses a mother")
        if daughters.intersection(fork.candidate_ids):
            raise ValueError("accepted fork assignment reuses a daughter")
        mothers.add(fork.mother_id)
        daughters.update(fork.candidate_ids)


def _rollback(
    fallback: TrackingGraph,
    attempted: tuple[AcceptedFork, ...],
    exc: Exception,
    *,
    fallback_hash: str | None,
) -> DivisionMutationResult:
    return DivisionMutationResult(
        graph=fallback,
        fallback_graph_sha256=fallback_hash,
        candidate_graph_sha256=fallback_hash,
        attempted_forks=attempted,
        applied_forks=(),
        rolled_back=True,
        diagnostic=f"{type(exc).__name__}: {exc}",
        restored_node_ids=(),
        removed_edges=(),
        added_edges=(),
    )


def apply_division_forks(
    fallback: TrackingGraph,
    raw: TrackingGraph,
    forks: Iterable[AcceptedFork],
) -> DivisionMutationResult:
    """Apply accepted forks atomically, returning the exact fallback on any failure."""
    attempted = tuple(forks)
    fallback_hash: str | None = None
    try:
        fallback_arrays = _graph_arrays(fallback, label="fallback graph")
        fallback_hash = logical_graph_sha256(fallback)
        _validate_graph_invariants(
            fallback_arrays,
            forks=(),
            max_outdegree=1,
        )
        if not attempted:
            return DivisionMutationResult(
                graph=fallback,
                fallback_graph_sha256=fallback_hash,
                candidate_graph_sha256=fallback_hash,
                attempted_forks=(),
                applied_forks=(),
                rolled_back=False,
                diagnostic=None,
                restored_node_ids=(),
                removed_edges=(),
                added_edges=(),
            )

        _validate_assignment(attempted)
        if any(fork.fallback_graph_sha256 != fallback_hash for fork in attempted):
            raise ValueError("accepted fork fallback identity does not match the mutation graph")
        raw_arrays = _graph_arrays(raw, label="raw score-cache graph")

        raw_row = {int(node_id): row for row, node_id in enumerate(raw_arrays.node_ids)}
        fallback_row = {
            int(node_id): row for row, node_id in enumerate(fallback_arrays.node_ids)
        }
        if not set(fallback_row) <= set(raw_row):
            raise ValueError("raw score-cache graph is missing fallback nodes")
        for node_id, row in fallback_row.items():
            source_row = raw_row[node_id]
            if fallback_arrays.t[row] != raw_arrays.t[source_row] or any(
                getattr(fallback_arrays, axis)[row] != getattr(raw_arrays, axis)[source_row]
                for axis in ("z", "y", "x")
            ):
                raise ValueError("shared fallback and raw node values differ")

        node_ids = fallback_arrays.node_ids.tolist()
        times = fallback_arrays.t.tolist()
        coordinates = {
            axis: getattr(fallback_arrays, axis).tolist() for axis in ("z", "y", "x")
        }
        present = set(node_ids)
        restored: set[int] = set()
        edges = [tuple(map(int, edge)) for edge in fallback_arrays.edges]
        removed: set[tuple[int, int]] = set()
        added: set[tuple[int, int]] = set()

        for fork in attempted:
            for member in (fork.mother_id, *fork.candidate_ids):
                if member in present:
                    continue
                if member not in raw_row:
                    raise ValueError(f"accepted fork node {member} is absent from the raw graph")
                row = raw_row[member]
                node_ids.append(member)
                times.append(int(raw_arrays.t[row]))
                for axis in ("z", "y", "x"):
                    coordinates[axis].append(float(getattr(raw_arrays, axis)[row]))
                present.add(member)
                restored.add(member)

            time_by_id = dict(zip(node_ids, times, strict=True))
            expected_time = time_by_id[fork.mother_id] + 1
            if any(time_by_id[daughter] != expected_time for daughter in fork.candidate_ids):
                raise ValueError("accepted fork daughters must occur at mother t+1")

            pair_ids = set(fork.candidate_ids)
            kept: list[tuple[int, int]] = []
            for edge in edges:
                source, target = edge
                competing_child = (
                    source == fork.mother_id
                    and time_by_id[target] == expected_time
                    and target not in pair_ids
                )
                wrong_parent = target in pair_ids and source != fork.mother_id
                if competing_child or wrong_parent:
                    removed.add(edge)
                else:
                    kept.append(edge)
            edges = kept
            edge_set = set(edges)
            for daughter in fork.candidate_ids:
                edge = (fork.mother_id, daughter)
                if edge not in edge_set:
                    edges.append(edge)
                    edge_set.add(edge)
                    added.add(edge)

        candidate = TrackingGraph(
            node_ids=np.asarray(node_ids, dtype=np.int64),
            t=np.asarray(times, dtype=np.int64),
            z=np.asarray(coordinates["z"], dtype=np.float64),
            y=np.asarray(coordinates["y"], dtype=np.float64),
            x=np.asarray(coordinates["x"], dtype=np.float64),
            edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        )
        candidate_arrays = _graph_arrays(candidate, label="mutated graph")
        _validate_graph_invariants(
            candidate_arrays,
            forks=attempted,
            max_outdegree=2,
        )
        candidate_hash = logical_graph_sha256(candidate)
        if candidate_hash == fallback_hash:
            raise ValueError("nonempty fork transaction produced no logical graph change")
    except Exception as exc:  # The whole-movie transaction deliberately fails closed.
        return _rollback(
            fallback,
            attempted,
            exc,
            fallback_hash=fallback_hash,
        )

    return DivisionMutationResult(
        graph=candidate,
        fallback_graph_sha256=fallback_hash,
        candidate_graph_sha256=candidate_hash,
        attempted_forks=attempted,
        applied_forks=attempted,
        rolled_back=False,
        diagnostic=None,
        restored_node_ids=tuple(sorted(restored)),
        removed_edges=tuple(sorted(removed)),
        added_edges=tuple(sorted(added)),
    )


def _predicted_triplets(graph: TrackingGraph) -> tuple[tuple[int, int, int], ...]:
    arrays = _graph_arrays(graph, label="predicted graph")
    successors: dict[int, list[int]] = {}
    time_by_id = {
        int(node_id): int(time) for node_id, time in zip(arrays.node_ids, arrays.t, strict=True)
    }
    for raw_source, raw_target in arrays.edges:
        source = int(raw_source)
        target = int(raw_target)
        successors.setdefault(source, []).append(target)
    triplets: list[tuple[int, int, int]] = []
    for mother, children in sorted(successors.items()):
        daughters = tuple(sorted(children))
        if len(daughters) < 2:
            continue
        if len(daughters) != 2:
            raise ValueError("predicted division mother must have exactly two daughters")
        if any(time_by_id[daughter] != time_by_id[mother] + 1 for daughter in daughters):
            raise ValueError("predicted division daughters must occur at mother t+1")
        triplets.append((mother, daughters[0], daughters[1]))
    return tuple(triplets)


def _evaluate_from_matches(
    pred: TrackingGraph,
    truth: TrackingGraph,
    matches: Mapping[int, int],
    *,
    division_weight: float = DIVISION_WEIGHT,
) -> MetricResult:
    """Reproduce ``metric.evaluate`` from one already-computed node match map."""
    pred_edges = {tuple(map(int, edge)) for edge in np.asarray(pred.edges).reshape(-1, 2)}
    truth_edges = {tuple(map(int, edge)) for edge in np.asarray(truth.edges).reshape(-1, 2)}
    edge_tp = edge_fp = edge_ignored = 0
    for source, target in pred_edges:
        if source in matches and target in matches:
            if (matches[source], matches[target]) in truth_edges:
                edge_tp += 1
            else:
                edge_fp += 1
        else:
            edge_ignored += 1
    edge_fn = len(truth_edges) - edge_tp
    edge_jaccard = _count_jaccard(edge_tp, edge_fp, edge_fn)

    truth_outdegree: dict[int, int] = {}
    pred_outdegree: dict[int, int] = {}
    for source, _ in truth_edges:
        truth_outdegree[source] = truth_outdegree.get(source, 0) + 1
    for source, _ in pred_edges:
        pred_outdegree[source] = pred_outdegree.get(source, 0) + 1
    truth_divisions_set = {
        node_id for node_id, degree in truth_outdegree.items() if degree >= 2
    }
    pred_divisions = {node_id for node_id, degree in pred_outdegree.items() if degree >= 2}
    matched_truth_divisions: set[int] = set()
    division_fp = 0
    for node_id in pred_divisions:
        mapped = matches.get(node_id)
        if mapped is None:
            continue
        if mapped in truth_divisions_set:
            matched_truth_divisions.add(mapped)
        else:
            division_fp += 1
    division_tp = len(matched_truth_divisions)
    division_fn = len(truth_divisions_set) - division_tp
    division_jaccard = _count_jaccard(division_tp, division_fp, division_fn)
    return MetricResult(
        score=(1.0 - division_weight) * edge_jaccard
        + division_weight * division_jaccard,
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        edge_ignored=edge_ignored,
        division_tp=division_tp,
        division_fp=division_fp,
        division_fn=division_fn,
        matched_nodes=len(matches),
        predicted_nodes=len(pred.node_ids),
        truth_nodes=len(truth.node_ids),
    )


def _evaluate_strict_from_matches(
    pred: TrackingGraph,
    truth: TrackingGraph,
    matches: Mapping[int, int],
) -> StrictTripletResult:
    predicted = _predicted_triplets(pred)
    truth_pairs = truth_divisions(truth)

    details: list[ForkMatch] = []
    matched_truth: set[tuple[int, int, int]] = set()
    matched_false_positives = 0
    wrong_forks = 0
    unmatched_forks = 0
    for mother, first, second in predicted:
        mapped_mother = matches.get(mother)
        mapped_first = matches.get(first)
        mapped_second = matches.get(second)
        mapped = (mapped_mother, mapped_first, mapped_second)
        truth_pair = truth_pairs.get(mapped_mother) if mapped_mother is not None else None
        truth_triplet = (
            (mapped_mother, truth_pair[0], truth_pair[1])
            if mapped_mother is not None and truth_pair is not None
            else None
        )
        if (
            truth_pair is not None
            and mapped_first is not None
            and mapped_second is not None
            and {mapped_first, mapped_second} == set(truth_pair)
        ):
            classification: ForkClassification = "true_positive"
            matched_truth.add((mapped_mother, truth_pair[0], truth_pair[1]))
        elif mapped_mother is not None and truth_pair is None:
            classification = "matched_division_false_positive"
            matched_false_positives += 1
        elif truth_pair is not None and mapped_first is not None and mapped_second is not None:
            classification = "fully_matched_wrong_fork"
            wrong_forks += 1
        else:
            classification = "unmatched_fork"
            unmatched_forks += 1
        details.append(
            ForkMatch(
                predicted_triplet=(mother, first, second),
                mapped_triplet=mapped,
                truth_triplet=truth_triplet,
                classification=classification,
            )
        )

    tp = len(matched_truth)
    return StrictTripletResult(
        tp=tp,
        fp=len(predicted) - tp,
        fn=len(truth_pairs) - tp,
        matched_division_false_positives=matched_false_positives,
        fully_matched_wrong_forks=wrong_forks,
        unmatched_forks=unmatched_forks,
        matched_truth_triplets=tuple(sorted(matched_truth)),
        details=tuple(details),
    )


def evaluate_strict_triplets(
    pred: TrackingGraph,
    truth: TrackingGraph,
    *,
    scale: dict[str, float] = VOXEL_SCALE_UM,
    max_distance: float = MATCH_RADIUS_UM,
) -> StrictTripletResult:
    """Evaluate forks only when mother and both daughters match one truth triplet."""
    canonical_pred = _canonical_graph(pred, label="predicted graph")
    canonical_truth = _canonical_graph(truth, label="truth graph")
    matches = match_nodes(
        canonical_pred,
        canonical_truth,
        scale=scale,
        max_distance=max_distance,
    )
    return _evaluate_strict_from_matches(canonical_pred, canonical_truth, matches)


def evaluate_division_movie(
    dataset: str,
    fallback: TrackingGraph,
    mutation: DivisionMutationResult,
    truth: TrackingGraph,
    *,
    estimated_nodes: int,
    scale: dict[str, float] = VOXEL_SCALE_UM,
    max_distance: float = MATCH_RADIUS_UM,
) -> MovieEvaluation:
    """Evaluate one fallback/candidate pair for later streaming aggregation."""
    if not isinstance(mutation, DivisionMutationResult):
        raise TypeError("movie evaluation requires a DivisionMutationResult")
    if mutation.rolled_back:
        raise ValueError(f"cannot evaluate rolled-back division mutation: {mutation.diagnostic}")
    canonical_fallback = _canonical_graph(fallback, label="fallback graph")
    canonical_candidate = _canonical_graph(mutation.graph, label="candidate graph")
    canonical_truth = _canonical_graph(truth, label="truth graph")
    fallback_hash = logical_graph_sha256(canonical_fallback)
    candidate_hash = logical_graph_sha256(canonical_candidate)
    truth_hash = logical_graph_sha256(canonical_truth)
    if mutation.fallback_graph_sha256 != fallback_hash:
        raise ValueError("movie evaluation received an unrelated fallback graph")
    if mutation.candidate_graph_sha256 != candidate_hash:
        raise ValueError("movie evaluation candidate graph disagrees with mutation identity")
    if any(
        fork.fallback_graph_sha256 != fallback_hash for fork in mutation.applied_forks
    ):
        raise ValueError("movie evaluation mutation forks reference another fallback graph")

    baseline_matches = match_nodes(
        canonical_fallback,
        canonical_truth,
        scale=scale,
        max_distance=max_distance,
    )
    candidate_matches = match_nodes(
        canonical_candidate,
        canonical_truth,
        scale=scale,
        max_distance=max_distance,
    )
    baseline_metric = _evaluate_from_matches(
        canonical_fallback,
        canonical_truth,
        baseline_matches,
    )
    candidate_metric = _evaluate_from_matches(
        canonical_candidate,
        canonical_truth,
        candidate_matches,
    )
    baseline_triplets = _evaluate_strict_from_matches(
        canonical_fallback,
        canonical_truth,
        baseline_matches,
    )
    candidate_triplets = _evaluate_strict_from_matches(
        canonical_candidate,
        canonical_truth,
        candidate_matches,
    )
    return MovieEvaluation(
        dataset=dataset,
        fallback_graph_sha256=fallback_hash,
        candidate_graph_sha256=candidate_hash,
        truth_graph_sha256=truth_hash,
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        baseline_triplets=baseline_triplets,
        candidate_triplets=candidate_triplets,
        accepted_forks=len(mutation.applied_forks),
        estimated_nodes=estimated_nodes,
        restored_node_count=mutation.restored_node_count,
        removed_edge_count=mutation.removed_edge_count,
        added_edge_count=mutation.added_edge_count,
    )


def _wins_ties_losses(values: np.ndarray) -> dict[str, int]:
    wins = int(np.sum(values > 0.0))
    losses = int(np.sum(values < 0.0))
    return {"wins": wins, "ties": len(values) - wins - losses, "losses": losses}


def aggregate_acceptance(
    rows: Iterable[MovieEvaluation],
    *,
    expected_datasets: Sequence[str],
    no_division_datasets: Iterable[str],
    expected_truth_graph_sha256: Mapping[str, str],
) -> AcceptanceSummary:
    """Apply every frozen acceptance guard to one exact caller-supplied cohort."""
    materialized = tuple(rows)
    expected = tuple(expected_datasets)
    controls = frozenset(no_division_datasets)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(not isinstance(dataset, str) or not dataset for dataset in expected)
    ):
        raise ValueError("expected_datasets must contain unique non-empty names")
    if len(expected) != int(ACCEPTANCE_COHORT_CONTRACT["movie_count"]):
        raise ValueError("expected_datasets does not match the frozen cohort size")
    if len(controls) != int(ACCEPTANCE_COHORT_CONTRACT["no_division_movies"]):
        raise ValueError("no_division_datasets does not match the frozen control cohort size")
    if not controls <= set(expected):
        raise ValueError("no_division_datasets must be a subset of expected_datasets")
    if not isinstance(expected_truth_graph_sha256, Mapping) or set(
        expected_truth_graph_sha256
    ) != set(expected):
        raise ValueError("truth graph SHA-256 mapping must match the exact expected cohort")
    expected_truth = {
        dataset: _require_sha256(
            expected_truth_graph_sha256[dataset],
            field=f"{dataset} truth graph SHA-256",
        )
        for dataset in expected
    }
    if any(not isinstance(row, MovieEvaluation) for row in materialized):
        raise TypeError("acceptance aggregation requires MovieEvaluation rows")
    if tuple(row.dataset for row in materialized) != expected:
        raise ValueError("movie rows do not contain the exact expected cohort")
    if any(row.truth_graph_sha256 != expected_truth[row.dataset] for row in materialized):
        raise ValueError("movie row truth graph SHA-256 disagrees with the expected cohort")

    observed_controls = {
        row.dataset for row in materialized if row.truth_division_count == 0
    }
    division_rows = [row for row in materialized if row.truth_division_count > 0]
    if observed_controls != controls:
        raise ValueError("caller control cohort disagrees with strict truth counts")
    if len(division_rows) != int(ACCEPTANCE_COHORT_CONTRACT["division_movies"]):
        raise ValueError("movie rows do not contain the frozen number of division movies")
    if sum(row.truth_division_count for row in materialized) != int(
        ACCEPTANCE_COHORT_CONTRACT["labelled_divisions"]
    ):
        raise ValueError("movie rows do not contain the frozen labelled division total")

    full_deltas = np.asarray(
        [row.candidate_metric.score - row.baseline_metric.score for row in materialized],
        dtype=np.float64,
    )
    edge_deltas = np.asarray(
        [
            row.candidate_metric.edge_jaccard - row.baseline_metric.edge_jaccard
            for row in materialized
        ],
        dtype=np.float64,
    )
    estimated_nodes = sum(row.estimated_nodes for row in materialized)
    baseline_ratio = (
        sum(row.baseline_metric.predicted_nodes for row in materialized) / estimated_nodes
    )
    candidate_ratio = (
        sum(row.candidate_metric.predicted_nodes for row in materialized) / estimated_nodes
    )
    node_ratio_delta = float(candidate_ratio - baseline_ratio)
    full_counts = _wins_ties_losses(full_deltas)
    edge_counts = _wins_ties_losses(edge_deltas)
    wrong_forks = sum(
        row.candidate_triplets.fully_matched_wrong_forks for row in materialized
    )
    strict_recoveries = sum(row.strict_true_recoveries for row in materialized)
    control_false_positives = {
        row.dataset: row.candidate_triplets.matched_division_false_positives
        for row in materialized
        if row.dataset in controls
    }
    no_fork_identity = all(
        row.accepted_forks != 0
        or row.candidate_graph_sha256 == row.fallback_graph_sha256
        for row in materialized
    )
    mean_full_delta = float(full_deltas.mean())
    mean_edge_delta = float(edge_deltas.mean())
    guards = {
        "aggregate_node_ratio_delta": node_ratio_delta
        <= float(ACCEPTANCE_GUARDS["aggregate_node_ratio_delta"]["value"]),
        "complete_cohort_movies": len(materialized)
        == int(ACCEPTANCE_GUARDS["complete_cohort_movies"]["value"]),
        "fully_matched_wrong_forks": wrong_forks
        == int(ACCEPTANCE_GUARDS["fully_matched_wrong_forks"]["value"]),
        "full_score_wins_vs_losses": full_counts["wins"] > full_counts["losses"],
        "matched_division_false_positives_per_no_division_movie": all(
            value
            == int(
                ACCEPTANCE_GUARDS[
                    "matched_division_false_positives_per_no_division_movie"
                ]["value"]
            )
            for value in control_false_positives.values()
        ),
        "mean_edge_jaccard_delta": mean_edge_delta
        >= float(ACCEPTANCE_GUARDS["mean_edge_jaccard_delta"]["value"]),
        "mean_full_score_delta": mean_full_delta
        >= float(ACCEPTANCE_GUARDS["mean_full_score_delta"]["value"]),
        "no_fork_fallback_identity": no_fork_identity,
        "strict_true_recoveries": strict_recoveries
        >= int(ACCEPTANCE_GUARDS["strict_true_recoveries"]["value"]),
    }
    if set(guards) != set(ACCEPTANCE_GUARDS):
        raise RuntimeError("acceptance implementation disagrees with the frozen guard contract")

    worst_index = min(
        range(len(materialized)),
        key=lambda index: (full_deltas[index], materialized[index].dataset),
    )
    metrics: dict[str, float | int | bool] = {
        "mean_full_score_delta": mean_full_delta,
        "mean_edge_jaccard_delta": mean_edge_delta,
        "aggregate_node_ratio_delta": node_ratio_delta,
        "fully_matched_wrong_forks": wrong_forks,
        "strict_true_recoveries": strict_recoveries,
        "no_fork_fallback_identity": no_fork_identity,
    }
    diagnostics = {
        "median_full_score_delta": float(np.median(full_deltas)),
        "median_edge_jaccard_delta": float(np.median(edge_deltas)),
        "edge_wins_ties_losses": edge_counts,
        "full_score_wins_ties_losses": full_counts,
        "worst_movie": {
            "dataset": materialized[worst_index].dataset,
            "full_score_delta": float(full_deltas[worst_index]),
        },
        "micro_division_counts": {
            "tp": sum(row.candidate_metric.division_tp for row in materialized),
            "fp": sum(row.candidate_metric.division_fp for row in materialized),
            "fn": sum(row.candidate_metric.division_fn for row in materialized),
        },
        "micro_triplet_counts": {
            "tp": sum(row.candidate_triplets.tp for row in materialized),
            "fp": sum(row.candidate_triplets.fp for row in materialized),
            "fn": sum(row.candidate_triplets.fn for row in materialized),
        },
        "matched_division_false_positives_by_control": control_false_positives,
        "unmatched_accepted_forks": sum(
            row.candidate_triplets.unmatched_forks for row in materialized
        ),
        "restored_nodes": sum(row.restored_node_count for row in materialized),
        "removed_edges": sum(row.removed_edge_count for row in materialized),
        "added_edges": sum(row.added_edge_count for row in materialized),
        "accepted_forks": sum(row.accepted_forks for row in materialized),
    }
    return AcceptanceSummary(
        passed=all(guards.values()),
        guards=guards,
        metrics=metrics,
        diagnostics=diagnostics,
    )


__all__ = [
    "MAX_FORKS_PER_MOVIE",
    "PROBABILITY_THRESHOLDS",
    "AcceptanceSummary",
    "AcceptedFork",
    "DivisionMutationResult",
    "ForkMatch",
    "MovieEvaluation",
    "ScoredProposal",
    "StrictTripletResult",
    "aggregate_acceptance",
    "apply_division_forks",
    "evaluate_division_movie",
    "evaluate_strict_triplets",
    "logit_to_probability",
    "scored_proposals_from_logits",
    "select_division_forks",
]
