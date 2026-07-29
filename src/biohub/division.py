"""Structured mother-to-daughter proposals for learned division recovery.

The accepted linker emits simple paths, so one mother can have only one linked
daughter.  This module exposes the triplets needed by a later appearance model:
the original linked-daughter plus orphan diagnostic and a broader mother plus
unordered candidate-pair representation in the next frame.

Proposal generation is deliberately broader than final fork acceptance.  Sparse
labels make unmatched candidates unknown, and Phase 7 showed that geometry alone
cannot safely decide whether a fork is real.  Callers can inspect the geometric
and track-context fields, then learn or audit a conservative acceptance rule.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from biohub.constants import (
    DIVISION_MAX_UM,
    DIVISION_MIN_ANGLE_DEG,
    DIVISION_MIN_CHILD_UM,
    VOXEL_SCALE_UM,
)
from biohub.metric import TrackingGraph


@dataclass(frozen=True)
class DivisionProposal:
    """One possible mother, linked-daughter, orphan-daughter triplet."""

    mother_id: int
    linked_daughter_id: int
    orphan_daughter_id: int
    daughter_time: int
    linked_distance_um: float
    orphan_distance_um: float
    daughter_angle_deg: float
    mother_has_history: bool
    orphan_has_future: bool


@dataclass(frozen=True)
class DivisionCandidateContext:
    """One possible daughter and its context in the filtered linked graph."""

    node_id: int
    distance_um: float
    displacement_um: tuple[float, float, float]
    present_in_linked_graph: bool
    is_current_child: bool
    is_parentless: bool
    has_wrong_parent: bool
    has_future: bool
    survives_pruning: bool


@dataclass(frozen=True)
class DivisionPairNeighborhood:
    """A filtered mother and nearby candidate daughters in the next frame."""

    mother_id: int
    daughter_time: int
    mother_has_history: bool
    mother_has_current_child: bool
    mother_survives_pruning: bool
    search_max_distance_um: float
    candidates: tuple[DivisionCandidateContext, ...]


@dataclass(frozen=True)
class DivisionPairProposal:
    """One mother and one canonical unordered pair of candidate daughters."""

    mother_id: int
    daughter_time: int
    candidate_1: DivisionCandidateContext
    candidate_2: DivisionCandidateContext
    daughter_angle_deg: float
    mother_has_history: bool
    mother_has_current_child: bool
    mother_survives_pruning: bool


def _as_int64_array(values, error: str) -> np.ndarray:
    array = np.asarray(values)
    int64 = np.iinfo(np.int64)
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(error)
    if np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.unsignedinteger) and np.any(array > int64.max):
            raise ValueError(error)
        return array.astype(np.int64)
    if np.issubdtype(array.dtype, np.floating):
        if (
            not np.all(np.isfinite(array))
            or not np.all(array == np.floor(array))
            or np.any(array < -(2**63))
            or np.any(array >= 2**63)
        ):
            raise ValueError(error)
        return array.astype(np.int64)

    converted: list[int] = []
    for value in array.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, Real)):
            raise ValueError(error)
        numeric = float(value)
        integer = int(value)
        if not np.isfinite(numeric) or numeric != integer or not int64.min <= integer <= int64.max:
            raise ValueError(error)
        converted.append(integer)
    return np.asarray(converted, dtype=np.int64).reshape(array.shape)


def _validate_graph(graph: TrackingGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_node_ids = np.asarray(graph.node_ids)
    if raw_node_ids.ndim != 1:
        raise ValueError("division proposal graph has invalid node identifiers")
    node_ids = _as_int64_array(
        raw_node_ids,
        "division proposal graph has invalid node identifiers",
    )
    expected = (len(node_ids),)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("division proposal graph contains duplicate node identifiers")
    coordinates = []
    for name in ("z", "y", "x"):
        try:
            values = np.asarray(getattr(graph, name), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"division proposal graph has invalid {name} coordinates"
            ) from exc
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"division proposal graph has invalid {name} coordinates")
        coordinates.append(values)
    raw_t = np.asarray(graph.t)
    if raw_t.shape != expected:
        raise ValueError("division proposal graph has invalid time coordinates")
    t = _as_int64_array(
        raw_t,
        "division proposal graph has invalid time coordinates",
    )
    raw_edges = np.asarray(graph.edges)
    if raw_edges.size % 2:
        raise ValueError("division proposal graph has invalid edge identifiers")
    try:
        raw_edges = raw_edges.reshape(-1, 2)
    except ValueError as exc:
        raise ValueError("division proposal graph has invalid edge identifiers") from exc
    edges = _as_int64_array(
        raw_edges,
        "division proposal graph has invalid edge identifiers",
    )
    known = set(int(node_id) for node_id in node_ids)
    if any(int(source) not in known or int(target) not in known for source, target in edges):
        raise ValueError("division proposal graph has an edge whose endpoint is missing")
    return node_ids, t, np.column_stack(coordinates)


def _oriented_adjacency(
    node_ids: np.ndarray,
    t: np.ndarray,
    edges: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, int]]:
    t_by_id = {int(node_id): int(time) for node_id, time in zip(node_ids, t, strict=True)}
    successors: dict[int, list[int]] = {}
    in_degree: dict[int, int] = {}
    for source, target in edges:
        source = int(source)
        target = int(target)
        if t_by_id[target] < t_by_id[source]:
            source, target = target, source
        successors.setdefault(source, []).append(target)
        in_degree[target] = in_degree.get(target, 0) + 1
    for children in successors.values():
        children.sort()
    return successors, in_degree


def _validate_scale(scale: dict[str, float]) -> np.ndarray:
    if set(scale) != {"z", "y", "x"} or any(
        not np.isfinite(float(scale[axis])) or float(scale[axis]) <= 0.0
        for axis in ("z", "y", "x")
    ):
        raise ValueError("scale must contain positive finite z, y, and x values")
    return np.array([scale["z"], scale["y"], scale["x"]], dtype=float)


def _validate_pair_rule(
    *,
    min_distance_um: float,
    max_distance_um: float,
    min_angle_deg: float,
    max_candidates_per_mother: int | None,
) -> None:
    if not np.isfinite(min_distance_um) or min_distance_um < 0.0:
        raise ValueError("min_distance_um must be non-negative and finite")
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("max_distance_um must be positive and finite")
    if min_distance_um > max_distance_um:
        raise ValueError("min_distance_um cannot exceed max_distance_um")
    if not np.isfinite(min_angle_deg) or not 0.0 <= min_angle_deg <= 180.0:
        raise ValueError("min_angle_deg must be finite and between 0 and 180")
    if max_candidates_per_mother is not None and (
        isinstance(max_candidates_per_mother, bool)
        or not isinstance(max_candidates_per_mother, int)
        or max_candidates_per_mother < 2
    ):
        raise ValueError("max_candidates_per_mother must be None or an integer of at least 2")


def generate_division_pair_neighborhoods(
    linked: TrackingGraph,
    candidates: TrackingGraph,
    *,
    pruned_node_ids: Collection[int] = frozenset(),
    max_distance_um: float = 16.0,
    scale: dict[str, float] = VOXEL_SCALE_UM,
) -> list[DivisionPairNeighborhood]:
    """Build local next-frame candidate neighborhoods around filtered mothers.

    ``linked`` supplies the mother population and all current ownership context.
    ``candidates`` may be the same filtered population or the broader raw-scored
    population. Nodes absent from ``linked`` retain explicit absent context rather
    than being described as parentless.
    """
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("max_distance_um must be positive and finite")
    scale_array = _validate_scale(scale)
    linked_ids, linked_t, linked_voxels = _validate_graph(linked)
    candidate_ids, candidate_t, candidate_voxels = _validate_graph(candidates)
    linked_id_set = {int(node_id) for node_id in linked_ids}
    candidate_id_set = {int(node_id) for node_id in candidate_ids}
    if not linked_id_set <= candidate_id_set:
        raise ValueError("candidate graph is missing nodes from the linked graph")

    pruned_ids = {int(node_id) for node_id in pruned_node_ids}
    if not pruned_ids <= linked_id_set:
        raise ValueError("pruned node identifiers must belong to the linked graph")
    if len(linked_ids) == 0:
        return []

    linked_row_of = {int(node_id): row for row, node_id in enumerate(linked_ids)}
    candidate_row_of = {
        int(node_id): row for row, node_id in enumerate(candidate_ids)
    }
    for node_id in linked_id_set:
        linked_row = linked_row_of[node_id]
        candidate_row = candidate_row_of[node_id]
        if (
            linked_t[linked_row] != candidate_t[candidate_row]
            or not np.array_equal(
                linked_voxels[linked_row],
                candidate_voxels[candidate_row],
            )
        ):
            raise ValueError("shared node coordinates differ between linked and candidates")

    linked_physical = linked_voxels * scale_array
    candidate_physical = candidate_voxels * scale_array
    linked_time_of = {
        int(node_id): int(time)
        for node_id, time in zip(linked_ids, linked_t, strict=True)
    }
    successors, in_degree = _oriented_adjacency(
        linked_ids,
        linked_t,
        np.asarray(linked.edges, dtype=np.int64).reshape(-1, 2),
    )

    candidate_ids_by_time: dict[int, np.ndarray] = {}
    candidate_trees: dict[int, cKDTree] = {}
    for time in sorted(set(int(value) for value in candidate_t)):
        rows = np.flatnonzero(np.asarray(candidate_t) == time)
        ids = candidate_ids[rows]
        order = np.argsort(ids, kind="stable")
        rows = rows[order]
        ids = ids[order]
        candidate_ids_by_time[time] = ids
        candidate_trees[time] = cKDTree(candidate_physical[rows])

    neighborhoods: list[DivisionPairNeighborhood] = []
    for mother_id in sorted(linked_id_set):
        daughter_time = linked_time_of[mother_id] + 1
        tree = candidate_trees.get(daughter_time)
        if tree is None:
            continue
        mother_position = linked_physical[linked_row_of[mother_id]]
        ids = candidate_ids_by_time[daughter_time]
        nearby_rows = tree.query_ball_point(mother_position, r=max_distance_um)
        contexts: list[DivisionCandidateContext] = []
        current_children = set(successors.get(mother_id, []))
        for nearby_row in nearby_rows:
            candidate_id = int(ids[int(nearby_row)])
            candidate_position = candidate_physical[candidate_row_of[candidate_id]]
            displacement = candidate_position - mother_position
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0 or distance > max_distance_um:
                continue
            present = candidate_id in linked_id_set
            current_child = candidate_id in current_children
            contexts.append(
                DivisionCandidateContext(
                    node_id=candidate_id,
                    distance_um=distance,
                    displacement_um=tuple(float(value) for value in displacement),
                    present_in_linked_graph=present,
                    is_current_child=current_child,
                    is_parentless=present and in_degree.get(candidate_id, 0) == 0,
                    has_wrong_parent=(
                        present
                        and in_degree.get(candidate_id, 0) > 0
                        and not current_child
                    ),
                    has_future=present and bool(successors.get(candidate_id)),
                    survives_pruning=candidate_id in pruned_ids,
                )
            )
        contexts.sort(key=lambda item: (item.distance_um, item.node_id))
        if len(contexts) < 2:
            continue
        neighborhoods.append(
            DivisionPairNeighborhood(
                mother_id=mother_id,
                daughter_time=daughter_time,
                mother_has_history=in_degree.get(mother_id, 0) > 0,
                mother_has_current_child=bool(current_children),
                mother_survives_pruning=mother_id in pruned_ids,
                search_max_distance_um=float(max_distance_um),
                candidates=tuple(contexts),
            )
        )
    return neighborhoods


def generate_division_pair_proposals(
    neighborhoods: Collection[DivisionPairNeighborhood],
    *,
    min_distance_um: float = 0.0,
    max_distance_um: float = 16.0,
    min_angle_deg: float = 0.0,
    max_candidates_per_mother: int | None = None,
) -> list[DivisionPairProposal]:
    """Apply local volume gates and emit deterministic unordered daughter pairs."""
    _validate_pair_rule(
        min_distance_um=min_distance_um,
        max_distance_um=max_distance_um,
        min_angle_deg=min_angle_deg,
        max_candidates_per_mother=max_candidates_per_mother,
    )
    proposals: list[DivisionPairProposal] = []
    for neighborhood in neighborhoods:
        if max_distance_um > neighborhood.search_max_distance_um:
            raise ValueError("pair max_distance_um exceeds the neighborhood search radius")
        candidates = [
            candidate
            for candidate in neighborhood.candidates
            if min_distance_um <= candidate.distance_um <= max_distance_um
        ]
        if max_candidates_per_mother is not None:
            candidates = candidates[:max_candidates_per_mother]
        for first, second in combinations(candidates, 2):
            first_vector = np.asarray(first.displacement_um, dtype=float)
            second_vector = np.asarray(second.displacement_um, dtype=float)
            cosine = float(
                np.dot(first_vector, second_vector)
                / (first.distance_um * second.distance_um)
            )
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if angle < min_angle_deg:
                continue
            candidate_1, candidate_2 = sorted((first, second), key=lambda item: item.node_id)
            proposals.append(
                DivisionPairProposal(
                    mother_id=neighborhood.mother_id,
                    daughter_time=neighborhood.daughter_time,
                    candidate_1=candidate_1,
                    candidate_2=candidate_2,
                    daughter_angle_deg=angle,
                    mother_has_history=neighborhood.mother_has_history,
                    mother_has_current_child=neighborhood.mother_has_current_child,
                    mother_survives_pruning=neighborhood.mother_survives_pruning,
                )
            )
    return sorted(
        proposals,
        key=lambda proposal: (
            proposal.mother_id,
            proposal.candidate_1.node_id,
            proposal.candidate_2.node_id,
        ),
    )


def generate_division_proposals(
    linked: TrackingGraph,
    *,
    max_distance_um: float = 14.0,
    scale: dict[str, float] = VOXEL_SCALE_UM,
) -> list[DivisionProposal]:
    """Enumerate broad, deterministic triplets from a one-successor linked graph.

    A proposal requires a mother with exactly one child in the next frame and a
    parentless node in that same daughter frame.  Both possible daughters must be
    within ``max_distance_um`` of the mother.  No minimum jump, angle, history, or
    future gate is applied here because those rules would erase evidence before a
    learned scorer or audit can inspect it.

    The default 14 micrometers covers the maximum labelled mother-to-daughter jump
    measured in Phase 7 while remaining a local spatial gate.
    """
    if not np.isfinite(max_distance_um) or max_distance_um <= 0.0:
        raise ValueError("max_distance_um must be positive and finite")
    scale_array = _validate_scale(scale)

    node_ids, t, voxel_coordinates = _validate_graph(linked)
    if len(node_ids) == 0:
        return []
    edges = np.asarray(linked.edges, dtype=np.int64).reshape(-1, 2)
    physical_coordinates = voxel_coordinates * scale_array
    row_of = {int(node_id): row for row, node_id in enumerate(node_ids)}
    time_of = {int(node_id): int(time) for node_id, time in zip(node_ids, t, strict=True)}
    successors, in_degree = _oriented_adjacency(node_ids, t, edges)

    orphan_ids_by_time: dict[int, np.ndarray] = {}
    orphan_trees: dict[int, cKDTree] = {}
    for time in sorted(set(int(value) for value in t)):
        ids = np.array(
            sorted(
                int(node_id)
                for node_id in node_ids
                if int(time_of[int(node_id)]) == time and in_degree.get(int(node_id), 0) == 0
            ),
            dtype=np.int64,
        )
        if len(ids):
            rows = np.array([row_of[int(node_id)] for node_id in ids], dtype=np.int64)
            orphan_ids_by_time[time] = ids
            orphan_trees[time] = cKDTree(physical_coordinates[rows])

    proposals: list[DivisionProposal] = []
    for mother in sorted(int(node_id) for node_id in node_ids):
        children = successors.get(mother, [])
        if len(children) != 1:
            continue
        linked_daughter = children[0]
        daughter_time = time_of[linked_daughter]
        if daughter_time != time_of[mother] + 1 or daughter_time not in orphan_trees:
            continue

        mother_position = physical_coordinates[row_of[mother]]
        linked_vector = physical_coordinates[row_of[linked_daughter]] - mother_position
        linked_distance = float(np.linalg.norm(linked_vector))
        if not 0.0 < linked_distance <= max_distance_um:
            continue

        orphan_ids = orphan_ids_by_time[daughter_time]
        nearby_rows = orphan_trees[daughter_time].query_ball_point(
            mother_position,
            r=max_distance_um,
        )
        for orphan_row in sorted(nearby_rows, key=lambda row: int(orphan_ids[row])):
            orphan = int(orphan_ids[orphan_row])
            if orphan == linked_daughter:
                continue
            orphan_vector = physical_coordinates[row_of[orphan]] - mother_position
            orphan_distance = float(np.linalg.norm(orphan_vector))
            if orphan_distance <= 0.0:
                continue
            cosine = float(
                np.dot(linked_vector, orphan_vector) / (linked_distance * orphan_distance)
            )
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            proposals.append(
                DivisionProposal(
                    mother_id=mother,
                    linked_daughter_id=linked_daughter,
                    orphan_daughter_id=orphan,
                    daughter_time=daughter_time,
                    linked_distance_um=linked_distance,
                    orphan_distance_um=orphan_distance,
                    daughter_angle_deg=angle,
                    mother_has_history=in_degree.get(mother, 0) > 0,
                    orphan_has_future=bool(successors.get(orphan)),
                )
            )

    return sorted(
        proposals,
        key=lambda proposal: (
            proposal.mother_id,
            -proposal.daughter_angle_deg,
            proposal.orphan_distance_um,
            proposal.orphan_daughter_id,
        ),
    )


def passes_phase7_gates(proposal: DivisionProposal) -> bool:
    """Return whether a broad proposal passes the dormant Phase 7 geometry rule."""
    return (
        proposal.linked_distance_um >= DIVISION_MIN_CHILD_UM
        and proposal.linked_distance_um <= DIVISION_MAX_UM
        and proposal.orphan_distance_um >= DIVISION_MIN_CHILD_UM
        and proposal.orphan_distance_um <= DIVISION_MAX_UM
        and proposal.daughter_angle_deg >= DIVISION_MIN_ANGLE_DEG
        and proposal.mother_has_history
        and proposal.orphan_has_future
    )
