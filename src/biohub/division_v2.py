"""Bounded Phase 11 v2 proposals and fit-only cross-validation primitives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from biohub.constants import VOXEL_SCALE_UM
from biohub.division import (
    DivisionCandidateContext,
    DivisionPairNeighborhood,
    DivisionPairProposal,
    _oriented_adjacency,
    _validate_graph,
    _validate_scale,
)
from biohub.metric import TrackingGraph

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without the ML extra
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_TorchModule = object if nn is None else nn.Module

V2_FOLD_SALT = "biohub-phase11-v2-movie-fold-v1"
V2_MODEL_NAMES = ("context_linear", "full_linear", "full_mlp16")


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _require_finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite and numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class V2ProposalRule:
    """A strict two-tier proposal rule with an exact per-mother bound."""

    max_distance_um: float = 18.0
    base_candidate_cap: int = 12
    anchor_candidate_cap: int = 6
    scan_candidate_cap: int = 24
    max_pairs_per_mother: int = 138

    def __post_init__(self) -> None:
        max_distance = _require_finite(
            self.max_distance_um,
            field="max_distance_um",
            minimum=np.finfo(float).tiny,
        )
        base = _require_int(self.base_candidate_cap, field="base_candidate_cap", minimum=2)
        anchors = _require_int(
            self.anchor_candidate_cap,
            field="anchor_candidate_cap",
            minimum=1,
        )
        scanned = _require_int(
            self.scan_candidate_cap,
            field="scan_candidate_cap",
            minimum=base,
        )
        maximum = _require_int(
            self.max_pairs_per_mother,
            field="max_pairs_per_mother",
            minimum=1,
        )
        if anchors > base:
            raise ValueError("anchor_candidate_cap cannot exceed base_candidate_cap")
        expected = math.comb(base, 2) + anchors * (scanned - base)
        if maximum != expected:
            raise ValueError(
                "max_pairs_per_mother must equal the exact two-tier proposal bound"
            )
        object.__setattr__(self, "max_distance_um", max_distance)
        object.__setattr__(self, "base_candidate_cap", base)
        object.__setattr__(self, "anchor_candidate_cap", anchors)
        object.__setattr__(self, "scan_candidate_cap", scanned)
        object.__setattr__(self, "max_pairs_per_mother", maximum)


DEFAULT_V2_PROPOSAL_RULE = V2ProposalRule()


def generate_targeted_division_pair_neighborhoods(
    linked: TrackingGraph,
    candidates: TrackingGraph,
    *,
    mother_node_ids: Collection[int],
    pruned_node_ids: Collection[int] = frozenset(),
    max_distance_um: float = 18.0,
    scale: dict[str, float] = VOXEL_SCALE_UM,
) -> list[DivisionPairNeighborhood]:
    """Build exact frozen-context neighborhoods for a validated mother subset."""
    maximum = _require_finite(
        max_distance_um,
        field="max_distance_um",
        minimum=np.finfo(float).tiny,
    )
    scale_array = _validate_scale(scale)
    linked_ids, linked_t, linked_voxels = _validate_graph(linked)
    candidate_ids, candidate_t, candidate_voxels = _validate_graph(candidates)
    linked_id_set = {int(node_id) for node_id in linked_ids}
    candidate_id_set = {int(node_id) for node_id in candidate_ids}
    if not linked_id_set <= candidate_id_set:
        raise ValueError("candidate graph is missing nodes from the linked graph")
    if any(
        isinstance(node_id, (bool, np.bool_)) or not isinstance(node_id, Integral)
        for node_id in mother_node_ids
    ):
        raise ValueError("mother node identifiers must be integers")
    selected_mothers = {int(node_id) for node_id in mother_node_ids}
    if not selected_mothers <= linked_id_set:
        raise ValueError("mother node identifiers must belong to the linked graph")
    pruned_ids = {int(node_id) for node_id in pruned_node_ids}
    if not pruned_ids <= linked_id_set:
        raise ValueError("pruned node identifiers must belong to the linked graph")
    if not selected_mothers:
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
    required_times = {linked_time_of[mother] + 1 for mother in selected_mothers}
    candidate_ids_by_time: dict[int, np.ndarray] = {}
    candidate_rows_by_time: dict[int, np.ndarray] = {}
    candidate_trees: dict[int, cKDTree] = {}
    for time in sorted(required_times):
        rows = np.flatnonzero(candidate_t == time)
        if not len(rows):
            continue
        ids = candidate_ids[rows]
        order = np.argsort(ids, kind="stable")
        rows = rows[order]
        ids = ids[order]
        candidate_ids_by_time[time] = ids
        candidate_rows_by_time[time] = rows
        candidate_trees[time] = cKDTree(candidate_physical[rows])

    neighborhoods: list[DivisionPairNeighborhood] = []
    for mother_id in sorted(selected_mothers):
        daughter_time = linked_time_of[mother_id] + 1
        tree = candidate_trees.get(daughter_time)
        if tree is None:
            continue
        mother_position = linked_physical[linked_row_of[mother_id]]
        ids = candidate_ids_by_time[daughter_time]
        rows = candidate_rows_by_time[daughter_time]
        nearby_rows = tree.query_ball_point(mother_position, r=maximum)
        contexts: list[DivisionCandidateContext] = []
        current_children = set(successors.get(mother_id, []))
        for nearby_row in nearby_rows:
            local_row = int(nearby_row)
            candidate_id = int(ids[local_row])
            candidate_position = candidate_physical[int(rows[local_row])]
            displacement = candidate_position - mother_position
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0 or distance > maximum:
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
                search_max_distance_um=maximum,
                candidates=tuple(contexts),
            )
        )
    return neighborhoods


def v2_pair_count(
    candidate_count: int,
    *,
    rule: V2ProposalRule = DEFAULT_V2_PROPOSAL_RULE,
) -> int:
    """Return the exact maximum emitted pairs for one candidate count."""
    if not isinstance(rule, V2ProposalRule):
        raise TypeError("rule must be a V2ProposalRule")
    count = _require_int(candidate_count, field="candidate_count")
    base = min(count, rule.base_candidate_cap)
    scanned = min(count, rule.scan_candidate_cap)
    anchors = min(base, rule.anchor_candidate_cap)
    return math.comb(base, 2) + anchors * max(scanned - rule.base_candidate_cap, 0)


def _validated_candidates(
    neighborhood: DivisionPairNeighborhood,
    *,
    rule: V2ProposalRule,
) -> tuple[DivisionCandidateContext, ...]:
    if not isinstance(neighborhood, DivisionPairNeighborhood):
        raise TypeError("v2 proposal generation requires DivisionPairNeighborhood values")
    if neighborhood.search_max_distance_um < rule.max_distance_um:
        raise ValueError("v2 neighborhood search radius is smaller than the proposal radius")
    if any(not isinstance(value, DivisionCandidateContext) for value in neighborhood.candidates):
        raise TypeError("v2 neighborhoods require DivisionCandidateContext candidates")
    identifiers = [candidate.node_id for candidate in neighborhood.candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("v2 neighborhood contains duplicate candidate identifiers")
    selected: list[DivisionCandidateContext] = []
    for candidate in neighborhood.candidates:
        distance = _require_finite(
            candidate.distance_um,
            field="candidate distance",
            minimum=np.finfo(float).tiny,
        )
        try:
            displacement = np.asarray(candidate.displacement_um, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate displacement must be numeric") from exc
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
            raise ValueError("candidate displacement must contain three finite values")
        if not math.isclose(
            float(np.linalg.norm(displacement)),
            distance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("candidate distance disagrees with its displacement")
        if distance <= rule.max_distance_um:
            selected.append(candidate)
    selected.sort(key=lambda item: (item.distance_um, item.node_id))
    return tuple(selected[: rule.scan_candidate_cap])


def _pair_proposal(
    neighborhood: DivisionPairNeighborhood,
    first: DivisionCandidateContext,
    second: DivisionCandidateContext,
) -> DivisionPairProposal:
    first_vector = np.asarray(first.displacement_um, dtype=float)
    second_vector = np.asarray(second.displacement_um, dtype=float)
    cosine = float(
        np.dot(first_vector, second_vector) / (first.distance_um * second.distance_um)
    )
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    candidate_1, candidate_2 = sorted((first, second), key=lambda value: value.node_id)
    return DivisionPairProposal(
        mother_id=neighborhood.mother_id,
        daughter_time=neighborhood.daughter_time,
        candidate_1=candidate_1,
        candidate_2=candidate_2,
        daughter_angle_deg=angle,
        mother_has_history=neighborhood.mother_has_history,
        mother_has_current_child=neighborhood.mother_has_current_child,
        mother_survives_pruning=neighborhood.mother_survives_pruning,
    )


def generate_division_pair_proposals_v2(
    neighborhoods: Collection[DivisionPairNeighborhood],
    *,
    rule: V2ProposalRule = DEFAULT_V2_PROPOSAL_RULE,
) -> list[DivisionPairProposal]:
    """Preserve the nearest-12 pairs and add only anchor-to-outer pairs."""
    if not isinstance(rule, V2ProposalRule):
        raise TypeError("rule must be a V2ProposalRule")
    proposals: list[DivisionPairProposal] = []
    for neighborhood in neighborhoods:
        candidates = _validated_candidates(neighborhood, rule=rule)
        base = candidates[: rule.base_candidate_cap]
        anchors = base[: rule.anchor_candidate_cap]
        outer = candidates[rule.base_candidate_cap :]
        pairs = [*combinations(base, 2)]
        pairs.extend((anchor, candidate) for anchor in anchors for candidate in outer)
        if len(pairs) != v2_pair_count(len(candidates), rule=rule):
            raise RuntimeError("v2 proposal generation violated its exact pair-count rule")
        if len(pairs) > rule.max_pairs_per_mother:
            raise RuntimeError("v2 proposal generation exceeded its per-mother bound")
        proposals.extend(
            _pair_proposal(neighborhood, first, second) for first, second in pairs
        )
    return sorted(
        proposals,
        key=lambda proposal: (
            proposal.mother_id,
            proposal.candidate_1.node_id,
            proposal.candidate_2.node_id,
        ),
    )


def require_fit_only_training_source(
    dataset: str,
    *,
    source_role: str,
    evaluation_unlocked: bool = False,
) -> None:
    """Reject every training input outside the sealed 6bba fit cohort."""
    if not isinstance(dataset, str) or not dataset.startswith("6bba_"):
        raise ValueError("Phase 11 v2 training accepts only 6bba datasets")
    if source_role != "fit":
        raise ValueError("Phase 11 v2 training is fit only")
    if evaluation_unlocked is not False:
        raise ValueError("evaluation_unlocked must remain false")


@dataclass(frozen=True, slots=True)
class MovieFoldInput:
    """Public metadata used to assign one complete movie to one fit fold."""

    dataset: str
    source_role: str
    truth_event_count: int

    def __post_init__(self) -> None:
        require_fit_only_training_source(self.dataset, source_role=self.source_role)
        count = _require_int(
            self.truth_event_count,
            field="truth_event_count",
        )
        object.__setattr__(self, "truth_event_count", count)


def _fold_rank(dataset: str, *, salt: str) -> bytes:
    if not isinstance(salt, str) or not salt:
        raise ValueError("fold salt must be a non-empty string")
    return hashlib.sha256(salt.encode("ascii") + b"\0" + dataset.encode("ascii")).digest()


def assign_movie_folds(
    movies: Iterable[MovieFoldInput],
    *,
    fold_count: int = 5,
    salt: str = V2_FOLD_SALT,
) -> dict[str, int]:
    """Assign whole fit movies while balancing labelled events and movie counts."""
    folds = _require_int(fold_count, field="fold_count", minimum=2)
    materialized = tuple(movies)
    if not materialized:
        raise ValueError("movie fold assignment requires at least one movie")
    if any(not isinstance(movie, MovieFoldInput) for movie in materialized):
        raise TypeError("movie fold assignment requires MovieFoldInput values")
    datasets = [movie.dataset for movie in materialized]
    if len(set(datasets)) != len(datasets):
        raise ValueError("movie fold assignment contains duplicate datasets")
    if len(materialized) < folds:
        raise ValueError("movie fold assignment requires at least one movie per fold")

    event_totals = [0] * folds
    movie_totals = [0] * folds
    assignment: dict[str, int] = {}
    positives = sorted(
        (movie for movie in materialized if movie.truth_event_count > 0),
        key=lambda movie: (
            -movie.truth_event_count,
            _fold_rank(movie.dataset, salt=salt),
            movie.dataset,
        ),
    )
    controls = sorted(
        (movie for movie in materialized if movie.truth_event_count == 0),
        key=lambda movie: (_fold_rank(movie.dataset, salt=salt), movie.dataset),
    )
    for movie in positives:
        fold = min(
            range(folds),
            key=lambda index: (event_totals[index], movie_totals[index], index),
        )
        assignment[movie.dataset] = fold
        event_totals[fold] += movie.truth_event_count
        movie_totals[fold] += 1
    for movie in controls:
        fold = min(
            range(folds),
            key=lambda index: (movie_totals[index], event_totals[index], index),
        )
        assignment[movie.dataset] = fold
        movie_totals[fold] += 1
    return dict(sorted(assignment.items()))


def event_balanced_listwise_loss(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Average one listwise softmax loss per truth mother."""
    if torch is None:  # pragma: no cover - exercised only without the ML extra
        raise ModuleNotFoundError("listwise ranking requires the ml optional dependency")
    if not all(isinstance(value, torch.Tensor) for value in (logits, group_ids, labels)):
        raise TypeError("listwise loss inputs must be torch tensors")
    if logits.ndim != 1 or group_ids.shape != logits.shape or labels.shape != logits.shape:
        raise ValueError("listwise loss inputs must share a one-dimensional shape")
    if logits.numel() == 0:
        raise ValueError("listwise loss requires at least one proposal")
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("listwise logits must be finite floating-point values")
    if group_ids.dtype == torch.bool or torch.is_floating_point(group_ids):
        raise ValueError("listwise group identifiers must be integers")
    if labels.dtype == torch.bool or torch.is_floating_point(labels):
        raise ValueError("listwise labels must be integer zero or one values")
    if logits.device != group_ids.device or logits.device != labels.device:
        raise ValueError("listwise loss inputs must share one device")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("listwise labels must contain only zero or one")

    losses: list[torch.Tensor] = []
    for group_id in torch.unique(group_ids, sorted=True):
        mask = group_ids == group_id
        group_labels = labels[mask]
        if int(group_labels.sum().item()) != 1:
            raise ValueError("each listwise group must contain exactly one truth pair")
        truth_index = int(torch.nonzero(group_labels, as_tuple=False)[0, 0].item())
        losses.append(-torch.log_softmax(logits[mask], dim=0)[truth_index])
    return torch.stack(losses).sum() / len(losses)


def heldout_control_threshold(scores: Sequence[Real]) -> float:
    """Return the smallest representable threshold above all observed controls."""
    if isinstance(scores, (str, bytes)) or not scores:
        raise ValueError("heldout control threshold requires at least one score")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("heldout control scores must be one-dimensional and finite")
    threshold = float(np.nextafter(np.max(values), np.inf))
    if not math.isfinite(threshold):
        raise ValueError("heldout control scores leave no finite higher threshold")
    return threshold


def gate_consensus(logits: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Require every fold-specific gate to clear its own heldout threshold."""
    if torch is None:  # pragma: no cover - exercised only without the ML extra
        raise ModuleNotFoundError("gate consensus requires the ml optional dependency")
    if not isinstance(logits, torch.Tensor) or not isinstance(thresholds, torch.Tensor):
        raise TypeError("gate consensus inputs must be torch tensors")
    if logits.ndim != 2 or thresholds.shape != (logits.shape[0],):
        raise ValueError("gate logits must be [folds, rows] with one threshold per fold")
    if logits.shape[0] < 2 or logits.shape[1] == 0:
        raise ValueError("gate consensus requires at least two folds and one row")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(thresholds):
        raise ValueError("gate consensus inputs must be floating-point tensors")
    if logits.device != thresholds.device:
        raise ValueError("gate consensus inputs must share one device")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(thresholds).all()):
        raise ValueError("gate consensus inputs must be finite")
    return torch.all(logits >= thresholds[:, None], dim=0)


@dataclass(frozen=True, slots=True)
class V2PairModelConfig:
    """One fixed small architecture for a v2 ranker or mother gate."""

    model_name: str
    image_feature_dim: int
    context_feature_dim: int

    def __post_init__(self) -> None:
        if self.model_name not in V2_MODEL_NAMES:
            raise ValueError(f"model_name must be one of {V2_MODEL_NAMES}")
        image_dim = _require_int(
            self.image_feature_dim,
            field="image_feature_dim",
            minimum=1,
        )
        context_dim = _require_int(
            self.context_feature_dim,
            field="context_feature_dim",
            minimum=1,
        )
        object.__setattr__(self, "image_feature_dim", image_dim)
        object.__setattr__(self, "context_feature_dim", context_dim)


class V2PairModel(_TorchModule):
    """Score symmetric cached pair features with one fixed v2 architecture."""

    def __init__(self, config: V2PairModelConfig) -> None:
        if torch is None or nn is None:  # pragma: no cover - exercised without ML extra
            raise ModuleNotFoundError("v2 pair models require the ml optional dependency")
        super().__init__()
        if not isinstance(config, V2PairModelConfig):
            raise TypeError("V2PairModel requires a V2PairModelConfig")
        self.config = config
        if config.model_name == "context_linear":
            self.head: nn.Module = nn.Linear(config.context_feature_dim, 1)
        else:
            input_dim = config.image_feature_dim + config.context_feature_dim
            if config.model_name == "full_linear":
                self.head = nn.Linear(input_dim, 1)
            else:
                self.head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                )

    def forward(
        self,
        image_features: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image_features, torch.Tensor) or not isinstance(
            normalized_context,
            torch.Tensor,
        ):
            raise TypeError("v2 model inputs must be torch tensors")
        batch = image_features.shape[0] if image_features.ndim == 2 else -1
        if image_features.shape != (batch, self.config.image_feature_dim):
            raise ValueError("v2 image features have an invalid shape")
        if normalized_context.shape != (batch, self.config.context_feature_dim):
            raise ValueError("v2 normalized context has an invalid shape")
        if batch == 0:
            raise ValueError("v2 models require at least one input row")
        if any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in (image_features, normalized_context)
        ):
            raise ValueError("v2 model inputs must contain finite floating-point values")
        if (
            image_features.dtype != normalized_context.dtype
            or image_features.device != normalized_context.device
        ):
            raise ValueError("v2 model inputs must share dtype and device")
        inputs = normalized_context
        if self.config.model_name != "context_linear":
            inputs = torch.cat((image_features, normalized_context), dim=1)
        return self.head(inputs).squeeze(1)


__all__ = [
    "DEFAULT_V2_PROPOSAL_RULE",
    "MovieFoldInput",
    "V2_MODEL_NAMES",
    "V2PairModel",
    "V2PairModelConfig",
    "V2ProposalRule",
    "assign_movie_folds",
    "event_balanced_listwise_loss",
    "gate_consensus",
    "generate_division_pair_proposals_v2",
    "generate_targeted_division_pair_neighborhoods",
    "heldout_control_threshold",
    "require_fit_only_training_source",
    "v2_pair_count",
]
