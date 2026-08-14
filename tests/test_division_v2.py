"""Tests for the bounded Phase 11 v2 proposal and cross-validation rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from biohub.division import (
    DivisionCandidateContext,
    DivisionPairNeighborhood,
    generate_division_pair_neighborhoods,
    generate_division_pair_proposals,
)
from biohub.division_v2 import (
    DEFAULT_V2_PROPOSAL_RULE,
    V2_MODEL_NAMES,
    MovieFoldInput,
    V2PairModel,
    V2PairModelConfig,
    assign_movie_folds,
    event_balanced_listwise_loss,
    gate_consensus,
    generate_division_pair_proposals_v2,
    generate_targeted_division_pair_neighborhoods,
    heldout_control_threshold,
    require_fit_only_training_source,
    v2_pair_count,
)
from biohub.metric import TrackingGraph
from scripts.run_division_v2_fit import proposal_volume_audit


def _candidate(rank: int) -> DivisionCandidateContext:
    distance = 1.0 + (rank - 1) * (17.0 / 23.0)
    angle = rank * math.pi / 12.0
    return DivisionCandidateContext(
        node_id=1000 + rank,
        distance_um=distance,
        displacement_um=(0.0, distance * math.sin(angle), distance * math.cos(angle)),
        present_in_linked_graph=True,
        is_current_child=rank == 1,
        is_parentless=rank != 1,
        has_wrong_parent=False,
        has_future=rank % 2 == 0,
        survives_pruning=rank <= 4,
    )


def _neighborhood(candidates: tuple[DivisionCandidateContext, ...]) -> DivisionPairNeighborhood:
    return DivisionPairNeighborhood(
        mother_id=91,
        daughter_time=8,
        mother_has_history=True,
        mother_has_current_child=True,
        mother_survives_pruning=True,
        search_max_distance_um=18.0,
        candidates=candidates,
    )


def _rank_pairs(proposals) -> set[tuple[int, int]]:
    return {
        tuple(sorted((proposal.candidate_1.node_id - 1000, proposal.candidate_2.node_id - 1000)))
        for proposal in proposals
    }


def test_v2_proposal_rule_preserves_v1_and_caps_outer_expansion() -> None:
    candidates = tuple(_candidate(rank) for rank in range(1, 25))

    proposals = generate_division_pair_proposals_v2([_neighborhood(candidates)])
    pairs = _rank_pairs(proposals)

    assert DEFAULT_V2_PROPOSAL_RULE.max_pairs_per_mother == 138
    assert len(proposals) == 138
    assert {
        (first, second)
        for first in range(1, 13)
        for second in range(first + 1, 13)
    } <= pairs
    assert {
        (anchor, outer)
        for anchor in range(1, 7)
        for outer in range(13, 25)
    } <= pairs
    assert (7, 13) not in pairs
    assert (13, 14) not in pairs


def test_v2_proposals_are_deterministic_under_candidate_input_order() -> None:
    candidates = tuple(_candidate(rank) for rank in range(1, 25))

    forward = generate_division_pair_proposals_v2([_neighborhood(candidates)])
    reversed_input = generate_division_pair_proposals_v2(
        [_neighborhood(tuple(reversed(candidates)))]
    )

    assert forward == reversed_input


@pytest.mark.parametrize("candidate_count", range(0, 31))
def test_v2_pair_count_matches_the_explicit_bound(candidate_count: int) -> None:
    base = min(candidate_count, 12)
    scanned = min(candidate_count, 24)
    expected = math.comb(base, 2) + min(base, 6) * max(scanned - 12, 0)

    assert v2_pair_count(candidate_count) == expected
    assert expected <= 138


def test_v2_proposals_require_a_neighborhood_covering_the_outer_radius() -> None:
    neighborhood = _neighborhood(tuple(_candidate(rank) for rank in range(1, 4)))
    too_small = DivisionPairNeighborhood(
        **{
            **neighborhood.__dict__,
            "search_max_distance_um": 16.0,
        }
    )

    with pytest.raises(ValueError, match="search radius"):
        generate_division_pair_proposals_v2([too_small])


def test_movie_grouped_folds_are_deterministic_balanced_and_disjoint() -> None:
    movies = tuple(
        MovieFoldInput(
            dataset=f"6bba_{index:08x}",
            source_role="fit",
            truth_event_count=(index % 4),
        )
        for index in range(30)
    )

    first = assign_movie_folds(movies, fold_count=5)
    second = assign_movie_folds(tuple(reversed(movies)), fold_count=5)

    assert first == second
    assert set(first) == {movie.dataset for movie in movies}
    assert set(first.values()) == set(range(5))
    fold_movies = [sum(fold == index for fold in first.values()) for index in range(5)]
    fold_events = [
        sum(movie.truth_event_count for movie in movies if first[movie.dataset] == index)
        for index in range(5)
    ]
    assert max(fold_movies) - min(fold_movies) <= 1
    assert max(fold_events) - min(fold_events) <= 2


@pytest.mark.parametrize("source_role", ["development", "evaluation"])
def test_movie_folds_and_training_source_reject_nonfit_roles(source_role: str) -> None:
    with pytest.raises(ValueError, match="fit only"):
        MovieFoldInput(
            dataset="6bba_00000001",
            source_role=source_role,
            truth_event_count=1,
        )
    with pytest.raises(ValueError, match="fit only"):
        require_fit_only_training_source("6bba_00000001", source_role=source_role)


def test_training_source_requires_6bba_and_a_closed_evaluation_gate() -> None:
    with pytest.raises(ValueError, match="6bba"):
        require_fit_only_training_source("44b6_deadbeef", source_role="fit")
    with pytest.raises(ValueError, match="evaluation_unlocked"):
        require_fit_only_training_source(
            "6bba_deadbeef",
            source_role="fit",
            evaluation_unlocked=True,
        )


def test_listwise_loss_weights_truth_mothers_equally() -> None:
    logits = torch.tensor([2.0, 0.0, 1.0, 0.5, -0.5], requires_grad=True)
    group_ids = torch.tensor([10, 10, 20, 20, 20])
    labels = torch.tensor([1, 0, 0, 1, 0])

    loss = event_balanced_listwise_loss(logits, group_ids, labels)
    expected = (
        -torch.log_softmax(logits[:2], dim=0)[0]
        - torch.log_softmax(logits[2:], dim=0)[1]
    ) / 2.0

    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_listwise_loss_requires_exactly_one_truth_pair_per_mother() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        event_balanced_listwise_loss(
            torch.tensor([0.0, 1.0]),
            torch.tensor([5, 5]),
            torch.tensor([0, 0]),
        )


def test_heldout_control_threshold_rejects_every_observed_control() -> None:
    threshold = heldout_control_threshold([0.1, -2.0, 1.25, 1.25])

    assert threshold > 1.25
    assert all(score < threshold for score in [0.1, -2.0, 1.25, 1.25])


def test_gate_consensus_requires_every_fold_to_clear_its_own_threshold() -> None:
    logits = torch.tensor(
        [
            [0.8, 0.8, 0.8],
            [1.2, 0.2, 1.2],
            [2.0, 2.0, -2.0],
        ]
    )
    thresholds = torch.tensor([0.5, 1.0, -1.0])

    assert torch.equal(gate_consensus(logits, thresholds), torch.tensor([True, False, False]))


def test_v2_model_ladder_has_fixed_small_architectures() -> None:
    image = torch.randn(4, 8)
    context = torch.randn(4, 3)

    assert V2_MODEL_NAMES == ("context_linear", "full_linear", "full_mlp16")
    for name in V2_MODEL_NAMES:
        model = V2PairModel(
            V2PairModelConfig(
                model_name=name,
                image_feature_dim=8,
                context_feature_dim=3,
            )
        )
        logits = model(image, context)
        assert logits.shape == (4,)
        assert torch.all(torch.isfinite(logits))

    context_only = V2PairModel(
        V2PairModelConfig(
            model_name="context_linear",
            image_feature_dim=8,
            context_feature_dim=3,
        )
    )
    assert torch.equal(context_only(image, context), context_only(image * 100.0, context))


def test_vectorized_volume_audit_matches_explicit_proposals() -> None:
    linked = TrackingGraph(
        node_ids=np.array([10, 11, 20, 30, 40, 50], dtype=np.int64),
        t=np.array([0, 0, 1, 1, 1, 1], dtype=np.int64),
        z=np.zeros(6),
        y=np.array([0.0, 100.0, 0.0, 0.0, 100.0, 100.0]),
        x=np.array([0.0, 0.0, 10.0, -10.0, 10.0, -10.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    raw = TrackingGraph(
        node_ids=linked.node_ids.copy(),
        t=linked.t.copy(),
        z=linked.z.copy(),
        y=linked.y.copy(),
        x=linked.x.copy(),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    neighborhoods = generate_division_pair_neighborhoods(
        linked,
        raw,
        max_distance_um=18.0,
    )

    audit = proposal_volume_audit(linked, raw)

    assert audit.v1_proposals == len(
        generate_division_pair_proposals(
            neighborhoods,
            max_distance_um=16.0,
            max_candidates_per_mother=12,
        )
    )
    assert audit.v2_proposals == len(generate_division_pair_proposals_v2(neighborhoods))
    assert audit.candidate_count_by_mother[10] == 2
    assert audit.candidate_count_by_mother[11] == 2


def test_targeted_v2_neighborhoods_equal_the_frozen_full_result() -> None:
    linked = TrackingGraph(
        node_ids=np.array([10, 11, 20, 30, 40, 50], dtype=np.int64),
        t=np.array([0, 0, 1, 1, 1, 1], dtype=np.int64),
        z=np.zeros(6),
        y=np.array([0.0, 100.0, 0.0, 0.0, 100.0, 100.0]),
        x=np.array([0.0, 0.0, 10.0, -10.0, 10.0, -10.0]),
        edges=np.array([(10, 20), (11, 40)], dtype=np.int64),
    )
    full = generate_division_pair_neighborhoods(
        linked,
        linked,
        pruned_node_ids={10, 20},
        max_distance_um=18.0,
    )

    targeted = generate_targeted_division_pair_neighborhoods(
        linked,
        linked,
        mother_node_ids={10},
        pruned_node_ids={10, 20},
        max_distance_um=18.0,
    )

    assert targeted == [item for item in full if item.mother_id == 10]
    with pytest.raises(ValueError, match="mother node identifiers"):
        generate_targeted_division_pair_neighborhoods(
            linked,
            linked,
            mother_node_ids={999},
        )
