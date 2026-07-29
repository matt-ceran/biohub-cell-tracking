from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from biohub import division_inference as inference_module
from biohub.constants import DIVISION_WEIGHT
from biohub.division import DivisionCandidateContext, DivisionPairProposal
from biohub.division_data import logical_graph_sha256
from biohub.division_inference import (
    PROBABILITY_THRESHOLDS,
    AcceptanceSummary,
    AcceptedFork,
    ForkMatch,
    MovieEvaluation,
    ScoredProposal,
    StrictTripletResult,
    aggregate_acceptance,
    apply_division_forks,
    evaluate_division_movie,
    evaluate_strict_triplets,
    scored_proposals_from_logits,
    select_division_forks,
)
from biohub.metric import MetricResult, TrackingGraph, evaluate, match_nodes

UNIT_SCALE = {"z": 1.0, "y": 1.0, "x": 1.0}
TEST_FALLBACK_HASH = "a" * 64


def graph(rows, edges=()) -> TrackingGraph:
    rows = list(rows)
    return TrackingGraph(
        node_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
        t=np.asarray([row[1] for row in rows], dtype=np.int64),
        z=np.asarray([row[2] for row in rows], dtype=float),
        y=np.asarray([row[3] for row in rows], dtype=float),
        x=np.asarray([row[4] for row in rows], dtype=float),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def graph_snapshot(value: TrackingGraph) -> dict[str, np.ndarray]:
    return {
        field: np.array(getattr(value, field), copy=True)
        for field in ("node_ids", "t", "z", "y", "x", "edges")
    }


def assert_graph_unchanged(value: TrackingGraph, snapshot: dict[str, np.ndarray]) -> None:
    for field, expected in snapshot.items():
        np.testing.assert_array_equal(getattr(value, field), expected)


def permute_graph(value: TrackingGraph, order: list[int]) -> TrackingGraph:
    return TrackingGraph(
        node_ids=np.asarray(value.node_ids)[order],
        t=np.asarray(value.t)[order],
        z=np.asarray(value.z)[order],
        y=np.asarray(value.y)[order],
        x=np.asarray(value.x)[order],
        edges=np.asarray(value.edges)[::-1].copy(),
    )


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def scored(
    mother: int,
    daughter_1: int,
    daughter_2: int,
    probability: float,
    *,
    removals: int = 0,
    restored: int = 0,
    fallback_hash: str = TEST_FALLBACK_HASH,
) -> ScoredProposal:
    return ScoredProposal(
        mother_id=mother,
        candidate_1_id=daughter_1,
        candidate_2_id=daughter_2,
        logit=logit(probability),
        edge_removal_count=removals,
        restored_node_count=restored,
        fallback_graph_sha256=fallback_hash,
    )


def accepted(
    mother: int,
    daughter_1: int,
    daughter_2: int,
    probability: float = 0.99,
    *,
    removals: int = 0,
    restored: int = 0,
    fallback: TrackingGraph | None = None,
    fallback_hash: str = TEST_FALLBACK_HASH,
) -> AcceptedFork:
    return AcceptedFork.from_scored(
        scored(
            mother,
            daughter_1,
            daughter_2,
            probability,
            removals=removals,
            restored=restored,
            fallback_hash=(
                logical_graph_sha256(fallback) if fallback is not None else fallback_hash
            ),
        )
    )


def candidate(node_id: int) -> DivisionCandidateContext:
    return DivisionCandidateContext(
        node_id=node_id,
        distance_um=1.0,
        displacement_um=(0.0, 0.0, 1.0),
        present_in_linked_graph=True,
        is_current_child=False,
        is_parentless=True,
        has_wrong_parent=False,
        has_future=False,
        survives_pruning=True,
    )


def pair(mother: int, daughter_1: int, daughter_2: int) -> DivisionPairProposal:
    return DivisionPairProposal(
        mother_id=mother,
        daughter_time=2,
        candidate_1=candidate(daughter_1),
        candidate_2=candidate(daughter_2),
        daughter_angle_deg=180.0,
        mother_has_history=True,
        mother_has_current_child=True,
        mother_survives_pruning=True,
    )


def test_scored_proposals_convert_logits_and_keep_only_highest_pair_per_mother() -> None:
    fallback = graph(
        [
            (10, 1, 0.0, 0.0, 0.0),
            (20, 2, 0.0, 0.0, 1.0),
            (21, 2, 0.0, 0.0, -1.0),
            (22, 2, 0.0, 1.0, 0.0),
        ],
        [(10, 20)],
    )
    proposals = [pair(10, 20, 21), pair(10, 20, 22)]

    values = scored_proposals_from_logits(proposals, [logit(0.8), logit(0.9)], fallback)
    forks = select_division_forks(values, probability_threshold=0.7)

    assert [value.probability for value in values] == pytest.approx([0.8, 0.9])
    assert [(fork.mother_id, fork.candidate_ids) for fork in forks] == [(10, (20, 22))]
    assert forks[0].edge_removal_count == 0
    assert forks[0].restored_node_count == 0
    assert {value.fallback_graph_sha256 for value in values} == {
        logical_graph_sha256(fallback)
    }
    assert forks[0].fallback_graph_sha256 == logical_graph_sha256(fallback)


def test_probability_thresholds_apply_directly_to_computed_sigmoid_probability() -> None:
    assert PROBABILITY_THRESHOLDS == (
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
    )
    boundary = 0.7
    one_ulp_below = math.nextafter(boundary, 0.0)
    at_boundary = scored(1, 10, 11, boundary)
    below_boundary = scored(2, 12, 13, one_ulp_below)
    below_upper = scored(3, 14, 15, math.nextafter(0.9999, 0.0))
    above_upper = scored(4, 16, 17, 0.9999)

    assert at_boundary.probability == boundary
    assert below_boundary.probability == one_ulp_below
    assert below_upper.probability == math.nextafter(0.9999, 0.0)
    assert above_upper.probability > 0.9999
    assert [fork.mother_id for fork in select_division_forks(
        [at_boundary, below_boundary], probability_threshold=boundary
    )] == [1]
    assert [fork.mother_id for fork in select_division_forks(
        [below_upper, above_upper], probability_threshold=0.9999
    )] == [4]


def test_scored_and_accepted_fork_records_are_frozen() -> None:
    proposal = scored(1, 10, 11, 0.8)
    fork = AcceptedFork.from_scored(proposal)

    with pytest.raises(FrozenInstanceError):
        proposal.logit = 0.0
    with pytest.raises(FrozenInstanceError):
        fork.mother_id = 2


def test_selection_rejects_proposals_from_different_fallback_graphs() -> None:
    proposals = [
        scored(1, 10, 11, 0.8, fallback_hash="a" * 64),
        scored(2, 12, 13, 0.8, fallback_hash="b" * 64),
    ]

    with pytest.raises(ValueError, match="one fallback graph"):
        select_division_forks(proposals, probability_threshold=0.7)


def test_raw_logit_orders_probabilities_that_both_saturate_to_one() -> None:
    proposals = [
        ScoredProposal(1, 10, 11, 40.0, 0, 0, TEST_FALLBACK_HASH),
        ScoredProposal(1, 12, 13, 100.0, 0, 0, TEST_FALLBACK_HASH),
    ]

    assert proposals[0].probability == proposals[1].probability == 1.0
    forks = select_division_forks(proposals, probability_threshold=0.9999)

    assert len(forks) == 1
    assert forks[0].candidate_ids == (12, 13)
    assert forks[0].logit == 100.0


def test_per_mother_and_global_conflict_ordering_is_deterministic() -> None:
    proposals = [
        scored(30, 300, 301, 0.9, removals=1, restored=0),
        scored(20, 200, 201, 0.9, removals=0, restored=2),
        scored(10, 100, 101, 0.9, removals=0, restored=1),
        scored(10, 102, 103, 0.9, removals=0, restored=0),
        scored(40, 400, 401, 0.95, removals=2, restored=2),
    ]

    forks = select_division_forks(proposals, probability_threshold=0.8)

    assert [(fork.mother_id, fork.candidate_ids) for fork in forks] == [
        (40, (400, 401)),
        (10, (102, 103)),
        (20, (200, 201)),
        (30, (300, 301)),
    ]


def test_global_terminal_tie_break_uses_mother_id() -> None:
    proposals = [
        scored(20, 200, 201, 0.8),
        scored(10, 100, 101, 0.8),
    ]

    forks = select_division_forks(proposals, probability_threshold=0.7)

    assert [fork.mother_id for fork in forks] == [10, 20]


def test_per_mother_terminal_tie_break_uses_candidate_ids() -> None:
    proposals = [
        scored(1, 12, 30, 0.8),
        scored(1, 11, 40, 0.8),
        scored(1, 11, 35, 0.8),
    ]

    forks = select_division_forks(proposals, probability_threshold=0.7)

    assert len(forks) == 1
    assert forks[0].candidate_ids == (11, 35)


def test_assignment_caps_forks_at_five() -> None:
    proposals = [
        scored(mother, 100 + 2 * mother, 101 + 2 * mother, 0.99 - mother / 10000)
        for mother in range(1, 7)
    ]

    forks = select_division_forks(proposals, probability_threshold=0.95)

    assert len(forks) == 5
    assert [fork.mother_id for fork in forks] == [1, 2, 3, 4, 5]


def test_assignment_rejects_daughter_reuse_but_allows_later_daughter_as_mother() -> None:
    proposals = [
        scored(1, 10, 11, 0.99),
        scored(2, 11, 12, 0.98),
        scored(10, 20, 21, 0.97),
    ]

    forks = select_division_forks(proposals, probability_threshold=0.95)

    assert [(fork.mother_id, fork.candidate_ids) for fork in forks] == [
        (1, (10, 11)),
        (10, (20, 21)),
    ]


def test_transaction_restores_nodes_rewires_only_fork_edges_and_preserves_context() -> None:
    rows = [
        (1, 0, 0.0, 0.0, 0.0),
        (10, 1, 0.0, 0.0, 1.0),
        (11, 1, 0.0, 3.0, 1.0),
        (20, 2, 0.0, 0.0, 2.0),
        (21, 2, 9.0, 8.0, 7.0),
        (30, 2, 0.0, 2.0, 2.0),
        (40, 3, 0.0, 0.0, 3.0),
        (41, 3, 0.0, 2.0, 3.0),
        (50, 0, 5.0, 5.0, 5.0),
        (51, 1, 5.0, 5.0, 6.0),
    ]
    fallback = graph(
        [row for row in rows if row[0] != 21],
        [(1, 10), (10, 20), (11, 30), (20, 40), (30, 41), (50, 51)],
    )
    raw = graph(rows)
    before = logical_graph_sha256(fallback)
    fallback_snapshot = graph_snapshot(fallback)
    raw_snapshot = graph_snapshot(raw)

    result = apply_division_forks(
        fallback,
        raw,
        [accepted(10, 21, 30, removals=2, restored=1, fallback=fallback)],
    )

    assert not result.rolled_back
    assert result.diagnostic is None
    assert result.fallback_graph_sha256 == logical_graph_sha256(fallback)
    assert result.candidate_graph_sha256 == logical_graph_sha256(result.graph)
    assert result.restored_node_ids == (21,)
    assert result.removed_edges == ((10, 20), (11, 30))
    assert result.added_edges == ((10, 21), (10, 30))
    assert set(result.graph.node_ids) == {row[0] for row in rows}
    restored_row = int(np.flatnonzero(result.graph.node_ids == 21)[0])
    assert (
        result.graph.z[restored_row],
        result.graph.y[restored_row],
        result.graph.x[restored_row],
    ) == (9.0, 8.0, 7.0)
    assert set(map(tuple, result.graph.edges)) == {
        (1, 10),
        (10, 21),
        (10, 30),
        (20, 40),
        (30, 41),
        (50, 51),
    }
    assert logical_graph_sha256(fallback) == before
    assert_graph_unchanged(fallback, fallback_snapshot)
    assert_graph_unchanged(raw, raw_snapshot)


def test_no_fork_returns_the_exact_untouched_fallback_object() -> None:
    fallback = graph(
        [(1, 0, 0.0, 0.0, 0.0), (2, 1, 0.0, 0.0, 1.0)],
        [(1, 2)],
    )
    before = logical_graph_sha256(fallback)

    result = apply_division_forks(fallback, fallback, [])

    assert result.graph is fallback
    assert result.applied_forks == ()
    assert not result.rolled_back
    assert result.fallback_graph_sha256 == before
    assert result.candidate_graph_sha256 == before
    assert logical_graph_sha256(fallback) == before


def test_mutation_rejects_a_fork_bound_to_another_fallback() -> None:
    fallback = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0),
            (3, 1, 0.0, 0.0, -1.0),
        ],
        [(1, 2)],
    )

    result = apply_division_forks(
        fallback,
        fallback,
        [accepted(1, 2, 3, fallback_hash="b" * 64)],
    )

    assert result.rolled_back
    assert result.graph is fallback
    assert result.diagnostic is not None and "fallback identity" in result.diagnostic


def test_divisions_off_fallback_rejects_a_preexisting_no_op_fork() -> None:
    fallback = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0),
            (3, 1, 0.0, 0.0, -1.0),
        ],
        [(1, 2), (1, 3)],
    )

    result = apply_division_forks(
        fallback,
        fallback,
        [accepted(1, 2, 3, fallback=fallback)],
    )

    assert result.rolled_back
    assert result.graph is fallback
    assert (
        result.diagnostic is not None
        and "fallback graph outdegree exceeds one" in result.diagnostic
    )


def test_nonempty_logical_no_op_is_rejected(monkeypatch) -> None:
    fallback = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0),
            (3, 1, 0.0, 0.0, -1.0),
        ],
        [(1, 2), (1, 3)],
    )
    original_validate = inference_module._validate_graph_invariants

    def permit_preexisting_fork(value, *, forks, max_outdegree):
        original_validate(value, forks=forks, max_outdegree=2)

    monkeypatch.setattr(
        inference_module,
        "_validate_graph_invariants",
        permit_preexisting_fork,
    )

    result = apply_division_forks(
        fallback,
        fallback,
        [accepted(1, 2, 3, fallback=fallback)],
    )

    assert result.rolled_back
    assert result.graph is fallback
    assert result.diagnostic is not None and "no logical graph change" in result.diagnostic


def test_later_invalid_fork_rolls_back_an_earlier_successful_fork() -> None:
    rows = [
        (1, 0, 0.0, 0.0, 0.0),
        (2, 1, 0.0, 0.0, 1.0),
        (3, 1, 0.0, 0.0, -1.0),
        (4, 1, 0.0, 2.0, 0.0),
        (5, 2, 0.0, 2.0, 1.0),
        (6, 3, 0.0, 2.0, -1.0),
    ]
    fallback = graph([row for row in rows if row[0] != 3], [(1, 2), (4, 5)])
    raw = graph(rows)
    first = accepted(1, 2, 3, fallback=fallback)
    invalid_later = accepted(4, 5, 6, fallback=fallback)
    fallback_snapshot = graph_snapshot(fallback)
    raw_snapshot = graph_snapshot(raw)

    first_only = apply_division_forks(fallback, raw, [first])
    result = apply_division_forks(fallback, raw, [first, invalid_later])

    assert not first_only.rolled_back
    assert first_only.restored_node_ids == (3,)
    assert set(map(tuple, first_only.graph.edges)) >= {(1, 2), (1, 3)}
    assert result.graph is fallback
    assert result.rolled_back
    assert result.applied_forks == ()
    assert result.restored_node_ids == ()
    assert result.removed_edges == ()
    assert result.added_edges == ()
    assert result.diagnostic is not None and "t+1" in result.diagnostic
    assert_graph_unchanged(fallback, fallback_snapshot)
    assert_graph_unchanged(raw, raw_snapshot)


@pytest.mark.parametrize(
    ("fallback_edges", "message"),
    [
        ([(1, 3), (2, 3)], "indegree"),
        ([(1, 3), (1, 4), (1, 5)], "outdegree"),
        ([(1, 3), (1, 3)], "duplicate edges"),
        ([(1, 99)], "missing endpoint"),
    ],
)
def test_each_graph_invariant_failure_rolls_back_without_mutating_inputs(
    fallback_edges: list[tuple[int, int]],
    message: str,
) -> None:
    rows = [
        (1, 0, 0.0, 0.0, 0.0),
        (2, 0, 0.0, 1.0, 0.0),
        (3, 1, 0.0, 0.0, 1.0),
        (4, 1, 0.0, 1.0, 1.0),
        (5, 1, 0.0, 2.0, 1.0),
        (10, 0, 0.0, 10.0, 0.0),
        (11, 1, 0.0, 10.0, 1.0),
        (12, 1, 0.0, 10.0, -1.0),
    ]
    fallback = graph(rows, fallback_edges)
    raw = graph(rows)
    fallback_snapshot = graph_snapshot(fallback)
    raw_snapshot = graph_snapshot(raw)

    result = apply_division_forks(fallback, raw, [accepted(10, 11, 12)])

    assert result.graph is fallback
    assert result.rolled_back
    assert result.applied_forks == ()
    assert result.diagnostic is not None and message in result.diagnostic
    assert_graph_unchanged(fallback, fallback_snapshot)
    assert_graph_unchanged(raw, raw_snapshot)


def test_cycle_validation_path_rejects_a_directed_cycle() -> None:
    arrays = inference_module._GraphArrays(
        node_ids=np.array([1, 2], dtype=np.int64),
        t=np.array([0, 1], dtype=np.int64),
        z=np.zeros(2),
        y=np.zeros(2),
        x=np.zeros(2),
        edges=np.array([[1, 2], [2, 1]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="cycle"):
        inference_module._validate_graph_invariants(arrays, forks=(), max_outdegree=2)


def test_integral_array_preserves_signed_int64_and_rejects_out_of_range_values() -> None:
    limits = np.iinfo(np.int64)
    values = np.array([limits.min, limits.max], dtype=np.int64)

    converted = inference_module._integral_array(values, field="identifiers")

    np.testing.assert_array_equal(converted, values)
    with pytest.raises(ValueError, match="signed 64-bit range"):
        inference_module._integral_array(
            np.array([2**63], dtype=np.uint64),
            field="identifiers",
        )
    with pytest.raises(ValueError, match="signed 64-bit range"):
        inference_module._integral_array(
            np.array([float(2**63)], dtype=np.float64),
            field="identifiers",
        )


@pytest.mark.parametrize("edges", [[(2, 1)], [(1, 3)]])
def test_graph_validation_requires_edges_to_be_strictly_forward(edges) -> None:
    value = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0),
            (3, 0, 0.0, 1.0, 0.0),
        ],
        edges,
    )

    with pytest.raises(ValueError, match="strictly forward"):
        inference_module._graph_arrays(value, label="test graph")


def strict_truth() -> TrackingGraph:
    return graph(
        [
            (100, 0, 0.0, 0.0, 0.0),
            (101, 1, 0.0, 0.0, -1.0),
            (102, 1, 0.0, 0.0, 1.0),
            (200, 0, 0.0, 10.0, 0.0),
            (201, 1, 0.0, 10.0, 1.0),
            (300, 1, 0.0, 0.0, 3.0),
        ],
        [(100, 101), (100, 102), (200, 201)],
    )


def test_strict_triplet_evaluation_does_not_count_a_mother_only_match_as_true() -> None:
    wrong = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -1.0),
            (3, 1, 0.0, 0.0, 3.0),
        ],
        [(1, 2), (1, 3)],
    )

    result = evaluate_strict_triplets(
        wrong,
        strict_truth(),
        scale=UNIT_SCALE,
        max_distance=0.1,
    )

    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.fully_matched_wrong_forks == 1
    assert result.matched_division_false_positives == 0
    assert result.unmatched_forks == 0
    assert result.details[0].mapped_triplet == (100, 101, 300)
    assert result.details[0].truth_triplet == (100, 101, 102)


def test_strict_triplet_evaluation_reports_true_control_and_unmatched_forks() -> None:
    exact = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -1.0),
            (3, 1, 0.0, 0.0, 1.0),
        ],
        [(1, 2), (1, 3)],
    )
    control = graph(
        [
            (4, 0, 0.0, 10.0, 0.0),
            (5, 1, 0.0, 10.0, 1.0),
            (6, 1, 0.0, 0.0, 3.0),
        ],
        [(4, 5), (4, 6)],
    )
    unmatched = graph(
        [
            (7, 0, 50.0, 50.0, 50.0),
            (8, 1, 50.0, 50.0, 49.0),
            (9, 1, 50.0, 50.0, 51.0),
        ],
        [(7, 8), (7, 9)],
    )

    true_result = evaluate_strict_triplets(
        exact, strict_truth(), scale=UNIT_SCALE, max_distance=0.1
    )
    control_result = evaluate_strict_triplets(
        control, strict_truth(), scale=UNIT_SCALE, max_distance=0.1
    )
    unmatched_result = evaluate_strict_triplets(
        unmatched, strict_truth(), scale=UNIT_SCALE, max_distance=0.1
    )

    assert (true_result.tp, true_result.fp, true_result.fn) == (1, 0, 0)
    assert true_result.matched_truth_triplets == ((100, 101, 102),)
    assert control_result.matched_division_false_positives == 1
    assert (control_result.tp, control_result.fp, control_result.fn) == (0, 1, 1)
    assert unmatched_result.unmatched_forks == 1
    assert (unmatched_result.tp, unmatched_result.fp, unmatched_result.fn) == (0, 1, 1)


def test_strict_and_metric_matching_are_row_order_invariant_under_distance_ties() -> None:
    truth = graph(
        [
            (100, 0, 0.0, 0.0, 0.0),
            (101, 1, 0.0, 0.0, 1.0),
            (102, 1, 0.0, 0.0, 1.0),
        ],
        [(100, 101), (100, 102)],
    )
    pred = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0),
            (3, 1, 0.0, 0.0, 1.0),
        ],
        [(1, 2), (1, 3)],
    )
    shuffled_truth = permute_graph(truth, [2, 0, 1])
    shuffled_pred = permute_graph(pred, [1, 2, 0])

    strict = evaluate_strict_triplets(pred, truth, scale=UNIT_SCALE, max_distance=0.1)
    shuffled_strict = evaluate_strict_triplets(
        shuffled_pred,
        shuffled_truth,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )
    canonical_pred = inference_module._canonical_graph(pred, label="predicted graph")
    canonical_truth = inference_module._canonical_graph(truth, label="truth graph")
    shuffled_canonical_pred = inference_module._canonical_graph(
        shuffled_pred,
        label="predicted graph",
    )
    shuffled_canonical_truth = inference_module._canonical_graph(
        shuffled_truth,
        label="truth graph",
    )
    metric_result = inference_module._evaluate_from_matches(
        canonical_pred,
        canonical_truth,
        match_nodes(canonical_pred, canonical_truth, scale=UNIT_SCALE, max_distance=0.1),
    )
    shuffled_metric_result = inference_module._evaluate_from_matches(
        shuffled_canonical_pred,
        shuffled_canonical_truth,
        match_nodes(
            shuffled_canonical_pred,
            shuffled_canonical_truth,
            scale=UNIT_SCALE,
            max_distance=0.1,
        ),
    )

    assert strict == shuffled_strict
    assert strict.details[0].mapped_triplet == (100, 101, 102)
    assert metric_result == shuffled_metric_result


def test_metric_from_precomputed_matches_is_equivalent_to_existing_evaluate() -> None:
    truth = strict_truth()
    pred = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -1.0),
            (3, 1, 0.0, 0.0, 1.0),
            (4, 0, 50.0, 50.0, 50.0),
            (5, 1, 50.0, 50.0, 51.0),
        ],
        [(1, 2), (1, 3), (4, 5)],
    )
    canonical_pred = inference_module._canonical_graph(pred, label="predicted graph")
    canonical_truth = inference_module._canonical_graph(truth, label="truth graph")
    matches = match_nodes(
        canonical_pred,
        canonical_truth,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )

    result = inference_module._evaluate_from_matches(canonical_pred, canonical_truth, matches)

    assert result == evaluate(
        canonical_pred,
        canonical_truth,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )


def count_jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator


def metric(
    *,
    edge_tp: int = 50,
    edge_fp: int = 0,
    edge_fn: int = 50,
    edge_ignored: int = 0,
    predicted_nodes: int = 100,
    division_tp: int = 0,
    division_fp: int = 0,
    division_fn: int = 0,
    matched_nodes: int = 80,
    truth_nodes: int = 100,
) -> MetricResult:
    edge_jaccard = count_jaccard(edge_tp, edge_fp, edge_fn)
    division_jaccard = count_jaccard(division_tp, division_fp, division_fn)
    return MetricResult(
        score=(1.0 - DIVISION_WEIGHT) * edge_jaccard
        + DIVISION_WEIGHT * division_jaccard,
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        edge_ignored=edge_ignored,
        division_tp=division_tp,
        division_fp=division_fp,
        division_fn=division_fn,
        matched_nodes=matched_nodes,
        predicted_nodes=predicted_nodes,
        truth_nodes=truth_nodes,
    )


def triplets(
    *,
    tp: int = 0,
    fp: int | None = None,
    fn: int = 0,
    matched_false_positives: int = 0,
    wrong_forks: int = 0,
    unmatched_forks: int = 0,
    matched: tuple[tuple[int, int, int], ...] = (),
) -> StrictTripletResult:
    details: list[ForkMatch] = []
    for index, truth_triplet in enumerate(matched):
        details.append(
            ForkMatch(
                predicted_triplet=(1000 + index, 2000 + 2 * index, 2001 + 2 * index),
                mapped_triplet=truth_triplet,
                truth_triplet=truth_triplet,
                classification="true_positive",
            )
        )
    for index in range(matched_false_positives):
        details.append(
            ForkMatch(
                predicted_triplet=(3000 + index, 4000 + 2 * index, 4001 + 2 * index),
                mapped_triplet=(5000 + index, 6000 + 2 * index, 6001 + 2 * index),
                truth_triplet=None,
                classification="matched_division_false_positive",
            )
        )
    for index in range(wrong_forks):
        truth_triplet = (7000 + index, 8000 + 2 * index, 8001 + 2 * index)
        details.append(
            ForkMatch(
                predicted_triplet=(9000 + index, 10000 + 2 * index, 10001 + 2 * index),
                mapped_triplet=(truth_triplet[0], truth_triplet[1], 11000 + index),
                truth_triplet=truth_triplet,
                classification="fully_matched_wrong_fork",
            )
        )
    for index in range(unmatched_forks):
        details.append(
            ForkMatch(
                predicted_triplet=(12000 + index, 13000 + 2 * index, 13001 + 2 * index),
                mapped_triplet=(None, None, None),
                truth_triplet=None,
                classification="unmatched_fork",
            )
        )
    derived_fp = matched_false_positives + wrong_forks + unmatched_forks
    return StrictTripletResult(
        tp=tp,
        fp=derived_fp if fp is None else fp,
        fn=fn,
        matched_division_false_positives=matched_false_positives,
        fully_matched_wrong_forks=wrong_forks,
        unmatched_forks=unmatched_forks,
        matched_truth_triplets=matched,
        details=tuple(details),
    )


def acceptance_rows() -> tuple[list[MovieEvaluation], tuple[str, ...], tuple[str, ...]]:
    datasets = tuple(f"6bba_synthetic_{index:02d}" for index in range(20))
    controls = datasets[:10]
    truth_counts = [0] * 10 + [1, 1, 1, 1, 1, 2, 2, 3, 3, 3]
    rows: list[MovieEvaluation] = []
    for index, (dataset, truth_count) in enumerate(zip(datasets, truth_counts, strict=True)):
        recovered = index == 10
        unmatched = index == 11
        accepted_count = int(recovered or unmatched)
        fallback_hash = f"{index:064x}"
        truth_hash = f"{index + 1000:064x}"
        candidate_hash = f"{index + 100:064x}" if recovered else fallback_hash
        if unmatched:
            candidate_hash = f"{index + 200:064x}"
        matched = ((100, 101, 102),) if recovered else ()
        candidate_triplets = triplets(
            tp=int(recovered),
            fp=int(unmatched),
            fn=truth_count - int(recovered),
            unmatched_forks=int(unmatched),
            matched=matched,
        )
        rows.append(
            MovieEvaluation(
                dataset=dataset,
                fallback_graph_sha256=fallback_hash,
                candidate_graph_sha256=candidate_hash,
                truth_graph_sha256=truth_hash,
                baseline_metric=metric(division_fn=truth_count),
                candidate_metric=metric(
                    predicted_nodes=100 + int(unmatched),
                    division_tp=int(recovered),
                    division_fn=truth_count - int(recovered),
                ),
                baseline_triplets=triplets(fn=truth_count),
                candidate_triplets=candidate_triplets,
                accepted_forks=accepted_count,
                estimated_nodes=100,
                restored_node_count=int(unmatched),
                removed_edge_count=int(recovered) + 2 * int(unmatched),
                added_edge_count=2 * accepted_count,
            )
        )
    return rows, datasets, controls


def summarize_acceptance(
    rows: list[MovieEvaluation],
    datasets: tuple[str, ...],
    controls: tuple[str, ...],
):
    return aggregate_acceptance(
        rows,
        expected_datasets=datasets,
        no_division_datasets=controls,
        expected_truth_graph_sha256={
            dataset: f"{index + 1000:064x}" for index, dataset in enumerate(datasets)
        },
    )


def assert_only_guard_failed(summary, guard: str) -> None:
    assert not summary.passed
    assert {name for name, passed in summary.guards.items() if not passed} == {guard}


def test_strict_triplet_result_rejects_incoherent_counts_and_details() -> None:
    valid = triplets(fn=1)

    with pytest.raises(ValueError, match="details|classifications"):
        replace(valid, fp=1)
    with pytest.raises(ValueError, match="TP count|matched truth triplets"):
        replace(valid, tp=1, fn=0, matched_truth_triplets=((1, 2, 3),))


def test_movie_evaluation_rejects_fabricated_no_fork_improvement() -> None:
    rows, _, _ = acceptance_rows()
    valid_improved_metric = metric(edge_tp=60, edge_fn=40)

    with pytest.raises(ValueError, match="no-fork.*metrics"):
        replace(rows[0], candidate_metric=valid_improved_metric)


def test_movie_evaluation_rejects_metric_score_and_mutation_count_fabrication() -> None:
    rows, _, _ = acceptance_rows()
    forged_score = replace(
        rows[0].baseline_metric,
        score=rows[0].baseline_metric.score + 0.01,
    )

    with pytest.raises(ValueError, match="full score"):
        replace(rows[0], baseline_metric=forged_score)
    with pytest.raises(ValueError, match="added edges"):
        replace(rows[10], added_edge_count=0)


def test_movie_evaluation_rejects_invalid_metric_counts_and_jaccards() -> None:
    rows, _, _ = acceptance_rows()
    baseline = rows[0].baseline_metric
    invalid = [
        (replace(baseline, edge_tp=-1), "non-negative"),
        (replace(baseline, edge_jaccard=0.6), "edge Jaccard"),
        (replace(baseline, division_jaccard=0.5), "division Jaccard"),
        (replace(baseline, matched_nodes=101), "matched node count"),
        (replace(baseline, score=float("nan")), "finite"),
    ]

    for metric_result, message in invalid:
        with pytest.raises(ValueError, match=message):
            replace(rows[0], baseline_metric=metric_result)


def test_movie_evaluation_rejects_cross_graph_truth_and_fallback_division_drift() -> None:
    rows, _, _ = acceptance_rows()
    wrong_truth_edges = metric(
        edge_tp=50,
        edge_fn=51,
        division_tp=1,
    )
    fallback_with_division = metric(division_tp=1)

    with pytest.raises(ValueError, match="truth totals"):
        replace(rows[10], candidate_metric=wrong_truth_edges)
    with pytest.raises(ValueError, match="divisions-off fallback"):
        replace(rows[10], baseline_metric=fallback_with_division)


def test_acceptance_requires_exact_truth_graph_hashes_for_every_dataset() -> None:
    rows, datasets, controls = acceptance_rows()
    expected = {
        dataset: f"{index + 1000:064x}" for index, dataset in enumerate(datasets)
    }
    expected[datasets[0]] = "f" * 64

    with pytest.raises(ValueError, match="truth graph SHA-256"):
        aggregate_acceptance(
            rows,
            expected_datasets=datasets,
            no_division_datasets=controls,
            expected_truth_graph_sha256=expected,
        )


def test_acceptance_summary_recursively_copies_and_freezes_nested_values() -> None:
    source = {"nested": {"values": [1, 2]}}
    summary = AcceptanceSummary(
        passed=True,
        guards={"guard": True},
        metrics={"metric": 1},
        diagnostics=source,
    )
    source["nested"]["values"].append(3)

    assert summary.diagnostics["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        summary.diagnostics["nested"]["new"] = 1


def test_acceptance_aggregation_applies_frozen_guards_and_reports_diagnostics() -> None:
    rows, datasets, controls = acceptance_rows()

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.passed
    assert all(summary.guards.values())
    assert summary.metrics["mean_full_score_delta"] == pytest.approx(0.005)
    assert summary.metrics["mean_edge_jaccard_delta"] == 0.0
    assert summary.metrics["aggregate_node_ratio_delta"] == pytest.approx(0.0005)
    assert summary.diagnostics["median_full_score_delta"] == 0.0
    assert summary.diagnostics["median_edge_jaccard_delta"] == 0.0
    assert summary.diagnostics["full_score_wins_ties_losses"] == {
        "wins": 1,
        "ties": 19,
        "losses": 0,
    }
    assert summary.diagnostics["edge_wins_ties_losses"] == {
        "wins": 0,
        "ties": 20,
        "losses": 0,
    }
    assert summary.diagnostics["worst_movie"] == {
        "dataset": datasets[0],
        "full_score_delta": 0.0,
    }
    assert summary.diagnostics["micro_division_counts"] == {"tp": 1, "fp": 0, "fn": 17}
    assert summary.diagnostics["micro_triplet_counts"] == {"tp": 1, "fp": 1, "fn": 17}
    assert summary.diagnostics["unmatched_accepted_forks"] == 1
    assert summary.diagnostics["restored_nodes"] == 1
    assert summary.diagnostics["removed_edges"] == 3
    assert summary.diagnostics["added_edges"] == 4


def test_acceptance_mean_full_delta_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[10] = replace(
        rows[10],
        candidate_graph_sha256=rows[10].fallback_graph_sha256,
        candidate_metric=rows[10].baseline_metric,
        candidate_triplets=rows[10].baseline_triplets,
        accepted_forks=0,
        removed_edge_count=0,
        added_edge_count=0,
    )
    rows[17] = replace(
        rows[17],
        candidate_graph_sha256="d" * 64,
        baseline_metric=metric(edge_tp=500, edge_fn=500, division_fn=3),
        candidate_metric=metric(
            edge_tp=481,
            edge_fn=519,
            division_tp=1,
            division_fn=2,
        ),
        candidate_triplets=triplets(
            tp=1,
            fn=2,
            matched=((100, 101, 102),),
        ),
        accepted_forks=1,
        removed_edge_count=1,
        added_edge_count=2,
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["mean_full_score_delta"] == pytest.approx(0.0008116666666666668)
    assert_only_guard_failed(summary, "mean_full_score_delta")


def test_acceptance_mean_edge_delta_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[10] = replace(
        rows[10],
        candidate_metric=metric(
            edge_tp=47,
            edge_fn=53,
            division_tp=1,
        ),
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["mean_edge_jaccard_delta"] == pytest.approx(-0.0015)
    assert_only_guard_failed(summary, "mean_edge_jaccard_delta")


def test_acceptance_aggregate_node_ratio_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[11] = replace(
        rows[11],
        candidate_metric=replace(rows[11].candidate_metric, predicted_nodes=103),
        restored_node_count=3,
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["aggregate_node_ratio_delta"] == pytest.approx(0.0015)
    assert_only_guard_failed(summary, "aggregate_node_ratio_delta")


def test_acceptance_control_matched_false_positive_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[0] = replace(
        rows[0],
        candidate_graph_sha256="a" * 64,
        candidate_metric=metric(edge_tp=70, edge_fn=30, division_fp=1),
        candidate_triplets=triplets(fp=1, matched_false_positives=1),
        accepted_forks=1,
        added_edge_count=2,
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.diagnostics["matched_division_false_positives_by_control"][datasets[0]] == 1
    assert_only_guard_failed(
        summary,
        "matched_division_false_positives_per_no_division_movie",
    )


def test_acceptance_strict_true_recovery_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[10] = replace(
        rows[10],
        candidate_metric=metric(edge_tp=60, edge_fn=40, division_fn=1),
        candidate_triplets=triplets(fp=1, fn=1, unmatched_forks=1),
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["strict_true_recoveries"] == 0
    assert_only_guard_failed(summary, "strict_true_recoveries")


def test_acceptance_full_score_wins_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()
    rows[11] = replace(
        rows[11],
        candidate_graph_sha256="b" * 64,
        baseline_metric=metric(edge_tp=500, edge_fn=500, division_fn=1),
        candidate_metric=metric(
            edge_tp=481,
            edge_fn=519,
            predicted_nodes=101,
            division_fn=1,
        ),
        candidate_triplets=triplets(fp=1, fn=1, unmatched_forks=1),
        accepted_forks=1,
        added_edge_count=2,
    )

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["mean_full_score_delta"] == pytest.approx(0.004145)
    assert summary.diagnostics["full_score_wins_ties_losses"] == {
        "wins": 1,
        "ties": 18,
        "losses": 1,
    }
    assert_only_guard_failed(summary, "full_score_wins_vs_losses")


def test_acceptance_no_fork_fallback_identity_guard_is_independent() -> None:
    rows, _, _ = acceptance_rows()

    with pytest.raises(ValueError, match="no-fork.*graph hash"):
        replace(rows[0], candidate_graph_sha256="c" * 64)


def test_acceptance_requires_the_exact_complete_cohort() -> None:
    rows, datasets, controls = acceptance_rows()

    summary = summarize_acceptance(rows, datasets, controls)
    assert summary.guards["complete_cohort_movies"]
    with pytest.raises(ValueError, match="exact expected cohort"):
        summarize_acceptance(rows[:-1], datasets, controls)


def test_acceptance_fully_matched_wrong_fork_guard_is_independent() -> None:
    rows, datasets, controls = acceptance_rows()

    wrong = replace(
        rows[11],
        accepted_forks=1,
        candidate_graph_sha256="f" * 64,
        candidate_metric=metric(
            predicted_nodes=101,
            division_tp=1,
        ),
        candidate_triplets=triplets(fp=1, fn=1, wrong_forks=1),
        added_edge_count=2,
    )
    rows[11] = wrong

    summary = summarize_acceptance(rows, datasets, controls)

    assert summary.metrics["fully_matched_wrong_forks"] == 1
    assert_only_guard_failed(summary, "fully_matched_wrong_forks")


def test_movie_evaluation_reuses_matches_and_reports_bound_graph_identity(monkeypatch) -> None:
    truth = strict_truth()
    fallback = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -1.0),
            (3, 1, 0.0, 0.0, 1.0),
        ],
        [(1, 2)],
    )
    mutation = apply_division_forks(fallback, fallback, [accepted(1, 2, 3, fallback=fallback)])
    original_match_nodes = inference_module.match_nodes
    calls = 0

    def counted_match_nodes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_match_nodes(*args, **kwargs)

    monkeypatch.setattr(inference_module, "match_nodes", counted_match_nodes)

    row = evaluate_division_movie(
        "6bba_synthetic",
        fallback,
        mutation,
        truth,
        estimated_nodes=10,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )

    assert row.strict_true_recoveries == 1
    assert row.accepted_forks == 1
    assert row.fallback_graph_sha256 != row.candidate_graph_sha256
    assert row.truth_graph_sha256 == logical_graph_sha256(truth)
    assert row.fallback_graph_sha256 == mutation.fallback_graph_sha256
    assert row.candidate_graph_sha256 == mutation.candidate_graph_sha256
    assert calls == 2
    assert row.baseline_metric == evaluate(
        fallback,
        truth,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )
    assert row.candidate_metric == evaluate(
        mutation.graph,
        truth,
        scale=UNIT_SCALE,
        max_distance=0.1,
    )


def test_movie_evaluation_rejects_a_fallback_unrelated_to_the_mutation() -> None:
    truth = strict_truth()
    fallback = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -1.0),
            (3, 1, 0.0, 0.0, 1.0),
        ],
        [(1, 2)],
    )
    unrelated = graph(
        [
            (1, 0, 0.0, 0.0, 0.0),
            (2, 1, 0.0, 0.0, -2.0),
            (3, 1, 0.0, 0.0, 1.0),
        ],
        [(1, 2)],
    )
    mutation = apply_division_forks(
        fallback,
        fallback,
        [accepted(1, 2, 3, fallback=fallback)],
    )

    with pytest.raises(ValueError, match="unrelated fallback"):
        evaluate_division_movie(
            "6bba_synthetic",
            unrelated,
            mutation,
            truth,
            estimated_nodes=10,
            scale=UNIT_SCALE,
            max_distance=0.1,
        )
