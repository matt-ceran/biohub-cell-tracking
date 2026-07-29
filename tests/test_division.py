"""Tests for Phase 11 mother-to-daughter proposal generation and auditing."""

from __future__ import annotations

import numpy as np
import pytest

from biohub.division import (
    generate_division_pair_neighborhoods,
    generate_division_pair_proposals,
    generate_division_proposals,
    passes_phase7_gates,
)
from biohub.link import prune_to_tracks
from biohub.metric import TrackingGraph
from scripts.audit_division_pair_proposals import (
    PairRule,
    aggregate_pair_neighborhoods,
    build_rules,
    select_pair_rule,
)
from scripts.audit_division_proposals import build_event_rows

UNIT_SCALE = {"z": 1.0, "y": 1.0, "x": 1.0}

DIVISION_ROWS = [
    (1, 0, 0.0, 0.0, 0.0),
    (2, 1, 0.0, 0.0, 0.0),
    (3, 2, 0.0, 0.0, 5.0),
    (4, 3, 0.0, 0.0, 9.0),
    (5, 2, 0.0, 0.0, -5.0),
    (6, 3, 0.0, 0.0, -9.0),
]
LINKED_EDGES = [(1, 2), (2, 3), (3, 4), (5, 6)]
TRUTH_EDGES = [*LINKED_EDGES, (2, 5)]


def graph(rows=DIVISION_ROWS, edges=()) -> TrackingGraph:
    rows = list(rows)
    return TrackingGraph(
        node_ids=np.array([row[0] for row in rows], dtype=np.int64),
        t=np.array([row[1] for row in rows]),
        z=np.array([row[2] for row in rows], dtype=float),
        y=np.array([row[3] for row in rows], dtype=float),
        x=np.array([row[4] for row in rows], dtype=float),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_broad_proposal_preserves_phase7_triplet_context() -> None:
    proposals = generate_division_proposals(
        graph(edges=LINKED_EDGES),
        max_distance_um=14.0,
        scale=UNIT_SCALE,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert (
        proposal.mother_id,
        proposal.linked_daughter_id,
        proposal.orphan_daughter_id,
    ) == (2, 3, 5)
    assert proposal.linked_distance_um == pytest.approx(5.0)
    assert proposal.orphan_distance_um == pytest.approx(5.0)
    assert proposal.daughter_angle_deg == pytest.approx(180.0)
    assert proposal.mother_has_history
    assert proposal.orphan_has_future
    assert passes_phase7_gates(proposal)


def test_broad_proposal_does_not_apply_phase7_acceptance_early() -> None:
    rows = list(DIVISION_ROWS)
    rows[2] = (3, 2, 0.0, 0.0, 2.0)
    rows[3] = (4, 3, 0.0, 0.0, 4.0)

    proposals = generate_division_proposals(
        graph(rows, LINKED_EDGES),
        max_distance_um=14.0,
        scale=UNIT_SCALE,
    )

    assert len(proposals) == 1
    assert proposals[0].linked_distance_um == pytest.approx(2.0)
    assert not passes_phase7_gates(proposals[0])


def test_pruning_can_erase_a_recoverable_second_daughter() -> None:
    linked = graph(edges=LINKED_EDGES)
    preprune = generate_division_proposals(linked, scale=UNIT_SCALE)
    pruned = prune_to_tracks(linked, min_track_length=3)
    postprune = generate_division_proposals(pruned, scale=UNIT_SCALE)

    assert len(preprune) == 1
    assert postprune == []
    assert set(pruned.node_ids) == {1, 2, 3, 4}


def test_event_audit_uses_stable_filtered_matches_across_pruning() -> None:
    nodes = graph()
    linked = graph(edges=LINKED_EDGES)
    pruned = prune_to_tracks(linked, min_track_length=3)
    truth = graph(edges=TRUTH_EDGES)
    preprune = generate_division_proposals(linked, scale=UNIT_SCALE)
    postprune = generate_division_proposals(pruned, scale=UNIT_SCALE)

    rows = build_event_rows(
        "synthetic",
        raw=nodes,
        filtered=nodes,
        linked=linked,
        pruned=pruned,
        truth=truth,
        preprune_proposals=preprune,
        postprune_proposals=postprune,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["raw_complete_triplet"]
    assert row["filtered_complete_triplet"]
    assert row["preprune_second_daughter_candidate"]
    assert row["preprune_broad_proposals"] == 1
    assert not row["pruned_complete_triplet_survives"]
    assert not row["postprune_second_daughter_candidate"]
    assert row["postprune_broad_proposals"] == 0


def test_proposal_generation_rejects_duplicate_node_identifiers() -> None:
    invalid = TrackingGraph(
        node_ids=np.array([1, 1], dtype=np.int64),
        t=np.array([0, 1]),
        z=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        x=np.array([0.0, 1.0]),
        edges=np.empty((0, 2), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="duplicate node"):
        generate_division_proposals(invalid, scale=UNIT_SCALE)


def test_pair_proposals_are_unordered_and_preserve_track_context() -> None:
    linked_rows = [
        (1, 0, 0.0, 0.0, 0.0),
        (10, 1, 0.0, 0.0, 0.0),
        (11, 1, 0.0, -4.0, 0.0),
        (20, 2, 0.0, -5.0, 0.0),
        (30, 2, 0.0, 0.0, 5.0),
        (40, 2, 0.0, 5.0, 0.0),
        (50, 3, 0.0, -9.0, 0.0),
        (60, 3, 0.0, 0.0, 9.0),
    ]
    raw_rows = [*linked_rows, (25, 2, 0.0, 0.0, -5.0)]
    linked = graph(
        linked_rows,
        [(1, 10), (10, 30), (11, 20), (20, 50), (30, 60)],
    )

    neighborhoods = generate_division_pair_neighborhoods(
        linked,
        graph(raw_rows),
        pruned_node_ids={1, 10, 30, 60},
        max_distance_um=14.0,
        scale=UNIT_SCALE,
    )
    mother = next(item for item in neighborhoods if item.mother_id == 10)
    contexts = {candidate.node_id: candidate for candidate in mother.candidates}

    assert mother.mother_has_history
    assert mother.mother_has_current_child
    assert mother.mother_survives_pruning
    assert contexts[30].is_current_child
    assert contexts[30].has_future
    assert contexts[30].survives_pruning
    assert contexts[20].has_wrong_parent
    assert not contexts[20].is_parentless
    assert contexts[40].is_parentless
    assert not contexts[40].has_future
    assert not contexts[25].present_in_linked_graph
    assert not contexts[25].is_parentless
    assert not contexts[25].has_wrong_parent

    proposals = generate_division_pair_proposals(
        [mother],
        min_distance_um=0.0,
        max_distance_um=14.0,
        min_angle_deg=0.0,
    )
    pairs = [
        (proposal.candidate_1.node_id, proposal.candidate_2.node_id)
        for proposal in proposals
    ]
    assert pairs == [(20, 25), (20, 30), (20, 40), (25, 30), (25, 40), (30, 40)]


def test_pair_proposals_allow_two_candidates_with_wrong_parents() -> None:
    rows = [
        (10, 1, 0.0, 0.0, 0.0),
        (11, 1, 0.0, -4.0, 0.0),
        (12, 1, 0.0, 4.0, 0.0),
        (20, 2, 0.0, -5.0, 0.0),
        (30, 2, 0.0, 5.0, 0.0),
    ]
    linked = graph(rows, [(11, 20), (12, 30)])
    neighborhoods = generate_division_pair_neighborhoods(
        linked,
        graph(rows),
        scale=UNIT_SCALE,
    )
    mother = next(item for item in neighborhoods if item.mother_id == 10)
    proposals = generate_division_pair_proposals([mother])

    assert len(proposals) == 1
    assert proposals[0].candidate_1.has_wrong_parent
    assert proposals[0].candidate_2.has_wrong_parent


def test_pair_distance_floor_is_applied_before_nearest_candidate_cap() -> None:
    rows = [
        (10, 1, 0.0, 0.0, 0.0),
        (20, 2, 0.0, 0.0, 1.0),
        (30, 2, 0.0, 0.0, 3.0),
        (40, 2, 0.0, 0.0, -4.0),
    ]
    neighborhoods = generate_division_pair_neighborhoods(
        graph(rows),
        graph(rows),
        max_distance_um=16.0,
        scale=UNIT_SCALE,
    )
    mother = next(item for item in neighborhoods if item.mother_id == 10)
    proposals = generate_division_pair_proposals(
        [mother],
        min_distance_um=2.0,
        max_distance_um=14.0,
        min_angle_deg=120.0,
        max_candidates_per_mother=2,
    )

    assert len(proposals) == 1
    assert (proposals[0].candidate_1.node_id, proposals[0].candidate_2.node_id) == (
        30,
        40,
    )
    assert proposals[0].daughter_angle_deg == pytest.approx(180.0)


def test_raw_candidate_pool_can_restore_a_filtered_pair() -> None:
    raw_rows = [
        (10, 1, 0.0, 0.0, 0.0),
        (20, 2, 0.0, 0.0, 5.0),
        (30, 2, 0.0, 0.0, -5.0),
    ]
    filtered_rows = raw_rows[:2]
    linked = graph(filtered_rows, [(10, 20)])

    filtered_neighborhoods = generate_division_pair_neighborhoods(
        linked,
        graph(filtered_rows),
        scale=UNIT_SCALE,
    )
    raw_neighborhoods = generate_division_pair_neighborhoods(
        linked,
        graph(raw_rows),
        scale=UNIT_SCALE,
    )

    assert filtered_neighborhoods == []
    assert len(raw_neighborhoods) == 1
    assert len(generate_division_pair_proposals(raw_neighborhoods)) == 1


def test_pair_neighborhoods_reject_inconsistent_shared_coordinates() -> None:
    linked = graph(
        [(10, 1, 0.0, 0.0, 0.0), (20, 2, 0.0, 0.0, 5.0)],
        [(10, 20)],
    )
    candidates = graph(
        [
            (10, 1, 0.0, 0.0, 0.0),
            (20, 2, 0.0, 0.0, 6.0),
            (30, 2, 0.0, 0.0, -5.0),
        ]
    )

    with pytest.raises(ValueError, match="shared node coordinates"):
        generate_division_pair_neighborhoods(linked, candidates, scale=UNIT_SCALE)


@pytest.mark.parametrize(
    ("node_ids", "times", "edges", "message"),
    [
        ([1.5, 2.0], [0, 1], [], "node identifiers"),
        ([1, 2], [0.5, 1.0], [], "time coordinates"),
        ([1, 2], [0, 1], [[1.0, 1.5]], "edge identifiers"),
    ],
)
def test_pair_graph_validation_rejects_fractional_identity_fields(
    node_ids,
    times,
    edges,
    message,
) -> None:
    invalid = TrackingGraph(
        node_ids=np.asarray(node_ids),
        t=np.asarray(times),
        z=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        x=np.array([0.0, 1.0]),
        edges=np.asarray(edges).reshape(-1, 2),
    )

    with pytest.raises(ValueError, match=message):
        generate_division_pair_neighborhoods(invalid, invalid, scale=UNIT_SCALE)


def test_pair_graph_validation_preserves_large_integer_identifiers() -> None:
    mother = 2**53 + 1
    rows = [
        (mother, 1, 0.0, 0.0, 0.0),
        (mother + 2, 2, 0.0, 0.0, 5.0),
        (mother + 4, 2, 0.0, 0.0, -5.0),
    ]
    neighborhoods = generate_division_pair_neighborhoods(
        graph(rows),
        graph(rows),
        scale=UNIT_SCALE,
    )
    proposals = generate_division_pair_proposals(neighborhoods)

    assert proposals[0].mother_id == mother
    assert proposals[0].candidate_1.node_id == mother + 2
    assert proposals[0].candidate_2.node_id == mother + 4


def test_pair_rule_grid_and_selection_are_frozen_and_deterministic() -> None:
    rules = build_rules()
    assert len(rules) == 480
    assert len({rule.rule_id for rule in rules}) == 480

    common = {
        "candidate_source": "raw_scored",
        "labelled_event_coverage": 18,
        "proposal_count": 100,
        "unique_proposed_mothers": 20,
    }
    restrictive = {
        **common,
        **PairRule(10.0, 4, 2.0, 60.0).as_dict(),
    }
    least_restrictive = {
        **common,
        **PairRule(16.0, None, 0.0, 0.0).as_dict(),
    }
    lower_volume = {
        **common,
        **PairRule(8.0, 2, 4.0, 120.0).as_dict(),
        "proposal_count": 99,
    }
    fewer_mothers = {
        **common,
        **PairRule(10.0, 4, 2.0, 60.0).as_dict(),
        "unique_proposed_mothers": 19,
    }
    filtered_decoy = {
        **common,
        **PairRule(8.0, 2, 4.0, 120.0).as_dict(),
        "candidate_source": "filtered",
        "proposal_count": 1,
    }

    assert select_pair_rule([restrictive, least_restrictive])["rule_id"] == (
        least_restrictive["rule_id"]
    )
    assert select_pair_rule([least_restrictive, lower_volume])["rule_id"] == (
        lower_volume["rule_id"]
    )
    assert select_pair_rule([least_restrictive, fewer_mothers])["rule_id"] == (
        fewer_mothers["rule_id"]
    )
    assert select_pair_rule([least_restrictive, filtered_decoy])["rule_id"] == (
        least_restrictive["rule_id"]
    )
    with pytest.raises(RuntimeError, match="18 of 18"):
        select_pair_rule([{**least_restrictive, "labelled_event_coverage": 17}])


def test_pair_rule_histogram_matches_direct_generation() -> None:
    rows = [
        (10, 1, 0.0, 0.0, 0.0),
        (20, 2, 0.0, 0.0, 1.0),
        (30, 2, 0.0, 0.0, 3.0),
        (40, 2, 0.0, 0.0, -4.0),
        (50, 2, 0.0, 8.0, 0.0),
    ]
    neighborhoods = generate_division_pair_neighborhoods(
        graph(rows),
        graph(rows),
        max_distance_um=16.0,
        scale=UNIT_SCALE,
    )
    counts = aggregate_pair_neighborhoods(
        neighborhoods,
        mother_matches={10: 100},
        candidate_matches={20: 200, 30: 201, 40: 202},
        truth_divisions={100: (200, 201)},
    )
    broad = counts[PairRule(16.0, None, 0.0, 0.0).rule_id]
    assert broad["labelled_event_coverage"] == 1
    assert broad["labelled_true_triplet_proposals"] == 1
    assert broad["labelled_division_mother_unknown_children_proposals"] == 3
    assert broad["labelled_division_mother_wrong_children_proposals"] == 2
    assert broad["unique_matched_nondivision_mothers"] == 0

    for rule in build_rules():
        direct = generate_division_pair_proposals(
            neighborhoods,
            min_distance_um=rule.min_distance_um,
            max_distance_um=rule.max_distance_um,
            min_angle_deg=rule.min_angle_deg,
            max_candidates_per_mother=rule.max_candidates_per_mother,
        )
        assert counts[rule.rule_id]["proposal_count"] == len(direct)
        assert counts[rule.rule_id]["unique_proposed_mothers"] == bool(direct)
