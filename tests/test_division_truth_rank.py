from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from biohub.division_inference import AcceptedFork, ScoredProposal
from biohub.division_truth_rank import (
    ASSIGNMENT_ACCEPTED,
    ASSIGNMENT_DAUGHTER_CONFLICT,
    ASSIGNMENT_FIVE_FORK_CAP,
    ASSIGNMENT_INCONSISTENT,
    ASSIGNMENT_PER_MOTHER_RANKING,
    ASSIGNMENT_THRESHOLD,
    classify_threshold_outcome,
    rank_truth_proposal,
)
from scripts.audit_division_truth_ranks import verify_existing_atlas

FALLBACK_SHA256 = hashlib.sha256(b"fallback").hexdigest()


def test_verify_existing_atlas_checks_manifest_and_contact_sheet_hashes(
    tmp_path: Path,
) -> None:
    from biohub.division_data import file_sha256, write_canonical_json

    atlas_dir = tmp_path / "atlas"
    images = atlas_dir / "images"
    images.mkdir(parents=True)
    contact = images / "event.png"
    contact.write_bytes(b"png payload")
    atlas = atlas_dir / "atlas.json"
    write_canonical_json(
        atlas,
        {
            "complete": True,
            "evaluation_boundary": {
                "evaluation_cache_count": 0,
                "evaluation_opened": False,
                "phase11_development_result_exists": False,
                "phase11_policy_exists": False,
            },
            "summary": {
                "reachable_truth_event_count": 1,
                "truth_event_count": 1,
            },
            "truth_rank_atlas_schema_version": 1,
        },
    )
    write_canonical_json(
        atlas_dir / "manifest.json",
        {
            "atlas": {"path": "atlas.json", "sha256": file_sha256(atlas)},
            "contact_sheets": [
                {"path": "images/event.png", "sha256": file_sha256(contact)}
            ],
            "logical_content_sha256": "unused-by-this-fixture",
            "manifest_schema_version": 1,
        },
    )

    payload = verify_existing_atlas(atlas_dir)

    assert payload["complete"]
    contact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="contact-sheet SHA-256"):
        verify_existing_atlas(atlas_dir)


def proposal(
    mother: int,
    first: int,
    second: int,
    logit: float,
    *,
    removals: int = 0,
    restored: int = 0,
) -> ScoredProposal:
    return ScoredProposal(
        mother_id=mother,
        candidate_1_id=first,
        candidate_2_id=second,
        logit=logit,
        edge_removal_count=removals,
        restored_node_count=restored,
        fallback_graph_sha256=FALLBACK_SHA256,
    )


def accepted(value: ScoredProposal) -> AcceptedFork:
    return AcceptedFork.from_scored(value)


def test_rank_truth_proposal_uses_exact_frozen_tie_breaks() -> None:
    truth = proposal(10, 20, 21, 5.0, removals=1)
    lower_logit = proposal(10, 22, 23, 4.0)
    same_logit_fewer_edits = proposal(10, 24, 25, 5.0, removals=0)
    same_logit_same_edits_later_ids = proposal(10, 26, 27, 5.0, removals=0)

    ranking = rank_truth_proposal(
        [lower_logit, truth, same_logit_same_edits_later_ids, same_logit_fewer_edits],
        true_candidate_ids=(21, 20),
    )

    assert ranking.mother_id == 10
    assert ranking.proposal_count == 4
    assert ranking.true_rank == 3
    assert ranking.true_proposal == truth
    assert ranking.best_proposal == same_logit_fewer_edits
    assert ranking.best_false_proposal == same_logit_fewer_edits
    assert ranking.ordered_proposals == (
        same_logit_fewer_edits,
        same_logit_same_edits_later_ids,
        truth,
        lower_logit,
    )
    assert ranking.false_minus_true_logit == 0.0
    assert ranking.false_minus_true_probability == pytest.approx(0.0)


def test_rank_truth_proposal_rejects_missing_or_ambiguous_inputs() -> None:
    first = proposal(10, 20, 21, 1.0)
    duplicate = proposal(10, 20, 21, 2.0)

    with pytest.raises(ValueError, match="exactly once"):
        rank_truth_proposal([first], true_candidate_ids=(22, 23))
    with pytest.raises(ValueError, match="duplicate"):
        rank_truth_proposal([first, duplicate], true_candidate_ids=(20, 21))
    with pytest.raises(ValueError, match="one mother"):
        rank_truth_proposal(
            [first, proposal(11, 22, 23, 2.0)],
            true_candidate_ids=(20, 21),
        )


def test_threshold_outcome_reports_below_threshold_before_ranking() -> None:
    truth = proposal(10, 20, 21, 0.0)
    false = proposal(10, 22, 23, 1.0)
    ranking = rank_truth_proposal([truth, false], true_candidate_ids=(20, 21))

    outcome = classify_threshold_outcome(ranking, threshold=0.7, accepted_forks=())

    assert outcome.stage == ASSIGNMENT_THRESHOLD
    assert outcome.blocking_fork is None
    assert outcome.true_probability == pytest.approx(0.5)


def test_threshold_outcome_reports_per_mother_ranking() -> None:
    truth = proposal(10, 20, 21, 2.0)
    false = proposal(10, 22, 23, 3.0)
    ranking = rank_truth_proposal([truth, false], true_candidate_ids=(20, 21))

    outcome = classify_threshold_outcome(ranking, threshold=0.5, accepted_forks=())

    assert outcome.stage == ASSIGNMENT_PER_MOTHER_RANKING
    assert outcome.blocking_fork == accepted(false)


def test_threshold_outcome_reports_accepted_truth() -> None:
    truth = proposal(10, 20, 21, 2.0)
    false = proposal(10, 22, 23, 1.0)
    ranking = rank_truth_proposal([truth, false], true_candidate_ids=(20, 21))

    outcome = classify_threshold_outcome(
        ranking,
        threshold=0.5,
        accepted_forks=(accepted(truth),),
    )

    assert outcome.stage == ASSIGNMENT_ACCEPTED
    assert outcome.blocking_fork is None


def test_threshold_outcome_reports_daughter_conflict() -> None:
    truth = proposal(10, 20, 21, 5.0)
    false = proposal(10, 22, 23, 4.0)
    blocker = proposal(11, 20, 30, 6.0)
    ranking = rank_truth_proposal([truth, false], true_candidate_ids=(20, 21))

    outcome = classify_threshold_outcome(
        ranking,
        threshold=0.5,
        accepted_forks=(accepted(blocker),),
    )

    assert outcome.stage == ASSIGNMENT_DAUGHTER_CONFLICT
    assert outcome.blocking_fork == accepted(blocker)


def test_threshold_outcome_reports_five_fork_cap() -> None:
    truth = proposal(10, 20, 21, 4.0)
    alternative = proposal(10, 22, 23, 3.0)
    ranking = rank_truth_proposal([truth, alternative], true_candidate_ids=(20, 21))
    earlier = tuple(
        accepted(proposal(100 + index, 200 + 2 * index, 201 + 2 * index, 10.0 - index))
        for index in range(5)
    )

    outcome = classify_threshold_outcome(
        ranking,
        threshold=0.5,
        accepted_forks=earlier,
    )

    assert outcome.stage == ASSIGNMENT_FIVE_FORK_CAP
    assert outcome.blocking_fork == earlier[-1]


def test_threshold_outcome_rejects_an_impossible_grid_trace() -> None:
    truth = proposal(10, 20, 21, math.log(9.0))
    alternative = proposal(10, 22, 23, 1.0)
    later_unrelated = accepted(proposal(11, 30, 31, 0.5))
    ranking = rank_truth_proposal([truth, alternative], true_candidate_ids=(20, 21))

    outcome = classify_threshold_outcome(
        ranking,
        threshold=0.5,
        accepted_forks=(later_unrelated,),
    )

    assert outcome.stage == ASSIGNMENT_INCONSISTENT
    assert outcome.blocking_fork is None
