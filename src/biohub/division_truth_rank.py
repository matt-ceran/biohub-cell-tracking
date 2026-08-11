"""Deterministic truth-ranking and assignment diagnostics for Phase 11 divisions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from biohub.division_inference import AcceptedFork, ScoredProposal

ASSIGNMENT_THRESHOLD = "below_threshold"
ASSIGNMENT_PER_MOTHER_RANKING = "per_mother_ranking"
ASSIGNMENT_DAUGHTER_CONFLICT = "daughter_conflict"
ASSIGNMENT_FIVE_FORK_CAP = "five_fork_cap"
ASSIGNMENT_ACCEPTED = "accepted"
ASSIGNMENT_INCONSISTENT = "assignment_inconsistent"

ASSIGNMENT_STAGES = (
    ASSIGNMENT_THRESHOLD,
    ASSIGNMENT_PER_MOTHER_RANKING,
    ASSIGNMENT_DAUGHTER_CONFLICT,
    ASSIGNMENT_FIVE_FORK_CAP,
    ASSIGNMENT_ACCEPTED,
    ASSIGNMENT_INCONSISTENT,
)


def _candidate_pair(values: Sequence[int]) -> tuple[int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError("true_candidate_ids must contain exactly two identifiers")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError("true candidate identifiers must be non-negative integers")
        parsed.append(int(value))
    first, second = sorted(parsed)
    if first == second:
        raise ValueError("true candidate identifiers must be distinct")
    return first, second


def proposal_rank_key(proposal: ScoredProposal) -> tuple[float, int, int, int, int]:
    """Return the exact frozen per-mother proposal ordering key."""
    if not isinstance(proposal, ScoredProposal):
        raise TypeError("proposal ranking requires ScoredProposal values")
    return (
        -proposal.logit,
        proposal.edge_removal_count,
        proposal.restored_node_count,
        proposal.candidate_1_id,
        proposal.candidate_2_id,
    )


def _global_key(
    proposal: ScoredProposal | AcceptedFork,
) -> tuple[float, int, int, int, int, int]:
    if not isinstance(proposal, (ScoredProposal, AcceptedFork)):
        raise TypeError("global assignment ranking requires a scored proposal or accepted fork")
    return (
        -proposal.logit,
        proposal.edge_removal_count,
        proposal.restored_node_count,
        proposal.mother_id,
        proposal.candidate_1_id,
        proposal.candidate_2_id,
    )


@dataclass(frozen=True, slots=True)
class TruthProposalRanking:
    """The exact position of one known true pair among one mother's proposals."""

    mother_id: int
    true_candidate_ids: tuple[int, int]
    ordered_proposals: tuple[ScoredProposal, ...]
    true_rank: int

    @property
    def proposal_count(self) -> int:
        return len(self.ordered_proposals)

    @property
    def true_proposal(self) -> ScoredProposal:
        return self.ordered_proposals[self.true_rank - 1]

    @property
    def best_proposal(self) -> ScoredProposal:
        return self.ordered_proposals[0]

    @property
    def best_false_proposal(self) -> ScoredProposal | None:
        return next(
            (
                proposal
                for proposal in self.ordered_proposals
                if proposal.candidate_ids != self.true_candidate_ids
            ),
            None,
        )

    @property
    def false_minus_true_logit(self) -> float | None:
        false = self.best_false_proposal
        return None if false is None else false.logit - self.true_proposal.logit

    @property
    def false_minus_true_probability(self) -> float | None:
        false = self.best_false_proposal
        return None if false is None else false.probability - self.true_proposal.probability


@dataclass(frozen=True, slots=True)
class ThresholdOutcome:
    """Why one reachable truth did or did not survive one tested threshold."""

    threshold: float
    stage: str
    true_probability: float
    accepted_fork_count: int
    blocking_fork: AcceptedFork | None = None

    def __post_init__(self) -> None:
        if self.stage not in ASSIGNMENT_STAGES:
            raise ValueError("threshold outcome has an unsupported assignment stage")


def rank_truth_proposal(
    proposals: Iterable[ScoredProposal],
    *,
    true_candidate_ids: Sequence[int],
) -> TruthProposalRanking:
    """Rank one known true daughter pair under the frozen per-mother ordering."""
    materialized = tuple(proposals)
    if not materialized:
        raise ValueError("truth ranking requires at least one proposal")
    if any(not isinstance(proposal, ScoredProposal) for proposal in materialized):
        raise TypeError("truth ranking requires ScoredProposal values")
    mothers = {proposal.mother_id for proposal in materialized}
    if len(mothers) != 1:
        raise ValueError("truth ranking proposals must all belong to one mother")
    fallback_hashes = {proposal.fallback_graph_sha256 for proposal in materialized}
    if len(fallback_hashes) != 1:
        raise ValueError("truth ranking proposals must reference one fallback graph")
    pairs = [proposal.candidate_ids for proposal in materialized]
    if len(set(pairs)) != len(pairs):
        raise ValueError("truth ranking proposals contain a duplicate candidate pair")

    truth_pair = _candidate_pair(true_candidate_ids)
    ordered = tuple(sorted(materialized, key=proposal_rank_key))
    matches = [
        index for index, proposal in enumerate(ordered) if proposal.candidate_ids == truth_pair
    ]
    if len(matches) != 1:
        raise ValueError("the true candidate pair must occur exactly once")
    return TruthProposalRanking(
        mother_id=next(iter(mothers)),
        true_candidate_ids=truth_pair,
        ordered_proposals=ordered,
        true_rank=matches[0] + 1,
    )


def _validated_threshold(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    return threshold


def classify_threshold_outcome(
    ranking: TruthProposalRanking,
    *,
    threshold: Real,
    accepted_forks: Iterable[AcceptedFork],
) -> ThresholdOutcome:
    """Trace one true proposal through thresholding and frozen movie assignment."""
    if not isinstance(ranking, TruthProposalRanking):
        raise TypeError("assignment tracing requires a TruthProposalRanking")
    cutoff = _validated_threshold(threshold)
    accepted = tuple(accepted_forks)
    if any(not isinstance(fork, AcceptedFork) for fork in accepted):
        raise TypeError("assignment tracing requires AcceptedFork values")
    if any(
        fork.fallback_graph_sha256 != ranking.true_proposal.fallback_graph_sha256
        for fork in accepted
    ):
        raise ValueError("accepted forks and truth ranking reference different fallback graphs")

    true = ranking.true_proposal
    common = {
        "threshold": cutoff,
        "true_probability": true.probability,
        "accepted_fork_count": len(accepted),
    }
    if true.probability < cutoff:
        return ThresholdOutcome(stage=ASSIGNMENT_THRESHOLD, **common)
    if ranking.true_rank != 1:
        blocker = ranking.best_proposal
        return ThresholdOutcome(
            stage=ASSIGNMENT_PER_MOTHER_RANKING,
            blocking_fork=AcceptedFork.from_scored(blocker),
            **common,
        )
    if any(
        fork.mother_id == true.mother_id and fork.candidate_ids == true.candidate_ids
        for fork in accepted
    ):
        return ThresholdOutcome(stage=ASSIGNMENT_ACCEPTED, **common)

    true_key = _global_key(true)
    earlier = tuple(
        sorted(
            (fork for fork in accepted if _global_key(fork) < true_key),
            key=_global_key,
        )
    )
    if len(earlier) >= 5:
        return ThresholdOutcome(
            stage=ASSIGNMENT_FIVE_FORK_CAP,
            blocking_fork=earlier[4],
            **common,
        )
    for fork in earlier:
        if set(fork.candidate_ids).intersection(true.candidate_ids):
            return ThresholdOutcome(
                stage=ASSIGNMENT_DAUGHTER_CONFLICT,
                blocking_fork=fork,
                **common,
            )
    return ThresholdOutcome(stage=ASSIGNMENT_INCONSISTENT, **common)


__all__ = [
    "ASSIGNMENT_ACCEPTED",
    "ASSIGNMENT_DAUGHTER_CONFLICT",
    "ASSIGNMENT_FIVE_FORK_CAP",
    "ASSIGNMENT_INCONSISTENT",
    "ASSIGNMENT_PER_MOTHER_RANKING",
    "ASSIGNMENT_STAGES",
    "ASSIGNMENT_THRESHOLD",
    "ThresholdOutcome",
    "TruthProposalRanking",
    "classify_threshold_outcome",
    "proposal_rank_key",
    "rank_truth_proposal",
]
