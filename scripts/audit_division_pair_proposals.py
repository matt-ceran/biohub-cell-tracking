#!/usr/bin/env python3
"""Audit bounded mother-plus-two-candidates proposals on the frozen Phase 10 cohort."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import filter_candidates  # noqa: E402
from biohub.appearance_evaluation import (  # noqa: E402
    checkpoint_dev_datasets,
    file_sha256,
    files_sha256,
    read_frozen_policy,
)
from biohub.appearance_model import load_cellness_checkpoint  # noqa: E402
from biohub.division import (  # noqa: E402
    DivisionPairNeighborhood,
    DivisionPairProposal,
    generate_division_pair_neighborhoods,
    generate_division_pair_proposals,
)
from biohub.io import SamplePaths, discover_split, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import MetricResult, TrackingGraph, evaluate  # noqa: E402
from scripts.audit_division_proposals import (  # noqa: E402
    BASELINE_EVALUATION_SOURCE_FILES,
    FROZEN_MIN_TRACK_LENGTH,
    FROZEN_SCORE_THRESHOLD,
    SCORING_SOURCE_FILES,
    _find_score_cache,
    _inverse_matches,
    _read_phase10_rows,
    _truth_divisions,
    _write_csv,
    _write_json,
    verify_phase10_reproduction,
)

PAIR_AUDIT_SCHEMA_VERSION = 1
REQUIRED_EVENT_COVERAGE = 18
EXPECTED_MOVIES = 13
EXPECTED_NO_DIVISION_CONTROLS = 3
EXPECTED_CHECKPOINT_SHA256 = (
    "bd35d082c985641728bc340817d08d4750a3c83c34d25f99d1507cbc8b7bb8d4"
)
EXPECTED_POLICY_SHA256 = (
    "1a333aa0cf3c1c5b92cdfc468fdbc483567a8bde2f4d2e73bd2332069144f3fc"
)
EXPECTED_PHASE10_RESULTS_SHA256 = (
    "acd59f7182050769922df05ae1d143ee1d97edbbb4558b186d8385bd9be82911"
)
EXPECTED_SCORING_SOURCE_SHA256 = (
    "426c09f04c99af55fd5457921fbf435de2f71bdf7ce08dbdf0e0fde6c02443c9"
)
EXPECTED_BASELINE_EVALUATION_SHA256 = (
    "be227a8d94434ba401209ba0ada2864e5ad4a49b8ace941b86c5c5f109d571f9"
)
EXPECTED_DATASETS = (
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
EXPECTED_SCORE_CACHE_SHA256 = {
    "6bba_474be664": "7ae309cd9c62322454dd18b57188365ee4bfede17fb5a695c88e86ccfdadba5c",
    "6bba_4f99ce20": "3a9406b399295aade1e9afa35b829cb296205774245220b77ddbe612f161aedd",
    "6bba_6ca87370": "b091579f90121e1564133900250c6bf6f8e23f720c1cf42e01a30f26d27b75a5",
    "6bba_789f8168": "207d6cf9f4057d87d02062151aaf0e9bfba1d632d2620f4e5bf92d744920468a",
    "6bba_7af54fde": "07ce97eb64a476d5e7cc67c15b1b0bf28991e5b7e38f88ed0481813d2906db1a",
    "6bba_825bd1c6": "621de711f04d47e060d34f56bb62dc78d910b4b7f04055e1b7afca90e5e8ef81",
    "6bba_907271db": "743889f8287d23f9d7483e097059ac6ca87ea906bc4d55a9d0dadb778ca41129",
    "6bba_afb141ff": "9151ccea90bc9a20fd9bef3855481a1b9a7daa5362b59f9c87422b0a637b3426",
    "6bba_d1acb6ff": "de2369306e9b8f5ab1751d17e9e13c2e0bb1907836fe96321822fe096c9e0c05",
    "6bba_ebdf3b34": "1fa2b5761ba3756a094751ddd99c85cd924722bf65e41d057e200472cad3465f",
    "6bba_ef7b4f7e": "8f85370b4f15238d607e5ca9e814c77777aff9eaa60e17c66134b29b1d921501",
    "6bba_f4ae811c": "c4a95c269aa688cb16fb786f9c6820651818f65d8a5427b49c1b77716b3b7c37",
    "6bba_fc5f39dc": "9c33d1d6dd8a8ff7f9a41bad7ee06711dd439b2dc9ba3adc9760838320773e48",
}

MAX_DISTANCES_UM = (8.0, 10.0, 12.0, 14.0, 16.0)
MAX_CANDIDATES = (2, 3, 4, 6, 8, 12, 16, None)
MIN_DISTANCES_UM = (0.0, 2.0, 4.0)
MIN_ANGLES_DEG = (0.0, 60.0, 90.0, 120.0)
CAP_VALUES = np.array(
    [float(value) if value is not None else np.inf for value in MAX_CANDIDATES],
    dtype=float,
)

ROLE_NAMES = (
    "labelled_true_triplet",
    "labelled_division_mother_unknown_children",
    "labelled_division_mother_wrong_children",
    "matched_nondivision_mother",
    "unmatched_mother_unknown",
)
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}

PAIR_AUDIT_SOURCE_FILES = [
    SRC / "biohub" / "division.py",
    Path(__file__).with_name("audit_division_proposals.py"),
    Path(__file__).resolve(),
]


def _number_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


@dataclass(frozen=True)
class PairRule:
    """One immutable geometry-only row in the predeclared proposal grid."""

    max_distance_um: float
    max_candidates_per_mother: int | None
    min_distance_um: float
    min_angle_deg: float

    @property
    def rule_id(self) -> str:
        cap = "all" if self.max_candidates_per_mother is None else str(
            self.max_candidates_per_mother
        )
        return (
            f"dmax{_number_token(self.max_distance_um)}_cap{cap}_"
            f"dmin{_number_token(self.min_distance_um)}_"
            f"angle{_number_token(self.min_angle_deg)}"
        )

    @property
    def indices(self) -> tuple[int, int, int, int]:
        return (
            MIN_DISTANCES_UM.index(self.min_distance_um),
            MAX_DISTANCES_UM.index(self.max_distance_um),
            MAX_CANDIDATES.index(self.max_candidates_per_mother),
            MIN_ANGLES_DEG.index(self.min_angle_deg),
        )

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "rule_id": self.rule_id,
            "max_distance_um": self.max_distance_um,
            "max_candidates_per_mother": (
                "all"
                if self.max_candidates_per_mother is None
                else self.max_candidates_per_mother
            ),
            "min_distance_um": self.min_distance_um,
            "min_angle_deg": self.min_angle_deg,
        }


def build_rules() -> list[PairRule]:
    """Return the exact 480-row geometry grid in stable order."""
    return [
        PairRule(maximum, cap, minimum, angle)
        for maximum, cap, minimum, angle in product(
            MAX_DISTANCES_UM,
            MAX_CANDIDATES,
            MIN_DISTANCES_UM,
            MIN_ANGLES_DEG,
        )
    ]


def _cap_for_sort(value: Any) -> float:
    if value is None or str(value).lower() in {"all", "none", ""}:
        return float("inf")
    return float(value)


def select_pair_rule(
    rows: list[dict[str, Any]],
    *,
    required_event_coverage: int = REQUIRED_EVENT_COVERAGE,
) -> dict[str, Any]:
    """Select the smallest raw-scored rule under the frozen deterministic ordering."""
    eligible = [
        row
        for row in rows
        if row.get("candidate_source") == "raw_scored"
        and int(row["labelled_event_coverage"]) == required_event_coverage
    ]
    if not eligible:
        raise RuntimeError(
            f"no raw-scored proposal rule retained {required_event_coverage} of "
            f"{REQUIRED_EVENT_COVERAGE} labelled events"
        )
    return min(
        eligible,
        key=lambda row: (
            int(row["proposal_count"]),
            int(row["unique_proposed_mothers"]),
            float(row["min_distance_um"]),
            float(row["min_angle_deg"]),
            -float(row["max_distance_um"]),
            -_cap_for_sort(row["max_candidates_per_mother"]),
            str(row["rule_id"]),
        ),
    )


def _cumulative_counts(exact: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(exact, axis=-3)
    cumulative = np.cumsum(cumulative, axis=-2)
    return np.flip(
        np.cumsum(np.flip(cumulative, axis=-1), axis=-1),
        axis=-1,
    )


def _cumulative_any(exact: np.ndarray) -> np.ndarray:
    cumulative = np.maximum.accumulate(exact, axis=-3)
    cumulative = np.maximum.accumulate(cumulative, axis=-2)
    return np.flip(
        np.maximum.accumulate(np.flip(cumulative, axis=-1), axis=-1),
        axis=-1,
    )


def _pair_role_codes(
    neighborhood: DivisionPairNeighborhood,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    *,
    mother_matches: dict[int, int],
    candidate_matches: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> tuple[np.ndarray, int | None, bool]:
    truth_mother = mother_matches.get(neighborhood.mother_id)
    count = len(first_indices)
    if truth_mother is None:
        return (
            np.full(count, ROLE_INDEX["unmatched_mother_unknown"], dtype=np.int8),
            None,
            False,
        )
    if truth_mother not in truth_divisions:
        return (
            np.full(count, ROLE_INDEX["matched_nondivision_mother"], dtype=np.int8),
            truth_mother,
            True,
        )

    candidate_truth = np.array(
        [candidate_matches.get(candidate.node_id, -1) for candidate in neighborhood.candidates],
        dtype=np.int64,
    )
    first_truth = candidate_truth[first_indices]
    second_truth = candidate_truth[second_indices]
    unknown = (first_truth < 0) | (second_truth < 0)
    daughters = np.array(truth_divisions[truth_mother], dtype=np.int64)
    true_pair = (
        ~unknown
        & (first_truth != second_truth)
        & np.isin(first_truth, daughters)
        & np.isin(second_truth, daughters)
    )
    roles = np.full(
        count,
        ROLE_INDEX["labelled_division_mother_wrong_children"],
        dtype=np.int8,
    )
    roles[unknown] = ROLE_INDEX["labelled_division_mother_unknown_children"]
    roles[true_pair] = ROLE_INDEX["labelled_true_triplet"]
    return roles, truth_mother, False


def aggregate_pair_neighborhoods(
    neighborhoods: list[DivisionPairNeighborhood],
    *,
    mother_matches: dict[int, int],
    candidate_matches: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> dict[str, dict[str, int]]:
    """Aggregate every frozen grid row while enumerating each broad pair once."""
    grid_shape = (
        len(MIN_DISTANCES_UM),
        len(MAX_DISTANCES_UM),
        len(MAX_CANDIDATES),
        len(MIN_ANGLES_DEG),
    )
    exact_roles = np.zeros((len(ROLE_NAMES), *grid_shape), dtype=np.int64)
    unique_mothers = np.zeros(grid_shape, dtype=np.int64)
    matched_nondivision_mothers = np.zeros(grid_shape, dtype=np.int64)
    event_mothers = tuple(sorted(truth_divisions))
    event_index = {mother: index for index, mother in enumerate(event_mothers)}
    exact_events = np.zeros((len(event_mothers), *grid_shape), dtype=bool)

    for neighborhood in neighborhoods:
        candidate_count = len(neighborhood.candidates)
        if candidate_count < 2:
            continue
        first_indices, second_indices = np.triu_indices(candidate_count, k=1)
        distances = np.array(
            [candidate.distance_um for candidate in neighborhood.candidates],
            dtype=float,
        )
        displacements = np.array(
            [candidate.displacement_um for candidate in neighborhood.candidates],
            dtype=float,
        )
        dots = np.einsum(
            "ij,ij->i",
            displacements[first_indices],
            displacements[second_indices],
        )
        denominators = distances[first_indices] * distances[second_indices]
        cosines = np.clip(dots / denominators, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        max_distance_indices = np.searchsorted(
            MAX_DISTANCES_UM,
            distances[second_indices],
            side="left",
        )
        angle_indices = (
            np.searchsorted(MIN_ANGLES_DEG, angles, side="right") - 1
        )
        roles, truth_mother, is_matched_nondivision = _pair_role_codes(
            neighborhood,
            first_indices,
            second_indices,
            mother_matches=mother_matches,
            candidate_matches=candidate_matches,
            truth_divisions=truth_divisions,
        )
        local_exact = np.zeros(grid_shape, dtype=bool)
        for minimum_index, minimum in enumerate(MIN_DISTANCES_UM):
            first_allowed = int(np.searchsorted(distances, minimum, side="left"))
            valid = first_indices >= first_allowed
            if not np.any(valid):
                continue
            cap_indices = np.searchsorted(
                CAP_VALUES,
                second_indices[valid] - first_allowed + 1,
                side="left",
            )
            minimum_indices = np.full(np.count_nonzero(valid), minimum_index, dtype=int)
            bins = (
                minimum_indices,
                max_distance_indices[valid],
                cap_indices,
                angle_indices[valid],
            )
            np.add.at(exact_roles, (roles[valid], *bins), 1)
            local_exact[bins] = True
            if truth_mother in event_index:
                true_valid = valid & (roles == ROLE_INDEX["labelled_true_triplet"])
                if np.any(true_valid):
                    true_cap_indices = np.searchsorted(
                        CAP_VALUES,
                        second_indices[true_valid] - first_allowed + 1,
                        side="left",
                    )
                    true_bins = (
                        np.full(np.count_nonzero(true_valid), minimum_index, dtype=int),
                        max_distance_indices[true_valid],
                        true_cap_indices,
                        angle_indices[true_valid],
                    )
                    exact_events[(event_index[truth_mother], *true_bins)] = True

        local_available = _cumulative_any(local_exact)
        unique_mothers += local_available
        if is_matched_nondivision:
            matched_nondivision_mothers += local_available

    role_counts = _cumulative_counts(exact_roles)
    event_available = _cumulative_any(exact_events)
    results: dict[str, dict[str, int]] = {}
    for rule in build_rules():
        indices = rule.indices
        current_roles = role_counts[(slice(None), *indices)]
        row = {
            "proposal_count": int(np.sum(current_roles)),
            "unique_proposed_mothers": int(unique_mothers[indices]),
            "unique_matched_nondivision_mothers": int(
                matched_nondivision_mothers[indices]
            ),
            "labelled_event_coverage": int(
                np.count_nonzero(event_available[(slice(None), *indices)])
            ),
        }
        row.update(
            {
                f"{role}_proposals": int(current_roles[role_index])
                for role_index, role in enumerate(ROLE_NAMES)
            }
        )
        results[rule.rule_id] = row
    return results


RULE_PARAMETER_FIELDS = [
    "rule_id",
    "max_distance_um",
    "max_candidates_per_mother",
    "min_distance_um",
    "min_angle_deg",
]
COUNT_FIELDS = [
    "labelled_event_coverage",
    "proposal_count",
    "unique_proposed_mothers",
    "unique_matched_nondivision_mothers",
    *[f"{role}_proposals" for role in ROLE_NAMES],
]
RULE_MOVIE_FIELDS = [
    "dataset",
    "candidate_source",
    *RULE_PARAMETER_FIELDS,
    "truth_divisions",
    *COUNT_FIELDS,
    "is_no_division_control",
    "raw_candidates",
    "filtered_candidates",
    "pruned_nodes",
    "link_seconds",
    "edge_jaccard",
    "score",
    "baseline_reproduced",
    "score_cache_sha256",
]
RULE_SWEEP_FIELDS = [
    "candidate_source",
    *RULE_PARAMETER_FIELDS,
    *COUNT_FIELDS,
    "no_division_control_movies",
    "no_division_control_movies_with_proposals",
    "no_division_control_proposals",
    "no_division_control_unique_mothers",
    "no_division_control_matched_nondivision_mothers",
]
EVENT_FIELDS = [
    "dataset",
    "truth_mother_id",
    "truth_daughter_ids",
    "truth_daughter_count",
    "predicted_mother_id",
    "filtered_daughters_matched",
    "raw_daughters_matched",
    "filtered_complete_triplet",
    "raw_complete_triplet",
    "filtered_selected_pair_proposals",
    "raw_scored_selected_pair_proposals",
    "filtered_selected_covered",
    "raw_scored_selected_covered",
    "raw_scored_restores_filtered_event",
    "raw_scored_restores_below_threshold_daughter",
    "selected_rule_id",
]
PROPOSAL_FIELDS = [
    "dataset",
    "candidate_source",
    "selected_rule_id",
    "movie_truth_divisions",
    "mother_id",
    "candidate_1_id",
    "candidate_2_id",
    "daughter_time",
    "candidate_1_distance_um",
    "candidate_2_distance_um",
    "daughter_angle_deg",
    "mother_cellness",
    "candidate_1_cellness",
    "candidate_2_cellness",
    "min_triplet_cellness",
    "mother_has_history",
    "mother_has_current_child",
    "mother_survives_pruning",
    "candidate_1_present_in_filtered_graph",
    "candidate_1_is_current_child",
    "candidate_1_is_parentless",
    "candidate_1_has_wrong_parent",
    "candidate_1_has_future",
    "candidate_1_survives_pruning",
    "candidate_2_present_in_filtered_graph",
    "candidate_2_is_current_child",
    "candidate_2_is_parentless",
    "candidate_2_has_wrong_parent",
    "candidate_2_has_future",
    "candidate_2_survives_pruning",
    "truth_mother_id",
    "truth_candidate_1_id",
    "truth_candidate_2_id",
    "sparse_label_role",
]


@dataclass(frozen=True)
class MovieState:
    sample: SamplePaths
    cache_path: Path
    cache_sha256: str
    raw: TrackingGraph
    scores: np.ndarray
    filtered: TrackingGraph
    linked: TrackingGraph
    pruned: TrackingGraph
    truth: TrackingGraph
    result: MetricResult
    link_seconds: float


def _load_movie_state(
    sample: SamplePaths,
    *,
    score_cache_dir: Path,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
    detector_threshold_k: float,
    patch_shape: tuple[int, int, int],
    expected_baseline: dict[str, str],
) -> MovieState:
    cache_path, raw, scores, _ = _find_score_cache(
        score_cache_dir,
        sample.dataset,
        checkpoint_sha256=checkpoint_sha256,
        scoring_source_sha256=scoring_source_sha256,
        detector_threshold_k=detector_threshold_k,
        patch_shape=patch_shape,
    )
    cache_sha256 = file_sha256(cache_path)
    expected_cache_sha256 = EXPECTED_SCORE_CACHE_SHA256.get(sample.dataset)
    if cache_sha256 != expected_cache_sha256:
        raise RuntimeError(
            f"score cache SHA-256 changed for {sample.dataset}: expected "
            f"{expected_cache_sha256}, found {cache_sha256}"
        )
    if sample.labels is None:
        raise RuntimeError(f"labelled training movie is missing: {sample.dataset}")
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    filtered = filter_candidates(raw, scores, min_score=FROZEN_SCORE_THRESHOLD)
    link_started = time.perf_counter()
    linked = link_graph_flow(filtered)
    link_seconds = time.perf_counter() - link_started
    pruned = prune_to_tracks(linked, min_track_length=FROZEN_MIN_TRACK_LENGTH)
    result = evaluate(pruned, truth)
    verify_phase10_reproduction(
        expected=expected_baseline,
        raw_candidates=len(raw.node_ids),
        filtered_candidates=len(filtered.node_ids),
        result=result,
    )
    return MovieState(
        sample=sample,
        cache_path=cache_path,
        cache_sha256=cache_sha256,
        raw=raw,
        scores=scores,
        filtered=filtered,
        linked=linked,
        pruned=pruned,
        truth=truth,
        result=result,
        link_seconds=link_seconds,
    )


def _movie_rule_rows(state: MovieState, rules: list[PairRule]) -> list[dict[str, Any]]:
    filtered_matches, _ = _inverse_matches(state.filtered, state.truth)
    raw_matches, _ = _inverse_matches(state.raw, state.truth)
    truth_divisions = _truth_divisions(state.truth)
    pruned_ids = {int(node_id) for node_id in state.pruned.node_ids}
    base = {
        "dataset": state.sample.dataset,
        "truth_divisions": len(truth_divisions),
        "is_no_division_control": len(truth_divisions) == 0,
        "raw_candidates": len(state.raw.node_ids),
        "filtered_candidates": len(state.filtered.node_ids),
        "pruned_nodes": len(state.pruned.node_ids),
        "link_seconds": state.link_seconds,
        "edge_jaccard": state.result.edge_jaccard,
        "score": state.result.score,
        "baseline_reproduced": True,
        "score_cache_sha256": state.cache_sha256,
    }
    rows: list[dict[str, Any]] = []
    for source, candidates, candidate_matches in (
        ("filtered", state.filtered, filtered_matches),
        ("raw_scored", state.raw, raw_matches),
    ):
        neighborhoods = generate_division_pair_neighborhoods(
            state.linked,
            candidates,
            pruned_node_ids=pruned_ids,
            max_distance_um=max(MAX_DISTANCES_UM),
        )
        counts = aggregate_pair_neighborhoods(
            neighborhoods,
            mother_matches=filtered_matches,
            candidate_matches=candidate_matches,
            truth_divisions=truth_divisions,
        )
        for rule in rules:
            rows.append(
                {
                    **base,
                    "candidate_source": source,
                    **rule.as_dict(),
                    **counts[rule.rule_id],
                }
            )
    return rows


def aggregate_rule_rows(movie_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sum disjoint movie counts into one stable row per source and rule."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for movie in movie_rows:
        key = (str(movie["candidate_source"]), str(movie["rule_id"]))
        row = grouped.setdefault(
            key,
            {
                "candidate_source": movie["candidate_source"],
                **{field: movie[field] for field in RULE_PARAMETER_FIELDS},
                **{field: 0 for field in COUNT_FIELDS},
                "no_division_control_movies": 0,
                "no_division_control_movies_with_proposals": 0,
                "no_division_control_proposals": 0,
                "no_division_control_unique_mothers": 0,
                "no_division_control_matched_nondivision_mothers": 0,
            },
        )
        for field in COUNT_FIELDS:
            row[field] += int(movie[field])
        if bool(movie["is_no_division_control"]):
            row["no_division_control_movies"] += 1
            row["no_division_control_movies_with_proposals"] += int(
                int(movie["proposal_count"]) > 0
            )
            row["no_division_control_proposals"] += int(movie["proposal_count"])
            row["no_division_control_unique_mothers"] += int(
                movie["unique_proposed_mothers"]
            )
            row["no_division_control_matched_nondivision_mothers"] += int(
                movie["unique_matched_nondivision_mothers"]
            )
    return [grouped[key] for key in sorted(grouped)]


def _rule_from_row(row: dict[str, Any]) -> PairRule:
    cap_value = row["max_candidates_per_mother"]
    cap = None if str(cap_value).lower() in {"all", "none", ""} else int(cap_value)
    return PairRule(
        max_distance_um=float(row["max_distance_um"]),
        max_candidates_per_mother=cap,
        min_distance_um=float(row["min_distance_um"]),
        min_angle_deg=float(row["min_angle_deg"]),
    )


def _pair_label(
    proposal: DivisionPairProposal,
    *,
    mother_matches: dict[int, int],
    candidate_matches: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> tuple[str, int | None, int | None, int | None]:
    truth_mother = mother_matches.get(proposal.mother_id)
    truth_first = candidate_matches.get(proposal.candidate_1.node_id)
    truth_second = candidate_matches.get(proposal.candidate_2.node_id)
    if truth_mother is None:
        role = "unmatched_mother_unknown"
    elif truth_mother not in truth_divisions:
        role = "matched_nondivision_mother"
    elif truth_first is None or truth_second is None:
        role = "labelled_division_mother_unknown_children"
    elif (
        truth_first != truth_second
        and truth_first in truth_divisions[truth_mother]
        and truth_second in truth_divisions[truth_mother]
    ):
        role = "labelled_true_triplet"
    else:
        role = "labelled_division_mother_wrong_children"
    return role, truth_mother, truth_first, truth_second


def _proposal_row(
    dataset: str,
    *,
    source: str,
    selected_rule_id: str,
    proposal: DivisionPairProposal,
    scores_by_id: dict[int, float],
    mother_matches: dict[int, int],
    candidate_matches: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    role, truth_mother, truth_first, truth_second = _pair_label(
        proposal,
        mother_matches=mother_matches,
        candidate_matches=candidate_matches,
        truth_divisions=truth_divisions,
    )
    first = proposal.candidate_1
    second = proposal.candidate_2
    scores = (
        scores_by_id[proposal.mother_id],
        scores_by_id[first.node_id],
        scores_by_id[second.node_id],
    )
    return {
        "dataset": dataset,
        "candidate_source": source,
        "selected_rule_id": selected_rule_id,
        "movie_truth_divisions": len(truth_divisions),
        "mother_id": proposal.mother_id,
        "candidate_1_id": first.node_id,
        "candidate_2_id": second.node_id,
        "daughter_time": proposal.daughter_time,
        "candidate_1_distance_um": first.distance_um,
        "candidate_2_distance_um": second.distance_um,
        "daughter_angle_deg": proposal.daughter_angle_deg,
        "mother_cellness": scores[0],
        "candidate_1_cellness": scores[1],
        "candidate_2_cellness": scores[2],
        "min_triplet_cellness": min(scores),
        "mother_has_history": proposal.mother_has_history,
        "mother_has_current_child": proposal.mother_has_current_child,
        "mother_survives_pruning": proposal.mother_survives_pruning,
        "candidate_1_present_in_filtered_graph": first.present_in_linked_graph,
        "candidate_1_is_current_child": first.is_current_child,
        "candidate_1_is_parentless": first.is_parentless,
        "candidate_1_has_wrong_parent": first.has_wrong_parent,
        "candidate_1_has_future": first.has_future,
        "candidate_1_survives_pruning": first.survives_pruning,
        "candidate_2_present_in_filtered_graph": second.present_in_linked_graph,
        "candidate_2_is_current_child": second.is_current_child,
        "candidate_2_is_parentless": second.is_parentless,
        "candidate_2_has_wrong_parent": second.has_wrong_parent,
        "candidate_2_has_future": second.has_future,
        "candidate_2_survives_pruning": second.survives_pruning,
        "truth_mother_id": truth_mother,
        "truth_candidate_1_id": truth_first,
        "truth_candidate_2_id": truth_second,
        "sparse_label_role": role,
    }


def _matching_event_proposals(
    proposals: list[DivisionPairProposal],
    *,
    truth_mother: int,
    truth_daughters: tuple[int, ...],
    mother_matches: dict[int, int],
    candidate_matches: dict[int, int],
) -> list[DivisionPairProposal]:
    daughter_set = set(truth_daughters)
    return [
        proposal
        for proposal in proposals
        if mother_matches.get(proposal.mother_id) == truth_mother
        and candidate_matches.get(proposal.candidate_1.node_id) in daughter_set
        and candidate_matches.get(proposal.candidate_2.node_id) in daughter_set
        and candidate_matches.get(proposal.candidate_1.node_id)
        != candidate_matches.get(proposal.candidate_2.node_id)
    ]


def _selected_event_rows(
    state: MovieState,
    *,
    selected_rule_id: str,
    proposals_by_source: dict[str, list[DivisionPairProposal]],
    filtered_matches: dict[int, int],
    raw_matches: dict[int, int],
) -> list[dict[str, Any]]:
    truth_divisions = _truth_divisions(state.truth)
    filtered_truth_to_predicted = {
        truth_id: predicted_id for predicted_id, truth_id in filtered_matches.items()
    }
    raw_truth_to_predicted = {
        truth_id: predicted_id for predicted_id, truth_id in raw_matches.items()
    }
    rows: list[dict[str, Any]] = []
    for truth_mother, truth_daughters in sorted(truth_divisions.items()):
        filtered_daughters = sum(
            daughter in filtered_truth_to_predicted for daughter in truth_daughters
        )
        raw_daughters = sum(daughter in raw_truth_to_predicted for daughter in truth_daughters)
        filtered_matching = _matching_event_proposals(
            proposals_by_source["filtered"],
            truth_mother=truth_mother,
            truth_daughters=truth_daughters,
            mother_matches=filtered_matches,
            candidate_matches=filtered_matches,
        )
        raw_matching = _matching_event_proposals(
            proposals_by_source["raw_scored"],
            truth_mother=truth_mother,
            truth_daughters=truth_daughters,
            mother_matches=filtered_matches,
            candidate_matches=raw_matches,
        )
        mother_predicted = filtered_truth_to_predicted.get(truth_mother)
        filtered_complete = mother_predicted is not None and filtered_daughters == len(
            truth_daughters
        )
        raw_complete = mother_predicted is not None and raw_daughters == len(truth_daughters)
        rows.append(
            {
                "dataset": state.sample.dataset,
                "truth_mother_id": truth_mother,
                "truth_daughter_ids": ";".join(str(value) for value in truth_daughters),
                "truth_daughter_count": len(truth_daughters),
                "predicted_mother_id": mother_predicted,
                "filtered_daughters_matched": filtered_daughters,
                "raw_daughters_matched": raw_daughters,
                "filtered_complete_triplet": filtered_complete,
                "raw_complete_triplet": raw_complete,
                "filtered_selected_pair_proposals": len(filtered_matching),
                "raw_scored_selected_pair_proposals": len(raw_matching),
                "filtered_selected_covered": bool(filtered_matching),
                "raw_scored_selected_covered": bool(raw_matching),
                "raw_scored_restores_filtered_event": bool(raw_matching)
                and not bool(filtered_matching),
                "raw_scored_restores_below_threshold_daughter": raw_complete
                and not filtered_complete,
                "selected_rule_id": selected_rule_id,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "data"
        / "working"
        / "appearance_checkpoints"
        / "cellness_cnn_phase10_full.pt",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_frozen_candidate_policy.json",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "data" / "working" / "phase10_internal_dev.csv",
    )
    parser.add_argument(
        "--score-cache-dir",
        type=Path,
        default=ROOT / "data" / "working" / "appearance_scores" / "phase10",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "data"
        / "working"
        / "phase11_division_audit"
        / "full13_pairs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required_files = (args.checkpoint, args.policy, args.baseline_results)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        parser.error(f"required Phase 10 artifacts are missing: {missing}")

    policy_sha256 = file_sha256(args.policy)
    if policy_sha256 != EXPECTED_POLICY_SHA256:
        parser.error("policy SHA-256 is not the frozen Phase 10 identity")
    policy = read_frozen_policy(args.policy)
    if not np.isclose(float(policy["score_threshold"]), FROZEN_SCORE_THRESHOLD):
        parser.error("the frozen policy does not use the expected 0.20 cellness threshold")
    if int(policy["min_track_length"]) != FROZEN_MIN_TRACK_LENGTH:
        parser.error("the frozen policy does not use the expected minimum track length 6")
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if (
        checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_sha256 != policy["checkpoint_sha256"]
    ):
        parser.error("checkpoint SHA-256 does not match the frozen Phase 10 policy")
    phase10_results_sha256 = file_sha256(args.baseline_results)
    if (
        phase10_results_sha256 != EXPECTED_PHASE10_RESULTS_SHA256
        or phase10_results_sha256 != policy["source_results_sha256"]
    ):
        parser.error("internal-development results SHA-256 does not match the frozen policy")
    scoring_source_sha256 = files_sha256(SCORING_SOURCE_FILES)
    if scoring_source_sha256 != EXPECTED_SCORING_SOURCE_SHA256:
        parser.error("frozen Phase 10 scoring source has changed")
    baseline_evaluation_sha256 = files_sha256(BASELINE_EVALUATION_SOURCE_FILES)
    if (
        baseline_evaluation_sha256 != EXPECTED_BASELINE_EVALUATION_SHA256
        or baseline_evaluation_sha256 != policy["evaluation_sha256"]
    ):
        parser.error("frozen Phase 10 evaluation source has changed")
    pair_audit_source_sha256 = files_sha256(PAIR_AUDIT_SOURCE_FILES)

    _, checkpoint = load_cellness_checkpoint(args.checkpoint, device="cpu")
    datasets = checkpoint_dev_datasets(checkpoint)
    if datasets != policy["selection_datasets"]:
        parser.error("checkpoint and frozen policy record different development movies")
    if tuple(datasets) != EXPECTED_DATASETS:
        parser.error("pair audit dataset tuple is not the frozen Phase 10 cohort")
    if set(EXPECTED_SCORE_CACHE_SHA256) != set(EXPECTED_DATASETS):
        parser.error("internal pair-audit cache manifest is incomplete")
    detector_threshold_k = float(checkpoint["detector_threshold_k"])
    patch_shape = tuple(int(value) for value in checkpoint["patch_shape"])

    samples_by_dataset = {
        sample.dataset: sample for sample in discover_split(args.data_root, "train")
    }
    samples: list[SamplePaths] = []
    for dataset in datasets:
        sample = samples_by_dataset.get(dataset)
        if sample is None or sample.labels is None:
            parser.error(f"labelled training movie is missing: {dataset}")
        samples.append(sample)
    phase10_rows = _read_phase10_rows(args.baseline_results)
    if set(datasets) - set(phase10_rows):
        parser.error("Phase 10 frozen-policy rows are missing for the exact cohort")

    targets = {
        "rule_sweep": args.output_dir / "rule_sweep.csv",
        "rule_movies": args.output_dir / "rule_movies.csv",
        "events": args.output_dir / "division_events.csv",
        "proposals": args.output_dir / "selected_pair_proposals.csv",
        "selected_movies": args.output_dir / "selected_movies.csv",
        "summary": args.output_dir / "summary.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            f"pair audit outputs already exist; use --overwrite for this target: {existing}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rules = build_rules()
    started = time.perf_counter()
    print(
        f"Phase 11 pair audit: movies={len(samples)} geometry_rules={len(rules)} "
        f"source_rules={len(rules) * 2} required_coverage={REQUIRED_EVENT_COVERAGE}",
        flush=True,
    )
    print(f"Checkpoint SHA-256: {checkpoint_sha256}", flush=True)
    print(f"Frozen evaluation SHA-256: {baseline_evaluation_sha256}", flush=True)

    movie_rows: list[dict[str, Any]] = []
    truth_division_total = 0
    no_division_controls = 0
    broad_rule_id = PairRule(max(MAX_DISTANCES_UM), None, 0.0, 0.0).rule_id
    for index, sample in enumerate(samples, start=1):
        movie_started = time.perf_counter()
        state = _load_movie_state(
            sample,
            score_cache_dir=args.score_cache_dir,
            checkpoint_sha256=checkpoint_sha256,
            scoring_source_sha256=scoring_source_sha256,
            detector_threshold_k=detector_threshold_k,
            patch_shape=patch_shape,
            expected_baseline=phase10_rows[sample.dataset],
        )
        current_rows = _movie_rule_rows(state, rules)
        movie_rows.extend(current_rows)
        truth_divisions = len(_truth_divisions(state.truth))
        truth_division_total += truth_divisions
        no_division_controls += int(truth_divisions == 0)
        broad = {
            row["candidate_source"]: row
            for row in current_rows
            if row["rule_id"] == broad_rule_id
        }
        print(
            f"[{index}/{len(samples)}] {sample.dataset} truth_div={truth_divisions} "
            f"broad_pairs={broad['filtered']['proposal_count']}"
            f"->{broad['raw_scored']['proposal_count']} "
            f"broad_coverage={broad['filtered']['labelled_event_coverage']}"
            f"->{broad['raw_scored']['labelled_event_coverage']} "
            f"baseline=exact elapsed={time.perf_counter() - movie_started:.1f}s",
            flush=True,
        )

    if truth_division_total != REQUIRED_EVENT_COVERAGE:
        raise RuntimeError(
            f"expected {REQUIRED_EVENT_COVERAGE} labelled divisions, found "
            f"{truth_division_total}"
        )
    if no_division_controls != EXPECTED_NO_DIVISION_CONTROLS:
        raise RuntimeError(
            f"expected {EXPECTED_NO_DIVISION_CONTROLS} no-division controls, found "
            f"{no_division_controls}"
        )
    expected_movie_rows = EXPECTED_MOVIES * len(rules) * 2
    if len(movie_rows) != expected_movie_rows:
        raise RuntimeError(
            f"expected {expected_movie_rows} per-movie rule rows, found {len(movie_rows)}"
        )

    rule_rows = aggregate_rule_rows(movie_rows)
    if len(rule_rows) != len(rules) * 2:
        raise RuntimeError("aggregate rule grid is incomplete")
    selected = select_pair_rule(rule_rows)
    selected_rule = _rule_from_row(selected)
    filtered_comparison = next(
        row
        for row in rule_rows
        if row["candidate_source"] == "filtered"
        and row["rule_id"] == selected_rule.rule_id
    )
    selected_movie_rows = [
        row for row in movie_rows if row["rule_id"] == selected_rule.rule_id
    ]
    if len(selected_movie_rows) != EXPECTED_MOVIES * 2:
        raise RuntimeError("selected-rule movie rows are incomplete")
    expected_selected = {
        (str(row["dataset"]), str(row["candidate_source"])): row
        for row in selected_movie_rows
    }

    print(
        f"Selected {selected_rule.rule_id}: raw_pairs={selected['proposal_count']} "
        f"raw_mothers={selected['unique_proposed_mothers']} "
        f"coverage={selected['labelled_event_coverage']}/{REQUIRED_EVENT_COVERAGE}",
        flush=True,
    )

    proposal_temporary = targets["proposals"].with_suffix(
        f"{targets['proposals'].suffix}.tmp"
    )
    event_rows: list[dict[str, Any]] = []
    with proposal_temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROPOSAL_FIELDS)
        writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            state = _load_movie_state(
                sample,
                score_cache_dir=args.score_cache_dir,
                checkpoint_sha256=checkpoint_sha256,
                scoring_source_sha256=scoring_source_sha256,
                detector_threshold_k=detector_threshold_k,
                patch_shape=patch_shape,
                expected_baseline=phase10_rows[sample.dataset],
            )
            first_pass_cache_sha256 = str(
                expected_selected[(sample.dataset, "raw_scored")]["score_cache_sha256"]
            )
            if state.cache_sha256 != first_pass_cache_sha256:
                raise RuntimeError(
                    f"score cache changed between passes for {sample.dataset}"
                )
            filtered_matches, _ = _inverse_matches(state.filtered, state.truth)
            raw_matches, _ = _inverse_matches(state.raw, state.truth)
            truth_divisions = _truth_divisions(state.truth)
            pruned_ids = {int(node_id) for node_id in state.pruned.node_ids}
            scores_by_id = {
                int(node_id): float(score)
                for node_id, score in zip(state.raw.node_ids, state.scores, strict=True)
            }
            proposals_by_source: dict[str, list[DivisionPairProposal]] = {}
            for source, candidates, candidate_matches in (
                ("filtered", state.filtered, filtered_matches),
                ("raw_scored", state.raw, raw_matches),
            ):
                neighborhoods = generate_division_pair_neighborhoods(
                    state.linked,
                    candidates,
                    pruned_node_ids=pruned_ids,
                    max_distance_um=max(MAX_DISTANCES_UM),
                )
                proposals = generate_division_pair_proposals(
                    neighborhoods,
                    min_distance_um=selected_rule.min_distance_um,
                    max_distance_um=selected_rule.max_distance_um,
                    min_angle_deg=selected_rule.min_angle_deg,
                    max_candidates_per_mother=(
                        selected_rule.max_candidates_per_mother
                    ),
                )
                expected = expected_selected[(sample.dataset, source)]
                if len(proposals) != int(expected["proposal_count"]):
                    raise RuntimeError(
                        f"selected proposal count changed for {sample.dataset} {source}: "
                        f"expected {expected['proposal_count']}, found {len(proposals)}"
                    )
                role_counts: Counter[str] = Counter()
                for proposal in proposals:
                    row = _proposal_row(
                        sample.dataset,
                        source=source,
                        selected_rule_id=selected_rule.rule_id,
                        proposal=proposal,
                        scores_by_id=scores_by_id,
                        mother_matches=filtered_matches,
                        candidate_matches=candidate_matches,
                        truth_divisions=truth_divisions,
                    )
                    role_counts[str(row["sparse_label_role"])] += 1
                    writer.writerow(row)
                for role in ROLE_NAMES:
                    if role_counts[role] != int(expected[f"{role}_proposals"]):
                        raise RuntimeError(
                            f"selected sparse role count changed for {sample.dataset} "
                            f"{source} {role}"
                        )
                proposals_by_source[source] = proposals

            event_rows.extend(
                _selected_event_rows(
                    state,
                    selected_rule_id=selected_rule.rule_id,
                    proposals_by_source=proposals_by_source,
                    filtered_matches=filtered_matches,
                    raw_matches=raw_matches,
                )
            )
            print(
                f"[write {index}/{len(samples)}] {sample.dataset} selected_pairs="
                f"{len(proposals_by_source['filtered'])}"
                f"->{len(proposals_by_source['raw_scored'])}",
                flush=True,
            )
    if len(event_rows) != REQUIRED_EVENT_COVERAGE:
        raise RuntimeError(
            f"expected {REQUIRED_EVENT_COVERAGE} selected event rows, found {len(event_rows)}"
        )
    raw_selected_coverage = sum(
        bool(row["raw_scored_selected_covered"]) for row in event_rows
    )
    filtered_selected_coverage = sum(
        bool(row["filtered_selected_covered"]) for row in event_rows
    )
    if raw_selected_coverage != REQUIRED_EVENT_COVERAGE:
        raise RuntimeError("selected raw-scored proposal rows did not reproduce 18-event coverage")
    if raw_selected_coverage != int(selected["labelled_event_coverage"]):
        raise RuntimeError("selected raw-scored event coverage changed during artifact writing")
    if filtered_selected_coverage != int(filtered_comparison["labelled_event_coverage"]):
        raise RuntimeError("selected filtered event coverage changed during artifact writing")
    if files_sha256(PAIR_AUDIT_SOURCE_FILES) != pair_audit_source_sha256:
        raise RuntimeError("pair-audit source changed while the audit was running")

    proposal_temporary.replace(targets["proposals"])
    _write_csv(targets["rule_sweep"], RULE_SWEEP_FIELDS, rule_rows)
    _write_csv(targets["rule_movies"], RULE_MOVIE_FIELDS, movie_rows)
    _write_csv(targets["events"], EVENT_FIELDS, event_rows)
    _write_csv(targets["selected_movies"], RULE_MOVIE_FIELDS, selected_movie_rows)

    controls = [
        {
            "dataset": row["dataset"],
            "candidate_source": row["candidate_source"],
            "proposal_count": row["proposal_count"],
            "unique_proposed_mothers": row["unique_proposed_mothers"],
            "unique_matched_nondivision_mothers": row[
                "unique_matched_nondivision_mothers"
            ],
        }
        for row in selected_movie_rows
        if bool(row["is_no_division_control"])
    ]
    if len(controls) != EXPECTED_NO_DIVISION_CONTROLS * 2:
        raise RuntimeError("selected-rule no-division control rows are incomplete")

    csv_outputs: dict[str, Any] = {}
    for name in ("rule_sweep", "rule_movies", "events", "proposals", "selected_movies"):
        path = targets[name]
        csv_outputs[f"{name}_csv"] = str(path)
        csv_outputs[f"{name}_sha256"] = file_sha256(path)
    summary = {
        "pair_audit_schema_version": PAIR_AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "cohort": "exact_phase10_internal_dev",
        "datasets": datasets,
        "configuration": {
            "cellness_threshold": FROZEN_SCORE_THRESHOLD,
            "min_track_length": FROZEN_MIN_TRACK_LENGTH,
            "detector_threshold_k": detector_threshold_k,
            "patch_shape": list(patch_shape),
            "mother_source": "cnn_filtered_preprune",
            "candidate_sources": ["filtered", "raw_scored"],
            "required_labelled_event_coverage": REQUIRED_EVENT_COVERAGE,
            "max_distances_um": list(MAX_DISTANCES_UM),
            "max_candidates_per_mother": [
                "all" if value is None else value for value in MAX_CANDIDATES
            ],
            "min_distances_um": list(MIN_DISTANCES_UM),
            "min_angles_deg": list(MIN_ANGLES_DEG),
            "geometry_rules": len(rules),
            "source_rules": len(rule_rows),
        },
        "baseline_reproduction": {
            "movies_checked": EXPECTED_MOVIES,
            "all_exact": all(bool(row["baseline_reproduced"]) for row in movie_rows),
        },
        "truth": {
            "labelled_divisions": truth_division_total,
            "division_movies": EXPECTED_MOVIES - no_division_controls,
            "no_labelled_division_controls": no_division_controls,
        },
        "selection": {
            "ordering": [
                "proposal_count",
                "unique_proposed_mothers",
                "least_restrictive_exact_tie",
                "rule_id",
            ],
            "selected_raw_scored_rule": selected,
            "filtered_same_geometry": filtered_comparison,
            "raw_scored_selected_event_coverage": raw_selected_coverage,
            "filtered_selected_event_coverage": filtered_selected_coverage,
            "raw_scored_restored_events": sum(
                bool(row["raw_scored_restores_filtered_event"]) for row in event_rows
            ),
            "raw_scored_below_threshold_restorations": sum(
                bool(row["raw_scored_restores_below_threshold_daughter"])
                for row in event_rows
            ),
        },
        "no_division_controls": controls,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "frozen_policy_sha256": policy_sha256,
            "phase10_results_sha256": phase10_results_sha256,
            "scoring_source_sha256": scoring_source_sha256,
            "frozen_evaluation_source_sha256": baseline_evaluation_sha256,
            "pair_audit_source_sha256": pair_audit_source_sha256,
        },
        "outputs": csv_outputs,
    }
    _write_json(targets["summary"], summary)
    print(f"Saved pair audit: {args.output_dir}", flush=True)
    print(
        json.dumps(
            {
                "selected_rule": selected_rule.rule_id,
                "raw_scored_coverage": raw_selected_coverage,
                "filtered_coverage": filtered_selected_coverage,
                "raw_scored_proposals": selected["proposal_count"],
                "filtered_proposals": filtered_comparison["proposal_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
