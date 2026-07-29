#!/usr/bin/env python3
"""Audit mother-to-daughter proposal recoverability after the frozen Phase 10 filter.

This is the first Phase 11 experiment slice.  It reuses provenance-checked Phase 10
candidate-score caches, applies the frozen 0.20 cellness threshold, reruns whole-movie
flow linking, and verifies that minimum-length-6 pruning exactly reproduces each saved
Phase 10 divisions-off result.

For every labelled division, the audit records whether the mother and two daughters are
available at the raw, filtered, linked, and pruned boundaries.  It also writes every
broad mother, linked-daughter, orphan-daughter proposal before and after pruning.  Sparse
labels are respected: an unmatched proposal is unknown, not a negative example.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import filter_candidates  # noqa: E402
from biohub.appearance_evaluation import (  # noqa: E402
    checkpoint_dev_datasets,
    file_sha256,
    files_sha256,
    read_frozen_policy,
    read_score_cache,
)
from biohub.appearance_model import load_cellness_checkpoint  # noqa: E402
from biohub.division import (  # noqa: E402
    DivisionProposal,
    generate_division_proposals,
    passes_phase7_gates,
)
from biohub.io import SamplePaths, discover_split, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, evaluate, match_nodes  # noqa: E402

AUDIT_SCHEMA_VERSION = 1
FROZEN_SCORE_THRESHOLD = 0.20
FROZEN_MIN_TRACK_LENGTH = 6
DEFAULT_BROAD_MAX_DISTANCE_UM = 14.0

SCORING_SOURCE_FILES = [
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "appearance_model.py",
    SRC / "biohub" / "detect.py",
    SRC / "biohub" / "io.py",
    SRC / "biohub" / "constants.py",
]
BASELINE_EVALUATION_SOURCE_FILES = [
    *SCORING_SOURCE_FILES,
    SRC / "biohub" / "appearance_evaluation.py",
    SRC / "biohub" / "link.py",
    SRC / "biohub" / "metric.py",
    Path(__file__).with_name("freeze_appearance_policy.py"),
    Path(__file__).with_name("validate_appearance.py"),
]
AUDIT_SOURCE_FILES = [
    SRC / "biohub" / "division.py",
    Path(__file__).resolve(),
]

EVENT_FIELDS = [
    "dataset",
    "truth_mother_id",
    "truth_daughter_ids",
    "truth_daughter_count",
    "raw_mother_matched",
    "raw_daughters_matched",
    "raw_complete_triplet",
    "filtered_mother_matched",
    "filtered_daughters_matched",
    "filtered_complete_triplet",
    "preprune_correct_linked_daughters",
    "preprune_orphan_truth_daughters",
    "preprune_second_daughter_candidate",
    "preprune_broad_proposals",
    "preprune_phase7_proposals",
    "pruned_mother_survives",
    "pruned_daughters_survive",
    "pruned_complete_triplet_survives",
    "postprune_correct_linked_daughters",
    "postprune_orphan_truth_daughters",
    "postprune_second_daughter_candidate",
    "postprune_broad_proposals",
    "postprune_phase7_proposals",
]

PROPOSAL_FIELDS = [
    "dataset",
    "stage",
    "movie_truth_divisions",
    "mother_id",
    "linked_daughter_id",
    "orphan_daughter_id",
    "daughter_time",
    "linked_distance_um",
    "orphan_distance_um",
    "daughter_angle_deg",
    "mother_has_history",
    "orphan_has_future",
    "passes_phase7_gates",
    "mother_cellness",
    "linked_daughter_cellness",
    "orphan_daughter_cellness",
    "min_triplet_cellness",
    "truth_mother_id",
    "truth_linked_daughter_id",
    "truth_orphan_daughter_id",
    "sparse_label_role",
]

MOVIE_FIELDS = [
    "dataset",
    "truth_divisions",
    "raw_candidates",
    "filtered_candidates",
    "pruned_nodes",
    "link_seconds",
    "edge_jaccard",
    "score",
    "baseline_reproduced",
    "preprune_proposals",
    "preprune_phase7_proposals",
    "preprune_labelled_true_triplets",
    "preprune_matched_nondivision_mothers",
    "preprune_unknown_mothers",
    "postprune_proposals",
    "postprune_phase7_proposals",
    "postprune_labelled_true_triplets",
    "postprune_matched_nondivision_mothers",
    "postprune_unknown_mothers",
    "raw_complete_truth_triplets",
    "filtered_complete_truth_triplets",
    "preprune_recoverable_truth_divisions",
    "postprune_recoverable_truth_divisions",
    "score_cache_sha256",
]

INTEGER_BASELINE_FIELDS = (
    "raw_candidates",
    "kept_candidates",
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
FLOAT_BASELINE_FIELDS = (
    "edge_jaccard",
    "score",
    "division_jaccard",
    "recall",
)


def _oriented_edges(graph: TrackingGraph) -> set[tuple[int, int]]:
    times = {
        int(node_id): int(time)
        for node_id, time in zip(graph.node_ids, graph.t, strict=True)
    }
    oriented: set[tuple[int, int]] = set()
    for source, target in np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2):
        source = int(source)
        target = int(target)
        if times[target] < times[source]:
            source, target = target, source
        oriented.add((source, target))
    return oriented


def _adjacency(graph: TrackingGraph) -> tuple[dict[int, set[int]], dict[int, int]]:
    successors: dict[int, set[int]] = defaultdict(set)
    in_degree: dict[int, int] = defaultdict(int)
    for source, target in _oriented_edges(graph):
        successors[source].add(target)
        in_degree[target] += 1
    return dict(successors), dict(in_degree)


def _truth_divisions(truth: TrackingGraph) -> dict[int, tuple[int, ...]]:
    successors, _ = _adjacency(truth)
    return {
        mother: tuple(sorted(children))
        for mother, children in successors.items()
        if len(children) >= 2
    }


def _inverse_matches(
    graph: TrackingGraph,
    truth: TrackingGraph,
) -> tuple[dict[int, int], dict[int, int]]:
    predicted_to_truth = match_nodes(graph, truth)
    truth_to_predicted = {
        truth_id: predicted_id for predicted_id, truth_id in predicted_to_truth.items()
    }
    if len(truth_to_predicted) != len(predicted_to_truth):
        raise RuntimeError("node matching unexpectedly assigned two predictions to one truth node")
    return predicted_to_truth, truth_to_predicted


def _event_proposals(
    proposals: list[DivisionProposal],
    predicted_to_truth: dict[int, int],
    truth_mother: int,
    truth_daughters: tuple[int, ...],
) -> list[DivisionProposal]:
    daughter_set = set(truth_daughters)
    return [
        proposal
        for proposal in proposals
        if predicted_to_truth.get(proposal.mother_id) == truth_mother
        and predicted_to_truth.get(proposal.linked_daughter_id) in daughter_set
        and predicted_to_truth.get(proposal.orphan_daughter_id) in daughter_set
        and predicted_to_truth.get(proposal.linked_daughter_id)
        != predicted_to_truth.get(proposal.orphan_daughter_id)
    ]


def build_event_rows(
    dataset: str,
    *,
    raw: TrackingGraph,
    filtered: TrackingGraph,
    linked: TrackingGraph,
    pruned: TrackingGraph,
    truth: TrackingGraph,
    preprune_proposals: list[DivisionProposal],
    postprune_proposals: list[DivisionProposal],
) -> list[dict[str, Any]]:
    """Build one stable recoverability funnel row per sparse labelled division."""
    _, raw_truth_to_predicted = _inverse_matches(raw, truth)
    filtered_predicted_to_truth, filtered_truth_to_predicted = _inverse_matches(filtered, truth)
    linked_successors, linked_in_degree = _adjacency(linked)
    pruned_successors, pruned_in_degree = _adjacency(pruned)
    pruned_ids = set(int(node_id) for node_id in pruned.node_ids)

    rows: list[dict[str, Any]] = []
    for truth_mother, truth_daughters in sorted(_truth_divisions(truth).items()):
        raw_mother = raw_truth_to_predicted.get(truth_mother)
        raw_daughters = [raw_truth_to_predicted.get(daughter) for daughter in truth_daughters]
        predicted_mother = filtered_truth_to_predicted.get(truth_mother)
        predicted_daughters = [
            filtered_truth_to_predicted.get(daughter) for daughter in truth_daughters
        ]
        available_daughters = [
            daughter for daughter in predicted_daughters if daughter is not None
        ]

        linked_children = linked_successors.get(predicted_mother, set())
        correct_linked = [
            daughter for daughter in available_daughters if daughter in linked_children
        ]
        linked_orphans = [
            daughter
            for daughter in available_daughters
            if linked_in_degree.get(daughter, 0) == 0
        ]
        pre_second_candidate = any(
            linked != orphan
            for linked in correct_linked
            for orphan in linked_orphans
        )

        pruned_children = pruned_successors.get(predicted_mother, set())
        post_correct_linked = [
            daughter
            for daughter in available_daughters
            if daughter in pruned_ids and daughter in pruned_children
        ]
        post_orphans = [
            daughter
            for daughter in available_daughters
            if daughter in pruned_ids and pruned_in_degree.get(daughter, 0) == 0
        ]
        post_second_candidate = any(
            linked != orphan
            for linked in post_correct_linked
            for orphan in post_orphans
        )

        matching_pre = _event_proposals(
            preprune_proposals,
            filtered_predicted_to_truth,
            truth_mother,
            truth_daughters,
        )
        matching_post = _event_proposals(
            postprune_proposals,
            filtered_predicted_to_truth,
            truth_mother,
            truth_daughters,
        )
        raw_daughter_count = sum(daughter is not None for daughter in raw_daughters)
        filtered_daughter_count = len(available_daughters)
        pruned_daughter_count = sum(daughter in pruned_ids for daughter in available_daughters)
        rows.append(
            {
                "dataset": dataset,
                "truth_mother_id": truth_mother,
                "truth_daughter_ids": ";".join(str(value) for value in truth_daughters),
                "truth_daughter_count": len(truth_daughters),
                "raw_mother_matched": raw_mother is not None,
                "raw_daughters_matched": raw_daughter_count,
                "raw_complete_triplet": raw_mother is not None
                and raw_daughter_count == len(truth_daughters),
                "filtered_mother_matched": predicted_mother is not None,
                "filtered_daughters_matched": filtered_daughter_count,
                "filtered_complete_triplet": predicted_mother is not None
                and filtered_daughter_count == len(truth_daughters),
                "preprune_correct_linked_daughters": len(correct_linked),
                "preprune_orphan_truth_daughters": len(linked_orphans),
                "preprune_second_daughter_candidate": pre_second_candidate,
                "preprune_broad_proposals": len(matching_pre),
                "preprune_phase7_proposals": sum(
                    passes_phase7_gates(proposal) for proposal in matching_pre
                ),
                "pruned_mother_survives": predicted_mother in pruned_ids,
                "pruned_daughters_survive": pruned_daughter_count,
                "pruned_complete_triplet_survives": predicted_mother in pruned_ids
                and pruned_daughter_count == len(truth_daughters),
                "postprune_correct_linked_daughters": len(post_correct_linked),
                "postprune_orphan_truth_daughters": len(post_orphans),
                "postprune_second_daughter_candidate": post_second_candidate,
                "postprune_broad_proposals": len(matching_post),
                "postprune_phase7_proposals": sum(
                    passes_phase7_gates(proposal) for proposal in matching_post
                ),
            }
        )
    return rows


def _proposal_label(
    proposal: DivisionProposal,
    predicted_to_truth: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> tuple[str, int | None, int | None, int | None]:
    truth_mother = predicted_to_truth.get(proposal.mother_id)
    truth_linked = predicted_to_truth.get(proposal.linked_daughter_id)
    truth_orphan = predicted_to_truth.get(proposal.orphan_daughter_id)
    if truth_mother is None:
        role = "unmatched_mother_unknown"
    elif truth_mother not in truth_divisions:
        role = "matched_nondivision_mother"
    elif truth_linked is None or truth_orphan is None:
        role = "labelled_division_mother_unknown_children"
    elif (
        truth_linked != truth_orphan
        and truth_linked in truth_divisions[truth_mother]
        and truth_orphan in truth_divisions[truth_mother]
    ):
        role = "labelled_true_triplet"
    else:
        role = "labelled_division_mother_wrong_children"
    return role, truth_mother, truth_linked, truth_orphan


def build_proposal_rows(
    dataset: str,
    *,
    stage: str,
    proposals: list[DivisionProposal],
    scores_by_id: dict[int, float],
    predicted_to_truth: dict[int, int],
    truth_divisions: dict[int, tuple[int, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        role, truth_mother, truth_linked, truth_orphan = _proposal_label(
            proposal,
            predicted_to_truth,
            truth_divisions,
        )
        triplet_scores = [
            scores_by_id[proposal.mother_id],
            scores_by_id[proposal.linked_daughter_id],
            scores_by_id[proposal.orphan_daughter_id],
        ]
        rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "movie_truth_divisions": len(truth_divisions),
                "mother_id": proposal.mother_id,
                "linked_daughter_id": proposal.linked_daughter_id,
                "orphan_daughter_id": proposal.orphan_daughter_id,
                "daughter_time": proposal.daughter_time,
                "linked_distance_um": proposal.linked_distance_um,
                "orphan_distance_um": proposal.orphan_distance_um,
                "daughter_angle_deg": proposal.daughter_angle_deg,
                "mother_has_history": proposal.mother_has_history,
                "orphan_has_future": proposal.orphan_has_future,
                "passes_phase7_gates": passes_phase7_gates(proposal),
                "mother_cellness": triplet_scores[0],
                "linked_daughter_cellness": triplet_scores[1],
                "orphan_daughter_cellness": triplet_scores[2],
                "min_triplet_cellness": min(triplet_scores),
                "truth_mother_id": truth_mother,
                "truth_linked_daughter_id": truth_linked,
                "truth_orphan_daughter_id": truth_orphan,
                "sparse_label_role": role,
            }
        )
    return rows


def _read_phase10_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if np.isclose(float(row["score_threshold"]), FROZEN_SCORE_THRESHOLD)
        and int(row["min_track_length"]) == FROZEN_MIN_TRACK_LENGTH
    ]
    by_dataset = {row["dataset"]: row for row in selected}
    if len(by_dataset) != len(selected):
        raise ValueError("Phase 10 baseline results contain duplicate frozen-policy rows")
    return by_dataset


def verify_phase10_reproduction(
    *,
    expected: dict[str, str],
    raw_candidates: int,
    filtered_candidates: int,
    result,
) -> None:
    actual: dict[str, int | float] = {
        "raw_candidates": raw_candidates,
        "kept_candidates": filtered_candidates,
        **result.as_dict(),
        "recall": result.matched_nodes / result.truth_nodes if result.truth_nodes else 0.0,
    }
    failures: list[str] = []
    for field in INTEGER_BASELINE_FIELDS:
        if int(expected[field]) != int(actual[field]):
            failures.append(f"{field}: expected {expected[field]}, got {actual[field]}")
    for field in FLOAT_BASELINE_FIELDS:
        if not np.isclose(float(expected[field]), float(actual[field]), rtol=0.0, atol=1e-12):
            failures.append(f"{field}: expected {expected[field]}, got {actual[field]}")
    if failures:
        raise RuntimeError("Phase 10 reproduction failed: " + "; ".join(failures))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _find_score_cache(
    directory: Path,
    dataset: str,
    *,
    checkpoint_sha256: str,
    scoring_source_sha256: str,
    detector_threshold_k: float,
    patch_shape: tuple[int, int, int],
) -> tuple[Path, TrackingGraph, np.ndarray, dict[str, Any]]:
    candidates = sorted(directory.glob(f"{dataset}.{checkpoint_sha256[:12]}.*.npz"))
    valid: list[tuple[Path, TrackingGraph, np.ndarray, dict[str, Any]]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            graph, scores, metadata = read_score_cache(
                path,
                expected_dataset=dataset,
                expected_data_split="train",
                expected_checkpoint_sha256=checkpoint_sha256,
                expected_scoring_source_sha256=scoring_source_sha256,
                expected_detector_threshold_k=detector_threshold_k,
                expected_patch_shape=patch_shape,
            )
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
        else:
            valid.append((path, graph, scores, metadata))
    if len(valid) != 1:
        detail = "; ".join(errors) if errors else "no matching cache files"
        raise RuntimeError(
            f"expected one exact Phase 10 score cache for {dataset}, found {len(valid)}: {detail}"
        )
    return valid[0]


def _stage_counts(rows: list[dict[str, Any]], stage: str) -> Counter[str]:
    return Counter(
        row["sparse_label_role"] for row in rows if row["stage"] == stage
    )


def _appearance_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['stage']}:{row['sparse_label_role']}"].append(row)
    summary: dict[str, dict[str, float | int]] = {}
    for key, current in sorted(grouped.items()):
        summary[key] = {
            "count": len(current),
            "mean_mother_cellness": float(
                np.mean([row["mother_cellness"] for row in current])
            ),
            "mean_linked_daughter_cellness": float(
                np.mean([row["linked_daughter_cellness"] for row in current])
            ),
            "mean_orphan_daughter_cellness": float(
                np.mean([row["orphan_daughter_cellness"] for row in current])
            ),
            "mean_min_triplet_cellness": float(
                np.mean([row["min_triplet_cellness"] for row in current])
            ),
        }
    return summary


def _sum_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row[field]) for row in rows)


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
        default=ROOT / "data" / "working" / "phase11_division_audit",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="ordered subset of frozen internal-development movie names; defaults to all 13",
    )
    parser.add_argument(
        "--broad-max-distance-um",
        type=float,
        default=DEFAULT_BROAD_MAX_DISTANCE_UM,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    required_files = (args.checkpoint, args.policy, args.baseline_results)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        parser.error(f"required Phase 10 artifacts are missing: {missing}")
    if not np.isfinite(args.broad_max_distance_um) or args.broad_max_distance_um <= 0.0:
        parser.error("--broad-max-distance-um must be positive and finite")

    policy = read_frozen_policy(args.policy)
    if not np.isclose(float(policy["score_threshold"]), FROZEN_SCORE_THRESHOLD):
        parser.error("the frozen policy does not use the expected 0.20 cellness threshold")
    if int(policy["min_track_length"]) != FROZEN_MIN_TRACK_LENGTH:
        parser.error("the frozen policy does not use the expected minimum track length 6")
    checkpoint_sha256 = file_sha256(args.checkpoint)
    if checkpoint_sha256 != policy["checkpoint_sha256"]:
        parser.error("checkpoint SHA-256 does not match the frozen Phase 10 policy")
    if file_sha256(args.baseline_results) != policy["source_results_sha256"]:
        parser.error("internal-development results SHA-256 does not match the frozen policy")
    scoring_source_sha256 = files_sha256(SCORING_SOURCE_FILES)
    baseline_evaluation_sha256 = files_sha256(BASELINE_EVALUATION_SOURCE_FILES)
    if baseline_evaluation_sha256 != policy["evaluation_sha256"]:
        parser.error("frozen Phase 10 evaluation source has changed")

    _, checkpoint = load_cellness_checkpoint(args.checkpoint, device="cpu")
    checkpoint_datasets = checkpoint_dev_datasets(checkpoint)
    if checkpoint_datasets != policy["selection_datasets"]:
        parser.error("checkpoint and frozen policy record different development movies")
    detector_threshold_k = float(checkpoint["detector_threshold_k"])
    patch_shape = tuple(int(value) for value in checkpoint["patch_shape"])

    datasets = args.datasets or checkpoint_datasets
    if len(set(datasets)) != len(datasets):
        parser.error("--datasets contains duplicates")
    unknown = sorted(set(datasets) - set(checkpoint_datasets))
    if unknown:
        parser.error(f"datasets are outside the frozen 6bba internal split: {unknown}")

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
    missing_rows = sorted(set(datasets) - set(phase10_rows))
    if missing_rows:
        parser.error(f"Phase 10 frozen-policy rows are missing: {missing_rows}")

    targets = {
        "events": args.output_dir / "division_events.csv",
        "proposals": args.output_dir / "division_proposals.csv",
        "movies": args.output_dir / "movies.csv",
        "summary": args.output_dir / "summary.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            f"audit outputs already exist; use --overwrite for this exact target: {existing}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Phase 11 division audit: movies={len(samples)} threshold={FROZEN_SCORE_THRESHOLD:.2f} "
        f"prune={FROZEN_MIN_TRACK_LENGTH} broad_gate={args.broad_max_distance_um:.1f}um",
        flush=True,
    )
    print(f"Checkpoint SHA-256: {checkpoint_sha256}", flush=True)
    print(f"Frozen evaluation SHA-256: {baseline_evaluation_sha256}", flush=True)

    all_event_rows: list[dict[str, Any]] = []
    all_proposal_rows: list[dict[str, Any]] = []
    movie_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        cache_path, raw, scores, _ = _find_score_cache(
            args.score_cache_dir,
            sample.dataset,
            checkpoint_sha256=checkpoint_sha256,
            scoring_source_sha256=scoring_source_sha256,
            detector_threshold_k=detector_threshold_k,
            patch_shape=patch_shape,
        )
        truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
        filtered = filter_candidates(raw, scores, min_score=FROZEN_SCORE_THRESHOLD)
        link_started = time.perf_counter()
        linked = link_graph_flow(filtered)
        link_seconds = time.perf_counter() - link_started
        pruned = prune_to_tracks(linked, min_track_length=FROZEN_MIN_TRACK_LENGTH)
        result = evaluate(pruned, truth)
        verify_phase10_reproduction(
            expected=phase10_rows[sample.dataset],
            raw_candidates=len(raw.node_ids),
            filtered_candidates=len(filtered.node_ids),
            result=result,
        )

        preprune_proposals = generate_division_proposals(
            linked,
            max_distance_um=args.broad_max_distance_um,
        )
        postprune_proposals = generate_division_proposals(
            pruned,
            max_distance_um=args.broad_max_distance_um,
        )
        event_rows = build_event_rows(
            sample.dataset,
            raw=raw,
            filtered=filtered,
            linked=linked,
            pruned=pruned,
            truth=truth,
            preprune_proposals=preprune_proposals,
            postprune_proposals=postprune_proposals,
        )
        predicted_to_truth, _ = _inverse_matches(filtered, truth)
        truth_divisions = _truth_divisions(truth)
        scores_by_id = {
            int(node_id): float(score)
            for node_id, score in zip(raw.node_ids, scores, strict=True)
        }
        proposal_rows = [
            *build_proposal_rows(
                sample.dataset,
                stage="preprune",
                proposals=preprune_proposals,
                scores_by_id=scores_by_id,
                predicted_to_truth=predicted_to_truth,
                truth_divisions=truth_divisions,
            ),
            *build_proposal_rows(
                sample.dataset,
                stage="postprune",
                proposals=postprune_proposals,
                scores_by_id=scores_by_id,
                predicted_to_truth=predicted_to_truth,
                truth_divisions=truth_divisions,
            ),
        ]
        pre_counts = _stage_counts(proposal_rows, "preprune")
        post_counts = _stage_counts(proposal_rows, "postprune")
        movie_rows.append(
            {
                "dataset": sample.dataset,
                "truth_divisions": len(truth_divisions),
                "raw_candidates": len(raw.node_ids),
                "filtered_candidates": len(filtered.node_ids),
                "pruned_nodes": len(pruned.node_ids),
                "link_seconds": link_seconds,
                "edge_jaccard": result.edge_jaccard,
                "score": result.score,
                "baseline_reproduced": True,
                "preprune_proposals": len(preprune_proposals),
                "preprune_phase7_proposals": sum(
                    passes_phase7_gates(proposal) for proposal in preprune_proposals
                ),
                "preprune_labelled_true_triplets": pre_counts["labelled_true_triplet"],
                "preprune_matched_nondivision_mothers": pre_counts[
                    "matched_nondivision_mother"
                ],
                "preprune_unknown_mothers": pre_counts["unmatched_mother_unknown"],
                "postprune_proposals": len(postprune_proposals),
                "postprune_phase7_proposals": sum(
                    passes_phase7_gates(proposal) for proposal in postprune_proposals
                ),
                "postprune_labelled_true_triplets": post_counts["labelled_true_triplet"],
                "postprune_matched_nondivision_mothers": post_counts[
                    "matched_nondivision_mother"
                ],
                "postprune_unknown_mothers": post_counts["unmatched_mother_unknown"],
                "raw_complete_truth_triplets": _sum_true(event_rows, "raw_complete_triplet"),
                "filtered_complete_truth_triplets": _sum_true(
                    event_rows, "filtered_complete_triplet"
                ),
                "preprune_recoverable_truth_divisions": _sum_true(
                    event_rows, "preprune_second_daughter_candidate"
                ),
                "postprune_recoverable_truth_divisions": _sum_true(
                    event_rows, "postprune_second_daughter_candidate"
                ),
                "score_cache_sha256": file_sha256(cache_path),
            }
        )
        all_event_rows.extend(event_rows)
        all_proposal_rows.extend(proposal_rows)
        elapsed = time.perf_counter() - started
        print(
            f"[{index}/{len(samples)}] {sample.dataset} truth_div={len(truth_divisions)} "
            f"proposals={len(preprune_proposals)}->{len(postprune_proposals)} "
            f"recoverable={_sum_true(event_rows, 'preprune_second_daughter_candidate')}"
            f"->{_sum_true(event_rows, 'postprune_second_daughter_candidate')} "
            f"baseline=exact elapsed={elapsed:.1f}s",
            flush=True,
        )

    _write_csv(targets["events"], EVENT_FIELDS, all_event_rows)
    _write_csv(targets["proposals"], PROPOSAL_FIELDS, all_proposal_rows)
    _write_csv(targets["movies"], MOVIE_FIELDS, movie_rows)
    pre_counts = _stage_counts(all_proposal_rows, "preprune")
    post_counts = _stage_counts(all_proposal_rows, "postprune")
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cohort": "phase10_internal_dev_subset",
        "datasets": datasets,
        "configuration": {
            "cellness_threshold": FROZEN_SCORE_THRESHOLD,
            "min_track_length": FROZEN_MIN_TRACK_LENGTH,
            "broad_max_distance_um": args.broad_max_distance_um,
            "detector_threshold_k": detector_threshold_k,
            "patch_shape": list(patch_shape),
        },
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "frozen_policy_sha256": file_sha256(args.policy),
            "phase10_results_sha256": file_sha256(args.baseline_results),
            "scoring_source_sha256": scoring_source_sha256,
            "frozen_evaluation_source_sha256": baseline_evaluation_sha256,
            "audit_source_sha256": files_sha256(AUDIT_SOURCE_FILES),
        },
        "baseline_reproduction": {
            "movies_checked": len(movie_rows),
            "all_exact": all(row["baseline_reproduced"] for row in movie_rows),
        },
        "truth_event_funnel": {
            "labelled_divisions": len(all_event_rows),
            "raw_complete_triplets": _sum_true(all_event_rows, "raw_complete_triplet"),
            "filtered_complete_triplets": _sum_true(
                all_event_rows, "filtered_complete_triplet"
            ),
            "preprune_second_daughter_candidates": _sum_true(
                all_event_rows, "preprune_second_daughter_candidate"
            ),
            "preprune_broad_proposals": sum(
                int(row["preprune_broad_proposals"]) > 0 for row in all_event_rows
            ),
            "preprune_phase7_proposals": sum(
                int(row["preprune_phase7_proposals"]) > 0 for row in all_event_rows
            ),
            "pruned_complete_triplets": _sum_true(
                all_event_rows, "pruned_complete_triplet_survives"
            ),
            "postprune_second_daughter_candidates": _sum_true(
                all_event_rows, "postprune_second_daughter_candidate"
            ),
            "postprune_broad_proposals": sum(
                int(row["postprune_broad_proposals"]) > 0 for row in all_event_rows
            ),
            "postprune_phase7_proposals": sum(
                int(row["postprune_phase7_proposals"]) > 0 for row in all_event_rows
            ),
        },
        "proposal_roles": {
            "preprune": dict(sorted(pre_counts.items())),
            "postprune": dict(sorted(post_counts.items())),
        },
        "proposal_appearance": _appearance_summary(all_proposal_rows),
        "outputs": {
            "division_events_csv": str(targets["events"]),
            "division_events_sha256": file_sha256(targets["events"]),
            "division_proposals_csv": str(targets["proposals"]),
            "division_proposals_sha256": file_sha256(targets["proposals"]),
            "movies_csv": str(targets["movies"]),
            "movies_sha256": file_sha256(targets["movies"]),
        },
    }
    _write_json(targets["summary"], summary)
    print(f"Saved audit: {args.output_dir}", flush=True)
    print(json.dumps(summary["truth_event_funnel"], indent=2), flush=True)


if __name__ == "__main__":
    main()
