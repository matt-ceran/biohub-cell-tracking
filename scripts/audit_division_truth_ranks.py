#!/usr/bin/env python3
"""Build the provenance-bound Phase 11 development truth-rank atlas."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import extract_patch  # noqa: E402
from biohub.division import (  # noqa: E402
    DivisionPairNeighborhood,
    DivisionPairProposal,
    generate_division_pair_neighborhoods,
    generate_division_pair_proposals,
)
from biohub.division_data import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
    logical_graph_sha256,
    read_division_contracts,
    source_bundle_sha256,
    truth_divisions,
    write_canonical_json,
)
from biohub.division_dataset import (  # noqa: E402
    _candidate_sample,
    _oriented_edges,
    _proposal_model_inputs,
)
from biohub.division_inference import (  # noqa: E402
    PROBABILITY_THRESHOLDS,
    AcceptedFork,
    ScoredProposal,
)
from biohub.division_model import DIVISION_MODEL_NAMES  # noqa: E402
from biohub.division_truth_rank import (  # noqa: E402
    ASSIGNMENT_INCONSISTENT,
    classify_threshold_outcome,
    rank_truth_proposal,
)
from biohub.io import discover_split, open_image_array  # noqa: E402
from biohub.metric import TrackingGraph, match_nodes  # noqa: E402
from scripts.freeze_division_policy import DEFAULT_POLICY, DEFAULT_RESULT  # noqa: E402
from scripts.validate_division_model import (  # noqa: E402
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_FIT_DATASET,
    DEFAULT_SUMMARY,
    DEFAULT_UPSTREAM_MANIFEST,
    _load_inference_movie,
    _load_truth,
    _load_validation_identity,
    _validate_existing_grid_artifacts,
    read_validation_grid,
)
from scripts.validate_division_model import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_GRID,
)

TRUTH_RANK_ATLAS_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = ROOT / "data" / "working" / "phase11_division_truth_rank_atlas"
AUDIT_SOURCE_PATHS = (
    SRC / "biohub" / "division_truth_rank.py",
    Path(__file__).resolve(),
)
PATCH_SHAPE = (9, 33, 33)
HEAD_LABELS = {
    "scalar": "scalar",
    "embedding_linear": "embedding linear",
    "embedding_mlp32": "embedding MLP-32",
}
ROLE_COLORS = {
    "mother": "#D39A19",
    "truth": "#188A63",
    "false": "#C84A3A",
}


def _project_relative(path: str | Path, *, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("truth-rank artifacts must remain inside the project root") from exc


def _proposal_payload(proposal: ScoredProposal | AcceptedFork | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "candidate_ids": [proposal.candidate_1_id, proposal.candidate_2_id],
        "edge_removal_count": proposal.edge_removal_count,
        "logit": proposal.logit,
        "mother_id": proposal.mother_id,
        "probability": proposal.probability,
        "restored_node_count": proposal.restored_node_count,
    }


def _accepted_forks_from_row(row: Mapping[str, Any]) -> tuple[AcceptedFork, ...]:
    fallback_hash = row["fallback_graph_sha256"]
    forks: list[AcceptedFork] = []
    for record in row["accepted_fork_records"]:
        fork = AcceptedFork(
            mother_id=record["mother_id"],
            candidate_1_id=record["candidate_1_id"],
            candidate_2_id=record["candidate_2_id"],
            logit=record["logit"],
            edge_removal_count=record["edge_removal_count"],
            restored_node_count=record["restored_node_count"],
            fallback_graph_sha256=fallback_hash,
        )
        if fork.probability != record["probability"]:
            raise ValueError("grid accepted-fork probability does not match its logit")
        forks.append(fork)
    return tuple(forks)


def _node_row(graph: TrackingGraph, node_id: int) -> int:
    rows = np.flatnonzero(np.asarray(graph.node_ids, dtype=np.int64) == int(node_id))
    if len(rows) != 1:
        raise ValueError(f"node {node_id} does not occur exactly once")
    return int(rows[0])


def _node_payload(graph: TrackingGraph, node_id: int) -> dict[str, Any]:
    row = _node_row(graph, node_id)
    return {
        "node_id": int(node_id),
        "t": int(graph.t[row]),
        "z": float(graph.z[row]),
        "y": float(graph.y[row]),
        "x": float(graph.x[row]),
    }


def _invert_unique(matches: Mapping[int, int]) -> dict[int, int]:
    inverted: dict[int, int] = {}
    for predicted, truth in matches.items():
        if truth in inverted:
            raise ValueError("node matching assigned two predictions to one truth node")
        inverted[int(truth)] = int(predicted)
    return inverted


def _proposal_context(
    movie: Any,
    *,
    neighborhood: DivisionPairNeighborhood,
    proposal_rule: Mapping[str, Any],
) -> tuple[tuple[DivisionPairProposal, ...], tuple[Any, ...], np.ndarray, np.ndarray]:
    proposals = tuple(
        generate_division_pair_proposals(
            (neighborhood,),
            min_distance_um=float(proposal_rule["min_distance_um"]),
            max_distance_um=float(proposal_rule["max_distance_um"]),
            min_angle_deg=float(proposal_rule["min_angle_deg"]),
            max_candidates_per_mother=int(proposal_rule["max_candidates_per_mother"]),
        )
    )
    if not proposals:
        raise ValueError("reachable truth mother produced no proposals")
    node_row = {int(node_id): index for index, node_id in enumerate(movie.raw.node_ids)}
    score_by_id = {
        int(node_id): float(score)
        for node_id, score in zip(movie.raw.node_ids, movie.cellness, strict=True)
    }
    fallback_ids = {int(node_id) for node_id in movie.fallback.node_ids}
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in _oriented_edges(movie.fallback):
        successors[source].add(target)
        predecessors[target].add(source)
    candidates = tuple(
        _candidate_sample(
            proposal,
            score_by_id=score_by_id,
            pruned_ids=fallback_ids,
            pruned_successors=successors,
            pruned_predecessors=predecessors,
        )
        for proposal in proposals
    )
    inputs = tuple(
        _proposal_model_inputs(
            candidate,
            node_row=node_row,
            cellness=movie.cellness,
            embeddings=movie.embeddings,
            intensity_summaries=movie.intensity_summaries,
        )
        for candidate in candidates
    )
    raw_context = np.stack([item[0] for item in inputs]).astype(np.float32, copy=False)
    image_features = np.stack([item[1] for item in inputs]).astype(np.float32, copy=False)
    return proposals, candidates, raw_context, image_features


def _score_proposals(
    proposals: Sequence[DivisionPairProposal],
    candidates: Sequence[Any],
    *,
    raw_context: np.ndarray,
    image_features: np.ndarray,
    feature_schema: Any,
    models: Mapping[str, Any],
    fallback_hash: str,
    device: torch.device,
) -> dict[str, tuple[ScoredProposal, ...]]:
    context = torch.tensor(
        feature_schema.standardize(raw_context),
        dtype=torch.float32,
        device=device,
    )
    image = torch.tensor(image_features, dtype=torch.float32, device=device)
    scored: dict[str, tuple[ScoredProposal, ...]] = {}
    for model_name in DIVISION_MODEL_NAMES:
        with torch.inference_mode():
            logits = (
                models[model_name].forward_cached_features(image, context).detach().cpu().numpy()
            )
        scored[model_name] = tuple(
            ScoredProposal(
                mother_id=proposal.mother_id,
                candidate_1_id=proposal.candidate_1.node_id,
                candidate_2_id=proposal.candidate_2.node_id,
                logit=float(logit),
                edge_removal_count=candidate.edge_removal_count,
                restored_node_count=candidate.restored_node_count,
                fallback_graph_sha256=fallback_hash,
            )
            for proposal, candidate, logit in zip(
                proposals,
                candidates,
                logits,
                strict=True,
            )
        )
    return scored


def _extract_node_patch(
    image_array: Any,
    graph: TrackingGraph,
    node_id: int,
    *,
    frame_cache: dict[int, np.ndarray],
) -> np.ndarray:
    row = _node_row(graph, node_id)
    timepoint = int(graph.t[row])
    volume = frame_cache.get(timepoint)
    if volume is None:
        volume = np.asarray(image_array[timepoint])
        frame_cache[timepoint] = volume
    return extract_patch(
        volume,
        (float(graph.z[row]), float(graph.y[row]), float(graph.x[row])),
        PATCH_SHAPE,
    )


def _plot_patch(
    axis: Any,
    patch: np.ndarray,
    *,
    title: str,
    color: str,
    vmin: float,
    vmax: float,
) -> None:
    projection = np.asarray(patch).max(axis=0)
    axis.imshow(projection, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_title(title, fontsize=8, color="#172126", pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2.2)
    inset = axis.inset_axes([0.66, 0.04, 0.30, 0.30])
    inset.imshow(
        np.asarray(patch)[patch.shape[0] // 2],
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)


def _write_contact_sheet(
    path: Path,
    *,
    dataset: str,
    truth_mother_id: int,
    movie: Any,
    image_path: Path,
    predicted_mother_id: int,
    true_candidate_ids: tuple[int, int],
    rankings: Mapping[str, Any],
) -> None:
    image_array = open_image_array(image_path)
    frame_cache: dict[int, np.ndarray] = {}
    patch_cache: dict[int, np.ndarray] = {}

    def patch(node_id: int) -> np.ndarray:
        value = patch_cache.get(node_id)
        if value is None:
            value = _extract_node_patch(
                image_array,
                movie.raw,
                node_id,
                frame_cache=frame_cache,
            )
            patch_cache[node_id] = value
        return value

    rows: list[tuple[str, tuple[int, int]]] = []
    for model_name in DIVISION_MODEL_NAMES:
        false = rankings[model_name].best_false_proposal
        if false is None:
            raise ValueError("truth-rank atlas requires one false comparison per head")
        rows.append((model_name, false.candidate_ids))
    displayed_ids = {predicted_mother_id, *true_candidate_ids}
    for _, false_ids in rows:
        displayed_ids.update(false_ids)
    patches = [patch(node_id) for node_id in sorted(displayed_ids)]
    pixels = np.concatenate([np.asarray(value, dtype=np.float32).ravel() for value in patches])
    vmin, vmax = (float(value) for value in np.percentile(pixels, (1.0, 99.7)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(pixels))
        vmax = float(np.max(pixels))
    if vmax <= vmin:
        vmax = vmin + 1.0

    figure, axes = plt.subplots(3, 5, figsize=(15.5, 8.2), facecolor="#F4F7F8")
    figure.suptitle(
        f"{dataset} - truth mother {truth_mother_id}",
        fontsize=16,
        fontweight="bold",
        color="#0E5F62",
        y=0.985,
    )
    column_labels = (
        ("mother", predicted_mother_id, "mother"),
        ("true daughter 1", true_candidate_ids[0], "truth"),
        ("true daughter 2", true_candidate_ids[1], "truth"),
    )
    for row_index, (model_name, false_ids) in enumerate(rows):
        ranking = rankings[model_name]
        cells = (
            *column_labels,
            ("best false 1", false_ids[0], "false"),
            ("best false 2", false_ids[1], "false"),
        )
        for column_index, (label, node_id, role) in enumerate(cells):
            _plot_patch(
                axes[row_index, column_index],
                patch(node_id),
                title=f"{label}\nnode {node_id}",
                color=ROLE_COLORS[role],
                vmin=vmin,
                vmax=vmax,
            )
        axes[row_index, 0].set_ylabel(
            f"{HEAD_LABELS[model_name]}\ntruth rank {ranking.true_rank}/{ranking.proposal_count}",
            fontsize=9,
            fontweight="bold",
            color="#172126",
            labelpad=18,
        )
    figure.text(
        0.5,
        0.015,
        "Large view: maximum-intensity projection. Small inset: center z-slice. "
        "Gold = mother, green = matched truth daughters, red = highest-scoring wrong pair.",
        ha="center",
        fontsize=9,
        color="#445258",
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.08, wspace=0.16, hspace=0.28)
    figure.savefig(
        path,
        dpi=180,
        facecolor=figure.get_facecolor(),
        metadata={"Software": "Biohub Phase 11 truth-rank atlas"},
    )
    plt.close(figure)


def _threshold_key(value: float) -> str:
    return format(float(value), ".4g")


def _event_reachability(
    *,
    truth_mother_id: int,
    truth_daughter_ids: tuple[int, int],
    mother_by_truth: Mapping[int, int],
    candidate_by_truth: Mapping[int, int],
    neighborhood_by_mother: Mapping[int, DivisionPairNeighborhood],
    proposal_rule: Mapping[str, Any],
) -> tuple[dict[str, Any], DivisionPairNeighborhood | None, tuple[int, int] | None]:
    predicted_mother = mother_by_truth.get(truth_mother_id)
    predicted_daughters = tuple(candidate_by_truth.get(value) for value in truth_daughter_ids)
    payload: dict[str, Any] = {
        "predicted_mother_id": predicted_mother,
        "predicted_true_candidate_ids": list(predicted_daughters),
        "reachable": False,
        "reason": None,
        "candidate_count_within_radius": 0,
        "true_candidate_ranks": [None, None],
    }
    if predicted_mother is None:
        payload["reason"] = "mother_missing_from_linked_source"
        return payload, None, None
    missing = [
        truth_id
        for truth_id, predicted_id in zip(
            truth_daughter_ids,
            predicted_daughters,
            strict=True,
        )
        if predicted_id is None
    ]
    if missing:
        payload["reason"] = "true_daughter_missing_from_raw_candidates"
        payload["missing_truth_daughter_ids"] = missing
        return payload, None, None
    pair = tuple(sorted(int(value) for value in predicted_daughters if value is not None))
    neighborhood = neighborhood_by_mother.get(predicted_mother)
    if neighborhood is None:
        payload["reason"] = "mother_has_fewer_than_two_candidates_within_radius"
        return payload, None, pair
    candidate_ids = [candidate.node_id for candidate in neighborhood.candidates]
    payload["candidate_count_within_radius"] = len(candidate_ids)
    ranks = [
        candidate_ids.index(node_id) + 1 if node_id in candidate_ids else None for node_id in pair
    ]
    payload["true_candidate_ranks"] = ranks
    if any(rank is None for rank in ranks):
        payload["reason"] = "true_daughter_outside_proposal_radius"
        return payload, neighborhood, pair
    cap = int(proposal_rule["max_candidates_per_mother"])
    if any(int(rank) > cap for rank in ranks if rank is not None):
        payload["reason"] = "true_daughter_beyond_candidate_cap"
        return payload, neighborhood, pair
    proposals = generate_division_pair_proposals(
        (neighborhood,),
        min_distance_um=float(proposal_rule["min_distance_um"]),
        max_distance_um=float(proposal_rule["max_distance_um"]),
        min_angle_deg=float(proposal_rule["min_angle_deg"]),
        max_candidates_per_mother=cap,
    )
    proposed_pairs = {
        (proposal.candidate_1.node_id, proposal.candidate_2.node_id) for proposal in proposals
    }
    if pair not in proposed_pairs:
        payload["reason"] = "true_pair_removed_by_proposal_rule"
        return payload, neighborhood, pair
    payload["reachable"] = True
    payload["reason"] = "reachable"
    return payload, neighborhood, pair


def _head_summary(events: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any]:
    reachable = [event for event in events if event["proposal_reachability"]["reachable"]]
    rows = [event["heads"][model_name] for event in reachable]
    ranks = [row["true_rank"] for row in rows]
    logit_gaps = [
        row["best_false_minus_true_logit"]
        for row in rows
        if row["best_false_minus_true_logit"] is not None
    ]
    probability_gaps = [
        row["best_false_minus_true_probability"]
        for row in rows
        if row["best_false_minus_true_probability"] is not None
    ]
    stage_counts: dict[str, dict[str, int]] = {}
    for threshold in PROBABILITY_THRESHOLDS:
        key = _threshold_key(threshold)
        stage_counts[key] = dict(
            sorted(Counter(row["threshold_outcomes"][key]["stage"] for row in rows).items())
        )
    return {
        "best_false_beats_truth_count": sum(gap > 0.0 for gap in logit_gaps),
        "best_false_minus_true_logit": {
            "maximum": max(logit_gaps),
            "mean": float(np.mean(logit_gaps)),
            "median": float(median(logit_gaps)),
            "minimum": min(logit_gaps),
        },
        "best_false_minus_true_probability": {
            "maximum": max(probability_gaps),
            "mean": float(np.mean(probability_gaps)),
            "median": float(median(probability_gaps)),
            "minimum": min(probability_gaps),
        },
        "events_ever_accepted": sum(row["ever_accepted"] for row in rows),
        "outcome_stage_counts_by_threshold": stage_counts,
        "reachable_event_count": len(rows),
        "top_1_truth_count": sum(rank <= 1 for rank in ranks),
        "top_3_truth_count": sum(rank <= 3 for rank in ranks),
        "top_5_truth_count": sum(rank <= 5 for rank in ranks),
        "top_10_truth_count": sum(rank <= 10 for rank in ranks),
        "true_rank": {
            "maximum": max(ranks),
            "mean": float(np.mean(ranks)),
            "median": float(median(ranks)),
            "minimum": min(ranks),
        },
    }


def _grid_summary(grid: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_name in DIVISION_MODEL_NAMES:
        threshold_rows: dict[str, Any] = {}
        for threshold in PROBABILITY_THRESHOLDS:
            rows = [
                row
                for row in grid["rows"]
                if row["model_name"] == model_name and row["score_threshold"] == threshold
            ]
            if len(rows) != 20:
                raise ValueError("grid replay summary is missing development movie rows")
            threshold_rows[_threshold_key(threshold)] = {
                "accepted_forks": sum(row["accepted_forks"] for row in rows),
                "fully_matched_wrong_forks": sum(row["fully_matched_wrong_forks"] for row in rows),
                "strict_true_recoveries": sum(row["strict_true_recoveries"] for row in rows),
            }
        summary[model_name] = threshold_rows
    return summary


def run_truth_rank_atlas(
    *,
    project_root: Path = ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fit_dataset: Path = DEFAULT_FIT_DATASET,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    upstream_manifest: Path = DEFAULT_UPSTREAM_MANIFEST,
    manifest_summary: Path = DEFAULT_SUMMARY,
    grid_path: Path = DEFAULT_GRID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the complete development-only truth-rank audit and write its atlas."""
    if output_dir.exists():
        raise FileExistsError(f"truth-rank atlas output already exists: {output_dir}")
    if DEFAULT_POLICY.exists() or DEFAULT_RESULT.exists():
        raise ValueError("truth-rank audit must run before policy or development-result freezing")
    audit_source_hash = source_bundle_sha256(AUDIT_SOURCE_PATHS)
    grid = read_validation_grid(
        grid_path,
        require_complete=True,
        verify_current_sources=True,
    )
    _validate_existing_grid_artifacts(grid, project_root=project_root)
    target_device = torch.device(device)
    identity, loaded, models, manifest_rows, checkpoint_metadata = _load_validation_identity(
        project_root=project_root,
        data_root=data_root,
        cache_dir=cache_dir,
        fit_dataset=fit_dataset,
        checkpoint_dir=checkpoint_dir,
        base_checkpoint=base_checkpoint,
        config_dir=config_dir,
        upstream_manifest=upstream_manifest,
        summary=manifest_summary,
        device=target_device,
    )
    grid = read_validation_grid(
        grid_path,
        expected_identity=identity,
        require_complete=True,
    )
    contracts = read_division_contracts(config_dir)
    proposal_rule = contracts["proposal"]["proposal_rule"]
    evaluation_datasets = tuple(contracts["split"]["role_datasets"]["evaluation"])
    evaluation_cache_paths = [
        cache_dir / f"{dataset}.npz"
        for dataset in evaluation_datasets
        if (cache_dir / f"{dataset}.npz").exists()
    ]
    if evaluation_cache_paths:
        raise ValueError("evaluation caches must remain absent during the truth-rank audit")
    development = tuple(identity["datasets"])
    manifest_by_dataset = {row.dataset: row for row in manifest_rows}
    division_datasets = tuple(
        dataset for dataset in development if manifest_by_dataset[dataset].division_count > 0
    )
    if len(division_datasets) != 10:
        raise ValueError("development truth-rank audit requires exactly 10 division movies")
    samples = discover_split(data_root, "train")
    sample_by_dataset = {sample.dataset: sample for sample in samples}
    grid_rows = {
        (row["dataset"], row["model_name"], float(row["score_threshold"])): row
        for row in grid["rows"]
    }
    if len(grid_rows) != len(grid["rows"]):
        raise ValueError("development grid contains duplicate row keys")

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_parent))
    image_dir = staging / "images"
    image_dir.mkdir()
    events: list[dict[str, Any]] = []
    try:
        for dataset in division_datasets:
            movie, cache_metadata, cache_file_hash = _load_inference_movie(
                dataset,
                role="development",
                cache_dir=cache_dir,
                cache_source_sha256=identity["cache_source_sha256"],
                contract_sha256=identity["contract_sha256"],
                upstream_manifest_sha256=identity["upstream_manifest_sha256"],
                proposal_contract=proposal_rule,
            )
            truth, _ = _load_truth(
                dataset,
                sample_by_dataset=sample_by_dataset,
                manifest_by_dataset=manifest_by_dataset,
            )
            truth_events = truth_divisions(truth)
            if len(truth_events) != manifest_by_dataset[dataset].division_count:
                raise ValueError(f"truth division count changed for {dataset}")
            mother_by_truth = _invert_unique(match_nodes(movie.linked, truth))
            candidate_by_truth = _invert_unique(match_nodes(movie.raw, truth))
            fallback_ids = {int(node_id) for node_id in movie.fallback.node_ids}
            neighborhoods = generate_division_pair_neighborhoods(
                movie.linked,
                movie.raw,
                pruned_node_ids=fallback_ids,
                max_distance_um=float(proposal_rule["max_distance_um"]),
            )
            neighborhood_by_mother = {
                neighborhood.mother_id: neighborhood for neighborhood in neighborhoods
            }
            if len(neighborhood_by_mother) != len(neighborhoods):
                raise ValueError("proposal generation returned duplicate mother neighborhoods")
            fallback_hash = logical_graph_sha256(movie.fallback)
            sample = sample_by_dataset[dataset]
            for truth_mother_id, truth_daughters in sorted(truth_events.items()):
                reachability, neighborhood, true_pair = _event_reachability(
                    truth_mother_id=truth_mother_id,
                    truth_daughter_ids=truth_daughters,
                    mother_by_truth=mother_by_truth,
                    candidate_by_truth=candidate_by_truth,
                    neighborhood_by_mother=neighborhood_by_mother,
                    proposal_rule=proposal_rule,
                )
                event: dict[str, Any] = {
                    "dataset": dataset,
                    "event_id": f"{dataset}:{truth_mother_id}",
                    "heads": {},
                    "proposal_reachability": reachability,
                    "truth_daughter_ids": list(truth_daughters),
                    "truth_mother_id": truth_mother_id,
                }
                if not reachability["reachable"]:
                    events.append(event)
                    continue
                if neighborhood is None or true_pair is None:
                    raise RuntimeError("reachable event is missing its proposal neighborhood")
                predicted_mother = int(reachability["predicted_mother_id"])
                proposals, candidates, raw_context, image_features = _proposal_context(
                    movie,
                    neighborhood=neighborhood,
                    proposal_rule=proposal_rule,
                )
                scored = _score_proposals(
                    proposals,
                    candidates,
                    raw_context=raw_context,
                    image_features=image_features,
                    feature_schema=loaded.feature_schema,
                    models=models,
                    fallback_hash=fallback_hash,
                    device=target_device,
                )
                ranking_objects: dict[str, Any] = {}
                for model_name in DIVISION_MODEL_NAMES:
                    ranking = rank_truth_proposal(
                        scored[model_name],
                        true_candidate_ids=true_pair,
                    )
                    ranking_objects[model_name] = ranking
                    outcomes: dict[str, Any] = {}
                    for threshold in PROBABILITY_THRESHOLDS:
                        row = grid_rows[(dataset, model_name, float(threshold))]
                        if row["fallback_graph_sha256"] != fallback_hash:
                            raise ValueError("truth-rank replay fallback disagrees with grid")
                        outcome = classify_threshold_outcome(
                            ranking,
                            threshold=threshold,
                            accepted_forks=_accepted_forks_from_row(row),
                        )
                        if outcome.stage == ASSIGNMENT_INCONSISTENT:
                            raise ValueError(
                                f"assignment trace is inconsistent for {dataset}, "
                                f"{model_name}, {threshold}"
                            )
                        outcomes[_threshold_key(threshold)] = {
                            "accepted_fork_count": outcome.accepted_fork_count,
                            "blocking_fork": _proposal_payload(outcome.blocking_fork),
                            "stage": outcome.stage,
                            "threshold": outcome.threshold,
                            "true_probability": outcome.true_probability,
                        }
                    head_payload = {
                        "best_false_minus_true_logit": ranking.false_minus_true_logit,
                        "best_false_minus_true_probability": (ranking.false_minus_true_probability),
                        "best_false_proposal": _proposal_payload(ranking.best_false_proposal),
                        "best_proposal": _proposal_payload(ranking.best_proposal),
                        "ever_accepted": any(
                            outcome["stage"] == "accepted" for outcome in outcomes.values()
                        ),
                        "proposal_count": ranking.proposal_count,
                        "threshold_outcomes": outcomes,
                        "true_proposal": _proposal_payload(ranking.true_proposal),
                        "true_rank": ranking.true_rank,
                    }
                    event["heads"][model_name] = head_payload
                event["predicted_mother"] = _node_payload(movie.raw, predicted_mother)
                event["predicted_true_candidates"] = [
                    _node_payload(movie.raw, node_id) for node_id in true_pair
                ]
                image_name = f"{dataset}_truth_{truth_mother_id}.png"
                image_path = image_dir / image_name
                _write_contact_sheet(
                    image_path,
                    dataset=dataset,
                    truth_mother_id=truth_mother_id,
                    movie=movie,
                    image_path=sample.image,
                    predicted_mother_id=predicted_mother,
                    true_candidate_ids=true_pair,
                    rankings=ranking_objects,
                )
                event["contact_sheet"] = {
                    "path": f"images/{image_name}",
                    "sha256": file_sha256(image_path),
                }
                events.append(event)
            if cache_metadata["dataset"] != dataset:
                raise ValueError("cache metadata dataset changed while auditing")
            expected_cache_hashes = {
                row["cache_file_sha256"] for row in grid["rows"] if row["dataset"] == dataset
            }
            if expected_cache_hashes != {cache_file_hash}:
                raise ValueError("truth-rank cache hash disagrees with development grid")

        if len(events) != 18:
            raise ValueError("truth-rank atlas requires exactly 18 development truth events")
        reachable_count = sum(event["proposal_reachability"]["reachable"] for event in events)
        if reachable_count != 10:
            raise ValueError("truth-rank atlas expected exactly 10 reachable truth events")
        if sum("contact_sheet" in event for event in events) != reachable_count:
            raise ValueError("each reachable event must have one contact sheet")
        reason_counts = Counter(event["proposal_reachability"]["reason"] for event in events)
        cache_files = sorted(cache_dir.glob("*.npz"))
        payload = {
            "artifacts": {
                "contact_sheet_count": reachable_count,
                "contact_sheet_directory": "images",
            },
            "audit_source_sha256": audit_source_hash,
            "checkpoint_epochs": {
                model_name: checkpoint_metadata[model_name]["epoch"]
                for model_name in DIVISION_MODEL_NAMES
            },
            "complete": True,
            "contract_sha256": identity["contract_sha256"],
            "datasets": {
                "development": list(development),
                "development_division_movies": list(division_datasets),
            },
            "evaluation_boundary": {
                "evaluation_cache_count": len(evaluation_cache_paths),
                "evaluation_opened": False,
                "phase11_development_result_exists": DEFAULT_RESULT.exists(),
                "phase11_policy_exists": DEFAULT_POLICY.exists(),
            },
            "events": events,
            "feature_schema_sha256": identity["feature_schema_sha256"],
            "grid": {
                "path": _project_relative(grid_path, project_root=project_root),
                "rows_sha256": grid["rows_sha256"],
                "sha256": file_sha256(grid_path),
                "validation_source_sha256": grid["validation_source_sha256"],
            },
            "grid_replay_summary": _grid_summary(grid),
            "model_inputs_sha256": identity["model_inputs_sha256"],
            "proposal_rule": dict(proposal_rule),
            "role": "development",
            "summary": {
                "development_movie_count": len(development),
                "division_movie_count": len(division_datasets),
                "head_results": {
                    model_name: _head_summary(events, model_name)
                    for model_name in DIVISION_MODEL_NAMES
                },
                "reachable_truth_event_count": reachable_count,
                "reachability_reason_counts": dict(sorted(reason_counts.items())),
                "truth_event_count": len(events),
                "unreachable_truth_event_count": len(events) - reachable_count,
            },
            "thresholds": list(PROBABILITY_THRESHOLDS),
            "truth_rank_atlas_schema_version": TRUTH_RANK_ATLAS_SCHEMA_VERSION,
            "working_cache_file_count": len(cache_files),
        }
        if payload["evaluation_boundary"] != {
            "evaluation_cache_count": 0,
            "evaluation_opened": False,
            "phase11_development_result_exists": False,
            "phase11_policy_exists": False,
        }:
            raise RuntimeError("truth-rank audit crossed the frozen evaluation boundary")
        if source_bundle_sha256(AUDIT_SOURCE_PATHS) != audit_source_hash:
            raise RuntimeError("truth-rank audit source changed while the audit was running")
        write_canonical_json(staging / "atlas.json", payload)
        manifest = {
            "atlas": {
                "path": "atlas.json",
                "sha256": file_sha256(staging / "atlas.json"),
            },
            "contact_sheets": [
                event["contact_sheet"] for event in events if "contact_sheet" in event
            ],
            "manifest_schema_version": 1,
        }
        manifest["logical_content_sha256"] = canonical_json_sha256(
            {
                "atlas_sha256": manifest["atlas"]["sha256"],
                "contact_sheet_sha256": [record["sha256"] for record in manifest["contact_sheets"]],
            }
        )
        write_canonical_json(staging / "manifest.json", manifest)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("development",), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fit-dataset", type=Path, default=DEFAULT_FIT_DATASET)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--upstream-manifest", type=Path, default=DEFAULT_UPSTREAM_MANIFEST)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_truth_rank_atlas(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        fit_dataset=args.fit_dataset,
        checkpoint_dir=args.checkpoint_dir,
        base_checkpoint=args.base_checkpoint,
        config_dir=args.config_dir,
        upstream_manifest=args.upstream_manifest,
        manifest_summary=args.manifest_summary,
        grid_path=args.grid,
        output_dir=args.output_dir,
        device=args.device,
    )
    atlas_path = args.output_dir / "atlas.json"
    print(
        "Saved Phase 11 truth-rank atlas: "
        f"events={payload['summary']['truth_event_count']} "
        f"reachable={payload['summary']['reachable_truth_event_count']} "
        f"sha256={file_sha256(atlas_path)} "
        f"path={atlas_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
