#!/usr/bin/env python3
"""Run one post-selection Phase 11 v2 diagnostic on development labels."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import filter_candidates  # noqa: E402
from biohub.division_data import (  # noqa: E402
    file_sha256,
    logical_graph_sha256,
    read_canonical_json,
    read_division_manifest,
    source_bundle_sha256,
    truth_divisions,
    write_canonical_json,
)
from biohub.division_dataset import (  # noqa: E402
    FROZEN_CELLNESS_THRESHOLD,
    division_cache_source_sha256,
)
from biohub.division_model import (  # noqa: E402
    FeatureSchema,
    canonical_model_state_sha256,
    current_division_contract_hashes,
    read_division_node_cache,
)
from biohub.division_v2 import (  # noqa: E402
    DEFAULT_V2_PROPOSAL_RULE,
    V2PairModel,
    V2PairModelConfig,
    generate_division_pair_proposals_v2,
    generate_targeted_division_pair_neighborhoods,
)
from biohub.io import discover_split, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, match_nodes  # noqa: E402
from scripts.run_division_v2_fit import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST,
    _proposal_inputs,
    load_experiment_config,
    proposal_volume_audit,
    verify_existing_result,
)

DEVELOPMENT_RESULT_SCHEMA_VERSION = 1
DEFAULT_FIT_DIR = ROOT / "data" / "working" / "phase11_v2_atlasforge" / "fit"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "working" / "phase11_v2_atlasforge" / "development"
)
SOURCE_PATHS = (
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "division.py",
    SRC / "biohub" / "division_data.py",
    SRC / "biohub" / "division_dataset.py",
    SRC / "biohub" / "division_model.py",
    SRC / "biohub" / "division_v2.py",
    SRC / "biohub" / "io.py",
    SRC / "biohub" / "link.py",
    SRC / "biohub" / "metric.py",
    Path(__file__).resolve(),
)
GROUP_EVENT = "truth_event"
GROUP_CONTROL = "matched_nondivision_control"
CONTROL_SALT = "biohub-phase11-v2-development-control-v1"


def _project_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("development artifacts must remain inside the project root") from exc


def _invert_unique(matches: Mapping[int, int]) -> dict[int, int]:
    inverted: dict[int, int] = {}
    for predicted, truth in matches.items():
        if truth in inverted:
            raise ValueError("node matching assigned two predictions to one truth node")
        inverted[int(truth)] = int(predicted)
    return inverted


def _control_rank(dataset: str, mother_id: int) -> bytes:
    return hashlib.sha256(
        CONTROL_SALT.encode("ascii")
        + b"\0"
        + dataset.encode("ascii")
        + b"\0"
        + str(mother_id).encode("ascii")
    ).digest()


@dataclass(frozen=True, slots=True)
class DiagnosticGroup:
    dataset: str
    mother_id: int
    kind: str
    candidate_ids: np.ndarray
    raw_context: np.ndarray
    image_features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if not self.dataset.startswith("6bba_"):
            raise ValueError("development groups accept only 6bba datasets")
        if self.kind not in {GROUP_EVENT, GROUP_CONTROL}:
            raise ValueError("development group has an unsupported kind")
        pairs = np.ascontiguousarray(self.candidate_ids, dtype=np.int64)
        context = np.ascontiguousarray(self.raw_context, dtype=np.float32)
        image = np.ascontiguousarray(self.image_features, dtype=np.float32)
        labels = np.ascontiguousarray(self.labels, dtype=np.uint8)
        count = len(pairs)
        if pairs.shape != (count, 2) or count < 1 or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("development group candidate identifiers are invalid")
        if context.ndim != 2 or context.shape[0] != count or not np.all(np.isfinite(context)):
            raise ValueError("development group context features are invalid")
        if image.shape != (count, 256) or not np.all(np.isfinite(image)):
            raise ValueError("development group image features are invalid")
        if labels.shape != (count,) or not np.all(np.isin(labels, (0, 1))):
            raise ValueError("development group labels are invalid")
        if self.kind == GROUP_EVENT and int(labels.sum()) != 1:
            raise ValueError("development event groups require exactly one truth pair")
        if self.kind == GROUP_CONTROL and int(labels.sum()) != 0:
            raise ValueError("development controls cannot contain a truth pair")
        for array in (pairs, context, image, labels):
            array.setflags(write=False)
        object.__setattr__(self, "candidate_ids", pairs)
        object.__setattr__(self, "raw_context", context)
        object.__setattr__(self, "image_features", image)
        object.__setattr__(self, "labels", labels)

    @property
    def key(self) -> tuple[str, str, int]:
        return self.dataset, self.kind, self.mother_id


@dataclass(frozen=True, slots=True)
class LoadedFold:
    fold: int
    schema: FeatureSchema
    ranker: V2PairModel
    gate: V2PairModel
    gate_threshold: float


def load_selected_folds(
    fit_result: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[LoadedFold, ...]:
    """Load the immutable fit-selected five-fold ensemble."""
    folds: list[LoadedFold] = []
    for record in fit_result["selected_checkpoints"]:
        checkpoint = ROOT / record["path"]
        if file_sha256(checkpoint) != record["file_sha256"]:
            raise ValueError("selected checkpoint file hash changed before development audit")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["evaluation_unlocked"] is not False:
            raise ValueError("selected checkpoint did not keep evaluation sealed")
        if canonical_model_state_sha256(payload["ranker_state_dict"]) != record[
            "ranker_state_sha256"
        ]:
            raise ValueError("selected ranker state hash changed")
        if canonical_model_state_sha256(payload["gate_state_dict"]) != record[
            "gate_state_sha256"
        ]:
            raise ValueError("selected gate state hash changed")
        ranker = V2PairModel(V2PairModelConfig(**payload["ranker_config"]))
        ranker.load_state_dict(payload["ranker_state_dict"], strict=True)
        gate = V2PairModel(V2PairModelConfig(**payload["gate_config"]))
        gate.load_state_dict(payload["gate_state_dict"], strict=True)
        folds.append(
            LoadedFold(
                fold=int(payload["fold"]),
                schema=FeatureSchema.from_dict(payload["feature_schema"]),
                ranker=ranker.to(device).eval(),
                gate=gate.to(device).eval(),
                gate_threshold=float(payload["gate_threshold"]),
            )
        )
    folds.sort(key=lambda value: value.fold)
    if [value.fold for value in folds] != list(range(5)):
        raise ValueError("development audit requires the exact five selected folds")
    return tuple(folds)


def _build_diagnostic_group(
    dataset: str,
    *,
    mother_id: int,
    kind: str,
    proposals,
    true_pair: tuple[int, int] | None,
    raw: TrackingGraph,
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
    pruned: TrackingGraph,
) -> DiagnosticGroup:
    candidate_ids, context, image = _proposal_inputs(
        proposals,
        raw=raw,
        cellness=cellness,
        embeddings=embeddings,
        intensity_summaries=intensity_summaries,
        pruned=pruned,
    )
    labels = np.zeros(len(candidate_ids), dtype=np.uint8)
    if true_pair is not None:
        pair = tuple(sorted(true_pair))
        matches = np.flatnonzero(
            (candidate_ids[:, 0] == pair[0]) & (candidate_ids[:, 1] == pair[1])
        )
        if len(matches) != 1:
            raise ValueError("development event group lost its truth pair")
        labels[int(matches[0])] = 1
    return DiagnosticGroup(
        dataset=dataset,
        mother_id=mother_id,
        kind=kind,
        candidate_ids=candidate_ids,
        raw_context=context,
        image_features=image,
        labels=labels,
    )


def build_development_movie(
    row,
    *,
    sample,
    cache_dir: Path,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
    control_mothers_per_movie: int,
) -> tuple[list[DiagnosticGroup], list[dict[str, Any]], dict[str, Any]]:
    """Build labelled event groups and independent matched controls for one movie."""
    if row.role != "development" or not row.dataset.startswith("6bba_"):
        raise ValueError("development audit accepts only frozen 6bba development rows")
    cache_path = cache_dir / f"{row.dataset}.npz"
    cache_hash = file_sha256(cache_path)
    raw, cellness, embeddings, intensity, metadata = read_division_node_cache(
        cache_path,
        expected_dataset=row.dataset,
        expected_role="development",
        expected_cache_source_sha256=cache_source_sha256,
        expected_contract_sha256=contract_sha256,
        expected_upstream_manifest_sha256=upstream_manifest_sha256,
    )
    if file_sha256(cache_path) != cache_hash:
        raise ValueError("development cache changed while reading")
    if sample.labels is None:
        raise FileNotFoundError(f"development labels are missing for {row.dataset}")
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    truth_hash = logical_graph_sha256(truth)
    if truth_hash != row.truth_graph_sha256:
        raise ValueError(f"development truth graph changed for {row.dataset}")
    filtered = filter_candidates(raw, cellness, min_score=FROZEN_CELLNESS_THRESHOLD)
    volume = proposal_volume_audit(filtered, raw)
    mother_matches = match_nodes(filtered, truth)
    candidate_matches = match_nodes(raw, truth)
    truth_to_mother = _invert_unique(mother_matches)
    truth_to_candidate = _invert_unique(candidate_matches)
    divisions = truth_divisions(truth)
    truth_mothers = {
        predicted
        for truth_mother in divisions
        if (predicted := truth_to_mother.get(truth_mother)) is not None
    }
    audit_neighborhoods = generate_targeted_division_pair_neighborhoods(
        filtered,
        raw,
        mother_node_ids=truth_mothers,
        max_distance_um=DEFAULT_V2_PROPOSAL_RULE.max_distance_um,
    )
    audit_by_mother = {item.mother_id: item for item in audit_neighborhoods}
    event_specs: list[tuple[int, tuple[int, int], int]] = []
    event_records: list[dict[str, Any]] = []
    for truth_mother, truth_daughters in sorted(divisions.items()):
        predicted_mother = truth_to_mother.get(truth_mother)
        predicted_daughters = tuple(truth_to_candidate.get(value) for value in truth_daughters)
        neighborhood = audit_by_mother.get(predicted_mother)
        rank_by_candidate = (
            {
                candidate.node_id: index + 1
                for index, candidate in enumerate(neighborhood.candidates)
            }
            if neighborhood is not None
            else {}
        )
        ranks = [
            None if daughter is None else rank_by_candidate.get(daughter)
            for daughter in predicted_daughters
        ]
        v1_reachable = bool(
            neighborhood is not None
            and all(daughter is not None for daughter in predicted_daughters)
            and all(rank is not None and rank <= 12 for rank in ranks)
            and all(
                next(
                    candidate.distance_um
                    for candidate in neighborhood.candidates
                    if candidate.node_id == daughter
                )
                <= 16.0
                for daughter in predicted_daughters
            )
        )
        v2_reachable = False
        pair: tuple[int, int] | None = None
        if neighborhood is not None and all(
            daughter is not None for daughter in predicted_daughters
        ):
            pair = tuple(sorted(int(value) for value in predicted_daughters))
            proposals = generate_division_pair_proposals_v2([neighborhood])
            v2_reachable = any(
                (proposal.candidate_1.node_id, proposal.candidate_2.node_id) == pair
                for proposal in proposals
            )
        if v1_reachable and not v2_reachable:
            raise RuntimeError("v2 regressed a v1-reachable development event")
        if v2_reachable and pair is not None:
            event_specs.append((int(predicted_mother), pair, truth_mother))
        event_records.append(
            {
                "candidate_ranks_within_18um": ranks,
                "dataset": row.dataset,
                "predicted_daughter_ids": list(predicted_daughters),
                "predicted_mother_id": predicted_mother,
                "truth_daughter_ids": list(truth_daughters),
                "truth_mother_id": truth_mother,
                "v1_reachable": v1_reachable,
                "v2_reachable": v2_reachable,
                "v2_newly_reachable": v2_reachable and not v1_reachable,
            }
        )

    controls = [
        int(predicted_mother)
        for predicted_mother, truth_mother in mother_matches.items()
        if truth_mother not in divisions
        and volume.candidate_count_by_mother.get(int(predicted_mother), 0) >= 2
    ]
    controls.sort(key=lambda mother: (_control_rank(row.dataset, mother), mother))
    controls = controls[:control_mothers_per_movie]
    target_mothers = {*(value[0] for value in event_specs), *controls}
    groups: list[DiagnosticGroup] = []
    if target_mothers:
        linked = link_graph_flow(filtered)
        pruned = prune_to_tracks(linked, min_track_length=6)
        neighborhoods = generate_targeted_division_pair_neighborhoods(
            linked,
            raw,
            mother_node_ids=target_mothers,
            pruned_node_ids={int(node_id) for node_id in pruned.node_ids},
            max_distance_um=DEFAULT_V2_PROPOSAL_RULE.max_distance_um,
        )
        by_mother = {item.mother_id: item for item in neighborhoods}
        for mother_id, pair, _ in event_specs:
            if mother_id not in by_mother:
                raise RuntimeError("reachable development event lost its linked neighborhood")
            groups.append(
                _build_diagnostic_group(
                    row.dataset,
                    mother_id=mother_id,
                    kind=GROUP_EVENT,
                    proposals=generate_division_pair_proposals_v2([by_mother[mother_id]]),
                    true_pair=pair,
                    raw=raw,
                    cellness=cellness,
                    embeddings=embeddings,
                    intensity_summaries=intensity,
                    pruned=pruned,
                )
            )
        for mother_id in controls:
            if mother_id not in by_mother:
                raise RuntimeError("eligible development control lost its neighborhood")
            groups.append(
                _build_diagnostic_group(
                    row.dataset,
                    mother_id=mother_id,
                    kind=GROUP_CONTROL,
                    proposals=generate_division_pair_proposals_v2([by_mother[mother_id]]),
                    true_pair=None,
                    raw=raw,
                    cellness=cellness,
                    embeddings=embeddings,
                    intensity_summaries=intensity,
                    pruned=pruned,
                )
            )
    groups.sort(key=lambda value: (value.kind, value.mother_id))
    return groups, event_records, {
        "cache_content_sha256": metadata["content_sha256"],
        "cache_file_sha256": cache_hash,
        "control_groups": len(controls),
        "dataset": row.dataset,
        "truth_graph_sha256": truth_hash,
        "v1_proposals": volume.v1_proposals,
        "v2_proposals": volume.v2_proposals,
    }


def score_diagnostic_groups(
    groups: Sequence[DiagnosticGroup],
    *,
    folds: Sequence[LoadedFold],
    device: torch.device,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Apply the frozen ranker ensemble and all-fold gate consensus."""
    if not groups:
        return {}
    image = np.concatenate([group.image_features for group in groups], axis=0)
    rank_logits: list[np.ndarray] = []
    for fold in folds:
        context = fold.schema.standardize(
            np.concatenate([group.raw_context for group in groups], axis=0)
        )
        with torch.inference_mode():
            values = (
                fold.ranker(
                    torch.tensor(image, dtype=torch.float32, device=device),
                    torch.tensor(context, dtype=torch.float32, device=device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        rank_logits.append(np.asarray(values, dtype=np.float64))

    selected_rows: list[int] = []
    preliminary: list[dict[str, Any]] = []
    offset = 0
    for group in groups:
        count = len(group.labels)
        fold_log_probabilities: list[np.ndarray] = []
        for values in rank_logits:
            logits = values[offset : offset + count]
            maximum = float(np.max(logits))
            log_normalizer = maximum + math.log(float(np.exp(logits - maximum).sum()))
            fold_log_probabilities.append(logits - log_normalizer)
        ensemble = np.mean(np.stack(fold_log_probabilities), axis=0)
        order = np.lexsort(
            (
                group.candidate_ids[:, 1],
                group.candidate_ids[:, 0],
                -ensemble,
            )
        )
        selected = int(order[0])
        selected_rows.append(selected)
        truth_rank = None
        if group.kind == GROUP_EVENT:
            truth_row = int(np.flatnonzero(group.labels == 1)[0])
            truth_rank = int(np.flatnonzero(order == truth_row)[0]) + 1
        preliminary.append(
            {
                "ensemble_selected_log_probability": float(ensemble[selected]),
                "selected_candidate_ids": group.candidate_ids[selected].tolist(),
                "truth_rank": truth_rank,
            }
        )
        offset += count
    if offset != len(image):
        raise RuntimeError("development group offsets disagree with model inputs")

    selected_image = np.stack(
        [group.image_features[row] for group, row in zip(groups, selected_rows, strict=True)]
    )
    gate_scores: list[np.ndarray] = []
    for fold in folds:
        selected_context = fold.schema.standardize(
            np.stack(
                [
                    group.raw_context[row]
                    for group, row in zip(groups, selected_rows, strict=True)
                ]
            )
        )
        with torch.inference_mode():
            values = (
                fold.gate(
                    torch.tensor(selected_image, dtype=torch.float32, device=device),
                    torch.tensor(selected_context, dtype=torch.float32, device=device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        gate_scores.append(np.asarray(values, dtype=np.float64))
    gate_matrix = np.stack(gate_scores)
    thresholds = np.asarray([fold.gate_threshold for fold in folds], dtype=np.float64)
    margins = gate_matrix - thresholds[:, None]
    accepted = np.all(margins >= 0.0, axis=0)
    return {
        group.key: {
            **row,
            "gate_consensus": bool(accepted[index]),
            "gate_logits": [float(value) for value in gate_matrix[:, index]],
            "gate_minimum_margin": float(np.min(margins[:, index])),
            "gate_thresholds": [float(value) for value in thresholds],
        }
        for index, (group, row) in enumerate(zip(groups, preliminary, strict=True))
    }


def verify_existing_development_result(
    output_dir: str | Path,
    *,
    fit_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    result = read_canonical_json(Path(output_dir) / "development_result.json")
    if result.get("development_result_schema_version") != DEVELOPMENT_RESULT_SCHEMA_VERSION:
        raise ValueError("existing development result has an unsupported schema")
    if result.get("evaluation_unlocked") is not False or result.get("policy_frozen") is not False:
        raise ValueError("existing development result crossed a sealed boundary")
    if result.get("config_sha256") != file_sha256(config_path):
        raise ValueError("existing development result config hash changed")
    if result.get("fit_result_sha256") != file_sha256(
        Path(fit_dir) / "fit_result.json"
    ):
        raise ValueError("existing development result fit hash changed")
    if result.get("source_sha256") != source_bundle_sha256(SOURCE_PATHS):
        raise ValueError("existing development result source hash changed")
    return result


def run_development_audit(
    *,
    config_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    data_root: Path,
    fit_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run the one authorized diagnostic after fit model selection is immutable."""
    config = load_experiment_config(config_path)
    if config["development_audit"] != {
        "can_freeze_policy": False,
        "mode": "single_post_selection_diagnostic",
        "uses_evaluation": False,
    }:
        raise ValueError("development audit contract changed")
    fit_result = verify_existing_result(
        fit_dir,
        config_path=config_path,
        manifest_path=manifest_path,
    )
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS device is unavailable")
        device = torch.device("mps")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cpu or mps")
    folds = load_selected_folds(fit_result, device=device)
    contract_hashes = current_division_contract_hashes(config_path.parent)
    manifest_hash = file_sha256(manifest_path)
    cache_source_hash = division_cache_source_sha256(ROOT)
    rows = read_division_manifest(manifest_path)
    development_rows = tuple(row for row in rows if row.role == "development")
    if len(development_rows) != 20 or sum(row.division_count for row in development_rows) != 18:
        raise ValueError("development audit requires the exact frozen 20-movie, 18-event cohort")
    samples = {sample.dataset: sample for sample in discover_split(data_root, "train")}
    if any(row.dataset not in samples for row in development_rows):
        raise FileNotFoundError("development audit is missing one or more development samples")

    event_records: list[dict[str, Any]] = []
    movie_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    total_v1_proposals = 0
    total_v2_proposals = 0
    control_cap = int(config["calibration"]["control_mothers_per_movie"])
    for index, row in enumerate(development_rows, start=1):
        groups, movie_events, movie_record = build_development_movie(
            row,
            sample=samples[row.dataset],
            cache_dir=cache_dir,
            cache_source_sha256=cache_source_hash,
            contract_sha256=contract_hashes,
            upstream_manifest_sha256=manifest_hash,
            control_mothers_per_movie=control_cap,
        )
        scores = score_diagnostic_groups(groups, folds=folds, device=device)
        for event in movie_events:
            predicted_mother = event["predicted_mother_id"]
            if event["v2_reachable"]:
                score = scores[(row.dataset, GROUP_EVENT, int(predicted_mother))]
                event.update(
                    {
                        "v2_gate_consensus": score["gate_consensus"],
                        "v2_gate_minimum_margin": score["gate_minimum_margin"],
                        "v2_selected_candidate_ids": score["selected_candidate_ids"],
                        "v2_true_rank": score["truth_rank"],
                        "v2_two_stage_recovered": (
                            score["truth_rank"] == 1 and score["gate_consensus"]
                        ),
                    }
                )
            else:
                event.update(
                    {
                        "v2_gate_consensus": None,
                        "v2_gate_minimum_margin": None,
                        "v2_selected_candidate_ids": None,
                        "v2_true_rank": None,
                        "v2_two_stage_recovered": False,
                    }
                )
            event_records.append(event)
        for group in groups:
            if group.kind != GROUP_CONTROL:
                continue
            score = scores[group.key]
            control_records.append(
                {
                    "dataset": group.dataset,
                    "gate_consensus": score["gate_consensus"],
                    "gate_minimum_margin": score["gate_minimum_margin"],
                    "mother_id": group.mother_id,
                    "selected_candidate_ids": score["selected_candidate_ids"],
                }
            )
        total_v1_proposals += int(movie_record["v1_proposals"])
        total_v2_proposals += int(movie_record["v2_proposals"])
        movie_records.append(movie_record)
        print(
            f"audited={index}/20 dataset={row.dataset} events={len(movie_events)} "
            f"controls={movie_record['control_groups']}"
        )
        if device.type == "mps":
            torch.mps.empty_cache()

    event_records.sort(key=lambda value: (value["dataset"], value["truth_mother_id"]))
    movie_records.sort(key=lambda value: value["dataset"])
    control_records.sort(key=lambda value: (value["dataset"], value["mother_id"]))
    v1_reachable = sum(event["v1_reachable"] for event in event_records)
    v2_reachable = sum(event["v2_reachable"] for event in event_records)
    if v1_reachable != 10:
        raise ValueError("development v1 reachability changed from the frozen truth-rank atlas")
    if any(event["v1_reachable"] and not event["v2_reachable"] for event in event_records):
        raise RuntimeError("v2 regressed a v1-reachable development event")
    top1 = sum(event["v2_true_rank"] == 1 for event in event_records)
    top3 = sum(
        event["v2_true_rank"] is not None and event["v2_true_rank"] <= 3
        for event in event_records
    )
    two_stage = sum(event["v2_two_stage_recovered"] for event in event_records)
    control_accepts = sum(record["gate_consensus"] for record in control_records)
    source_hash = source_bundle_sha256(SOURCE_PATHS)
    result = {
        "cache_directory": _project_relative(cache_dir),
        "cache_source_sha256": cache_source_hash,
        "config_path": _project_relative(config_path),
        "config_sha256": file_sha256(config_path),
        "contract_sha256": contract_hashes,
        "control_records": control_records,
        "data_root": _project_relative(data_root),
        "development_result_schema_version": DEVELOPMENT_RESULT_SCHEMA_VERSION,
        "device": str(device),
        "evaluation_cache_files_opened": 0,
        "evaluation_unlocked": False,
        "event_records": event_records,
        "fit_result_path": _project_relative(fit_dir / "fit_result.json"),
        "fit_result_sha256": file_sha256(fit_dir / "fit_result.json"),
        "manifest_path": _project_relative(manifest_path),
        "manifest_sha256": manifest_hash,
        "movie_records": movie_records,
        "policy_frozen": False,
        "proposal_audit": {
            "development_truth_events": len(event_records),
            "total_v1_proposals": total_v1_proposals,
            "total_v2_proposals": total_v2_proposals,
            "total_volume_ratio": total_v2_proposals / total_v1_proposals,
            "v1_reachable_events": v1_reachable,
            "v2_reachable_events": v2_reachable,
        },
        "selected_gate": fit_result["selected_gate"],
        "selected_ranker": fit_result["selected_ranker"],
        "source_sha256": source_hash,
        "targeted_diagnostic": {
            "gate_consensus_control_accepts": control_accepts,
            "matched_control_groups": len(control_records),
            "reachable_top1": top1,
            "reachable_top3": top3,
            "two_stage_recoveries": two_stage,
        },
    }
    if output_dir.exists():
        raise FileExistsError(
            f"development output already exists: {output_dir}; use --verify-existing"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
    ) as temporary_name:
        staging = Path(temporary_name)
        result_hash = write_canonical_json(staging / "development_result.json", result)
        os.replace(staging, output_dir)
    print(
        f"development_result={output_dir / 'development_result.json'} "
        f"sha256={result_hash}"
    )
    return verify_existing_development_result(
        output_dir,
        fit_dir=fit_dir,
        config_path=config_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.verify_existing:
        result = verify_existing_development_result(
            args.output_dir,
            fit_dir=args.fit_dir,
            config_path=args.config,
        )
        print(
            f"verified={args.output_dir / 'development_result.json'} "
            f"two_stage={result['targeted_diagnostic']['two_stage_recoveries']}"
        )
        return
    run_development_audit(
        config_path=args.config,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        data_root=args.data_root,
        fit_dir=args.fit_dir,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
