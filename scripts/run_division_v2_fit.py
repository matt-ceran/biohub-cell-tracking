#!/usr/bin/env python3
"""Run the fit-only, movie-grouped Phase 11 v2 ATLASFORGE experiment."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance import filter_candidates  # noqa: E402
from biohub.constants import VOXEL_SCALE_UM  # noqa: E402
from biohub.division_data import (  # noqa: E402
    file_sha256,
    logical_graph_sha256,
    read_canonical_json,
    read_division_contracts,
    read_division_manifest,
    source_bundle_sha256,
    truth_divisions,
    write_canonical_json,
)
from biohub.division_dataset import (  # noqa: E402
    FROZEN_CELLNESS_THRESHOLD,
    MATCHED_NONDIVISION_CONTROL,
    POSITIVE,
    SAME_MOTHER_NEGATIVE,
    _candidate_sample,
    _oriented_edges,
    _proposal_model_inputs,
    division_cache_source_sha256,
    read_division_dataset,
)
from biohub.division_model import (  # noqa: E402
    CONTEXT_FEATURE_KINDS,
    CONTEXT_FEATURE_NAMES,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    FeatureSchema,
    canonical_model_state_sha256,
    current_division_contract_hashes,
    read_division_node_cache,
)
from biohub.division_v2 import (  # noqa: E402
    DEFAULT_V2_PROPOSAL_RULE,
    V2_MODEL_NAMES,
    MovieFoldInput,
    V2PairModel,
    V2PairModelConfig,
    V2ProposalRule,
    assign_movie_folds,
    event_balanced_listwise_loss,
    generate_division_pair_proposals_v2,
    generate_targeted_division_pair_neighborhoods,
    heldout_control_threshold,
    require_fit_only_training_source,
)
from biohub.io import discover_split, read_geff_graph  # noqa: E402
from biohub.link import link_graph_flow, prune_to_tracks  # noqa: E402
from biohub.metric import TrackingGraph, match_nodes  # noqa: E402

FIT_RESULT_SCHEMA_VERSION = 1
DEFAULT_CONFIG = ROOT / "configs" / "phase11_v2_division.json"
DEFAULT_MANIFEST = ROOT / "data" / "working" / "phase11_division_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "working" / "phase11_division_manifest_summary.json"
DEFAULT_CACHE_DIR = ROOT / "data" / "working" / "phase11_division_cache"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw"
DEFAULT_FIT_DATASET = ROOT / "data" / "working" / "phase11_division_dataset" / "fit_complete"
DEFAULT_BASE_CHECKPOINT = (
    ROOT / "data" / "working" / "appearance_checkpoints" / "cellness_cnn_phase10_full.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "working" / "phase11_v2_atlasforge" / "fit"
SOURCE_PATHS = (
    SRC / "biohub" / "appearance.py",
    SRC / "biohub" / "constants.py",
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
GROUP_HASH_SCHEMA = "biohub-phase11-v2-fit-groups-v1"
CHECKPOINT_SCHEMA_VERSION = 1
GROUP_DIVISION = "division"
GROUP_CONTROL = "matched_nondivision_control"


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{field} has wrong fields; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def _require_finite(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return float(value)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Read and fail-closed validate the separately versioned v2 contract."""
    config = read_canonical_json(path)
    _require_exact_fields(
        config,
        {
            "activation_keyword",
            "calibration",
            "cross_validation",
            "development_audit",
            "evaluation_unlocked",
            "experiment_name",
            "model_ladder",
            "optimization",
            "promotion_guards",
            "proposal_rule",
            "ranking",
            "source_cohort",
            "training_data",
            "v1_reference",
            "version",
        },
        field="v2 experiment config",
    )
    if config["activation_keyword"] != "ATLASFORGE" or config["version"] != 1:
        raise ValueError("v2 config has an unsupported activation or version")
    if config["evaluation_unlocked"] is not False:
        raise ValueError("evaluation_unlocked must remain false")
    if config["experiment_name"] != "phase11_v2_atlasforge":
        raise ValueError("v2 config has an unsupported experiment name")
    if config["model_ladder"] != list(V2_MODEL_NAMES):
        raise ValueError("v2 config must use the exact fixed model ladder")
    if _require_mapping(config["source_cohort"], field="source_cohort") != {
        "embryo": "6bba",
        "role": "fit",
    }:
        raise ValueError("v2 source cohort must be 6bba fit only")
    if _require_mapping(config["training_data"], field="training_data") != {
        "base_dataset": "data/working/phase11_division_dataset/fit_complete",
        "base_strata": [
            "positive",
            "same_mother_negative",
            "matched_nondivision_control",
        ],
        "v2_expansion": "all_bounded_pairs_for_newly_reachable_fit_events",
    }:
        raise ValueError("v2 training data contract is unsupported")

    proposal = _require_mapping(config["proposal_rule"], field="proposal_rule")
    _require_exact_fields(
        proposal,
        {
            "anchor_candidate_cap",
            "base_candidate_cap",
            "candidate_order",
            "candidate_source",
            "max_distance_um",
            "max_pairs_per_mother",
            "mother_source",
            "outer_pair_rule",
            "scan_candidate_cap",
        },
        field="proposal_rule",
    )
    rule = V2ProposalRule(
        max_distance_um=proposal["max_distance_um"],
        base_candidate_cap=proposal["base_candidate_cap"],
        anchor_candidate_cap=proposal["anchor_candidate_cap"],
        scan_candidate_cap=proposal["scan_candidate_cap"],
        max_pairs_per_mother=proposal["max_pairs_per_mother"],
    )
    if rule != DEFAULT_V2_PROPOSAL_RULE:
        raise ValueError("v2 config proposal rule differs from the implemented v2 rule")
    if proposal["candidate_order"] != ["distance_um:ascending", "node_id:ascending"]:
        raise ValueError("v2 candidate order is not canonical")
    if proposal["candidate_source"] != "raw_scored":
        raise ValueError("v2 candidate source must remain raw_scored")
    if proposal["mother_source"] != "cnn_filtered_preprune":
        raise ValueError("v2 mother source must remain cnn_filtered_preprune")
    if proposal["outer_pair_rule"] != "anchor_to_outer_only":
        raise ValueError("v2 outer pair rule must remain anchor_to_outer_only")

    cross_validation = _require_mapping(
        config["cross_validation"],
        field="cross_validation",
    )
    if cross_validation != {
        "fold_count": 5,
        "group": "movie",
        "salt": "biohub-phase11-v2-movie-fold-v1",
    }:
        raise ValueError("v2 cross-validation contract is not the exact movie-grouped rule")

    calibration = _require_mapping(config["calibration"], field="calibration")
    _require_exact_fields(
        calibration,
        {
            "acceptance_rule",
            "control_mothers_per_movie",
            "loss",
            "threshold_rule",
        },
        field="calibration",
    )
    _require_int(
        calibration["control_mothers_per_movie"],
        field="control_mothers_per_movie",
        minimum=1,
    )
    if calibration["loss"] != "balanced_binary_cross_entropy":
        raise ValueError("v2 calibration loss is unsupported")
    if calibration["threshold_rule"] != "next_float_above_max_heldout_matched_control":
        raise ValueError("v2 calibration threshold rule is unsupported")
    if calibration["acceptance_rule"] != "all_fold_gates_clear_fold_threshold":
        raise ValueError("v2 calibration acceptance rule is unsupported")

    optimization = _require_mapping(config["optimization"], field="optimization")
    _require_exact_fields(
        optimization,
        {"gate_epochs", "learning_rate", "ranker_epochs", "seed", "weight_decay"},
        field="optimization",
    )
    for name in ("gate_epochs", "ranker_epochs", "seed"):
        _require_int(optimization[name], field=name, minimum=1)
    _require_finite(optimization["learning_rate"], field="learning_rate", minimum=1e-12)
    _require_finite(optimization["weight_decay"], field="weight_decay")

    guards = _require_mapping(config["promotion_guards"], field="promotion_guards")
    _require_exact_fields(
        guards,
        {
            "max_control_proposal_ratio_vs_v1",
            "max_pairs_per_mother",
            "max_total_proposal_ratio_vs_v1",
            "minimum_fit_reachable_events",
            "preserve_all_v1_reachable_fit_events",
            "require_zero_heldout_selected_control_accepts",
        },
        field="promotion_guards",
    )
    if guards["max_pairs_per_mother"] != rule.max_pairs_per_mother:
        raise ValueError("v2 proposal and promotion pair bounds disagree")
    if guards["preserve_all_v1_reachable_fit_events"] is not True:
        raise ValueError("v2 must preserve every v1-reachable fit event")
    if guards["require_zero_heldout_selected_control_accepts"] is not True:
        raise ValueError("v2 must reject every selected heldout control")

    development = _require_mapping(config["development_audit"], field="development_audit")
    if development != {
        "can_freeze_policy": False,
        "mode": "single_post_selection_diagnostic",
        "uses_evaluation": False,
    }:
        raise ValueError("v2 development audit boundary is not sealed")
    reference = _require_mapping(config["v1_reference"], field="v1_reference")
    if reference != {
        "development_truth_rank_atlas_sha256": (
            "6d177858354dc4f94d88571d5ec320819fe74883cf1b60355a755f33bc0f5d0a"
        ),
        "fit_reachable_events": 76,
        "max_candidates_per_mother": 12,
        "max_distance_um": 16.0,
    }:
        raise ValueError("v2 config has an unsupported v1 reference")
    ranking = _require_mapping(config["ranking"], field="ranking")
    if ranking["loss"] != "event_balanced_listwise_softmax":
        raise ValueError("v2 ranking loss is unsupported")
    return config


@dataclass(frozen=True, slots=True)
class FitGroup:
    """All bounded pair proposals for one matched fit mother."""

    dataset: str
    mother_id: int
    kind: str
    candidate_ids: np.ndarray
    raw_context: np.ndarray
    image_features: np.ndarray
    labels: np.ndarray
    v1_reachable: bool

    def __post_init__(self) -> None:
        require_fit_only_training_source(self.dataset, source_role="fit")
        if self.kind not in {GROUP_DIVISION, GROUP_CONTROL}:
            raise ValueError("fit group has an unsupported kind")
        if isinstance(self.mother_id, bool) or not isinstance(self.mother_id, int):
            raise ValueError("fit group mother identifier must be an integer")
        pairs = np.asarray(self.candidate_ids, dtype=np.int64)
        context = np.asarray(self.raw_context, dtype=np.float32)
        image = np.asarray(self.image_features, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.uint8)
        count = len(pairs)
        if pairs.shape != (count, 2) or count < 1 or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("fit group candidate identifiers are invalid")
        if len({tuple(pair) for pair in pairs.tolist()}) != count:
            raise ValueError("fit group contains duplicate candidate pairs")
        if context.shape != (count, len(CONTEXT_FEATURE_NAMES)) or not np.all(
            np.isfinite(context)
        ):
            raise ValueError("fit group context features are invalid")
        if image.ndim != 2 or image.shape[0] != count or not np.all(np.isfinite(image)):
            raise ValueError("fit group image features are invalid")
        if labels.shape != (count,) or not np.all(np.isin(labels, (0, 1))):
            raise ValueError("fit group labels are invalid")
        positive_count = int(labels.sum())
        if self.kind == GROUP_DIVISION and positive_count != 1:
            raise ValueError("division fit groups require exactly one truth pair")
        if self.kind == GROUP_CONTROL and positive_count != 0:
            raise ValueError("control fit groups cannot contain a truth pair")
        if not isinstance(self.v1_reachable, bool):
            raise ValueError("v1_reachable must be boolean")
        if self.kind == GROUP_CONTROL and self.v1_reachable:
            raise ValueError("control groups cannot be v1 reachable")
        pairs = np.ascontiguousarray(pairs)
        context = np.ascontiguousarray(context)
        image = np.ascontiguousarray(image)
        labels = np.ascontiguousarray(labels)
        for array in (pairs, context, image, labels):
            array.setflags(write=False)
        object.__setattr__(self, "candidate_ids", pairs)
        object.__setattr__(self, "raw_context", context)
        object.__setattr__(self, "image_features", image)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True, slots=True)
class MovieBuildResult:
    groups: tuple[FitGroup, ...]
    cache_record: dict[str, Any]
    event_records: tuple[dict[str, Any], ...]
    v1_proposals: int
    v2_proposals: int
    max_candidates_in_radius: int
    max_pairs_per_mother: int


@dataclass(frozen=True, slots=True)
class ProposalVolumeAudit:
    candidate_count_by_mother: dict[int, int]
    v1_proposals: int
    v2_proposals: int
    max_candidates_in_radius: int
    max_pairs_per_mother: int


def proposal_volume_audit(
    linked: TrackingGraph,
    raw: TrackingGraph,
) -> ProposalVolumeAudit:
    """Count all v1 and v2 pairs by timepoint without materializing proposals."""
    scale = np.asarray(
        [VOXEL_SCALE_UM[axis] for axis in ("z", "y", "x")],
        dtype=np.float64,
    )
    linked_ids = np.asarray(linked.node_ids, dtype=np.int64)
    linked_times = np.asarray(linked.t, dtype=np.int64)
    linked_positions = np.column_stack((linked.z, linked.y, linked.x)) * scale
    raw_times = np.asarray(raw.t, dtype=np.int64)
    raw_positions = np.column_stack((raw.z, raw.y, raw.x)) * scale
    if len(np.unique(linked_ids)) != len(linked_ids):
        raise ValueError("proposal volume audit received duplicate linked node identifiers")

    count_by_mother: dict[int, int] = {int(node_id): 0 for node_id in linked_ids}
    total_v1 = 0
    total_v2 = 0
    maximum_candidates = 0
    maximum_pairs = 0
    for mother_time in sorted(set(int(value) for value in linked_times)):
        mother_rows = np.flatnonzero(linked_times == mother_time)
        daughter_rows = np.flatnonzero(raw_times == mother_time + 1)
        if len(mother_rows) == 0 or len(daughter_rows) == 0:
            continue
        distances = cKDTree(linked_positions[mother_rows]).sparse_distance_matrix(
            cKDTree(raw_positions[daughter_rows]),
            DEFAULT_V2_PROPOSAL_RULE.max_distance_um,
            output_type="coo_matrix",
        )
        positive = distances.data > 0.0
        row_indices = distances.row[positive]
        values = distances.data[positive]
        counts_v2 = np.bincount(row_indices, minlength=len(mother_rows)).astype(
            np.int64,
            copy=False,
        )
        counts_v1 = np.bincount(
            row_indices[values <= 16.0],
            minlength=len(mother_rows),
        ).astype(np.int64, copy=False)
        for local_row, mother_row in enumerate(mother_rows):
            count_by_mother[int(linked_ids[mother_row])] = int(counts_v2[local_row])
        base_v1 = np.minimum(counts_v1, 12)
        pair_counts_v1 = base_v1 * (base_v1 - 1) // 2
        base_v2 = np.minimum(counts_v2, 12)
        scanned_v2 = np.minimum(counts_v2, 24)
        anchors_v2 = np.minimum(base_v2, 6)
        pair_counts_v2 = (
            base_v2 * (base_v2 - 1) // 2
            + anchors_v2 * np.maximum(scanned_v2 - 12, 0)
        )
        total_v1 += int(pair_counts_v1.sum())
        total_v2 += int(pair_counts_v2.sum())
        maximum_candidates = max(maximum_candidates, int(counts_v2.max(initial=0)))
        maximum_pairs = max(maximum_pairs, int(pair_counts_v2.max(initial=0)))
    return ProposalVolumeAudit(
        candidate_count_by_mother=count_by_mother,
        v1_proposals=total_v1,
        v2_proposals=total_v2,
        max_candidates_in_radius=maximum_candidates,
        max_pairs_per_mother=maximum_pairs,
    )


def _invert_unique(matches: Mapping[int, int]) -> dict[int, int]:
    inverted: dict[int, int] = {}
    for predicted, truth in matches.items():
        if truth in inverted:
            raise ValueError("node matching assigned two predictions to one truth node")
        inverted[int(truth)] = int(predicted)
    return inverted


def _proposal_inputs(
    proposals,
    *,
    raw: TrackingGraph,
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
    pruned: TrackingGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_row = {int(node_id): index for index, node_id in enumerate(raw.node_ids)}
    score_by_id = {
        int(node_id): float(score)
        for node_id, score in zip(raw.node_ids, cellness, strict=True)
    }
    pruned_ids = {int(node_id) for node_id in pruned.node_ids}
    successors: dict[int, set[int]] = defaultdict(set)
    predecessors: dict[int, set[int]] = defaultdict(set)
    for source, target in _oriented_edges(pruned):
        successors[source].add(target)
        predecessors[target].add(source)
    candidates = tuple(
        _candidate_sample(
            proposal,
            score_by_id=score_by_id,
            pruned_ids=pruned_ids,
            pruned_successors=successors,
            pruned_predecessors=predecessors,
        )
        for proposal in proposals
    )
    model_inputs = tuple(
        _proposal_model_inputs(
            candidate,
            node_row=node_row,
            cellness=cellness,
            embeddings=embeddings,
            intensity_summaries=intensity_summaries,
        )
        for candidate in candidates
    )
    raw_context = np.stack([values[0] for values in model_inputs]).astype(
        np.float32,
        copy=False,
    )
    image_features = np.stack([values[1] for values in model_inputs]).astype(
        np.float32,
        copy=False,
    )
    candidate_ids = np.asarray(
        [
            (proposal.candidate_1.node_id, proposal.candidate_2.node_id)
            for proposal in proposals
        ],
        dtype=np.int64,
    )
    return candidate_ids, raw_context, image_features


def _build_group(
    dataset: str,
    *,
    mother_id: int,
    kind: str,
    proposals,
    true_pair: tuple[int, int] | None,
    v1_reachable: bool,
    raw: TrackingGraph,
    cellness: np.ndarray,
    embeddings: np.ndarray,
    intensity_summaries: np.ndarray,
    pruned: TrackingGraph,
) -> FitGroup:
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
        canonical = tuple(sorted(true_pair))
        matches = np.flatnonzero(
            (candidate_ids[:, 0] == canonical[0]) & (candidate_ids[:, 1] == canonical[1])
        )
        if len(matches) != 1:
            raise ValueError("reachable division group does not contain exactly one truth pair")
        labels[int(matches[0])] = 1
    return FitGroup(
        dataset=dataset,
        mother_id=mother_id,
        kind=kind,
        candidate_ids=candidate_ids,
        raw_context=context,
        image_features=image,
        labels=labels,
        v1_reachable=v1_reachable,
    )


def groups_from_v1_dataset(loaded) -> tuple[FitGroup, ...]:
    """Reuse the provenance-checked v1 sampled groups without reopening labels."""
    division_records: dict[tuple[str, int], list[Any]] = defaultdict(list)
    control_records: list[Any] = []
    for record in loaded.records:
        if record.sampling_stratum in {POSITIVE, SAME_MOTHER_NEGATIVE}:
            division_records[(record.dataset, record.mother_id)].append(record)
        elif record.sampling_stratum == MATCHED_NONDIVISION_CONTROL:
            control_records.append(record)
    groups: list[FitGroup] = []
    for (dataset, mother_id), records in sorted(division_records.items()):
        ordered = sorted(
            records,
            key=lambda record: (
                -record.label,
                record.candidate_1_id,
                record.candidate_2_id,
            ),
        )
        groups.append(
            FitGroup(
                dataset=dataset,
                mother_id=mother_id,
                kind=GROUP_DIVISION,
                candidate_ids=np.asarray(
                    [(record.candidate_1_id, record.candidate_2_id) for record in ordered],
                    dtype=np.int64,
                ),
                raw_context=np.asarray(
                    [record.raw_context for record in ordered],
                    dtype=np.float32,
                ),
                image_features=np.asarray(
                    [record.image_features for record in ordered],
                    dtype=np.float32,
                ),
                labels=np.asarray([record.label for record in ordered], dtype=np.uint8),
                v1_reachable=True,
            )
        )
    for record in sorted(
        control_records,
        key=lambda value: (value.dataset, value.mother_id),
    ):
        groups.append(
            FitGroup(
                dataset=record.dataset,
                mother_id=record.mother_id,
                kind=GROUP_CONTROL,
                candidate_ids=np.asarray(
                    [(record.candidate_1_id, record.candidate_2_id)],
                    dtype=np.int64,
                ),
                raw_context=np.asarray([record.raw_context], dtype=np.float32),
                image_features=np.asarray([record.image_features], dtype=np.float32),
                labels=np.zeros(1, dtype=np.uint8),
                v1_reachable=False,
            )
        )
    groups.sort(key=lambda value: (value.dataset, value.kind, value.mother_id))
    if len(division_records) != 76:
        raise ValueError("v1 fit dataset no longer contains exactly 76 truth groups")
    if len(control_records) != 2816:
        raise ValueError("v1 fit dataset no longer contains exactly 2816 matched controls")
    keys = [(group.dataset, group.kind, group.mother_id) for group in groups]
    if len(set(keys)) != len(keys):
        raise ValueError("v1 fit dataset produced duplicate v2 group keys")
    return tuple(groups)


def build_fit_movie(
    row,
    *,
    sample,
    cache_dir: Path,
    cache_source_sha256: str,
    contract_sha256: Mapping[str, str],
    upstream_manifest_sha256: str,
) -> MovieBuildResult:
    """Audit one fit movie and materialize only its newly reachable v2 groups."""
    require_fit_only_training_source(row.dataset, source_role=str(row.role))
    cache_path = cache_dir / f"{row.dataset}.npz"
    cache_file_before = file_sha256(cache_path)
    raw, cellness, embeddings, intensity, metadata = read_division_node_cache(
        cache_path,
        expected_dataset=row.dataset,
        expected_role="fit",
        expected_cache_source_sha256=cache_source_sha256,
        expected_contract_sha256=contract_sha256,
        expected_upstream_manifest_sha256=upstream_manifest_sha256,
    )
    if file_sha256(cache_path) != cache_file_before:
        raise ValueError(f"division cache changed while reading {row.dataset}")
    if sample.labels is None:
        raise FileNotFoundError(f"fit labels are missing for {row.dataset}")
    truth = TrackingGraph.from_geff(read_geff_graph(sample.labels))
    truth_hash = logical_graph_sha256(truth)
    if truth_hash != row.truth_graph_sha256:
        raise ValueError(f"fit truth graph changed for {row.dataset}")
    filtered = filter_candidates(
        raw,
        cellness,
        min_score=FROZEN_CELLNESS_THRESHOLD,
    )
    volume = proposal_volume_audit(filtered, raw)
    mother_matches = match_nodes(filtered, truth)
    candidate_matches = match_nodes(raw, truth)
    truth_to_mother = _invert_unique(mother_matches)
    truth_to_candidate = _invert_unique(candidate_matches)
    divisions = truth_divisions(truth)
    target_mothers = {
        *(
            predicted
            for truth_mother in divisions
            if (predicted := truth_to_mother.get(truth_mother)) is not None
        ),
    }
    neighborhoods = generate_targeted_division_pair_neighborhoods(
        filtered,
        raw,
        mother_node_ids=target_mothers,
        max_distance_um=DEFAULT_V2_PROPOSAL_RULE.max_distance_um,
    )
    neighborhood_by_mother = {item.mother_id: item for item in neighborhoods}
    if len(neighborhood_by_mother) != len(neighborhoods):
        raise RuntimeError("v2 neighborhood generation produced duplicate mothers")

    expansion_specs: list[tuple[int, tuple[int, int], bool]] = []
    events: list[dict[str, Any]] = []
    for truth_mother, truth_daughters in sorted(divisions.items()):
        predicted_mother = truth_to_mother.get(truth_mother)
        predicted_daughters = tuple(truth_to_candidate.get(value) for value in truth_daughters)
        neighborhood = neighborhood_by_mother.get(predicted_mother)
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
        if neighborhood is not None and all(
            daughter is not None for daughter in predicted_daughters
        ):
            proposals = generate_division_pair_proposals_v2([neighborhood])
            pair = tuple(sorted(int(value) for value in predicted_daughters))
            if any(
                (proposal.candidate_1.node_id, proposal.candidate_2.node_id) == pair
                for proposal in proposals
            ):
                v2_reachable = True
                if not v1_reachable:
                    expansion_specs.append((int(predicted_mother), pair, v1_reachable))
        if v1_reachable and not v2_reachable:
            raise RuntimeError("v2 proposal rule regressed a v1-reachable fit event")
        events.append(
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
    groups: list[FitGroup] = []
    if expansion_specs:
        linked = link_graph_flow(filtered)
        pruned = prune_to_tracks(linked, min_track_length=6)
        actual_neighborhoods = generate_targeted_division_pair_neighborhoods(
            linked,
            raw,
            mother_node_ids={value[0] for value in expansion_specs},
            pruned_node_ids={int(node_id) for node_id in pruned.node_ids},
            max_distance_um=DEFAULT_V2_PROPOSAL_RULE.max_distance_um,
        )
        actual_by_mother = {item.mother_id: item for item in actual_neighborhoods}
        for mother_id, pair, v1_reachable in expansion_specs:
            neighborhood = actual_by_mother.get(mother_id)
            if neighborhood is None:
                raise RuntimeError("new v2 fit event lost its linked neighborhood")
            proposals = generate_division_pair_proposals_v2([neighborhood])
            groups.append(
                _build_group(
                    row.dataset,
                    mother_id=mother_id,
                    kind=GROUP_DIVISION,
                    proposals=proposals,
                    true_pair=pair,
                    v1_reachable=v1_reachable,
                    raw=raw,
                    cellness=cellness,
                    embeddings=embeddings,
                    intensity_summaries=intensity,
                    pruned=pruned,
                )
            )
    groups.sort(key=lambda value: (value.kind, value.mother_id))
    return MovieBuildResult(
        groups=tuple(groups),
        cache_record={
            "content_sha256": metadata["content_sha256"],
            "dataset": row.dataset,
            "file_sha256": cache_file_before,
            "truth_graph_sha256": truth_hash,
        },
        event_records=tuple(events),
        v1_proposals=volume.v1_proposals,
        v2_proposals=volume.v2_proposals,
        max_candidates_in_radius=volume.max_candidates_in_radius,
        max_pairs_per_mother=volume.max_pairs_per_mother,
    )


def fit_groups_sha256(groups: Sequence[FitGroup]) -> str:
    """Hash logical fit groups independently of local paths or array containers."""
    digest = hashlib.sha256(GROUP_HASH_SCHEMA.encode("ascii") + b"\0")
    for group in sorted(groups, key=lambda item: (item.dataset, item.kind, item.mother_id)):
        digest.update(group.dataset.encode("ascii") + b"\0")
        digest.update(group.kind.encode("ascii") + b"\0")
        digest.update(str(group.mother_id).encode("ascii") + b"\0")
        digest.update(b"1\0" if group.v1_reachable else b"0\0")
        for name, array in (
            ("candidate_ids", group.candidate_ids),
            ("raw_context", group.raw_context),
            ("image_features", group.image_features),
            ("labels", group.labels),
        ):
            values = np.ascontiguousarray(array)
            digest.update(name.encode("ascii") + b"\0")
            digest.update(values.dtype.str.encode("ascii") + b"\0")
            digest.update(",".join(str(value) for value in values.shape).encode("ascii") + b"\0")
            digest.update(values.tobytes(order="C") + b"\0")
    return digest.hexdigest()


def _fold_feature_schema(
    groups: Sequence[FitGroup],
    *,
    fold_by_dataset: Mapping[str, int],
    heldout_fold: int,
) -> FeatureSchema:
    training_context = np.concatenate(
        [
            group.raw_context
            for group in groups
            if fold_by_dataset[group.dataset] != heldout_fold
        ],
        axis=0,
    )
    return FeatureSchema.fit(
        CONTEXT_FEATURE_NAMES,
        training_context,
        kinds=CONTEXT_FEATURE_KINDS,
        source_role="fit",
    )


def _model_config(model_name: str, groups: Sequence[FitGroup]) -> V2PairModelConfig:
    if not groups:
        raise ValueError("v2 model configuration requires at least one group")
    return V2PairModelConfig(
        model_name=model_name,
        image_feature_dim=int(groups[0].image_features.shape[1]),
        context_feature_dim=len(CONTEXT_FEATURE_NAMES),
    )


def _seeded_model(
    config: V2PairModelConfig,
    *,
    seed: int,
    device: torch.device,
) -> V2PairModel:
    torch.manual_seed(seed)
    model = V2PairModel(config)
    return model.to(device)


def _train_ranker(
    groups: Sequence[FitGroup],
    *,
    schema: FeatureSchema,
    model_name: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[V2PairModel, float]:
    division_groups = [group for group in groups if group.kind == GROUP_DIVISION]
    if not division_groups:
        raise ValueError("ranker training requires division groups")
    image = np.concatenate([group.image_features for group in division_groups], axis=0)
    context = schema.standardize(
        np.concatenate([group.raw_context for group in division_groups], axis=0)
    )
    labels = np.concatenate([group.labels for group in division_groups]).astype(
        np.int64,
        copy=False,
    )
    group_ids = np.concatenate(
        [
            np.full(len(group.labels), index, dtype=np.int64)
            for index, group in enumerate(division_groups)
        ]
    )
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device)
    context_tensor = torch.tensor(context, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
    group_tensor = torch.tensor(group_ids, dtype=torch.int64, device=device)
    model = _seeded_model(
        _model_config(model_name, division_groups),
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    final_loss = math.inf
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor, context_tensor)
        loss = event_balanced_listwise_loss(logits, group_tensor, label_tensor)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("v2 ranker loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return model.eval(), final_loss


def _score_groups(
    model: V2PairModel,
    schema: FeatureSchema,
    groups: Sequence[FitGroup],
    *,
    device: torch.device,
    max_rows_per_batch: int = 65536,
) -> dict[tuple[str, str, int], np.ndarray]:
    results: dict[tuple[str, str, int], np.ndarray] = {}
    pending: list[FitGroup] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal pending, pending_rows
        if not pending:
            return
        image = np.concatenate([group.image_features for group in pending], axis=0)
        context = schema.standardize(
            np.concatenate([group.raw_context for group in pending], axis=0)
        )
        with torch.inference_mode():
            values = (
                model(
                    torch.tensor(image, dtype=torch.float32, device=device),
                    torch.tensor(context, dtype=torch.float32, device=device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        offset = 0
        for group in pending:
            count = len(group.labels)
            key = (group.dataset, group.kind, group.mother_id)
            results[key] = np.asarray(values[offset : offset + count], dtype=np.float64)
            offset += count
        if offset != len(values):
            raise RuntimeError("v2 group scoring split disagrees with the model output")
        pending = []
        pending_rows = 0

    for group in groups:
        if pending and pending_rows + len(group.labels) > max_rows_per_batch:
            flush()
        pending.append(group)
        pending_rows += len(group.labels)
    flush()
    return results


def _ordered_indices(group: FitGroup, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(group.labels),) or not np.all(np.isfinite(values)):
        raise ValueError("v2 group scores are invalid")
    return np.lexsort(
        (
            group.candidate_ids[:, 1],
            group.candidate_ids[:, 0],
            -values,
        )
    )


def _selected_rows(
    groups: Sequence[FitGroup],
    scores: Mapping[tuple[str, str, int], np.ndarray],
) -> dict[tuple[str, str, int], int]:
    return {
        (group.dataset, group.kind, group.mother_id): int(
            _ordered_indices(
                group,
                scores[(group.dataset, group.kind, group.mother_id)],
            )[0]
        )
        for group in groups
    }


def _train_gate(
    groups: Sequence[FitGroup],
    *,
    selected_rows: Mapping[tuple[str, str, int], int],
    schema: FeatureSchema,
    model_name: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[V2PairModel, float]:
    image_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    labels: list[float] = []
    for group in groups:
        key = (group.dataset, group.kind, group.mother_id)
        if group.kind == GROUP_DIVISION:
            row = int(np.flatnonzero(group.labels == 1)[0])
            label = 1.0
        else:
            row = selected_rows[key]
            label = 0.0
        image_rows.append(group.image_features[row])
        context_rows.append(group.raw_context[row])
        labels.append(label)
    label_array = np.asarray(labels, dtype=np.float32)
    positives = int(label_array.sum())
    negatives = len(label_array) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("gate training requires both division and control mothers")
    image_tensor = torch.tensor(
        np.stack(image_rows),
        dtype=torch.float32,
        device=device,
    )
    context_tensor = torch.tensor(
        schema.standardize(np.stack(context_rows)),
        dtype=torch.float32,
        device=device,
    )
    label_tensor = torch.tensor(label_array, dtype=torch.float32, device=device)
    model = _seeded_model(
        _model_config(model_name, groups),
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(negatives) / positives, device=device)
    )
    final_loss = math.inf
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor, context_tensor)
        loss = loss_function(logits, label_tensor)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("v2 gate loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return model.eval(), final_loss


def _score_selected_rows(
    model: V2PairModel,
    schema: FeatureSchema,
    groups: Sequence[FitGroup],
    selected_rows: Mapping[tuple[str, str, int], int],
    *,
    device: torch.device,
) -> dict[tuple[str, str, int], float]:
    keys = [(group.dataset, group.kind, group.mother_id) for group in groups]
    image = np.stack(
        [group.image_features[selected_rows[key]] for group, key in zip(groups, keys, strict=True)]
    )
    context = schema.standardize(
        np.stack(
            [group.raw_context[selected_rows[key]] for group, key in zip(groups, keys, strict=True)]
        )
    )
    with torch.inference_mode():
        logits = (
            model(
                torch.tensor(image, dtype=torch.float32, device=device),
                torch.tensor(context, dtype=torch.float32, device=device),
            )
            .detach()
            .cpu()
            .numpy()
        )
    return {key: float(value) for key, value in zip(keys, logits, strict=True)}


def _cpu_state_dict(model: V2PairModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _parameter_count(model: V2PairModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(slots=True)
class FoldModelArtifact:
    schema: FeatureSchema
    ranker_state: dict[str, torch.Tensor]
    ranker_loss: float
    ranker_parameter_count: int
    gate_state: dict[str, torch.Tensor]
    gate_loss: float
    gate_parameter_count: int
    gate_threshold: float


def run_cross_validation(
    groups: Sequence[FitGroup],
    *,
    fold_by_dataset: Mapping[str, int],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], tuple[str, str], dict[tuple[str, str, int], FoldModelArtifact]]:
    """Train and compare the exact 3 by 3 model ladder on fit folds only."""
    optimization = config["optimization"]
    fold_count = int(config["cross_validation"]["fold_count"])
    candidate_accumulators: dict[tuple[str, str], dict[str, Any]] = {
        (ranker, gate): {
            "folds": [],
            "all_reachable_oof_top1": 0,
            "all_reachable_oof_top3": 0,
            "shared_v1_oof_top1": 0,
            "shared_v1_oof_total": 0,
            "two_stage_oof_recoveries": 0,
            "heldout_selected_control_accepts": 0,
            "heldout_selected_control_total": 0,
            "reachable_oof_total": 0,
        }
        for ranker in V2_MODEL_NAMES
        for gate in V2_MODEL_NAMES
    }
    artifacts: dict[tuple[str, str, int], FoldModelArtifact] = {}

    for fold in range(fold_count):
        schema = _fold_feature_schema(
            groups,
            fold_by_dataset=fold_by_dataset,
            heldout_fold=fold,
        )
        training = [group for group in groups if fold_by_dataset[group.dataset] != fold]
        heldout = [group for group in groups if fold_by_dataset[group.dataset] == fold]
        heldout_divisions = [group for group in heldout if group.kind == GROUP_DIVISION]
        heldout_controls = [group for group in heldout if group.kind == GROUP_CONTROL]
        if not heldout_divisions or not heldout_controls:
            raise ValueError("each v2 fold requires heldout division and control groups")
        for ranker_index, ranker_name in enumerate(V2_MODEL_NAMES):
            ranker_seed = int(optimization["seed"]) + fold * 100 + ranker_index * 10
            ranker, ranker_loss = _train_ranker(
                training,
                schema=schema,
                model_name=ranker_name,
                epochs=int(optimization["ranker_epochs"]),
                learning_rate=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]),
                seed=ranker_seed,
                device=device,
            )
            rank_scores = _score_groups(ranker, schema, groups, device=device)
            selected = _selected_rows(groups, rank_scores)
            rank_by_group: dict[tuple[str, str, int], int] = {}
            for group in heldout_divisions:
                key = (group.dataset, group.kind, group.mother_id)
                order = _ordered_indices(group, rank_scores[key])
                truth_row = int(np.flatnonzero(group.labels == 1)[0])
                rank_by_group[key] = int(np.flatnonzero(order == truth_row)[0]) + 1

            for gate_index, gate_name in enumerate(V2_MODEL_NAMES):
                gate_seed = ranker_seed + gate_index + 1
                gate, gate_loss = _train_gate(
                    training,
                    selected_rows=selected,
                    schema=schema,
                    model_name=gate_name,
                    epochs=int(optimization["gate_epochs"]),
                    learning_rate=float(optimization["learning_rate"]),
                    weight_decay=float(optimization["weight_decay"]),
                    seed=gate_seed,
                    device=device,
                )
                gate_scores = _score_selected_rows(
                    gate,
                    schema,
                    heldout,
                    selected,
                    device=device,
                )
                control_scores = [
                    gate_scores[(group.dataset, group.kind, group.mother_id)]
                    for group in heldout_controls
                ]
                threshold = heldout_control_threshold(control_scores)
                control_accepts = sum(value >= threshold for value in control_scores)
                if control_accepts:
                    raise RuntimeError("v2 heldout control threshold accepted a control")
                top1 = sum(rank_value == 1 for rank_value in rank_by_group.values())
                top3 = sum(rank_value <= 3 for rank_value in rank_by_group.values())
                shared_groups = [group for group in heldout_divisions if group.v1_reachable]
                shared_top1 = sum(
                    rank_by_group[(group.dataset, group.kind, group.mother_id)] == 1
                    for group in shared_groups
                )
                two_stage = sum(
                    rank_by_group[(group.dataset, group.kind, group.mother_id)] == 1
                    and gate_scores[(group.dataset, group.kind, group.mother_id)] >= threshold
                    for group in heldout_divisions
                )
                accumulator = candidate_accumulators[(ranker_name, gate_name)]
                accumulator["all_reachable_oof_top1"] += top1
                accumulator["all_reachable_oof_top3"] += top3
                accumulator["shared_v1_oof_top1"] += shared_top1
                accumulator["shared_v1_oof_total"] += len(shared_groups)
                accumulator["two_stage_oof_recoveries"] += two_stage
                accumulator["heldout_selected_control_accepts"] += control_accepts
                accumulator["heldout_selected_control_total"] += len(heldout_controls)
                accumulator["reachable_oof_total"] += len(heldout_divisions)
                accumulator["folds"].append(
                    {
                        "fold": fold,
                        "gate_loss": gate_loss,
                        "gate_threshold": threshold,
                        "heldout_controls": len(heldout_controls),
                        "heldout_divisions": len(heldout_divisions),
                        "ranker_loss": ranker_loss,
                        "shared_v1_top1": shared_top1,
                        "shared_v1_total": len(shared_groups),
                        "top1": top1,
                        "top3": top3,
                        "two_stage_recoveries": two_stage,
                    }
                )
                artifacts[(ranker_name, gate_name, fold)] = FoldModelArtifact(
                    schema=schema,
                    ranker_state=_cpu_state_dict(ranker),
                    ranker_loss=ranker_loss,
                    ranker_parameter_count=_parameter_count(ranker),
                    gate_state=_cpu_state_dict(gate),
                    gate_loss=gate_loss,
                    gate_parameter_count=_parameter_count(gate),
                    gate_threshold=threshold,
                )
                print(
                    f"fold={fold} ranker={ranker_name} gate={gate_name} "
                    f"top1={top1}/{len(heldout_divisions)} two_stage={two_stage}"
                )
            del ranker
            if device.type == "mps":
                torch.mps.empty_cache()

    candidates: list[dict[str, Any]] = []
    for (ranker_name, gate_name), accumulator in candidate_accumulators.items():
        first_artifact = artifacts[(ranker_name, gate_name, 0)]
        candidates.append(
            {
                "gate_name": gate_name,
                "gate_parameter_count": first_artifact.gate_parameter_count,
                "ranker_name": ranker_name,
                "ranker_parameter_count": first_artifact.ranker_parameter_count,
                **accumulator,
            }
        )
    candidates.sort(key=lambda value: (value["ranker_name"], value["gate_name"]))
    selected_row = min(
        candidates,
        key=lambda value: (
            -value["two_stage_oof_recoveries"],
            -value["shared_v1_oof_top1"],
            -value["all_reachable_oof_top1"],
            -value["all_reachable_oof_top3"],
            value["ranker_parameter_count"],
            value["gate_parameter_count"],
            value["ranker_name"],
            value["gate_name"],
        ),
    )
    selected = (str(selected_row["ranker_name"]), str(selected_row["gate_name"]))
    return candidates, selected, artifacts


def _project_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("v2 artifacts must remain inside the project root") from exc


def _write_selected_checkpoints(
    staging_dir: Path,
    *,
    final_output_dir: Path,
    selected: tuple[str, str],
    artifacts: Mapping[tuple[str, str, int], FoldModelArtifact],
    fold_count: int,
    config_sha256: str,
    fit_groups_sha256_value: str,
    source_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    ranker_name, gate_name = selected
    records: list[dict[str, Any]] = []
    for fold in range(fold_count):
        artifact = artifacts[(ranker_name, gate_name, fold)]
        ranker_config = V2PairModelConfig(
            model_name=ranker_name,
            image_feature_dim=256,
            context_feature_dim=len(CONTEXT_FEATURE_NAMES),
        )
        gate_config = V2PairModelConfig(
            model_name=gate_name,
            image_feature_dim=256,
            context_feature_dim=len(CONTEXT_FEATURE_NAMES),
        )
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "evaluation_unlocked": False,
            "feature_schema": artifact.schema.to_dict(),
            "fit_groups_sha256": fit_groups_sha256_value,
            "fold": fold,
            "gate_config": asdict(gate_config),
            "gate_loss": artifact.gate_loss,
            "gate_state_dict": artifact.gate_state,
            "gate_state_sha256": canonical_model_state_sha256(artifact.gate_state),
            "gate_threshold": artifact.gate_threshold,
            "manifest_sha256": manifest_sha256,
            "ranker_config": asdict(ranker_config),
            "ranker_loss": artifact.ranker_loss,
            "ranker_state_dict": artifact.ranker_state,
            "ranker_state_sha256": canonical_model_state_sha256(artifact.ranker_state),
            "source_sha256": source_sha256,
        }
        filename = f"fold_{fold}_{ranker_name}_{gate_name}.pt"
        staging_path = staging_dir / filename
        torch.save(payload, staging_path)
        with staging_path.open("rb") as handle:
            os.fsync(handle.fileno())
        records.append(
            {
                "file_sha256": file_sha256(staging_path),
                "fold": fold,
                "gate_state_sha256": payload["gate_state_sha256"],
                "gate_threshold": artifact.gate_threshold,
                "path": _project_relative(final_output_dir / filename),
                "ranker_state_sha256": payload["ranker_state_sha256"],
            }
        )
    return records


def verify_existing_result(
    output_dir: str | Path,
    *,
    config_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify a completed fit result without opening any training or evaluation labels."""
    output = Path(output_dir)
    result_path = output / "fit_result.json"
    result = read_canonical_json(result_path)
    if result.get("fit_result_schema_version") != FIT_RESULT_SCHEMA_VERSION:
        raise ValueError("existing v2 fit result has an unsupported schema")
    if result.get("evaluation_unlocked") is not False:
        raise ValueError("existing v2 fit result did not keep evaluation sealed")
    if result.get("config_sha256") != file_sha256(config_path):
        raise ValueError("existing v2 fit result config hash changed")
    if result.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("existing v2 fit result manifest hash changed")
    if result.get("source_sha256") != source_bundle_sha256(SOURCE_PATHS):
        raise ValueError("existing v2 fit result source hash changed")
    checkpoints = result.get("selected_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise ValueError("existing v2 fit result has an invalid checkpoint set")
    for record in checkpoints:
        checkpoint = ROOT / str(record["path"])
        if file_sha256(checkpoint) != record["file_sha256"]:
            raise ValueError(f"existing v2 checkpoint hash changed: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["evaluation_unlocked"] is not False:
            raise ValueError("existing v2 checkpoint did not keep evaluation sealed")
        if canonical_model_state_sha256(payload["ranker_state_dict"]) != record[
            "ranker_state_sha256"
        ]:
            raise ValueError("existing v2 ranker state hash changed")
        if canonical_model_state_sha256(payload["gate_state_dict"]) != record[
            "gate_state_sha256"
        ]:
            raise ValueError("existing v2 gate state hash changed")
    return result


def run_fit_experiment(
    *,
    config_path: Path,
    manifest_path: Path,
    summary_path: Path,
    cache_dir: Path,
    data_root: Path,
    fit_dataset: Path,
    base_checkpoint: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Execute the complete fit-only ATLASFORGE experiment transaction."""
    config = load_experiment_config(config_path)
    config_hash = file_sha256(config_path)
    source_hash = source_bundle_sha256(SOURCE_PATHS)
    manifest_hash = file_sha256(manifest_path)
    summary_hash = file_sha256(summary_path)
    contracts = read_division_contracts(config_path.parent)
    contract_hashes = current_division_contract_hashes(config_path.parent)
    if contract_hashes != EXPECTED_DIVISION_CONTRACT_SHA256:
        raise ValueError("frozen v1 division contract hashes changed")
    observed_contract_hashes = {
        name: file_sha256(config_path.parent / f"phase11_division_{name}.json")
        for name in contracts
    }
    if observed_contract_hashes != contract_hashes:
        raise ValueError("frozen v1 division contract files changed during preflight")
    cache_source_hash = division_cache_source_sha256(ROOT)
    rows = read_division_manifest(manifest_path)
    fit_rows = tuple(row for row in rows if row.role == "fit")
    if len(fit_rows) != 88 or sum(row.division_count for row in fit_rows) != 89:
        raise ValueError("v2 fit experiment requires the exact frozen 88-movie, 89-event cohort")
    samples = {sample.dataset: sample for sample in discover_split(data_root, "train")}
    if any(row.dataset not in samples for row in fit_rows):
        raise FileNotFoundError("v2 fit experiment is missing one or more fit samples")
    if _project_relative(fit_dataset) != config["training_data"]["base_dataset"]:
        raise ValueError("v2 fit dataset path differs from its tracked training-data contract")
    loaded_v1 = read_division_dataset(
        fit_dataset,
        project_root=ROOT,
        data_root=data_root,
        config_dir=config_path.parent,
        upstream_manifest_path=manifest_path,
        summary_path=summary_path,
        cache_dir=cache_dir,
        checkpoint_path=base_checkpoint,
    )
    fold_inputs = tuple(
        MovieFoldInput(
            dataset=row.dataset,
            source_role="fit",
            truth_event_count=row.division_count,
        )
        for row in fit_rows
    )
    fold_by_dataset = assign_movie_folds(
        fold_inputs,
        fold_count=int(config["cross_validation"]["fold_count"]),
        salt=str(config["cross_validation"]["salt"]),
    )

    groups = list(groups_from_v1_dataset(loaded_v1))
    cache_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    total_v1 = 0
    total_v2 = 0
    control_v1 = 0
    control_v2 = 0
    maximum_candidates = 0
    maximum_pairs = 0
    upstream_manifest_hash = manifest_hash
    for index, row in enumerate(fit_rows, start=1):
        built = build_fit_movie(
            row,
            sample=samples[row.dataset],
            cache_dir=cache_dir,
            cache_source_sha256=cache_source_hash,
            contract_sha256=contract_hashes,
            upstream_manifest_sha256=upstream_manifest_hash,
        )
        groups.extend(built.groups)
        cache_records.append(built.cache_record)
        event_records.extend(built.event_records)
        total_v1 += built.v1_proposals
        total_v2 += built.v2_proposals
        if row.no_division:
            control_v1 += built.v1_proposals
            control_v2 += built.v2_proposals
        maximum_candidates = max(maximum_candidates, built.max_candidates_in_radius)
        maximum_pairs = max(maximum_pairs, built.max_pairs_per_mother)
        reachable = sum(event["v2_reachable"] for event in built.event_records)
        print(
            f"audited={index}/88 dataset={row.dataset} expansion_groups={len(built.groups)} "
            f"reachable={reachable}/{row.division_count}"
        )

    groups.sort(key=lambda value: (value.dataset, value.kind, value.mother_id))
    cache_records.sort(key=lambda value: value["dataset"])
    event_records.sort(key=lambda value: (value["dataset"], value["truth_mother_id"]))
    v1_reachable = sum(event["v1_reachable"] for event in event_records)
    v2_reachable = sum(event["v2_reachable"] for event in event_records)
    guards = config["promotion_guards"]
    if v1_reachable != int(config["v1_reference"]["fit_reachable_events"]):
        raise ValueError(
            f"fit v1 reachability changed: observed {v1_reachable}, expected 76"
        )
    if v2_reachable < int(guards["minimum_fit_reachable_events"]):
        raise ValueError("v2 did not meet its predeclared fit reachability guard")
    if any(event["v1_reachable"] and not event["v2_reachable"] for event in event_records):
        raise ValueError("v2 regressed at least one v1-reachable fit event")
    total_ratio = total_v2 / total_v1
    control_ratio = control_v2 / control_v1
    if total_ratio > float(guards["max_total_proposal_ratio_vs_v1"]):
        raise ValueError("v2 exceeded its total proposal-volume guard")
    if control_ratio > float(guards["max_control_proposal_ratio_vs_v1"]):
        raise ValueError("v2 exceeded its control proposal-volume guard")
    if maximum_pairs > int(guards["max_pairs_per_mother"]):
        raise ValueError("v2 exceeded its exact per-mother pair guard")
    division_groups = [group for group in groups if group.kind == GROUP_DIVISION]
    control_groups = [group for group in groups if group.kind == GROUP_CONTROL]
    if len(division_groups) != v2_reachable:
        raise RuntimeError("v2 reachable events and division groups disagree")
    if not control_groups:
        raise RuntimeError("v2 fit experiment produced no matched control groups")
    group_hash = fit_groups_sha256(groups)
    print(
        f"fit groups ready: divisions={len(division_groups)} controls={len(control_groups)} "
        f"rows={sum(len(group.labels) for group in groups)} sha256={group_hash}"
    )

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS device is unavailable")
        device = torch.device("mps")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cpu or mps")
    candidates, selected, artifacts = run_cross_validation(
        groups,
        fold_by_dataset=fold_by_dataset,
        config=config,
        device=device,
    )
    selected_row = next(
        row
        for row in candidates
        if (row["ranker_name"], row["gate_name"]) == selected
    )
    if selected_row["heldout_selected_control_accepts"] != 0:
        raise RuntimeError("selected v2 model accepted a heldout matched control")

    if output_dir.exists():
        raise FileExistsError(
            f"v2 fit output already exists: {output_dir}; use --verify-existing"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
    ) as temporary_name:
        staging = Path(temporary_name)
        checkpoints = _write_selected_checkpoints(
            staging,
            final_output_dir=output_dir,
            selected=selected,
            artifacts=artifacts,
            fold_count=int(config["cross_validation"]["fold_count"]),
            config_sha256=config_hash,
            fit_groups_sha256_value=group_hash,
            source_sha256=source_hash,
            manifest_sha256=manifest_hash,
        )
        fold_summary = {
            str(fold): {
                "division_events": sum(
                    group.kind == GROUP_DIVISION and fold_by_dataset[group.dataset] == fold
                    for group in groups
                ),
                "movies": sum(value == fold for value in fold_by_dataset.values()),
                "selected_controls": sum(
                    group.kind == GROUP_CONTROL and fold_by_dataset[group.dataset] == fold
                    for group in groups
                ),
            }
            for fold in range(int(config["cross_validation"]["fold_count"]))
        }
        result = {
            "cache_directory": _project_relative(cache_dir),
            "cache_records": cache_records,
            "cache_source_sha256": cache_source_hash,
            "candidate_results": candidates,
            "config_path": _project_relative(config_path),
            "config_sha256": config_hash,
            "contract_sha256": contract_hashes,
            "data_root": _project_relative(data_root),
            "device": str(device),
            "evaluation_unlocked": False,
            "event_records": event_records,
            "fit_groups": {
                "base_dataset_manifest_sha256": loaded_v1.dataset_manifest_sha256,
                "base_model_inputs_sha256": loaded_v1.model_inputs_sha256,
                "control_groups": len(control_groups),
                "division_groups": len(division_groups),
                "logical_sha256": group_hash,
                "proposal_rows": sum(len(group.labels) for group in groups),
            },
            "fit_result_schema_version": FIT_RESULT_SCHEMA_VERSION,
            "fold_assignment": fold_by_dataset,
            "fold_summary": fold_summary,
            "manifest_path": _project_relative(manifest_path),
            "manifest_sha256": manifest_hash,
            "proposal_audit": {
                "control_v1_proposals": control_v1,
                "control_v2_proposals": control_v2,
                "control_volume_ratio": control_ratio,
                "fit_truth_events": len(event_records),
                "max_candidates_within_18um": maximum_candidates,
                "max_pairs_per_mother": maximum_pairs,
                "total_v1_proposals": total_v1,
                "total_v2_proposals": total_v2,
                "total_volume_ratio": total_ratio,
                "v1_reachable_events": v1_reachable,
                "v2_reachable_events": v2_reachable,
            },
            "selected_checkpoints": checkpoints,
            "selected_gate": selected[1],
            "selected_ranker": selected[0],
            "selected_result": selected_row,
            "source_sha256": source_hash,
            "summary_path": _project_relative(summary_path),
            "summary_sha256": summary_hash,
        }
        result_hash = write_canonical_json(staging / "fit_result.json", result)
        os.replace(staging, output_dir)
    print(f"fit_result={output_dir / 'fit_result.json'} sha256={result_hash}")
    return verify_existing_result(
        output_dir,
        config_path=config_path,
        manifest_path=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fit-dataset", type=Path, default=DEFAULT_FIT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the existing result and checkpoints without rerunning fit labels",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.verify_existing:
        result = verify_existing_result(
            args.output_dir,
            config_path=args.config,
            manifest_path=args.manifest,
        )
        print(
            f"verified={args.output_dir / 'fit_result.json'} "
            f"ranker={result['selected_ranker']} gate={result['selected_gate']}"
        )
        return
    run_fit_experiment(
        config_path=args.config,
        manifest_path=args.manifest,
        summary_path=args.summary,
        cache_dir=args.cache_dir,
        data_root=args.data_root,
        fit_dataset=args.fit_dataset,
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
