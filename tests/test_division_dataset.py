from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import weakref
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biohub import division_dataset as division_dataset_module
from biohub.division_data import canonical_json_bytes, file_sha256, read_division_contracts
from biohub.division_dataset import (
    BUILD_MODE_FIT_COMPLETE,
    BUILD_MODE_SMOKE,
    DATASET_ARRAY_FIELDS,
    DATASET_FILES,
    DIAGNOSTIC_UNKNOWN,
    LABEL_UNKNOWN,
    MATCHED_NONDIVISION_CONTROL,
    POSITIVE,
    SAME_MOTHER_NEGATIVE,
    DatasetBuildIdentity,
    InferenceProposalBatch,
    ProposalRecord,
    balanced_epoch_indices,
    build_dataset_artifacts,
    canonical_model_inputs_sha256,
    dataset_lock_path,
    dataset_transaction_path,
    exclusive_dataset_builder_lock,
    fit_feature_schema,
    invariant_image_features,
    iter_inference_proposal_batches,
    prepare_inference_movie,
    proposal_hardness_key,
    read_dataset_artifacts,
    read_division_dataset,
    reconstruct_upstream,
    sample_movie_proposals,
    select_matched_nondivision_controls,
    select_same_mother_negatives,
    select_unknown_diagnostics,
    validate_complete_role_tuple,
    validate_node_cache_identity_cohort,
)
from biohub.division_model import (
    CONTEXT_FEATURE_KINDS,
    CONTEXT_FEATURE_NAMES,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    FIXED_DIVISION_MODEL_CONFIGS,
    DivisionModel,
    build_symmetric_context_features,
)
from biohub.metric import TrackingGraph
from scripts import build_division_dataset

ROOT = Path(__file__).resolve().parents[1]


def _context(*, mother_cellness: float = 0.8, intensity_offset: float = 0.0) -> tuple[float, ...]:
    values = build_symmetric_context_features(
        mother_cellness=np.array([mother_cellness], dtype=np.float32),
        daughter_cellness=np.array([[0.7, 0.6]], dtype=np.float32),
        daughter_distance_um=np.array([[2.0, 3.0]], dtype=np.float32),
        daughter_angle_deg=np.array([120.0], dtype=np.float32),
        candidate_flags={
            "present_in_linked_graph": np.array([[1, 1]], dtype=np.uint8),
            "is_current_child": np.array([[1, 0]], dtype=np.uint8),
            "is_parentless": np.array([[0, 1]], dtype=np.uint8),
            "has_wrong_parent": np.array([[0, 0]], dtype=np.uint8),
            "has_future": np.array([[1, 1]], dtype=np.uint8),
            "survives_pruning": np.array([[1, 0]], dtype=np.uint8),
        },
        mother_flags={
            "mother_has_history": np.array([1], dtype=np.uint8),
            "mother_has_current_child": np.array([1], dtype=np.uint8),
            "mother_survives_pruning": np.array([1], dtype=np.uint8),
        },
        restored_node_count=np.array([1], dtype=np.uint8),
        edge_removal_count=np.array([0], dtype=np.uint8),
        mother_intensity_summaries=np.array(
            [[5.0, 1.0, 0.5, 1.0, 9.0]], dtype=np.float32
        )
        + intensity_offset,
        daughter_intensity_summaries=np.array(
            [[[4.0, 1.0, 0.4, 1.0, 8.0], [6.0, 2.0, 0.6, 2.0, 10.0]]],
            dtype=np.float32,
        )
        + intensity_offset,
    )
    return tuple(float(value) for value in values[0])


def _record(
    index: int,
    stratum: str,
    *,
    mother_id: int | None = None,
    minimum_cellness: float = 0.5,
    restored_node_count: int = 0,
    edge_removal_count: int = 0,
    context: tuple[float, ...] | None = None,
) -> ProposalRecord:
    labels = {
        POSITIVE: (1, 1.0),
        SAME_MOTHER_NEGATIVE: (0, 1.0),
        MATCHED_NONDIVISION_CONTROL: (0, 1.0),
        DIAGNOSTIC_UNKNOWN: (LABEL_UNKNOWN, 0.0),
    }
    label, weight = labels[stratum]
    mother = 1000 + index if mother_id is None else mother_id
    first = 10_000 + 2 * index
    second = first + 1
    return ProposalRecord(
        dataset="6bba_synthetic",
        mother_id=mother,
        candidate_1_id=first,
        candidate_2_id=second,
        mother_node_index=index * 3,
        candidate_1_node_index=index * 3 + 1,
        candidate_2_node_index=index * 3 + 2,
        sampling_stratum=stratum,
        source_role="fit",
        label=label,
        loss_weight=weight,
        minimum_cellness=minimum_cellness,
        restored_node_count=restored_node_count,
        edge_removal_count=edge_removal_count,
        deterministic_rank=hashlib.sha256(f"rank:{index}".encode()).hexdigest(),
        raw_context=_context() if context is None else context,
        image_features=tuple(float(value) for value in np.arange(256, dtype=np.float32)),
    )


def test_proposal_record_is_immutable_strict_and_identifier_free_in_features() -> None:
    record = _record(0, POSITIVE)

    assert len(record.raw_context) == len(CONTEXT_FEATURE_NAMES)
    assert len(record.image_features) == 256
    assert not any("id" in name.lower() for name in CONTEXT_FEATURE_NAMES)
    with pytest.raises(FrozenInstanceError):
        record.label = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.raw_context[0] = 10.0  # type: ignore[index]

    invalid = [
        {"dataset": "other_synthetic"},
        {"source_role": "development"},
        {"candidate_2_id": record.candidate_1_id},
        {"candidate_1_node_index": record.mother_node_index},
        {"sampling_stratum": POSITIVE, "label": 0},
        {"sampling_stratum": DIAGNOSTIC_UNKNOWN, "label": LABEL_UNKNOWN, "loss_weight": 1.0},
        {"deterministic_rank": "not-a-hash"},
        {"raw_context": record.raw_context[:-1]},
        {"image_features": record.image_features[:-1]},
    ]
    for changes in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(record, **changes)


def test_precomputed_image_features_are_exactly_256d_and_daughter_symmetric() -> None:
    mother = np.arange(64, dtype=np.float32)
    first = np.linspace(-2.0, 3.0, 64, dtype=np.float32)
    second = np.linspace(4.0, -1.0, 64, dtype=np.float32)

    observed = invariant_image_features(mother, first, second)
    swapped = invariant_image_features(mother, second, first)
    expected = np.concatenate(
        [mother, (first + second) / 2.0, np.abs(first - second), first * second]
    ).astype(np.float32)

    assert observed.shape == (256,)
    assert observed.dtype == np.float32
    assert np.array_equal(observed, expected)
    assert np.array_equal(observed, swapped)


def test_hardness_ties_and_exact_same_mother_cap_are_deterministic() -> None:
    records = [
        _record(
            index,
            SAME_MOTHER_NEGATIVE,
            mother_id=77,
            minimum_cellness=0.8 if index < 14 else 0.7,
            restored_node_count=index % 2,
            edge_removal_count=(index // 2) % 2,
        )
        for index in reversed(range(20))
    ]

    selected = select_same_mother_negatives(records, max_pairs_per_mother=12)
    expected = sorted(records, key=proposal_hardness_key)[:12]

    assert selected == expected
    assert len(selected) == 12
    assert [proposal_hardness_key(row) for row in selected] == sorted(
        proposal_hardness_key(row) for row in selected
    )


def test_control_and_unknown_sampling_apply_exact_32_caps() -> None:
    control_candidates: list[ProposalRecord] = []
    for mother in range(40):
        control_candidates.extend(
            [
                _record(
                    mother * 2,
                    MATCHED_NONDIVISION_CONTROL,
                    mother_id=mother,
                    minimum_cellness=0.4,
                ),
                _record(
                    mother * 2 + 1,
                    MATCHED_NONDIVISION_CONTROL,
                    mother_id=mother,
                    minimum_cellness=0.9,
                ),
            ]
        )
    unknown = [_record(index, DIAGNOSTIC_UNKNOWN) for index in reversed(range(50))]
    sampling = read_division_contracts(ROOT / "configs")["sampling"]

    controls = select_matched_nondivision_controls(
        control_candidates,
        dataset="6bba_synthetic",
        contract=sampling["matched_nondivision_controls"],
    )
    diagnostics = select_unknown_diagnostics(
        unknown,
        dataset="6bba_synthetic",
        contract=sampling["unmatched_diagnostics"],
    )

    assert len(controls) == 32
    assert len({record.mother_id for record in controls}) == 32
    assert all(record.minimum_cellness == pytest.approx(0.9) for record in controls)
    assert [record.deterministic_rank for record in controls] == sorted(
        record.deterministic_rank for record in controls
    )
    assert len(diagnostics) == 32
    assert [record.deterministic_rank for record in diagnostics] == sorted(
        record.deterministic_rank for record in diagnostics
    )
    assert all(
        record.label == LABEL_UNKNOWN and record.loss_weight == 0.0
        for record in diagnostics
    )


def test_fit_schema_uses_supervised_rows_only_and_keeps_binary_counts_raw() -> None:
    supervised_a = _record(0, POSITIVE, context=_context(mother_cellness=0.2))
    supervised_b = _record(1, SAME_MOTHER_NEGATIVE, context=_context(mother_cellness=0.6))
    supervised_c = _record(2, MATCHED_NONDIVISION_CONTROL, context=_context(mother_cellness=1.0))
    unknown = _record(
        3,
        DIAGNOSTIC_UNKNOWN,
        context=_context(mother_cellness=0.9, intensity_offset=10_000.0),
    )

    schema = fit_feature_schema([supervised_a, supervised_b, supervised_c, unknown])
    mother_index = CONTEXT_FEATURE_NAMES.index("mother_cellness")

    assert schema.means[mother_index] == pytest.approx(0.6)
    assert schema.scales[mother_index] == pytest.approx(np.std([0.2, 0.6, 1.0]))
    assert all(
        mean == 0.0 and scale == 1.0
        for kind, mean, scale in zip(
            CONTEXT_FEATURE_KINDS,
            schema.means,
            schema.scales,
            strict=True,
        )
        if kind != "continuous"
    )

    with pytest.raises(ValueError, match="supervised"):
        fit_feature_schema([unknown])


def test_epoch_indices_are_deterministic_balanced_and_exclude_unknowns() -> None:
    records = [
        *[_record(index, POSITIVE) for index in range(2)],
        *[_record(10 + index, SAME_MOTHER_NEGATIVE) for index in range(5)],
        *[_record(20 + index, MATCHED_NONDIVISION_CONTROL) for index in range(3)],
        *[_record(30 + index, DIAGNOSTIC_UNKNOWN) for index in range(7)],
    ]

    first = balanced_epoch_indices(records, seed=17, epoch=4)
    repeated = balanced_epoch_indices(records, seed=17, epoch=4)
    next_epoch = balanced_epoch_indices(records, seed=17, epoch=5)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)
    assert first.dtype == np.int64
    assert len(first) == 15
    selected_strata = [records[int(index)].sampling_stratum for index in first]
    assert selected_strata == [
        value
        for _ in range(5)
        for value in (POSITIVE, SAME_MOTHER_NEGATIVE, MATCHED_NONDIVISION_CONTROL)
    ]
    assert not any(records[int(index)].sampling_stratum == DIAGNOSTIC_UNKNOWN for index in first)
    assert all(records[int(index)].loss_weight == 1.0 for index in first)


def test_reconstruct_upstream_uses_exact_public_phase10_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = TrackingGraph(
        node_ids=np.array([1, 2], dtype=np.int64),
        t=np.array([0, 1], dtype=np.int64),
        z=np.zeros(2),
        y=np.zeros(2),
        x=np.zeros(2),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    filtered = replace(raw, node_ids=np.array([1], dtype=np.int64), t=np.array([0]))
    linked = replace(filtered, edges=np.empty((0, 2), dtype=np.int64))
    pruned = replace(linked)
    calls: list[tuple[str, object]] = []

    def fake_filter(graph, scores, *, min_score):
        calls.append(("filter", min_score))
        assert graph is raw
        assert np.array_equal(scores, np.array([0.2, 0.1], dtype=np.float32))
        return filtered

    def fake_link(graph):
        calls.append(("link", graph))
        assert graph is filtered
        return linked

    def fake_prune(graph, *, min_track_length):
        calls.append(("prune", min_track_length))
        assert graph is linked
        return pruned

    monkeypatch.setattr("biohub.division_dataset.filter_candidates", fake_filter)
    monkeypatch.setattr("biohub.division_dataset.link_graph_flow", fake_link)
    monkeypatch.setattr("biohub.division_dataset.prune_to_tracks", fake_prune)

    observed = reconstruct_upstream(raw, np.array([0.2, 0.1], dtype=np.float32))

    assert observed == (filtered, linked, pruned)
    assert calls == [("filter", 0.2), ("link", filtered), ("prune", 6)]


def _synthetic_movie() -> tuple[
    TrackingGraph,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TrackingGraph,
    TrackingGraph,
    TrackingGraph,
]:
    # One truth division mother, one matched non-division mother, and one unmatched mother.
    node_ids = [10, *range(20, 34), 100, 101, 102, 200, 201, 202]
    times = [0, *([1] * 14), 0, 1, 1, 0, 1, 1]
    x = [
        0.0,
        -1.0,
        1.0,
        -2.0,
        2.0,
        -3.0,
        3.0,
        -4.0,
        4.0,
        -5.0,
        5.0,
        -6.0,
        6.0,
        -7.0,
        7.0,
        100.0,
        99.0,
        101.0,
        200.0,
        199.0,
        201.0,
    ]
    raw = TrackingGraph(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        t=np.asarray(times, dtype=np.int64),
        z=np.zeros(len(node_ids), dtype=np.float64),
        y=np.zeros(len(node_ids), dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    linked = replace(
        raw,
        edges=np.array([[10, 20], [100, 101], [200, 201]], dtype=np.int64),
    )
    truth = TrackingGraph(
        node_ids=np.array([1000, 1001, 1002, 1100], dtype=np.int64),
        t=np.array([0, 1, 1, 0], dtype=np.int64),
        z=np.zeros(4),
        y=np.zeros(4),
        x=np.array([0.0, -1.0, 1.0, 100.0]),
        edges=np.array([[1000, 1001], [1000, 1002]], dtype=np.int64),
    )
    scores = np.linspace(0.4, 0.99, len(node_ids), dtype=np.float32)
    embeddings = np.arange(len(node_ids) * 64, dtype=np.float32).reshape(len(node_ids), 64)
    intensity = np.column_stack(
        [
            np.full(len(node_ids), 5.0),
            np.full(len(node_ids), 1.0),
            np.zeros(len(node_ids)),
            np.full(len(node_ids), 1.0),
            np.full(len(node_ids), 9.0),
        ]
    ).astype(np.float32)
    return raw, scores, embeddings, intensity, linked, linked, truth


def test_synthetic_movie_builds_true_pair_negatives_control_and_unknown() -> None:
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    contracts = read_division_contracts(ROOT / "configs")

    records, funnel = sample_movie_proposals(
        "6bba_synthetic",
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        filtered=linked,
        linked=linked,
        pruned=pruned,
        truth=truth,
        proposal_contract=contracts["proposal"]["proposal_rule"],
        sampling_contract=contracts["sampling"],
        source_role="fit",
    )
    by_stratum = {
        stratum: [record for record in records if record.sampling_stratum == stratum]
        for stratum in (
            POSITIVE,
            SAME_MOTHER_NEGATIVE,
            MATCHED_NONDIVISION_CONTROL,
            DIAGNOSTIC_UNKNOWN,
        )
    }

    assert len(by_stratum[POSITIVE]) == 1
    assert len(by_stratum[SAME_MOTHER_NEGATIVE]) == 12
    assert len(by_stratum[MATCHED_NONDIVISION_CONTROL]) == 1
    assert 1 <= len(by_stratum[DIAGNOSTIC_UNKNOWN]) <= 32
    positive = by_stratum[POSITIVE][0]
    assert {positive.candidate_1_id, positive.candidate_2_id} == {20, 21}
    assert funnel["truth_events"] == 1
    assert funnel["recoverable_positives"] == 1
    assert funnel["missing_mothers"] == 0
    assert funnel["missing_daughters"] == 0
    assert funnel["events"][0]["candidate_ranks"] == [1, 2]
    assert funnel["selected_role_counts"] == {
        DIAGNOSTIC_UNKNOWN: len(by_stratum[DIAGNOSTIC_UNKNOWN]),
        MATCHED_NONDIVISION_CONTROL: 1,
        POSITIVE: 1,
        SAME_MOTHER_NEGATIVE: 12,
    }
    assert np.array_equal(
        np.asarray(positive.image_features),
        invariant_image_features(
            embeddings[positive.mother_node_index],
            embeddings[positive.candidate_1_node_index],
            embeddings[positive.candidate_2_node_index],
        ),
    )


def test_label_free_inference_stream_matches_fit_features_and_direct_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    contracts = read_division_contracts(ROOT / "configs")
    records, _ = sample_movie_proposals(
        "6bba_synthetic",
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        filtered=linked,
        linked=linked,
        pruned=pruned,
        truth=truth,
        proposal_contract=contracts["proposal"]["proposal_rule"],
        sampling_contract=contracts["sampling"],
        source_role="fit",
    )
    schema = fit_feature_schema(records)
    monkeypatch.setattr(
        division_dataset_module,
        "reconstruct_upstream",
        lambda observed_raw, observed_scores: (linked, linked, pruned),
    )
    movie = prepare_inference_movie(
        "6bba_synthetic",
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        proposal_contract=contracts["proposal"]["proposal_rule"],
        source_role="development",
    )
    batches = tuple(
        iter_inference_proposal_batches(
            movie,
            feature_schema=schema,
            proposal_contract=contracts["proposal"]["proposal_rule"],
            max_mothers_per_batch=2,
        )
    )

    assert batches
    assert all(isinstance(batch, InferenceProposalBatch) for batch in batches)
    assert all(batch.mother_count <= 2 for batch in batches)
    assert not {
        "truth",
        "label",
        "labels",
        "loss_weight",
        "sampling_stratum",
    } & set(InferenceProposalBatch.__dataclass_fields__)
    assert all(not batch.raw_context.flags.writeable for batch in batches)
    assert all(not batch.image_features.flags.writeable for batch in batches)
    assert all(not batch.normalized_context.flags.writeable for batch in batches)

    streamed = {
        (
            proposal.mother_id,
            proposal.candidate_1.node_id,
            proposal.candidate_2.node_id,
        ): (batch, index)
        for batch in batches
        for index, proposal in enumerate(batch.proposals)
    }
    for record in records:
        key = (record.mother_id, record.candidate_1_id, record.candidate_2_id)
        batch, index = streamed[key]
        assert np.array_equal(batch.raw_context[index], np.asarray(record.raw_context))
        assert np.array_equal(batch.image_features[index], np.asarray(record.image_features))
        assert batch.restored_node_counts[index] == record.restored_node_count
        assert batch.edge_removal_counts[index] == record.edge_removal_count

    batch = batches[0]
    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS["embedding_linear"])
    model.eval()
    row_by_id = {int(node_id): row for row, node_id in enumerate(raw.node_ids)}
    mothers = torch.from_numpy(
        np.stack([embeddings[row_by_id[item.mother_id]] for item in batch.proposals])
    )
    first = torch.from_numpy(
        np.stack(
            [embeddings[row_by_id[item.candidate_1.node_id]] for item in batch.proposals]
        )
    )
    second = torch.from_numpy(
        np.stack(
            [embeddings[row_by_id[item.candidate_2.node_id]] for item in batch.proposals]
        )
    )
    context = torch.from_numpy(batch.normalized_context.copy())
    with torch.no_grad():
        direct = model(mothers, first, second, context)
        streamed_scores = model.forward_cached_features(
            torch.from_numpy(batch.image_features.copy()),
            context,
        )
    assert torch.equal(direct, streamed_scores)


def test_inference_stream_is_daughter_symmetric_and_validates_role_and_bound() -> None:
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    contracts = read_division_contracts(ROOT / "configs")
    records, _ = sample_movie_proposals(
        "6bba_synthetic",
        raw=raw,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
        filtered=linked,
        linked=linked,
        pruned=pruned,
        truth=truth,
        proposal_contract=contracts["proposal"]["proposal_rule"],
        sampling_contract=contracts["sampling"],
        source_role="fit",
    )
    schema = fit_feature_schema(records)
    movie = division_dataset_module.InferenceMovie(
        dataset="6bba_synthetic",
        source_role="development",
        raw=raw,
        filtered=linked,
        linked=linked,
        fallback=pruned,
        cellness=scores,
        embeddings=embeddings,
        intensity_summaries=intensity,
    )
    batch = next(
        iter_inference_proposal_batches(
            movie,
            feature_schema=schema,
            proposal_contract=contracts["proposal"]["proposal_rule"],
            max_mothers_per_batch=1,
        )
    )
    proposal = batch.proposals[0]
    row_by_id = {int(node_id): row for row, node_id in enumerate(raw.node_ids)}
    swapped = invariant_image_features(
        embeddings[row_by_id[proposal.mother_id]],
        embeddings[row_by_id[proposal.candidate_2.node_id]],
        embeddings[row_by_id[proposal.candidate_1.node_id]],
    )
    assert np.array_equal(batch.image_features[0], swapped)

    with pytest.raises(ValueError, match="development or evaluation"):
        replace(movie, source_role="fit")
    with pytest.raises(ValueError, match="positive integer"):
        tuple(
            iter_inference_proposal_batches(
                movie,
                feature_schema=schema,
                proposal_contract=contracts["proposal"]["proposal_rule"],
                max_mothers_per_batch=0,
            )
        )


def test_synthetic_sampling_is_repeatable_and_rejects_any_contract_drift() -> None:
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    contracts = read_division_contracts(ROOT / "configs")
    kwargs = {
        "raw": raw,
        "cellness": scores,
        "embeddings": embeddings,
        "intensity_summaries": intensity,
        "filtered": linked,
        "linked": linked,
        "pruned": pruned,
        "truth": truth,
        "proposal_contract": contracts["proposal"]["proposal_rule"],
        "sampling_contract": contracts["sampling"],
        "source_role": "fit",
    }

    first = sample_movie_proposals("6bba_synthetic", **kwargs)
    second = sample_movie_proposals("6bba_synthetic", **kwargs)
    assert first == second

    changed_proposal = deepcopy(contracts["proposal"]["proposal_rule"])
    changed_proposal["max_distance_um"] = 15.0
    with pytest.raises(ValueError, match="exact frozen proposal"):
        sample_movie_proposals(
            "6bba_synthetic",
            **{**kwargs, "proposal_contract": changed_proposal},
        )

    changed_sampling = deepcopy(contracts["sampling"])
    changed_sampling["same_mother_alternatives"]["max_pairs_per_mother"] = 11
    with pytest.raises(ValueError, match="exact frozen sampling"):
        sample_movie_proposals(
            "6bba_synthetic",
            **{**kwargs, "sampling_contract": changed_sampling},
        )


def test_development_funnel_cannot_materialize_records_or_fit_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    contracts = read_division_contracts(ROOT / "configs")
    kwargs = {
        "raw": raw,
        "cellness": scores,
        "embeddings": embeddings,
        "intensity_summaries": intensity,
        "filtered": linked,
        "linked": linked,
        "pruned": pruned,
        "truth": truth,
        "proposal_contract": contracts["proposal"]["proposal_rule"],
        "sampling_contract": contracts["sampling"],
        "source_role": "development",
    }

    with pytest.raises(ValueError, match="cannot materialize"):
        sample_movie_proposals("6bba_synthetic", **kwargs)

    def unexpected_record(*args, **kwargs):
        raise AssertionError("development funnel must not create ProposalRecord values")

    monkeypatch.setattr("biohub.division_dataset._proposal_record", unexpected_record)
    records, funnel = sample_movie_proposals(
        "6bba_synthetic",
        **kwargs,
        materialize_records=False,
    )

    assert records == []
    assert funnel["source_role"] == "development"
    assert funnel["recoverable_positives"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "backend",
        "backend_policy",
        "cache_source_sha256",
        "checkpoint_sha256",
        "contract_sha256",
        "upstream_manifest_sha256",
        "model_state_sha256",
        "role",
        "scoring_source_sha256",
        "torch_version",
    ],
)
def test_cache_cohort_rejects_mixed_backend_source_manifest_role_and_model(
    field: str,
) -> None:
    metadata = {
        "backend": "pytorch_mps",
        "backend_policy": "phase11_local_mps_float32_v1",
        "cache_source_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "contract_sha256": dict(EXPECTED_DIVISION_CONTRACT_SHA256),
        "data_split": "train",
        "detector_method": "dog",
        "detector_threshold_k": 3.0,
        "device": "mps",
        "embedding_dim": 64,
        "inference_dtype": "float32",
        "upstream_manifest_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
        "normalization": "per_patch_mean_std_clip_6",
        "patch_shape": [9, 33, 33],
        "role": "fit",
        "scoring_source_sha256": "e" * 64,
        "torch_version": "2.13.0",
    }
    changed = deepcopy(metadata)
    if field == "contract_sha256":
        changed[field]["sampling"] = "f" * 64
    elif field == "role":
        changed[field] = "development"
    else:
        changed[field] = "changed"

    with pytest.raises(ValueError, match="mixes"):
        validate_node_cache_identity_cohort([metadata, changed], expected_role="fit")


def _identity(
    records: list[ProposalRecord],
    *,
    build_mode: str = BUILD_MODE_SMOKE,
    datasets: tuple[str, ...] | None = None,
) -> DatasetBuildIdentity:
    selected = (
        tuple(sorted({record.dataset for record in records}))
        if datasets is None
        else datasets
    )
    return DatasetBuildIdentity(
        build_mode=build_mode,
        cache_files={
            dataset: {
                "content_sha256": hashlib.sha256(f"{dataset}:content".encode()).hexdigest(),
                "file_sha256": hashlib.sha256(f"{dataset}:file".encode()).hexdigest(),
                "image_sha256": hashlib.sha256(f"{dataset}:image".encode()).hexdigest(),
                "role": "fit",
            }
            for dataset in selected
        },
        cache_source_sha256="c" * 64,
        contract_sha256=dict(EXPECTED_DIVISION_CONTRACT_SHA256),
        dataset_source_sha256="e" * 64,
        datasets=selected,
        upstream_manifest_sha256="d" * 64,
    )


def test_dataset_bundle_round_trip_is_hashed_strict_and_resume_safe(tmp_path: Path) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
        _record(3, DIAGNOSTIC_UNKNOWN),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    funnels = {"fit": {"truth_events": 1, "recoverable_positives": 1}}

    first = build_dataset_artifacts(output, records, identity=identity, funnels=funnels)
    resumed = build_dataset_artifacts(output, records, identity=identity, funnels=funnels)
    loaded = read_dataset_artifacts(
        output,
        expected_identity=identity,
        expected_dataset_source_sha256="e" * 64,
    )

    assert first.skipped is False
    assert resumed.skipped is True
    assert first.build_mode == BUILD_MODE_SMOKE
    assert first.dataset_manifest_sha256 == resumed.dataset_manifest_sha256
    assert {path.name for path in output.iterdir()} == set(DATASET_FILES)
    assert loaded.records == tuple(records)
    assert loaded.manifest["sample_tuple_sha256"] == first.sample_tuple_sha256
    assert loaded.manifest["model_inputs_sha256"] == first.model_inputs_sha256
    assert loaded.feature_schema == fit_feature_schema(records)

    with np.load(output / "records.npz", allow_pickle=False) as payload:
        assert set(payload.files) == DATASET_ARRAY_FIELDS
        assert payload["raw_context"].shape == (4, len(CONTEXT_FEATURE_NAMES))
        assert payload["image_features"].shape == (4, 256)
        assert payload["labels"].tolist() == [1, 0, 0, LABEL_UNKNOWN]
        assert payload["loss_weights"].tolist() == [1.0, 1.0, 1.0, 0.0]
        raw_context = np.array(payload["raw_context"], copy=True)
        image_features = np.array(payload["image_features"], copy=True)
    assert loaded.model_inputs_sha256 == canonical_model_inputs_sha256(
        raw_context,
        image_features,
    )
    assert np.array_equal(loaded.image_features, image_features)
    assert np.array_equal(
        loaded.normalized_context,
        loaded.feature_schema.standardize(raw_context),
    )
    assert not loaded.image_features.flags.writeable
    assert not loaded.normalized_context.flags.writeable


def test_model_input_hash_binds_values_shape_dtype_and_row_order() -> None:
    context = np.arange(12, dtype="<f4").reshape(3, 4)
    image = np.arange(24, dtype="<f4").reshape(3, 8)
    baseline = canonical_model_inputs_sha256(context, image)

    changed_value = image.copy()
    changed_value[0, 0] += 1.0
    assert canonical_model_inputs_sha256(context, changed_value) != baseline
    assert canonical_model_inputs_sha256(context[::-1], image[::-1]) != baseline
    assert canonical_model_inputs_sha256(context.astype("<f8"), image) != baseline
    assert canonical_model_inputs_sha256(context.reshape(1, 12), image.reshape(1, 24)) != baseline


def test_feature_mutation_with_same_sample_tuple_and_schema_rejects_resume_and_artifact(
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    changed_features = list(records[0].image_features)
    changed_features[0] += 1.0
    changed_records = [
        replace(records[0], image_features=tuple(changed_features)),
        *records[1:],
    ]

    with pytest.raises(FileExistsError, match="does not match requested inputs"):
        build_dataset_artifacts(
            output,
            changed_records,
            identity=identity,
            funnels={"fit": {}},
        )

    with np.load(output / "records.npz", allow_pickle=False) as loaded:
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    arrays["image_features"][0, 0] += 1.0
    with (output / "records.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"]["records.npz"]["sha256"] = file_sha256(output / "records.npz")
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="model inputs SHA-256"):
        read_dataset_artifacts(
            output,
            expected_identity=identity,
            expected_dataset_source_sha256="e" * 64,
        )


def test_production_identity_and_reader_reject_smoke_or_subset_artifacts(
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    with pytest.raises(ValueError, match="exactly 88"):
        _identity(records, build_mode=BUILD_MODE_FIT_COMPLETE)

    output = tmp_path / "smoke"
    build_dataset_artifacts(
        output,
        records,
        identity=_identity(records),
        funnels={"fit": {}},
    )
    with pytest.raises(ValueError, match="production training reader rejects"):
        read_division_dataset(
            output,
            project_root=tmp_path,
            data_root=tmp_path / "data",
            config_dir=tmp_path / "configs",
            upstream_manifest_path=tmp_path / "manifest.csv",
            summary_path=tmp_path / "summary.json",
            cache_dir=tmp_path / "caches",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )


def test_complete_role_tuple_rejects_wrong_membership_or_order() -> None:
    fit = tuple(f"6bba_fit_{index:03d}" for index in range(88))
    development = tuple(f"6bba_dev_{index:03d}" for index in range(20))

    assert validate_complete_role_tuple(
        fit,
        fit,
        role="fit",
        expected_count=88,
    ) == fit
    assert validate_complete_role_tuple(
        development,
        development,
        role="development",
        expected_count=20,
    ) == development
    with pytest.raises(ValueError, match="exact frozen cohort"):
        validate_complete_role_tuple(
            fit[::-1],
            fit,
            role="fit",
            expected_count=88,
        )
    changed = (*fit[:-1], "6bba_wrong_member")
    with pytest.raises(ValueError, match="exact frozen cohort"):
        validate_complete_role_tuple(
            changed,
            fit,
            role="fit",
            expected_count=88,
        )


def test_exact_88_movie_synthetic_complete_artifact_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    datasets = tuple(f"6bba_complete_{index:03d}" for index in range(88))
    strata = (POSITIVE, SAME_MOTHER_NEGATIVE, MATCHED_NONDIVISION_CONTROL)
    records = [
        replace(_record(index, strata[index % len(strata)]), dataset=dataset)
        for index, dataset in enumerate(datasets)
    ]
    upstream_manifest = tmp_path / "manifest.csv"
    upstream_manifest.write_bytes(b"synthetic frozen manifest\n")
    identity = replace(
        _identity(
            records,
            build_mode=BUILD_MODE_FIT_COMPLETE,
            datasets=datasets,
        ),
        upstream_manifest_sha256=file_sha256(upstream_manifest),
    )
    output = tmp_path / "fit-complete"

    result = build_dataset_artifacts(
        output,
        records,
        identity=identity,
        funnels={"fit": {"movies": 88}},
    )
    loaded = read_dataset_artifacts(
        output,
        expected_identity=identity,
        expected_dataset_source_sha256="e" * 64,
    )

    assert result.build_mode == BUILD_MODE_FIT_COMPLETE
    assert loaded.manifest["build_mode"] == BUILD_MODE_FIT_COMPLETE
    assert tuple(loaded.manifest["datasets"]) == datasets
    assert len(loaded.records) == 88

    monkeypatch.setattr(
        division_dataset_module,
        "verify_division_contract_bundle",
        lambda *args, **kwargs: {"split": {"role_datasets": {"fit": datasets}}},
    )
    monkeypatch.setattr(
        division_dataset_module,
        "current_division_contract_hashes",
        lambda _: dict(EXPECTED_DIVISION_CONTRACT_SHA256),
    )
    monkeypatch.setattr(
        division_dataset_module,
        "read_division_manifest",
        lambda _: tuple(SimpleNamespace(dataset=dataset, role="fit") for dataset in datasets),
    )
    monkeypatch.setattr(division_dataset_module, "division_cache_source_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(
        division_dataset_module,
        "division_dataset_source_sha256",
        lambda _: "e" * 64,
    )

    def unexpected_cache_load(*args, **kwargs):
        raise AssertionError("strict dataset readers must not load node caches")

    monkeypatch.setattr(division_dataset_module, "read_division_node_cache", unexpected_cache_load)
    production = read_division_dataset(
        output,
        project_root=tmp_path,
        data_root=tmp_path / "missing-raw-data",
        config_dir=tmp_path / "configs",
        upstream_manifest_path=upstream_manifest,
        summary_path=tmp_path / "summary.json",
        cache_dir=tmp_path / "missing-caches",
        checkpoint_path=tmp_path / "checkpoint.pt",
    )
    assert production.dataset_manifest_sha256 == result.dataset_manifest_sha256
    assert production.model_inputs_sha256 == result.model_inputs_sha256

    wrong_role_tuple = (*datasets[:-1], "6bba_complete_999")
    monkeypatch.setattr(
        division_dataset_module,
        "verify_division_contract_bundle",
        lambda *args, **kwargs: {"split": {"role_datasets": {"fit": wrong_role_tuple}}},
    )
    monkeypatch.setattr(
        division_dataset_module,
        "read_division_manifest",
        lambda _: tuple(
            SimpleNamespace(dataset=dataset, role="fit") for dataset in wrong_role_tuple
        ),
    )
    with pytest.raises(ValueError, match="fit role tuple"):
        read_division_dataset(
            output,
            project_root=tmp_path,
            data_root=tmp_path / "data",
            config_dir=tmp_path / "configs",
            upstream_manifest_path=upstream_manifest,
            summary_path=tmp_path / "summary.json",
            cache_dir=tmp_path / "caches",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "array",
        "csv",
        "schema",
        "manifest",
        "legacy_manifest_field",
        "source",
        "cache",
        "role",
        "sample_tuple",
    ],
)
def test_dataset_reader_rejects_malformed_records_and_every_identity_mismatch(
    corruption: str,
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})

    expected_identity = identity
    expected_source = "e" * 64
    if corruption == "array":
        with np.load(output / "records.npz", allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
        arrays["labels"][0] = 0
        with (output / "records.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    elif corruption == "csv":
        (output / "records.csv").write_bytes((output / "records.csv").read_bytes() + b"junk\n")
    elif corruption == "schema":
        schema = json.loads((output / "feature_schema.json").read_text())
        schema["means"][0] += 1.0
        (output / "feature_schema.json").write_bytes(canonical_json_bytes(schema))
    elif corruption == "manifest":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["record_count"] += 1
        (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    elif corruption == "legacy_manifest_field":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["manifest_sha256"] = manifest.pop("upstream_manifest_sha256")
        (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    elif corruption == "source":
        expected_source = "f" * 64
    elif corruption == "cache":
        changed = dict(identity.cache_files)
        changed[identity.datasets[0]] = {
            **changed[identity.datasets[0]],
            "content_sha256": "f" * 64,
        }
        expected_identity = replace(identity, cache_files=changed)
    elif corruption == "role":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["source_role"] = "development"
        (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["sample_tuple_sha256"] = "f" * 64
        (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises((OSError, TypeError, ValueError)):
        read_dataset_artifacts(
            output,
            expected_identity=expected_identity,
            expected_dataset_source_sha256=expected_source,
        )


def test_atomic_failure_preserves_destination_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    output = tmp_path / "dataset"
    identity = _identity(records)
    original = build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail_after_partial_write(path, arrays):
        path.write_bytes(b"partial")
        raise OSError("injected dataset write failure")

    monkeypatch.setattr("biohub.division_dataset._write_records_npz", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        build_dataset_artifacts(
            output,
            records,
            identity=identity,
            funnels={"fit": {}},
            overwrite=True,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not dataset_transaction_path(output).exists()
    assert read_dataset_artifacts(
        output,
        expected_identity=identity,
        expected_dataset_source_sha256="e" * 64,
    ).dataset_manifest_sha256 == original.dataset_manifest_sha256


def test_failed_post_promotion_validation_restores_previous_complete_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    output = tmp_path / "dataset"
    identity = _identity(records)
    build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    original_reader = division_dataset_module._read_dataset_artifacts_unlocked

    def fail_only_promoted(path, **kwargs):
        if Path(path) == output:
            raise ValueError("injected post-promotion validation failure")
        return original_reader(path, **kwargs)

    monkeypatch.setattr(
        "biohub.division_dataset._read_dataset_artifacts_unlocked",
        fail_only_promoted,
    )
    with pytest.raises(ValueError, match="post-promotion"):
        build_dataset_artifacts(
            output,
            records,
            identity=identity,
            funnels={"fit": {}},
            overwrite=True,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not dataset_transaction_path(output).exists()


def test_interrupted_promotion_is_recovered_before_next_reader(tmp_path: Path) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    original = build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    transaction = dataset_transaction_path(output)
    code = (
        "import os;"
        "from pathlib import Path;"
        "from biohub.division_dataset import ("
        "TRANSACTION_PROMOTING,dataset_transaction_path,exclusive_dataset_builder_lock,"
        "_write_transaction_marker);"
        f"output=Path({str(output)!r});transaction=dataset_transaction_path(output);"
        "\nwith exclusive_dataset_builder_lock(output):\n"
        " transaction.mkdir()\n"
        " (transaction/'staged').mkdir()\n"
        " _write_transaction_marker(transaction,state=TRANSACTION_PROMOTING,"
        "had_destination=True,"
        f"staged_dataset_manifest_sha256={original.dataset_manifest_sha256!r})\n"
        " os.replace(output,transaction/'backup')\n"
        " os._exit(23)\n"
    )
    interrupted = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
        check=False,
    )

    assert interrupted.returncode == 23
    assert not output.exists()
    recovered = read_dataset_artifacts(
        output,
        expected_identity=identity,
        expected_dataset_source_sha256="e" * 64,
    )

    assert recovered.dataset_manifest_sha256 == original.dataset_manifest_sha256
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not transaction.exists()


def test_reader_blocks_across_writer_missing_destination_window(tmp_path: Path) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    output = tmp_path / "dataset"
    identity = _identity(records)
    build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    transaction = dataset_transaction_path(output)
    signal = tmp_path / "reader-result.txt"
    code = (
        "from pathlib import Path;"
        "from biohub.division_dataset import shared_dataset_reader_lock;"
        f"output=Path({str(output)!r});signal=Path({str(signal)!r});"
        "\nwith shared_dataset_reader_lock(output):\n"
        " signal.write_text('complete' if (output/'manifest.json').is_file() else 'missing')\n"
    )

    with exclusive_dataset_builder_lock(output):
        transaction.mkdir()
        (transaction / "staged").mkdir()
        division_dataset_module._write_transaction_marker(
            transaction,
            state=division_dataset_module.TRANSACTION_PROMOTING,
            had_destination=True,
            staged_dataset_manifest_sha256=file_sha256(output / "manifest.json"),
        )
        os.replace(output, transaction / "backup")
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.25)
        assert process.poll() is None
        assert not signal.exists()
        os.replace(transaction / "backup", output)
        division_dataset_module._remove_path(transaction)

    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr
    assert signal.read_text() == "complete"


def test_cache_replacement_before_promotion_aborts_and_preserves_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    dataset = identity.datasets[0]
    cache = cache_dir / f"{dataset}.npz"
    cache.write_bytes(b"original-cache")
    metadata = {
        "content_sha256": "a" * 64,
        "image_sha256": "b" * 64,
    }
    context = division_dataset_module._CohortContext(
        caches={
            dataset: SimpleNamespace(
                path=cache,
                file_sha256=file_sha256(cache),
                metadata=metadata,
            )
        },
        contracts={},
        contract_sha256=dict(EXPECTED_DIVISION_CONTRACT_SHA256),
        rows=(SimpleNamespace(dataset=dataset),),
        samples={},
        cache_source_sha256="c" * 64,
        upstream_manifest_sha256="d" * 64,
    )
    monkeypatch.setattr(
        division_dataset_module,
        "read_division_node_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pre-promotion drift checks must not reload cache arrays")
        ),
    )

    def replace_cache_then_validate() -> None:
        cache.write_bytes(b"replacement-cache")
        division_dataset_module._revalidate_input_cache_files(
            context,
        )

    with pytest.raises(ValueError, match="changed after sampling"):
        build_dataset_artifacts(
            output,
            records,
            identity=identity,
            funnels={"fit": {}},
            overwrite=True,
            before_promotion=replace_cache_then_validate,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not dataset_transaction_path(output).exists()


def test_high_level_construction_loads_each_cache_once_and_never_hashes_raw_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = "6bba_synthetic"
    output = tmp_path / "dataset"
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{dataset}.npz"
    cache_path.write_bytes(b"one immutable synthetic cache")
    upstream_manifest = tmp_path / "manifest.csv"
    upstream_manifest.write_bytes(b"synthetic upstream manifest\n")
    upstream_hash = file_sha256(upstream_manifest)
    raw, scores, embeddings, intensity, linked, pruned, truth = _synthetic_movie()
    row = SimpleNamespace(dataset=dataset, role="fit", truth_graph_sha256="f" * 64)
    sample = SimpleNamespace(dataset=dataset, embryo="6bba", labels=tmp_path / "labels")
    contracts = read_division_contracts(ROOT / "configs")
    metadata = {
        "backend": "pytorch_mps",
        "backend_policy": "phase11_local_mps_float32_v1",
        "cache_source_sha256": "c" * 64,
        "checkpoint_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "contract_sha256": dict(EXPECTED_DIVISION_CONTRACT_SHA256),
        "data_split": "train",
        "detector_method": "dog",
        "detector_threshold_k": 3.0,
        "device": "mps",
        "embedding_dim": 64,
        "image_sha256": "d" * 64,
        "inference_dtype": "float32",
        "model_state_sha256": "e" * 64,
        "normalization": "per_patch_mean_std_clip_6",
        "patch_shape": [9, 33, 33],
        "role": "fit",
        "scoring_source_sha256": "f" * 64,
        "torch_version": "2.13.0",
        "upstream_manifest_sha256": upstream_hash,
    }
    cache_reads = 0
    metadata_reads = 0

    def read_cache_metadata(*args, **kwargs):
        nonlocal metadata_reads
        metadata_reads += 1
        return dict(metadata), file_sha256(cache_path)

    def read_cache(*args, **kwargs):
        nonlocal cache_reads
        cache_reads += 1
        return raw, scores, embeddings, intensity, dict(metadata)

    monkeypatch.setattr(
        division_dataset_module,
        "verify_division_contract_bundle",
        lambda *args, **kwargs: contracts,
    )
    monkeypatch.setattr(
        division_dataset_module,
        "current_division_contract_hashes",
        lambda _: dict(EXPECTED_DIVISION_CONTRACT_SHA256),
    )
    monkeypatch.setattr(division_dataset_module, "read_division_manifest", lambda _: (row,))
    monkeypatch.setattr(
        division_dataset_module,
        "read_division_node_cache_metadata",
        read_cache_metadata,
    )
    monkeypatch.setattr(division_dataset_module, "read_division_node_cache", read_cache)
    monkeypatch.setattr(division_dataset_module, "discover_split", lambda *args: [sample])
    monkeypatch.setattr(division_dataset_module, "division_cache_source_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(
        division_dataset_module,
        "division_dataset_source_sha256",
        lambda _: "e" * 64,
    )
    monkeypatch.setattr(division_dataset_module, "_load_truth", lambda *args, **kwargs: truth)
    monkeypatch.setattr(
        division_dataset_module,
        "reconstruct_upstream",
        lambda *args, **kwargs: (linked, linked, pruned),
    )

    result = division_dataset_module.build_smoke_dataset(
        output,
        dataset=dataset,
        project_root=tmp_path,
        data_root=tmp_path / "raw",
        config_dir=tmp_path / "configs",
        upstream_manifest_path=upstream_manifest,
        summary_path=tmp_path / "summary.json",
        cache_dir=cache_dir,
        checkpoint_path=tmp_path / "checkpoint.pt",
    )
    resumed = division_dataset_module.build_smoke_dataset(
        output,
        dataset=dataset,
        project_root=tmp_path,
        data_root=tmp_path / "missing-raw-on-resume",
        config_dir=tmp_path / "configs",
        upstream_manifest_path=upstream_manifest,
        summary_path=tmp_path / "summary.json",
        cache_dir=tmp_path / "missing-caches-on-resume",
        checkpoint_path=tmp_path / "checkpoint.pt",
    )

    assert result.record_count > 0
    assert resumed.skipped is True
    assert metadata_reads == 1
    assert cache_reads == 1
    assert not hasattr(division_dataset_module, "open_image_array")


def test_multi_movie_sampling_releases_each_full_cache_before_loading_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    datasets = tuple(f"6bba_memory_{index}" for index in range(4))
    cache_dir = tmp_path / "caches"
    cache_dir.mkdir()
    upstream_manifest = tmp_path / "manifest.csv"
    upstream_manifest.write_bytes(b"synthetic upstream manifest\n")
    upstream_hash = file_sha256(upstream_manifest)
    rows = tuple(
        SimpleNamespace(dataset=dataset, role="fit", truth_graph_sha256="f" * 64)
        for dataset in datasets
    )
    samples = [
        SimpleNamespace(dataset=dataset, embryo="6bba", labels=tmp_path / f"{dataset}.labels")
        for dataset in datasets
    ]
    paths = {dataset: cache_dir / f"{dataset}.npz" for dataset in datasets}
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"metadata-only-cache-{index}".encode("ascii"))
    contracts = read_division_contracts(ROOT / "configs")
    metadata_by_dataset = {
        dataset: {
            "backend": "pytorch_mps",
            "backend_policy": "phase11_local_mps_float32_v1",
            "cache_source_sha256": "c" * 64,
            "checkpoint_sha256": "a" * 64,
            "content_sha256": hashlib.sha256(f"{dataset}:content".encode()).hexdigest(),
            "contract_sha256": dict(EXPECTED_DIVISION_CONTRACT_SHA256),
            "data_split": "train",
            "dataset": dataset,
            "detector_method": "dog",
            "detector_threshold_k": 3.0,
            "device": "mps",
            "embedding_dim": 64,
            "image_sha256": hashlib.sha256(f"{dataset}:image".encode()).hexdigest(),
            "inference_dtype": "float32",
            "model_state_sha256": "e" * 64,
            "normalization": "per_patch_mean_std_clip_6",
            "patch_shape": [9, 33, 33],
            "role": "fit",
            "scoring_source_sha256": "f" * 64,
            "torch_version": "2.13.0",
            "upstream_manifest_sha256": upstream_hash,
        }
        for dataset in datasets
    }
    metadata_loads = Counter()
    full_loads = Counter()
    payload_groups: list[tuple[weakref.ReferenceType[np.ndarray], ...]] = []
    maximum_live_groups = 0

    def read_metadata(path, *, expected_dataset, **kwargs):
        metadata_loads[expected_dataset] += 1
        return dict(metadata_by_dataset[expected_dataset]), file_sha256(path)

    def read_full(path, *, expected_dataset, **kwargs):
        nonlocal maximum_live_groups
        gc.collect()
        assert not any(
            any(reference() is not None for reference in group)
            for group in payload_groups
        )
        full_loads[expected_dataset] += 1
        raw = TrackingGraph(
            node_ids=np.array([1, 2, 3], dtype=np.int64),
            t=np.array([0, 1, 1], dtype=np.int64),
            z=np.zeros(3),
            y=np.zeros(3),
            x=np.zeros(3),
            edges=np.empty((0, 2), dtype=np.int64),
        )
        scores = np.ones(3, dtype=np.float32)
        embeddings = np.ones((3, 64), dtype=np.float32)
        intensity = np.ones((3, 5), dtype=np.float32)
        group = tuple(
            weakref.ref(array)
            for array in (raw.node_ids, scores, embeddings, intensity)
        )
        payload_groups.append(group)
        maximum_live_groups = max(
            maximum_live_groups,
            sum(any(reference() is not None for reference in item) for item in payload_groups),
        )
        return raw, scores, embeddings, intensity, dict(metadata_by_dataset[expected_dataset])

    monkeypatch.setattr(
        division_dataset_module,
        "verify_division_contract_bundle",
        lambda *args, **kwargs: contracts,
    )
    monkeypatch.setattr(
        division_dataset_module,
        "current_division_contract_hashes",
        lambda _: dict(EXPECTED_DIVISION_CONTRACT_SHA256),
    )
    monkeypatch.setattr(division_dataset_module, "read_division_manifest", lambda _: rows)
    monkeypatch.setattr(division_dataset_module, "discover_split", lambda *args: samples)
    monkeypatch.setattr(division_dataset_module, "division_cache_source_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(
        division_dataset_module,
        "read_division_node_cache_metadata",
        read_metadata,
    )
    monkeypatch.setattr(division_dataset_module, "read_division_node_cache", read_full)
    monkeypatch.setattr(division_dataset_module, "_load_truth", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        division_dataset_module,
        "reconstruct_upstream",
        lambda raw, scores: (raw, raw, raw),
    )
    monkeypatch.setattr(
        division_dataset_module,
        "sample_movie_proposals",
        lambda dataset, **kwargs: ([], {"dataset": dataset}),
    )

    context = division_dataset_module._prepare_cohort(
        project_root=tmp_path,
        data_root=tmp_path / "raw",
        config_dir=tmp_path / "configs",
        upstream_manifest_path=upstream_manifest,
        summary_path=tmp_path / "summary.json",
        cache_dir=cache_dir,
        checkpoint_path=tmp_path / "checkpoint.pt",
        role="fit",
        datasets=datasets,
    )
    assert not full_loads
    assert all(
        not any(
            hasattr(identity, field)
            for field in ("raw", "cellness", "embeddings", "intensity_summaries")
        )
        for identity in context.caches.values()
    )

    for row in context.rows:
        records, funnel = division_dataset_module._sample_cached_movie(
            row,
            context=context,
            source_role="fit",
        )
        assert records == []
        assert funnel == {"dataset": row.dataset}
    gc.collect()

    assert metadata_loads == Counter({dataset: 1 for dataset in datasets})
    assert full_loads == Counter({dataset: 1 for dataset in datasets})
    assert maximum_live_groups == 1
    assert not any(any(reference() is not None for reference in group) for group in payload_groups)
    division_dataset_module._revalidate_input_cache_files(context)
    assert full_loads == Counter({dataset: 1 for dataset in datasets})


def test_builder_restart_recovers_transaction_before_cohort_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records = [
        _record(0, POSITIVE),
        _record(1, SAME_MOTHER_NEGATIVE),
        _record(2, MATCHED_NONDIVISION_CONTROL),
    ]
    identity = _identity(records)
    output = tmp_path / "dataset"
    original = build_dataset_artifacts(output, records, identity=identity, funnels={"fit": {}})
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    transaction = dataset_transaction_path(output)
    with exclusive_dataset_builder_lock(output):
        transaction.mkdir()
        (transaction / "staged").mkdir()
        division_dataset_module._write_transaction_marker(
            transaction,
            state=division_dataset_module.TRANSACTION_PROMOTING,
            had_destination=True,
            staged_dataset_manifest_sha256=original.dataset_manifest_sha256,
        )
        os.replace(output, transaction / "backup")

    monkeypatch.setattr(
        division_dataset_module,
        "division_dataset_source_sha256",
        lambda _: "e" * 64,
    )

    def stop_at_cohort_preparation(**kwargs):
        assert output.is_dir()
        assert not transaction.exists()
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before
        raise RuntimeError("cohort preparation reached after recovery")

    monkeypatch.setattr(
        division_dataset_module,
        "_prepare_cohort",
        stop_at_cohort_preparation,
    )
    with pytest.raises(RuntimeError, match="reached after recovery"):
        division_dataset_module.build_smoke_dataset(
            output,
            dataset="6bba_synthetic",
            project_root=tmp_path,
            data_root=tmp_path / "raw",
            config_dir=tmp_path / "configs",
            upstream_manifest_path=tmp_path / "manifest.csv",
            summary_path=tmp_path / "summary.json",
            cache_dir=tmp_path / "caches",
            checkpoint_path=tmp_path / "checkpoint.pt",
            overwrite=True,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not transaction.exists()


def test_dataset_builder_lock_rejects_active_process_and_reuses_stale_file(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    lock = dataset_lock_path(output)

    with exclusive_dataset_builder_lock(output):
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_dataset_builder_lock(output):
                pass

    assert lock.is_file()
    lock.write_text("stale but unlocked")
    with exclusive_dataset_builder_lock(output):
        pass


def test_cli_exposes_only_fit_build_or_read_only_development_funnel() -> None:
    parser = build_division_dataset.build_parser()

    assert parser.parse_args(["--fit"]).fit is True
    smoke = parser.parse_args(
        ["--smoke-dataset", "6bba_synthetic", "--output-dir", "smoke-output"]
    )
    assert smoke.smoke_dataset == "6bba_synthetic"
    assert parser.parse_args(["--development-funnel"]).development_funnel is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "6bba_synthetic"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--role", "evaluation"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--fit", "--development-funnel"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--fit", "--manifest", "legacy.csv"])
    with pytest.raises(SystemExit):
        build_division_dataset.main(["--smoke-dataset", "6bba_synthetic"])


def test_cli_help_is_clean_clone_safe_without_optional_torch() -> None:
    script = ROOT / "scripts" / "build_division_dataset.py"
    command = (
        "import runpy,sys;"
        "sys.modules['torch']=None;"
        f"sys.argv=[{str(script)!r},'--help'];"
        f"runpy.run_path({str(script)!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "development-funnel" in result.stdout


def test_division_dataset_imports_when_torch_is_unavailable() -> None:
    command = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch blocked for import test")
        return None

sys.meta_path.insert(0, BlockTorch())
import biohub.division_dataset
from biohub.division_model import DivisionModel, DivisionModelConfig
try:
    DivisionModel(DivisionModelConfig("scalar"))
except ModuleNotFoundError as exc:
    assert "require PyTorch" in str(exc)
else:
    raise AssertionError("model construction unexpectedly worked without torch")
print("imported-without-torch")
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported-without-torch"


def test_cli_clean_clone_failure_creates_no_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_division_dataset.py"),
            "--fit",
            "--config-dir",
            str(tmp_path / "missing-configs"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--upstream-manifest",
            str(tmp_path / "missing-manifest.csv"),
            "--summary",
            str(tmp_path / "missing-summary.json"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint.pt"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "No such file" in result.stderr or "does not exist" in result.stderr
    assert not output.exists()
