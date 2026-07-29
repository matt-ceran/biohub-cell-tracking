from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biohub import division_data as division_data_module
from biohub.division_data import (
    GRAPH_HASH_SCHEMA,
    SPLIT_QUOTAS,
    SPLIT_RANK_PREFIX,
    DivisionManifestRow,
    assign_division_roles,
    canonical_json_bytes,
    canonical_json_sha256,
    division_count_manifest,
    file_sha256,
    logical_graph_sha256,
    read_canonical_json,
    read_division_contract,
    read_division_contracts,
    read_division_manifest,
    truth_divisions,
    verify_division_contract_bundle,
    write_canonical_json,
    write_division_manifest,
)
from biohub.io import GeffGraph, SamplePaths
from biohub.metric import TrackingGraph
from scripts import freeze_division_contracts

ROOT = Path(__file__).resolve().parents[1]
EXPOSED_DATASETS = (
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
EXPOSED_COUNTS = {
    "6bba_474be664": 1,
    "6bba_4f99ce20": 2,
    "6bba_6ca87370": 1,
    "6bba_789f8168": 0,
    "6bba_7af54fde": 2,
    "6bba_825bd1c6": 0,
    "6bba_907271db": 1,
    "6bba_afb141ff": 4,
    "6bba_d1acb6ff": 1,
    "6bba_ebdf3b34": 2,
    "6bba_ef7b4f7e": 2,
    "6bba_f4ae811c": 0,
    "6bba_fc5f39dc": 2,
}


def division_graph() -> TrackingGraph:
    return TrackingGraph(
        node_ids=np.array([3, 1, 2], dtype=np.int32),
        t=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        z=np.array([3.0, 1.0, 2.0], dtype=np.float32),
        y=np.array([30.0, 10.0, 20.0], dtype=np.float32),
        x=np.array([300.0, 100.0, 200.0], dtype=np.float32),
        edges=np.array([[1, 3], [1, 2]], dtype=np.int32),
    )


def _tagged_array(name: str, values: np.ndarray) -> bytes:
    shape = ",".join(str(value) for value in values.shape).encode("ascii")
    return (
        b"field\0"
        + name.encode("ascii")
        + b"\0dtype\0"
        + values.dtype.str.encode("ascii")
        + b"\0shape\0"
        + shape
        + b"\0data\0"
        + values.tobytes(order="C")
        + b"\0"
    )


def test_logical_graph_sha256_has_an_exact_tagged_canonical_encoding() -> None:
    expected_bytes = GRAPH_HASH_SCHEMA.encode("ascii") + b"\0"
    expected_bytes += _tagged_array("node_id", np.array([1, 2, 3], dtype="<i8"))
    expected_bytes += _tagged_array("t", np.array([0, 1, 1], dtype="<i8"))
    expected_bytes += _tagged_array("z", np.array([1.0, 2.0, 3.0], dtype="<f8"))
    expected_bytes += _tagged_array("y", np.array([10.0, 20.0, 30.0], dtype="<f8"))
    expected_bytes += _tagged_array("x", np.array([100.0, 200.0, 300.0], dtype="<f8"))
    expected_bytes += _tagged_array(
        "edges",
        np.array([[1, 2], [1, 3]], dtype="<i8"),
    )

    assert logical_graph_sha256(division_graph()) == hashlib.sha256(expected_bytes).hexdigest()


def test_logical_graph_hash_and_truth_divisions_ignore_storage_order() -> None:
    equivalent = TrackingGraph(
        node_ids=np.array([2, 3, 1], dtype=np.int64),
        t=np.array([1, 1, 0], dtype=np.int16),
        z=np.array([2.0, 3.0, 1.0], dtype=np.float64),
        y=np.array([20.0, 30.0, 10.0], dtype=np.float64),
        x=np.array([200.0, 300.0, 100.0], dtype=np.float64),
        edges=np.array([[1, 2], [1, 3]], dtype=np.int64),
    )

    assert logical_graph_sha256(division_graph()) == logical_graph_sha256(equivalent)
    assert truth_divisions(equivalent) == {1: (2, 3)}


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (replace(division_graph(), node_ids=np.array([3.5, 1, 2])), "node identifiers"),
        (replace(division_graph(), t=np.array([1, 0.5, 1])), "time values"),
        (replace(division_graph(), node_ids=np.array([1, 1, 2])), "duplicate node"),
        (replace(division_graph(), z=np.array([np.nan, 1, 2])), "finite z"),
        (replace(division_graph(), edges=np.array([[1, 2], [1, 99]])), "missing endpoint"),
        (replace(division_graph(), edges=np.array([[1, 2], [1, 2]])), "duplicate edge"),
    ],
)
def test_graph_contract_rejects_malformed_values(
    graph: TrackingGraph,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        logical_graph_sha256(graph)


@pytest.mark.parametrize(
    "edges",
    [
        np.array([[3, 1], [1, 2]]),
        np.array([[2, 3], [1, 2]]),
    ],
)
def test_graph_contract_rejects_backward_and_same_frame_edges(edges: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strictly forward"):
        logical_graph_sha256(replace(division_graph(), edges=edges))


@pytest.mark.parametrize(
    "graph",
    [
        TrackingGraph(
            node_ids=np.array([1, 2, 3, 4]),
            t=np.array([0, 1, 1, 1]),
            z=np.zeros(4),
            y=np.zeros(4),
            x=np.zeros(4),
            edges=np.array([[1, 2], [1, 3], [1, 4]]),
        ),
        TrackingGraph(
            node_ids=np.array([1, 2, 3]),
            t=np.array([0, 1, 2]),
            z=np.zeros(3),
            y=np.zeros(3),
            x=np.zeros(3),
            edges=np.array([[1, 2], [1, 3]]),
        ),
    ],
)
def test_truth_divisions_requires_exactly_two_daughters_at_the_next_frame(
    graph: TrackingGraph,
) -> None:
    with pytest.raises(ValueError, match=r"exactly two daughters at t\+1"):
        truth_divisions(graph)


def test_canonical_json_hash_matches_exact_atomic_file_bytes(tmp_path: Path) -> None:
    payload = {"z": [3, 2], "a": {"b": True}}
    expected = b'{\n  "a": {\n    "b": true\n  },\n  "z": [\n    3,\n    2\n  ]\n}\n'
    output = tmp_path / "identity.json"

    digest = write_canonical_json(output, payload)

    assert canonical_json_bytes(payload) == expected
    assert output.read_bytes() == expected
    assert digest == canonical_json_sha256(payload) == hashlib.sha256(expected).hexdigest()
    assert read_canonical_json(output) == payload
    assert list(tmp_path.iterdir()) == [output]


def test_canonical_json_rejects_nonfinite_and_noncanonical_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_json_bytes({"bad": np.nan})

    path = tmp_path / "not-canonical.json"
    path.write_text('{"b": 2, "a": 1}\n')
    with pytest.raises(ValueError, match="not canonical"):
        read_canonical_json(path)


def test_division_count_manifest_extracts_truth_and_hash(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "6bba_movie.zarr"
    image.mkdir()
    labels = tmp_path / "6bba_movie.geff"
    labels.mkdir()
    sample = SamplePaths(
        dataset="6bba_movie",
        image=image,
        labels=labels,
        embryo="6bba",
    )
    graph = division_graph()
    geff = GeffGraph(graph.node_ids, graph.t, graph.z, graph.y, graph.x, graph.edges)
    monkeypatch.setattr(
        "biohub.division_data.open_image_array",
        lambda _: SimpleNamespace(shape=(2, 3, 4, 5), dtype=np.dtype("uint16")),
    )
    monkeypatch.setattr("biohub.division_data.read_geff_graph", lambda _: geff)

    rows = division_count_manifest([sample])

    assert rows == (
        DivisionManifestRow(
            dataset="6bba_movie",
            embryo="6bba",
            division_count=1,
            no_division=False,
            truth_graph_sha256=logical_graph_sha256(graph),
        ),
    )


def test_manifest_rejects_non_6bba_before_loading_any_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def unexpected_load(path: Path) -> GeffGraph:
        calls.append(path)
        raise AssertionError("labels must not be loaded before cohort preflight")

    monkeypatch.setattr("biohub.division_data.read_geff_graph", unexpected_load)
    samples = [
        SamplePaths("6bba_safe", tmp_path / "safe.zarr", tmp_path / "safe.geff", "6bba"),
        SamplePaths("44b6_forbidden", tmp_path / "bad.zarr", tmp_path / "bad.geff", "44b6"),
    ]

    with pytest.raises(ValueError, match="only 6bba"):
        division_count_manifest(samples)
    assert calls == []


def test_manifest_rejects_missing_image_before_loading_labels(monkeypatch, tmp_path: Path) -> None:
    labels = tmp_path / "6bba_missing.geff"
    labels.mkdir()
    label_calls: list[Path] = []
    monkeypatch.setattr(
        "biohub.division_data.read_geff_graph",
        lambda path: label_calls.append(path),
    )
    sample = SamplePaths(
        "6bba_missing",
        tmp_path / "6bba_missing.zarr",
        labels,
        "6bba",
    )

    with pytest.raises(FileNotFoundError, match="image"):
        division_count_manifest([sample])
    assert label_calls == []


def test_manifest_rejects_unreadable_zarr_before_loading_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "6bba_unreadable.zarr"
    image.mkdir()
    labels = tmp_path / "6bba_unreadable.geff"
    labels.mkdir()
    label_calls: list[Path] = []

    def unreadable(_: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr("biohub.division_data.open_image_array", unreadable)
    monkeypatch.setattr(
        "biohub.division_data.read_geff_graph",
        lambda path: label_calls.append(path),
    )
    sample = SamplePaths("6bba_unreadable", image, labels, "6bba")

    with pytest.raises(ValueError, match="readable Zarr"):
        division_count_manifest([sample])
    assert label_calls == []


def synthetic_manifest_rows() -> tuple[DivisionManifestRow, ...]:
    totals = {division_count: sum(quotas) for division_count, quotas in SPLIT_QUOTAS.items()}
    names_by_count: dict[int, list[str]] = {value: [] for value in totals}
    for dataset, division_count in EXPOSED_COUNTS.items():
        names_by_count[division_count].append(dataset)
    rows: list[DivisionManifestRow] = []
    for division_count, total in totals.items():
        names = names_by_count[division_count]
        names.extend(
            f"6bba_s{division_count}_{index:03d}"
            for index in range(total - len(names))
        )
        for dataset in names:
            rows.append(
                DivisionManifestRow(
                    dataset=dataset,
                    embryo="6bba",
                    division_count=division_count,
                    no_division=division_count == 0,
                    truth_graph_sha256=hashlib.sha256(dataset.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(sorted(rows, key=lambda row: row.dataset))


def test_assign_division_roles_has_exact_quotas_events_and_forced_fit() -> None:
    assigned = assign_division_roles(synthetic_manifest_rows(), EXPOSED_DATASETS)
    by_role = {
        role: [row for row in assigned if row.role == role]
        for role in ("fit", "development", "evaluation")
    }

    assert {role: len(rows) for role, rows in by_role.items()} == {
        "fit": 88,
        "development": 20,
        "evaluation": 20,
    }
    assert {
        role: sum(row.division_count for row in rows) for role, rows in by_role.items()
    } == {"fit": 89, "development": 18, "evaluation": 18}
    for role in ("development", "evaluation"):
        assert sum(row.no_division for row in by_role[role]) == 10
        assert sum(not row.no_division for row in by_role[role]) == 10
    assert set(EXPOSED_DATASETS) <= {row.dataset for row in by_role["fit"]}
    assert all(row.proposal_exposed == (row.dataset in EXPOSED_DATASETS) for row in assigned)
    assert tuple(row.dataset for row in assigned) == tuple(
        sorted(row.dataset for row in assigned)
    )


def test_assign_division_roles_uses_the_exact_sha256_rank() -> None:
    assigned = assign_division_roles(synthetic_manifest_rows(), EXPOSED_DATASETS)
    assigned_role = {row.dataset: row.role for row in assigned}

    for division_count, (_, development_quota, evaluation_quota) in SPLIT_QUOTAS.items():
        remaining = [
            row.dataset
            for row in synthetic_manifest_rows()
            if row.division_count == division_count and row.dataset not in EXPOSED_DATASETS
        ]
        ranked = sorted(
            remaining,
            key=lambda dataset: (
                hashlib.sha256(SPLIT_RANK_PREFIX + dataset.encode("utf-8")).digest(),
                dataset,
            ),
        )
        assert all(assigned_role[name] == "evaluation" for name in ranked[:evaluation_quota])
        assert all(
            assigned_role[name] == "development"
            for name in ranked[evaluation_quota : evaluation_quota + development_quota]
        )


def test_assign_division_roles_rejects_bad_cohorts() -> None:
    rows = synthetic_manifest_rows()
    with pytest.raises(ValueError, match="exact division-count histogram"):
        assign_division_roles(rows[:-1], EXPOSED_DATASETS)
    with pytest.raises(ValueError, match="only 6bba"):
        assign_division_roles((replace(rows[0], dataset="44b6_bad"), *rows[1:]), EXPOSED_DATASETS)
    with pytest.raises(ValueError, match="exactly 13"):
        assign_division_roles(rows, EXPOSED_DATASETS[:-1])


def test_tracked_contracts_have_exact_frozen_values_and_valid_cross_hashes() -> None:
    contracts = read_division_contracts(ROOT / "configs")

    assert set(contracts) == {"split", "proposal", "sampling", "acceptance", "assignment"}
    assert contracts["proposal"]["proposal_rule"] == {
        "candidate_order": ["distance_um:ascending", "node_id:ascending"],
        "candidate_source": "raw_scored",
        "max_candidates_per_mother": 12,
        "max_distance_um": 16.0,
        "min_angle_deg": 0.0,
        "min_distance_um": 0.0,
        "mother_source": "cnn_filtered_preprune",
        "rule_id": "dmax16_cap12_dmin0_angle0",
    }
    assert contracts["sampling"]["same_mother_alternatives"]["max_pairs_per_mother"] == 12
    assert contracts["sampling"]["matched_nondivision_controls"]["max_mothers_per_movie"] == 32
    for stratum in (
        "positive_records",
        "same_mother_alternatives",
        "matched_nondivision_controls",
        "unmatched_diagnostics",
    ):
        assert contracts["sampling"][stratum]["source_role"] == "fit"
    assert contracts["sampling"]["unmatched_diagnostics"] == {
        "loss_weight": 0.0,
        "max_proposals_per_movie": 32,
        "proposal_rank": {
            "algorithm": "sha256",
            "encoding": (
                "salt_nul_dataset_nul_mother_id_nul_candidate_1_id_nul_candidate_2_id_ascii"
            ),
            "salt": "biohub-phase11-division-unmatched-diagnostic-v1",
        },
        "source_role": "fit",
        "supervised": False,
    }
    assert contracts["sampling"]["supervised_batches"]["stratum_ratio"] == {
        "matched_nondivision_control": 1,
        "positive": 1,
        "same_mother_negative": 1,
    }
    assert contracts["acceptance"]["guards"]["mean_full_score_delta"] == {
        "operator": ">=",
        "value": 0.001,
    }
    assert contracts["acceptance"]["guards"]["mean_edge_jaccard_delta"] == {
        "operator": ">=",
        "value": -0.001,
    }
    assert contracts["acceptance"]["guards"]["aggregate_node_ratio_delta"] == {
        "operator": "<=",
        "value": 0.001,
    }
    assert contracts["assignment"]["threshold_grid"] == [
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
    ]
    assert contracts["assignment"]["max_forks_per_movie"] == 5


def test_contract_reader_rejects_noncanonical_and_wrong_static_values(tmp_path: Path) -> None:
    source = ROOT / "configs" / "phase11_division_assignment.json"
    payload = json.loads(source.read_text())
    payload["max_forks_per_movie"] = 6
    changed = tmp_path / source.name
    write_canonical_json(changed, payload)

    with pytest.raises(ValueError, match="assignment contract"):
        read_division_contract(changed, contract_kind="assignment")

    changed.write_text(json.dumps(payload, sort_keys=False))
    with pytest.raises(ValueError, match="not canonical"):
        read_division_contract(changed, contract_kind="assignment")


def test_contract_bundle_rejects_cross_hash_tampering(tmp_path: Path) -> None:
    names = ("split", "proposal", "sampling", "acceptance", "assignment")
    for name in names:
        source = ROOT / "configs" / f"phase11_division_{name}.json"
        payload = read_canonical_json(source)
        if name == "sampling":
            payload["source_fingerprints"]["division_data_sha256"] = "f" * 64
        write_canonical_json(tmp_path / source.name, payload)

    with pytest.raises(ValueError, match="cross-hash"):
        read_division_contracts(tmp_path)


def _write_synthetic_contract_bundle(
    directory: Path,
    *,
    source_fingerprints: dict[str, str],
) -> tuple[Path, Path, Path, tuple[DivisionManifestRow, ...]]:
    rows = assign_division_roles(synthetic_manifest_rows(), EXPOSED_DATASETS)
    manifest = directory / "manifest.csv"
    manifest_hash = write_division_manifest(manifest, rows)
    configs = directory / "configs"
    contracts = freeze_division_contracts.build_contract_payloads(
        rows,
        manifest_sha256=manifest_hash,
        source_fingerprints=source_fingerprints,
    )
    for kind, payload in contracts.items():
        write_canonical_json(configs / f"phase11_division_{kind}.json", payload)
    contract_hashes = {
        kind: canonical_json_sha256(payload) for kind, payload in contracts.items()
    }
    summary = directory / "summary.json"
    write_canonical_json(
        summary,
        freeze_division_contracts.build_summary_payload(
            role_summary=contracts["split"]["role_summary"],
            source_fingerprints=source_fingerprints,
            manifest_sha256=manifest_hash,
            contract_hashes=contract_hashes,
        ),
    )
    return configs, manifest, summary, rows


def test_high_level_verifier_rejects_fake_source_and_manifest_hashes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    actual_fingerprints = {
        "division_data_sha256": "a" * 64,
        "freeze_division_contracts_sha256": "b" * 64,
        "io_sha256": "c" * 64,
        "metric_sha256": "d" * 64,
    }
    configs, manifest, summary, rows = _write_synthetic_contract_bundle(
        tmp_path / "valid",
        source_fingerprints=actual_fingerprints,
    )
    monkeypatch.setattr(
        "biohub.division_data._current_contract_source_fingerprints",
        lambda _: actual_fingerprints,
    )
    monkeypatch.setattr(
        "biohub.division_data._verify_required_frozen_sources",
        lambda *args, **kwargs: None,
    )

    verified = verify_division_contract_bundle(
        configs,
        manifest_path=manifest,
        summary_path=summary,
        project_root=ROOT,
        expected_manifest_sha256=file_sha256(manifest),
        expected_datasets=tuple(row.dataset for row in rows),
    )
    assert set(verified) == {"split", "proposal", "sampling", "acceptance", "assignment"}

    fake_configs, fake_manifest, _, _ = _write_synthetic_contract_bundle(
        tmp_path / "fake-source",
        source_fingerprints={field: "f" * 64 for field in actual_fingerprints},
    )
    with pytest.raises(ValueError, match="current contract-generation source fingerprint"):
        verify_division_contract_bundle(
            fake_configs,
            manifest_path=fake_manifest,
            project_root=ROOT,
        )

    changed_rows = (replace(rows[0], truth_graph_sha256="f" * 64), *rows[1:])
    changed_manifest = tmp_path / "changed-manifest.csv"
    write_division_manifest(changed_manifest, changed_rows)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify_division_contract_bundle(
            configs,
            manifest_path=changed_manifest,
            project_root=ROOT,
        )

    with pytest.raises(ValueError, match="exact dataset tuple"):
        verify_division_contract_bundle(
            configs,
            manifest_path=manifest,
            project_root=ROOT,
            expected_datasets=tuple(row.dataset for row in rows[:-1]),
        )


def test_high_level_verifier_rejects_stale_or_tampered_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fingerprints = {
        "division_data_sha256": "a" * 64,
        "freeze_division_contracts_sha256": "b" * 64,
        "io_sha256": "c" * 64,
        "metric_sha256": "d" * 64,
    }
    configs, manifest, summary, _ = _write_synthetic_contract_bundle(
        tmp_path / "bundle",
        source_fingerprints=fingerprints,
    )
    monkeypatch.setattr(
        "biohub.division_data._current_contract_source_fingerprints",
        lambda _: fingerprints,
    )
    monkeypatch.setattr(
        "biohub.division_data._verify_required_frozen_sources",
        lambda *args, **kwargs: None,
    )

    stale = read_canonical_json(summary)
    stale["configs"]["sampling"]["sha256"] = "f" * 64
    write_canonical_json(summary, stale)
    with pytest.raises(ValueError, match="summary is stale or tampered"):
        verify_division_contract_bundle(
            configs,
            manifest_path=manifest,
            summary_path=summary,
            project_root=ROOT,
        )

    stale["configs"]["sampling"]["sha256"] = file_sha256(
        configs / "phase11_division_sampling.json"
    )
    stale["source_fingerprints"]["metric_sha256"] = "f" * 64
    write_canonical_json(summary, stale)
    with pytest.raises(ValueError, match="summary is stale or tampered"):
        verify_division_contract_bundle(
            configs,
            manifest_path=manifest,
            summary_path=summary,
            project_root=ROOT,
        )


def test_checked_in_contracts_match_tracked_source_bundles() -> None:
    """This clean-clone audit intentionally fails whenever tracked Task 1 source drifts."""
    contracts = read_division_contracts(ROOT / "configs")
    expected_fingerprints = {
        field: file_sha256(ROOT / path)
        for field, path in division_data_module.CONTRACT_SOURCE_PATHS.items()
    }
    assert all(
        contract["source_fingerprints"] == expected_fingerprints
        for contract in contracts.values()
    )

    phase10 = contracts["proposal"]["phase10_inputs"]
    scoring_paths = [ROOT / path for path in phase10["scoring_source_files"]]
    evaluation_paths = [ROOT / path for path in phase10["evaluation_source_files"]]
    pair_audit = contracts["proposal"]["pair_audit"]
    pair_paths = [ROOT / path for path in pair_audit["source_files"]]
    assert division_data_module.source_bundle_sha256(scoring_paths) == phase10[
        "scoring_source_sha256"
    ]
    assert division_data_module.source_bundle_sha256(evaluation_paths) == phase10[
        "evaluation_source_sha256"
    ]
    assert division_data_module.source_bundle_sha256(pair_paths) == pair_audit["source_sha256"]


def test_local_artifact_bundle_matches_full_frozen_provenance_or_skips() -> None:
    """Run the full local audit before fitting when ignored artifacts are available."""
    manifest = ROOT / "data" / "working" / "phase11_division_manifest.csv"
    summary = ROOT / "data" / "working" / "phase11_division_manifest_summary.json"
    required = [
        manifest,
        summary,
        freeze_division_contracts.DEFAULT_CHECKPOINT,
        freeze_division_contracts.DEFAULT_POLICY,
        freeze_division_contracts.DEFAULT_PAIR_SUMMARY,
        *(
            ROOT / output["path"]
            for output in division_data_module.PAIR_AUDIT_OUTPUTS.values()
        ),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.skip("local Phase 11 artifact audit unavailable: " + ", ".join(map(str, missing)))
    rows = read_division_manifest(manifest)

    contracts = verify_division_contract_bundle(
        ROOT / "configs",
        manifest_path=manifest,
        summary_path=summary,
        project_root=ROOT,
        expected_manifest_sha256=file_sha256(manifest),
        expected_datasets=tuple(row.dataset for row in rows),
    )

    assert tuple(contracts) == ("split", "proposal", "sampling", "acceptance", "assignment")


@pytest.mark.parametrize("drift", ["checkpoint", "scoring", "evaluation"])
def test_freezer_rejects_each_current_phase10_hash_drift(monkeypatch, drift: str) -> None:
    checkpoint_hash = (
        "f" * 64
        if drift == "checkpoint"
        else freeze_division_contracts.EXPECTED_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(
        freeze_division_contracts,
        "phase10_file_sha256",
        lambda _: checkpoint_hash,
    )

    def bundle_hash(paths) -> str:
        if paths == freeze_division_contracts.SCORING_SOURCE_FILES:
            return (
                "f" * 64
                if drift == "scoring"
                else freeze_division_contracts.EXPECTED_SCORING_SOURCE_SHA256
            )
        assert paths == freeze_division_contracts.BASELINE_EVALUATION_SOURCE_FILES
        return (
            "f" * 64
            if drift == "evaluation"
            else freeze_division_contracts.EXPECTED_EVALUATION_SOURCE_SHA256
        )

    monkeypatch.setattr(freeze_division_contracts, "files_sha256", bundle_hash)

    with pytest.raises(ValueError, match=f"Phase 10 {drift}"):
        freeze_division_contracts._verify_phase10_inputs(
            freeze_division_contracts.DEFAULT_CHECKPOINT
        )


def _transaction_fixture(tmp_path: Path):
    transaction = tmp_path / "transaction"
    staged = transaction / "staged"
    destinations = tmp_path / "destinations"
    staged_paths: dict[str, Path] = {}
    destination_paths: dict[str, Path] = {}
    old_bytes: dict[str, bytes] = {}
    new_bytes: dict[str, bytes] = {}
    for name in freeze_division_contracts.PROMOTION_ORDER:
        staged_path = staged / name
        destination = destinations / name
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        old_bytes[name] = f"old-{name}".encode()
        new_bytes[name] = f"new-{name}".encode()
        staged_path.write_bytes(new_bytes[name])
        destination.write_bytes(old_bytes[name])
        staged_paths[name] = staged_path
        destination_paths[name] = destination
    return transaction, staged_paths, destination_paths, old_bytes, new_bytes


@pytest.mark.parametrize("failure", ["promotion", "verification"])
def test_bundle_transaction_restores_old_bytes_after_normal_failure(
    monkeypatch,
    tmp_path: Path,
    failure: str,
) -> None:
    transaction, staged, destinations, old_bytes, new_bytes = _transaction_fixture(tmp_path)
    promoted: list[str] = []
    original_promote = freeze_division_contracts._promote_file

    def promote_with_optional_failure(source: Path, destination: Path) -> None:
        name = next(name for name, path in destinations.items() if path == destination)
        promoted.append(name)
        if failure == "promotion" and name == "summary":
            raise OSError("simulated summary promotion failure")
        original_promote(source, destination)

    monkeypatch.setattr(
        freeze_division_contracts,
        "_promote_file",
        promote_with_optional_failure,
    )

    def validate_staged() -> None:
        assert all(staged[name].read_bytes() == new_bytes[name] for name in staged)

    def validate_final() -> None:
        if failure == "verification":
            raise ValueError("simulated final verification failure")

    with pytest.raises((OSError, ValueError), match="simulated"):
        freeze_division_contracts._promote_staged_bundle(
            transaction,
            staged_paths=staged,
            destinations=destinations,
            validate_staged=validate_staged,
            validate_final=validate_final,
        )

    assert all(destinations[name].read_bytes() == old_bytes[name] for name in destinations)
    assert not transaction.exists()
    if failure == "promotion":
        assert tuple(promoted) == freeze_division_contracts.PROMOTION_ORDER


def test_interrupted_bundle_transaction_is_recovered_on_startup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transaction, staged, destinations, old_bytes, new_bytes = _transaction_fixture(tmp_path)
    original_promote = freeze_division_contracts._promote_file

    def interrupt_promotion(source: Path, destination: Path) -> None:
        name = next(name for name, path in destinations.items() if path == destination)
        if name == "proposal":
            raise KeyboardInterrupt("simulated process interruption")
        original_promote(source, destination)

    monkeypatch.setattr(
        freeze_division_contracts,
        "_promote_file",
        interrupt_promotion,
    )

    with pytest.raises(KeyboardInterrupt, match="simulated"):
        freeze_division_contracts._promote_staged_bundle(
            transaction,
            staged_paths=staged,
            destinations=destinations,
            validate_staged=lambda: None,
            validate_final=lambda: None,
        )

    assert (transaction / "marker.json").is_file()
    current = {name: path.read_bytes() for name, path in destinations.items()}
    assert current != old_bytes
    assert current != new_bytes

    def accept_only_complete_bundle() -> None:
        observed = {name: path.read_bytes() for name, path in destinations.items()}
        if observed not in (old_bytes, new_bytes):
            raise ValueError("mixed bundle")

    with pytest.raises(ValueError, match="mixed bundle"):
        accept_only_complete_bundle()
    monkeypatch.setattr(freeze_division_contracts, "_promote_file", original_promote)
    freeze_division_contracts._recover_transaction(transaction)

    assert all(destinations[name].read_bytes() == old_bytes[name] for name in destinations)
    assert not transaction.exists()
    accept_only_complete_bundle()


def test_cross_process_lock_blocks_second_freezer_without_touching_bundle(tmp_path: Path) -> None:
    working = tmp_path / "working"
    config_dir = tmp_path / "configs"
    manifest = working / "manifest.csv"
    summary = working / "summary.json"
    transaction = working / freeze_division_contracts.TRANSACTION_DIRECTORY_NAME
    transaction.mkdir(parents=True)
    (transaction / "active-staging").write_bytes(b"active-transaction")
    manifest.write_bytes(b"old-manifest")
    summary.write_bytes(b"old-summary")
    config_dir.mkdir()
    config_paths = []
    for kind in freeze_division_contracts.CONFIG_KINDS:
        path = config_dir / f"phase11_division_{kind}.json"
        path.write_bytes(f"old-{kind}".encode())
        config_paths.append(path)
    watched = [transaction / "active-staging", manifest, summary, *config_paths]
    before = {path: path.read_bytes() for path in watched}
    lock_path = freeze_division_contracts.FREEZER_LOCK_PATH
    code = (
        "from pathlib import Path;"
        "from scripts.freeze_division_contracts import freeze_contracts;"
        "\ntry:\n"
        f" freeze_contracts(data_root=Path({str(tmp_path / 'raw')!r}),"
        f"checkpoint_path=Path({str(tmp_path / 'checkpoint')!r}),"
        f"policy_path=Path({str(tmp_path / 'policy')!r}),"
        f"pair_summary_path=Path({str(tmp_path / 'pair-summary')!r}),"
        f"manifest_path=Path({str(manifest)!r}),summary_path=Path({str(summary)!r}),"
        f"config_dir=Path({str(config_dir)!r}),overwrite=True)\n"
        "except RuntimeError as exc:\n print(exc)\n raise SystemExit(23)\n"
    )

    with freeze_division_contracts._exclusive_freezer_lock(lock_path):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    assert result.returncode == 23, result.stderr
    assert "already active" in result.stdout
    assert {path: path.read_bytes() for path in watched} == before


def test_stale_unlocked_lock_file_is_reused_for_recovery(tmp_path: Path) -> None:
    working = tmp_path / "working"
    working.mkdir()
    lock_path = working / freeze_division_contracts.LOCK_FILE_NAME
    lock_path.write_text("stale but unlocked")
    transaction = working / freeze_division_contracts.TRANSACTION_DIRECTORY_NAME
    transaction.mkdir()
    (transaction / "abandoned-stage").write_bytes(b"partial")

    with freeze_division_contracts._exclusive_freezer_lock(lock_path):
        freeze_division_contracts._recover_transaction(transaction)

    assert not transaction.exists()
    with freeze_division_contracts._exclusive_freezer_lock(lock_path):
        pass


def test_freezer_help_does_not_import_optional_torch() -> None:
    script = ROOT / "scripts" / "freeze_division_contracts.py"
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
