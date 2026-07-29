#!/usr/bin/env python3
"""Freeze the Phase 11 6bba truth manifest and all five pre-model contracts."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biohub.appearance_evaluation import (  # noqa: E402
    file_sha256 as phase10_file_sha256,
)
from biohub.appearance_evaluation import (  # noqa: E402
    files_sha256,
)
from biohub.division_data import (  # noqa: E402
    ACCEPTANCE_COHORT_CONTRACT,
    ACCEPTANCE_DIAGNOSTICS,
    ACCEPTANCE_GUARDS,
    ASSIGNMENT_STATIC_CONTRACT,
    BASELINE_EVALUATION_SOURCE_PATHS,
    CONTRACT_SCHEMA_VERSION,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DIVISION_HISTOGRAM,
    EXPECTED_EVALUATION_SOURCE_SHA256,
    EXPECTED_EXPOSED_DATASETS,
    EXPECTED_PAIR_AUDIT_SOURCE_SHA256,
    EXPECTED_PAIR_AUDIT_SUMMARY_SHA256,
    EXPECTED_POLICY_SHA256,
    EXPECTED_SCORING_SOURCE_SHA256,
    LOGICAL_GRAPH_HASH_CONTRACT,
    MANIFEST_CSV_SCHEMA_VERSION,
    PAIR_AUDIT_CONTRACT,
    PAIR_AUDIT_SOURCE_PATHS,
    PHASE10_INPUT_CONTRACT,
    PROPOSAL_RULE_CONTRACT,
    ROLE_NAMES,
    SAMPLING_STATIC_CONTRACT,
    SCORING_SOURCE_PATHS,
    SPLIT_QUOTAS,
    SPLIT_RANK_SALT,
    assign_division_roles,
    canonical_json_sha256,
    canonical_manifest_bytes,
    division_count_manifest,
    division_role_summary,
    file_sha256,
    read_canonical_json,
    source_bundle_sha256,
    validate_6bba_image_inputs,
    validate_division_contract,
    verify_division_contract_bundle,
    write_canonical_json,
    write_division_manifest,
)
from biohub.io import SamplePaths, dataset_name  # noqa: E402

DEFAULT_CHECKPOINT = (
    ROOT
    / "data"
    / "working"
    / "appearance_checkpoints"
    / "cellness_cnn_phase10_full.pt"
)
DEFAULT_POLICY = ROOT / "data" / "working" / "phase10_frozen_candidate_policy.json"
DEFAULT_PAIR_SUMMARY = (
    ROOT / "data" / "working" / "phase11_division_audit" / "full13_pairs" / "summary.json"
)
DEFAULT_MANIFEST = ROOT / "data" / "working" / "phase11_division_manifest.csv"
DEFAULT_SUMMARY = ROOT / "data" / "working" / "phase11_division_manifest_summary.json"
DEFAULT_CONFIG_DIR = ROOT / "configs"
CONFIG_KINDS = ("split", "proposal", "sampling", "acceptance", "assignment")
LOGICAL_MANIFEST_PATH = "data/working/phase11_division_manifest.csv"
LOGICAL_POLICY_PATH = "data/working/phase10_frozen_candidate_policy.json"
SCORING_SOURCE_FILES = [ROOT / path for path in SCORING_SOURCE_PATHS]
BASELINE_EVALUATION_SOURCE_FILES = [
    ROOT / path for path in BASELINE_EVALUATION_SOURCE_PATHS
]
TRANSACTION_SCHEMA_VERSION = 1
TRANSACTION_DIRECTORY_NAME = ".phase11_division_contracts.transaction"
LOCK_FILE_NAME = ".phase11_division_contracts.lock"
FREEZER_LOCK_PATH = ROOT / "data" / "working" / LOCK_FILE_NAME
PROMOTION_ORDER = ("manifest", *CONFIG_KINDS, "summary")


def discover_6bba_samples(data_root: str | Path) -> tuple[SamplePaths, ...]:
    """Discover and validate only 6bba images without opening any labels."""
    train_dir = Path(data_root) / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"missing training data directory: {train_dir}")
    images = sorted(train_dir.glob("6bba_*.zarr"))
    if len(images) != 128:
        raise ValueError(f"expected exactly 128 6bba images, found {len(images)}")
    samples: list[SamplePaths] = []
    for image in images:
        dataset = dataset_name(image)
        labels = train_dir / f"{dataset}.geff"
        samples.append(
            SamplePaths(
                dataset=dataset,
                image=image,
                labels=labels,
                embryo="6bba",
            )
        )
    labels = sorted(train_dir.glob("6bba_*.geff"))
    image_names = {sample.dataset for sample in samples}
    label_names = {dataset_name(path) for path in labels}
    if label_names != image_names:
        raise ValueError(
            "6bba image/label identities differ; "
            f"missing_labels={sorted(image_names - label_names)}, "
            f"orphan_labels={sorted(label_names - image_names)}"
        )
    validate_6bba_image_inputs(samples)
    return tuple(samples)


def _logical_source_paths(paths: list[Path]) -> tuple[str, ...]:
    try:
        return tuple(path.resolve().relative_to(ROOT).as_posix() for path in paths)
    except ValueError as exc:
        raise ValueError("frozen Phase 10 source path is outside the project root") from exc


def _verify_phase10_inputs(checkpoint_path: Path) -> None:
    """Hash current Phase 10 bytes rather than trusting copied provenance metadata."""
    checkpoint_hash = phase10_file_sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Phase 10 checkpoint hash is {checkpoint_hash}, expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    if _logical_source_paths(SCORING_SOURCE_FILES) != SCORING_SOURCE_PATHS:
        raise ValueError("Phase 10 scoring source file tuple has drifted")
    scoring_hash = files_sha256(SCORING_SOURCE_FILES)
    if scoring_hash != EXPECTED_SCORING_SOURCE_SHA256:
        raise ValueError(
            f"Phase 10 scoring source hash is {scoring_hash}, "
            f"expected {EXPECTED_SCORING_SOURCE_SHA256}"
        )
    if (
        _logical_source_paths(BASELINE_EVALUATION_SOURCE_FILES)
        != BASELINE_EVALUATION_SOURCE_PATHS
    ):
        raise ValueError("Phase 10 evaluation source file tuple has drifted")
    evaluation_hash = files_sha256(BASELINE_EVALUATION_SOURCE_FILES)
    if evaluation_hash != EXPECTED_EVALUATION_SOURCE_SHA256:
        raise ValueError(
            f"Phase 10 evaluation source hash is {evaluation_hash}, "
            f"expected {EXPECTED_EVALUATION_SOURCE_SHA256}"
        )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_freezer_lock(lock_path: Path) -> Iterator[None]:
    """Hold one nonblocking cross-process lock for the complete freezer invocation."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Phase 11 contract freezer is already active") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _promote_file(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)


def _transaction_marker_path(transaction_dir: Path) -> Path:
    return transaction_dir / "marker.json"


def _read_transaction_marker(transaction_dir: Path) -> dict[str, Any]:
    marker = read_canonical_json(_transaction_marker_path(transaction_dir))
    if marker.get("transaction_schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 11 contract transaction marker")
    if marker.get("state") not in {"promoting", "committed"}:
        raise RuntimeError("invalid Phase 11 contract transaction state")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or [item.get("name") for item in artifacts] != list(
        PROMOTION_ORDER
    ):
        raise RuntimeError("invalid Phase 11 contract transaction artifact list")
    return marker


def _rollback_transaction(transaction_dir: Path) -> None:
    marker = _read_transaction_marker(transaction_dir)
    for artifact in reversed(marker["artifacts"]):
        destination = Path(artifact["destination"])
        if artifact["existed"]:
            backup = transaction_dir / artifact["backup"]
            if not backup.is_file() or file_sha256(backup) != artifact["old_sha256"]:
                raise RuntimeError(f"cannot restore transaction backup for {artifact['name']}")
            _atomic_write_bytes(destination, backup.read_bytes())
            if file_sha256(destination) != artifact["old_sha256"]:
                raise RuntimeError(f"transaction rollback failed for {artifact['name']}")
        else:
            destination.unlink(missing_ok=True)
    shutil.rmtree(transaction_dir)


def _recover_transaction(transaction_dir: Path) -> None:
    """Deterministically finish cleanup or restore backups from an interrupted run."""
    if not transaction_dir.exists():
        return
    marker_path = _transaction_marker_path(transaction_dir)
    if not marker_path.is_file():
        shutil.rmtree(transaction_dir)
        return
    marker = _read_transaction_marker(transaction_dir)
    if marker["state"] == "committed" and all(
        Path(artifact["destination"]).is_file()
        and file_sha256(artifact["destination"]) == artifact["new_sha256"]
        for artifact in marker["artifacts"]
    ):
        shutil.rmtree(transaction_dir)
        return
    _rollback_transaction(transaction_dir)


def _promote_staged_bundle(
    transaction_dir: Path,
    *,
    staged_paths: Mapping[str, Path],
    destinations: Mapping[str, Path],
    validate_staged: Callable[[], None],
    validate_final: Callable[[], None],
) -> None:
    """Promote one validated bundle with rollback and restart recovery metadata."""
    if set(staged_paths) != set(PROMOTION_ORDER) or set(destinations) != set(
        PROMOTION_ORDER
    ):
        raise ValueError("transaction must contain the complete Phase 11 artifact set")
    try:
        validate_staged()
        backup_dir = transaction_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[dict[str, Any]] = []
        for name in PROMOTION_ORDER:
            staged = staged_paths[name]
            if not staged.is_file():
                raise FileNotFoundError(f"staged transaction artifact is missing: {name}")
            destination = destinations[name].resolve()
            existed = destination.is_file()
            backup_relative = f"backup/{name}"
            old_sha256 = None
            if existed:
                backup = transaction_dir / backup_relative
                _atomic_write_bytes(backup, destination.read_bytes())
                old_sha256 = file_sha256(backup)
            artifacts.append(
                {
                    "backup": backup_relative,
                    "destination": str(destination),
                    "existed": existed,
                    "name": name,
                    "new_sha256": file_sha256(staged),
                    "old_sha256": old_sha256,
                }
            )
        marker = {
            "artifacts": artifacts,
            "state": "promoting",
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
        }
        write_canonical_json(_transaction_marker_path(transaction_dir), marker)
    except Exception:
        shutil.rmtree(transaction_dir, ignore_errors=True)
        raise

    try:
        for name in PROMOTION_ORDER:
            _promote_file(staged_paths[name], destinations[name])
        validate_final()
        marker["state"] = "committed"
        write_canonical_json(_transaction_marker_path(transaction_dir), marker)
    except Exception:
        _rollback_transaction(transaction_dir)
        raise
    shutil.rmtree(transaction_dir, ignore_errors=True)


def _verify_policy(path: Path) -> tuple[str, ...]:
    policy = read_canonical_json(path, expected_sha256=EXPECTED_POLICY_SHA256)
    expected_fields = {
        "candidate_policy_schema_version": 1,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "evaluation_sha256": EXPECTED_EVALUATION_SOURCE_SHA256,
        "selection_cohort": "internal-dev",
    }
    for field, expected in expected_fields.items():
        if policy.get(field) != expected:
            raise ValueError(f"frozen Phase 10 policy has unexpected {field}")
    datasets = policy.get("selection_datasets")
    if datasets != list(EXPECTED_EXPOSED_DATASETS):
        raise ValueError("frozen Phase 10 policy does not contain the exact 13 exposed movies")
    return tuple(datasets)


def _verify_pair_summary(path: Path) -> None:
    summary = read_canonical_json(path, expected_sha256=EXPECTED_PAIR_AUDIT_SUMMARY_SHA256)
    if summary.get("pair_audit_schema_version") != 1:
        raise ValueError("pair-audit summary uses an unsupported schema")
    if summary.get("cohort") != "exact_phase10_internal_dev":
        raise ValueError("pair-audit summary uses the wrong cohort")
    if summary.get("datasets") != list(EXPECTED_EXPOSED_DATASETS):
        raise ValueError("pair-audit summary does not contain the exact exposed movie tuple")
    if summary.get("truth") != {
        "division_movies": 10,
        "labelled_divisions": 18,
        "no_labelled_division_controls": 3,
    }:
        raise ValueError("pair-audit summary has unexpected truth totals")
    provenance = summary.get("provenance")
    expected_provenance = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "frozen_evaluation_source_sha256": EXPECTED_EVALUATION_SOURCE_SHA256,
        "frozen_policy_sha256": EXPECTED_POLICY_SHA256,
        "pair_audit_source_sha256": EXPECTED_PAIR_AUDIT_SOURCE_SHA256,
        "phase10_results_sha256": (
            "acd59f7182050769922df05ae1d143ee1d97edbbb4558b186d8385bd9be82911"
        ),
        "scoring_source_sha256": EXPECTED_SCORING_SOURCE_SHA256,
    }
    if provenance != expected_provenance:
        raise ValueError("pair-audit summary provenance does not match the frozen identities")
    selection = summary.get("selection", {}).get("selected_raw_scored_rule")
    for field, expected in {
        "candidate_source": "raw_scored",
        "labelled_event_coverage": 18,
        "max_candidates_per_mother": 12,
        "max_distance_um": 16.0,
        "min_angle_deg": 0.0,
        "min_distance_um": 0.0,
        "rule_id": "dmax16_cap12_dmin0_angle0",
    }.items():
        if not isinstance(selection, dict) or selection.get(field) != expected:
            raise ValueError(f"pair-audit summary has unexpected selected-rule {field}")
    expected_outputs = {
        "events_csv": PAIR_AUDIT_CONTRACT["outputs"]["events_csv"]["path"],
        "events_sha256": PAIR_AUDIT_CONTRACT["outputs"]["events_csv"]["sha256"],
        "proposals_csv": PAIR_AUDIT_CONTRACT["outputs"]["proposals_csv"]["path"],
        "proposals_sha256": PAIR_AUDIT_CONTRACT["outputs"]["proposals_csv"]["sha256"],
        "rule_movies_csv": PAIR_AUDIT_CONTRACT["outputs"]["rule_movies_csv"]["path"],
        "rule_movies_sha256": PAIR_AUDIT_CONTRACT["outputs"]["rule_movies_csv"]["sha256"],
        "rule_sweep_csv": PAIR_AUDIT_CONTRACT["outputs"]["rule_sweep_csv"]["path"],
        "rule_sweep_sha256": PAIR_AUDIT_CONTRACT["outputs"]["rule_sweep_csv"]["sha256"],
        "selected_movies_csv": PAIR_AUDIT_CONTRACT["outputs"]["selected_movies_csv"]["path"],
        "selected_movies_sha256": PAIR_AUDIT_CONTRACT["outputs"]["selected_movies_csv"]["sha256"],
    }
    if summary.get("outputs") != expected_outputs:
        raise ValueError("pair-audit summary output identities have drifted")
    for output in PAIR_AUDIT_CONTRACT["outputs"].values():
        output_path = ROOT / output["path"]
        if not output_path.is_file():
            raise FileNotFoundError(f"missing frozen pair-audit output: {output['path']}")
        observed = file_sha256(output_path)
        if observed != output["sha256"]:
            raise ValueError(
                f"pair-audit output {output['path']} has SHA-256 {observed}, "
                f"expected {output['sha256']}"
            )

    source_hash = source_bundle_sha256(
        [
            ROOT / "src" / "biohub" / "division.py",
            ROOT / "scripts" / "audit_division_proposals.py",
            ROOT / "scripts" / "audit_division_pair_proposals.py",
        ]
    )
    if source_hash != EXPECTED_PAIR_AUDIT_SOURCE_SHA256:
        raise ValueError(
            f"pair-audit source hash is {source_hash}, expected {EXPECTED_PAIR_AUDIT_SOURCE_SHA256}"
        )


def _source_fingerprints() -> dict[str, str]:
    return {
        "division_data_sha256": file_sha256(ROOT / "src" / "biohub" / "division_data.py"),
        "freeze_division_contracts_sha256": file_sha256(Path(__file__).resolve()),
        "io_sha256": file_sha256(ROOT / "src" / "biohub" / "io.py"),
        "metric_sha256": file_sha256(ROOT / "src" / "biohub" / "metric.py"),
    }


def _quota_json() -> dict[str, dict[str, int]]:
    return {
        str(division_count): {
            "development": quotas[1],
            "evaluation": quotas[2],
            "fit": quotas[0],
        }
        for division_count, quotas in SPLIT_QUOTAS.items()
    }


def build_contract_payloads(
    rows,
    *,
    manifest_sha256: str,
    source_fingerprints: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build and validate the acyclic five-contract identity chain."""
    role_summary = division_role_summary(rows)
    role_datasets = {
        role: sorted(row.dataset for row in rows if row.role == role) for role in ROLE_NAMES
    }
    strata = {
        str(division_count): {
            role: sorted(
                row.dataset
                for row in rows
                if row.division_count == division_count and row.role == role
            )
            for role in ROLE_NAMES
        }
        for division_count in SPLIT_QUOTAS
    }
    split = {
        "cohort": {
            "division_count_histogram": {
                str(key): value for key, value in EXPECTED_DIVISION_HISTOGRAM.items()
            },
            "division_movies": 66,
            "embryo": "6bba",
            "labelled_divisions": 125,
            "movies": 128,
            "no_division_movies": 62,
        },
        "contract_name": "phase11_division_split",
        "cross_contract_sha256": {},
        "division_split_schema_version": CONTRACT_SCHEMA_VERSION,
        "forced_fit": {
            "policy_path": LOGICAL_POLICY_PATH,
            "policy_sha256": EXPECTED_POLICY_SHA256,
            "selection_datasets": list(EXPECTED_EXPOSED_DATASETS),
        },
        "logical_graph_hash": copy.deepcopy(LOGICAL_GRAPH_HASH_CONTRACT),
        "manifest": {
            "csv_schema_version": MANIFEST_CSV_SCHEMA_VERSION,
            "path": LOGICAL_MANIFEST_PATH,
            "sha256": manifest_sha256,
        },
        "quotas_by_division_count": _quota_json(),
        "ranking": {
            "algorithm": "sha256",
            "assignment_order": ["evaluation", "development", "fit"],
            "encoding": "salt_nul_then_dataset_utf8",
            "salt": SPLIT_RANK_SALT,
        },
        "role_datasets": role_datasets,
        "role_summary": role_summary,
        "source_fingerprints": source_fingerprints,
        "strata": strata,
    }
    validate_division_contract("split", split)
    split_hash = canonical_json_sha256(split)

    proposal = {
        "contract_name": "phase11_division_proposal",
        "cross_contract_sha256": {"split": split_hash},
        "division_proposal_schema_version": CONTRACT_SCHEMA_VERSION,
        "pair_audit": copy.deepcopy(PAIR_AUDIT_CONTRACT),
        "phase10_inputs": copy.deepcopy(PHASE10_INPUT_CONTRACT),
        "proposal_rule": copy.deepcopy(PROPOSAL_RULE_CONTRACT),
        "source_fingerprints": source_fingerprints,
    }
    validate_division_contract("proposal", proposal)
    proposal_hash = canonical_json_sha256(proposal)

    sampling = {
        "contract_name": "phase11_division_sampling",
        "cross_contract_sha256": {"proposal": proposal_hash, "split": split_hash},
        "division_sampling_schema_version": CONTRACT_SCHEMA_VERSION,
        "source_fingerprints": source_fingerprints,
        **copy.deepcopy(SAMPLING_STATIC_CONTRACT),
    }
    validate_division_contract("sampling", sampling)
    sampling_hash = canonical_json_sha256(sampling)

    acceptance = {
        "applies_to": ["development_unlock", "locked_evaluation_acceptance"],
        "cohort": copy.deepcopy(ACCEPTANCE_COHORT_CONTRACT),
        "contract_name": "phase11_division_acceptance",
        "cross_contract_sha256": {
            "proposal": proposal_hash,
            "sampling": sampling_hash,
            "split": split_hash,
        },
        "diagnostics": copy.deepcopy(ACCEPTANCE_DIAGNOSTICS),
        "division_acceptance_schema_version": CONTRACT_SCHEMA_VERSION,
        "evaluation_access": {
            "development_must_pass_every_guard": True,
            "evaluation_remains_unread_if_development_fails": True,
            "evaluation_tuning_permitted": False,
        },
        "guards": copy.deepcopy(ACCEPTANCE_GUARDS),
        "source_fingerprints": source_fingerprints,
    }
    validate_division_contract("acceptance", acceptance)
    acceptance_hash = canonical_json_sha256(acceptance)

    assignment = {
        "contract_name": "phase11_division_assignment",
        "cross_contract_sha256": {
            "acceptance": acceptance_hash,
            "proposal": proposal_hash,
            "sampling": sampling_hash,
            "split": split_hash,
        },
        "division_assignment_schema_version": CONTRACT_SCHEMA_VERSION,
        "source_fingerprints": source_fingerprints,
        **copy.deepcopy(ASSIGNMENT_STATIC_CONTRACT),
    }
    validate_division_contract("assignment", assignment)
    return {
        "split": split,
        "proposal": proposal,
        "sampling": sampling,
        "acceptance": acceptance,
        "assignment": assignment,
    }


def build_summary_payload(
    *,
    role_summary: dict[str, dict[str, int]],
    source_fingerprints: dict[str, str],
    manifest_sha256: str,
    contract_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the canonical generated summary without an impossible self-hash."""
    return {
        "cohort": {
            "division_count_histogram": {
                str(key): value for key, value in EXPECTED_DIVISION_HISTOGRAM.items()
            },
            "division_movies": 66,
            "embryo": "6bba",
            "labelled_divisions": 125,
            "movies": 128,
            "no_division_movies": 62,
        },
        "configs": {
            kind: {
                "path": f"configs/phase11_division_{kind}.json",
                "sha256": contract_hashes[kind],
            }
            for kind in CONFIG_KINDS
        },
        "division_manifest_summary_schema_version": 1,
        "outputs": {
            "manifest_csv": {
                "path": LOGICAL_MANIFEST_PATH,
                "sha256": manifest_sha256,
            }
        },
        "role_summary": role_summary,
        "source_fingerprints": source_fingerprints,
    }


def _verify_manifest_facts(rows) -> None:
    histogram = Counter(row.division_count for row in rows)
    if len(rows) != 128:
        raise ValueError(f"truth manifest has {len(rows)} movies, expected 128")
    if dict(sorted(histogram.items())) != EXPECTED_DIVISION_HISTOGRAM:
        raise ValueError(
            f"truth manifest histogram is {dict(sorted(histogram.items()))}, "
            f"expected {EXPECTED_DIVISION_HISTOGRAM}"
        )
    if sum(row.division_count for row in rows) != 125:
        raise ValueError("truth manifest does not contain exactly 125 labelled divisions")
    if sum(not row.no_division for row in rows) != 66:
        raise ValueError("truth manifest does not contain exactly 66 division movies")
    if sum(row.no_division for row in rows) != 62:
        raise ValueError("truth manifest does not contain exactly 62 no-division movies")


def freeze_contracts(
    *,
    data_root: Path,
    checkpoint_path: Path,
    policy_path: Path,
    pair_summary_path: Path,
    manifest_path: Path,
    summary_path: Path,
    config_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    """Acquire the freezer lock before recovery, staging, or destination access."""
    with _exclusive_freezer_lock(FREEZER_LOCK_PATH):
        return _freeze_contracts_locked(
            data_root=data_root,
            checkpoint_path=checkpoint_path,
            policy_path=policy_path,
            pair_summary_path=pair_summary_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            config_dir=config_dir,
            overwrite=overwrite,
        )


def _freeze_contracts_locked(
    *,
    data_root: Path,
    checkpoint_path: Path,
    policy_path: Path,
    pair_summary_path: Path,
    manifest_path: Path,
    summary_path: Path,
    config_dir: Path,
    overwrite: bool,
) -> dict[str, str]:
    """Run the complete pre-model freeze and return generated artifact hashes."""
    transaction_dir = summary_path.parent / TRANSACTION_DIRECTORY_NAME
    _recover_transaction(transaction_dir)
    config_paths = {
        kind: config_dir / f"phase11_division_{kind}.json" for kind in CONFIG_KINDS
    }
    outputs = [manifest_path, summary_path, *config_paths.values()]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite frozen outputs: {joined}")

    _verify_phase10_inputs(checkpoint_path)
    exposed_datasets = _verify_policy(policy_path)
    _verify_pair_summary(pair_summary_path)
    samples = discover_6bba_samples(data_root)
    base_rows = division_count_manifest(samples)
    _verify_manifest_facts(base_rows)
    rows = assign_division_roles(base_rows, exposed_datasets)

    # Compute the identity before promotion so every contract can be validated first.
    manifest_content_hash = hashlib.sha256(canonical_manifest_bytes(rows)).hexdigest()
    fingerprints = _source_fingerprints()
    contracts = build_contract_payloads(
        rows,
        manifest_sha256=manifest_content_hash,
        source_fingerprints=fingerprints,
    )
    contract_hashes = {
        kind: canonical_json_sha256(payload) for kind, payload in contracts.items()
    }
    role_summary = division_role_summary(rows)
    summary = build_summary_payload(
        role_summary=role_summary,
        source_fingerprints=fingerprints,
        manifest_sha256=manifest_content_hash,
        contract_hashes=contract_hashes,
    )
    summary_hash = canonical_json_sha256(summary)

    stage_root = transaction_dir / "staged"
    staged_config_dir = stage_root / "configs"
    staged_paths = {
        "manifest": stage_root / "phase11_division_manifest.csv",
        **{
            kind: staged_config_dir / f"phase11_division_{kind}.json"
            for kind in CONFIG_KINDS
        },
        "summary": stage_root / "phase11_division_manifest_summary.json",
    }
    destinations = {
        "manifest": manifest_path,
        **config_paths,
        "summary": summary_path,
    }
    transaction_dir.mkdir(parents=True)
    try:
        observed_manifest_hash = write_division_manifest(staged_paths["manifest"], rows)
        if observed_manifest_hash != manifest_content_hash:
            raise RuntimeError("manifest hash changed during transaction staging")
        for kind in CONFIG_KINDS:
            observed = write_canonical_json(staged_paths[kind], contracts[kind])
            if observed != contract_hashes[kind]:
                raise RuntimeError(f"{kind} contract hash changed during transaction staging")
        observed_summary_hash = write_canonical_json(staged_paths["summary"], summary)
        if observed_summary_hash != summary_hash:
            raise RuntimeError("summary hash changed during transaction staging")
    except Exception:
        shutil.rmtree(transaction_dir, ignore_errors=True)
        raise

    verification_options = {
        "project_root": ROOT,
        "expected_manifest_sha256": manifest_content_hash,
        "expected_datasets": tuple(sample.dataset for sample in samples),
        "checkpoint_path": checkpoint_path,
        "policy_path": policy_path,
        "pair_summary_path": pair_summary_path,
        "scoring_source_files": SCORING_SOURCE_FILES,
        "evaluation_source_files": BASELINE_EVALUATION_SOURCE_FILES,
        "pair_audit_source_files": tuple(ROOT / path for path in PAIR_AUDIT_SOURCE_PATHS),
    }

    def validate_staged() -> None:
        verify_division_contract_bundle(
            staged_config_dir,
            manifest_path=staged_paths["manifest"],
            summary_path=staged_paths["summary"],
            **verification_options,
        )

    def validate_final() -> None:
        verify_division_contract_bundle(
            config_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            **verification_options,
        )

    _promote_staged_bundle(
        transaction_dir,
        staged_paths=staged_paths,
        destinations=destinations,
        validate_staged=validate_staged,
        validate_final=validate_final,
    )
    return {
        "manifest": manifest_content_hash,
        **contract_hashes,
        "summary": summary_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--pair-summary", type=Path, default=DEFAULT_PAIR_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        hashes = freeze_contracts(
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            policy_path=args.policy,
            pair_summary_path=args.pair_summary,
            manifest_path=args.manifest,
            summary_path=args.summary,
            config_dir=args.config_dir,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("Frozen Phase 11 division contracts")
    print("Movies: 128 (88 fit / 20 development / 20 evaluation)")
    print("Labelled divisions: 125 (89 / 18 / 18)")
    for name, digest in hashes.items():
        print(f"{name}_sha256={digest}")


if __name__ == "__main__":
    main()
