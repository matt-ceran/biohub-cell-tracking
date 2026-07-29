from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from biohub.division_dataset import (
    BUILD_MODE_FIT_COMPLETE,
    BUILD_MODE_SMOKE,
    DIAGNOSTIC_UNKNOWN,
    LABEL_UNKNOWN,
    MATCHED_NONDIVISION_CONTROL,
    POSITIVE,
    SAME_MOTHER_NEGATIVE,
    LoadedDivisionDataset,
    ProposalRecord,
    fit_feature_schema,
)
from biohub.division_model import (
    CONTEXT_FEATURE_KINDS,
    CONTEXT_FEATURE_NAMES,
    DIVISION_CHECKPOINT_SCHEMA_VERSION,
    DIVISION_MODEL_NAMES,
    EXPECTED_DIVISION_CONTRACT_SHA256,
    FIXED_DIVISION_MODEL_CONFIGS,
    DivisionModel,
    load_division_checkpoint,
)
from scripts import train_division

ROOT = Path(__file__).resolve().parents[1]


def _record(index: int, stratum: str) -> ProposalRecord:
    label, weight = {
        POSITIVE: (1, 1.0),
        SAME_MOTHER_NEGATIVE: (0, 1.0),
        MATCHED_NONDIVISION_CONTROL: (0, 1.0),
        DIAGNOSTIC_UNKNOWN: (LABEL_UNKNOWN, 0.0),
    }[stratum]
    context = np.zeros(len(CONTEXT_FEATURE_NAMES), dtype=np.float32)
    continuous = np.asarray([kind == "continuous" for kind in CONTEXT_FEATURE_KINDS])
    context[continuous] = np.float32(index / 10.0)
    image = np.linspace(-0.25, 0.25, 256, dtype=np.float32) + np.float32(index / 100.0)
    return ProposalRecord(
        dataset="6bba_synthetic",
        mother_id=1000 + index,
        candidate_1_id=10_000 + 2 * index,
        candidate_2_id=10_001 + 2 * index,
        mother_node_index=20_000 + 3 * index,
        candidate_1_node_index=20_001 + 3 * index,
        candidate_2_node_index=20_002 + 3 * index,
        sampling_stratum=stratum,
        source_role="fit",
        label=label,
        loss_weight=weight,
        minimum_cellness=0.5,
        restored_node_count=0,
        edge_removal_count=0,
        deterministic_rank=hashlib.sha256(str(index).encode("ascii")).hexdigest(),
        raw_context=tuple(float(value) for value in context),
        image_features=tuple(float(value) for value in image),
    )


def _loaded_dataset(*, build_mode: str = BUILD_MODE_SMOKE) -> LoadedDivisionDataset:
    strata = (
        POSITIVE,
        POSITIVE,
        SAME_MOTHER_NEGATIVE,
        SAME_MOTHER_NEGATIVE,
        SAME_MOTHER_NEGATIVE,
        MATCHED_NONDIVISION_CONTROL,
        DIAGNOSTIC_UNKNOWN,
    )
    records = tuple(_record(index, stratum) for index, stratum in enumerate(strata))
    schema = fit_feature_schema(records)
    raw_context = np.asarray([record.raw_context for record in records], dtype=np.float32)
    image_features = np.asarray(
        [record.image_features for record in records], dtype=np.float32
    )
    normalized_context = np.ascontiguousarray(schema.standardize(raw_context), dtype=np.float32)
    image_features.setflags(write=False)
    normalized_context.setflags(write=False)
    counts = Counter(record.sampling_stratum for record in records)
    manifest = {
        "build_mode": build_mode,
        "contract_sha256": dict(EXPECTED_DIVISION_CONTRACT_SHA256),
        "datasets": ["6bba_synthetic"],
        "sample_tuple_sha256": "b" * 64,
        "stratum_counts": {
            stratum: counts[stratum]
            for stratum in (
                POSITIVE,
                SAME_MOTHER_NEGATIVE,
                MATCHED_NONDIVISION_CONTROL,
                DIAGNOSTIC_UNKNOWN,
            )
        },
    }
    return LoadedDivisionDataset(
        records=records,
        feature_schema=schema,
        image_features=image_features,
        manifest=MappingProxyType(manifest),
        dataset_manifest_sha256="a" * 64,
        model_inputs_sha256="c" * 64,
        normalized_context=normalized_context,
    )


def _accepted_base_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    checkpoint = tmp_path / "cellness.pt"
    checkpoint.write_bytes(b"accepted synthetic base checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr("biohub.division_model.EXPECTED_BASE_CHECKPOINT_SHA256", digest)
    return checkpoint


def _smoke_settings(epochs: int) -> train_division.TrainingSettings:
    return train_division.TrainingSettings.for_mode(nonproduction=True, epochs=epochs)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_fixed_ladder_and_training_settings_are_not_configurable() -> None:
    assert train_division.MODEL_NAMES == DIVISION_MODEL_NAMES == (
        "scalar",
        "embedding_linear",
        "embedding_mlp32",
    )
    assert train_division.SEED == 0
    assert train_division.LEARNING_RATE == 0.001
    assert train_division.WEIGHT_DECAY == 0.001
    assert train_division.BATCH_SIZE == 64
    assert train_division.GRADIENT_CLIP_NORM == 1.0
    assert train_division.PRODUCTION_EPOCHS == 100
    assert train_division.NUM_WORKERS == 0

    production = train_division.TrainingSettings.for_mode(nonproduction=False, epochs=None)
    assert production.epochs == 100
    assert production.optimizer == "AdamW"
    assert production.loss == "BCEWithLogitsLoss"
    with pytest.raises(ValueError, match="production training requires exactly 100 epochs"):
        train_division.TrainingSettings.for_mode(nonproduction=False, epochs=2)
    assert _smoke_settings(2).epochs == 2

    parser = train_division.build_parser()
    assert parser.parse_args(["--all"]).all is True
    assert parser.parse_args(["--model", "scalar"]).model == "scalar"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "other"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--all", "--model", "scalar"])


@pytest.mark.parametrize("model_name", DIVISION_MODEL_NAMES)
def test_optimizer_contains_head_parameters_only(model_name: str) -> None:
    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS[model_name])
    optimizer = train_division.build_optimizer(model)

    train_division.validate_head_only_optimizer(model, optimizer)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in model.head.parameters()}
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == 0.001

    optimizer.add_param_group({"params": [torch.nn.Parameter(torch.zeros(()))]})
    with pytest.raises(ValueError, match="head parameters only"):
        train_division.validate_head_only_optimizer(model, optimizer)

    optimizer = train_division.build_optimizer(model)
    optimizer.param_groups[0]["lr"] = 0.5
    with pytest.raises(ValueError, match="fixed AdamW"):
        train_division.validate_head_only_optimizer(model, optimizer)


def test_epoch_batches_are_deterministic_balanced_and_exclude_unknown() -> None:
    loaded = _loaded_dataset()
    first = train_division.epoch_batches(loaded.records, epoch=7, batch_size=4)
    second = train_division.epoch_batches(loaded.records, epoch=7, batch_size=4)
    changed_epoch = train_division.epoch_batches(loaded.records, epoch=8, batch_size=4)

    assert [batch.tolist() for batch in first] == [batch.tolist() for batch in second]
    assert [batch.tolist() for batch in first] != [batch.tolist() for batch in changed_epoch]
    indices = np.concatenate(first)
    counts = Counter(loaded.records[int(index)].sampling_stratum for index in indices)
    assert len(indices) == 9
    assert counts == {
        POSITIVE: 3,
        SAME_MOTHER_NEGATIVE: 3,
        MATCHED_NONDIVISION_CONTROL: 3,
    }
    assert all(loaded.records[int(index)].label in (0, 1) for index in indices)
    assert all(loaded.records[int(index)].loss_weight == 1.0 for index in indices)


def test_run_epoch_records_only_three_supervised_stratum_diagnostics() -> None:
    loaded = _loaded_dataset()
    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS["embedding_linear"])
    optimizer = train_division.build_optimizer(model)

    metrics = train_division.run_epoch(
        model,
        optimizer,
        loaded,
        epoch=1,
        settings=_smoke_settings(1),
        device=torch.device("cpu"),
    )

    assert np.isfinite(metrics["loss"])
    assert set(metrics["strata"]) == {
        POSITIVE,
        SAME_MOTHER_NEGATIVE,
        MATCHED_NONDIVISION_CONTROL,
    }
    assert {value["count"] for value in metrics["strata"].values()} == {3}
    assert "unknown" not in json.dumps(metrics)


def test_artifacts_are_separate_and_fresh_start_refuses_any_existing_file(
    tmp_path: Path,
) -> None:
    paths = train_division.model_artifact_paths(tmp_path, "scalar")
    assert len({paths.checkpoint, paths.history, paths.state, paths.lock}) == 4
    assert paths.checkpoint.name == "scalar.pt"
    assert paths.history.name == "scalar.history.json"
    assert paths.state.name == "scalar.state.pt"

    for existing in (paths.checkpoint, paths.history, paths.state):
        existing.write_bytes(b"existing")
        with pytest.raises(FileExistsError, match="already exists"):
            train_division.validate_artifact_start(paths, resume=False)
        existing.unlink()

    paths.history.write_bytes(b"orphan")
    with pytest.raises(FileNotFoundError, match="resume state"):
        train_division.validate_artifact_start(paths, resume=True)


def test_atomic_writes_preserve_destination_and_clean_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "state.pt"
    destination.write_bytes(b"known-good")

    def fail_after_partial_write(payload, handle) -> None:
        handle.write(b"partial")
        raise OSError("injected save failure")

    monkeypatch.setattr(train_division.torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        train_division.atomic_torch_save(destination, {"value": torch.ones(1)})

    assert destination.read_bytes() == b"known-good"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.pt"]


def test_atomic_state_remains_resumable_if_history_write_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = _accepted_base_checkpoint(monkeypatch, tmp_path)
    output = tmp_path / "history-failure"
    original_write = train_division.atomic_json_write
    failed = False

    def fail_first_history(path, payload):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected history failure")
        return original_write(path, payload)

    monkeypatch.setattr(train_division, "atomic_json_write", fail_first_history)
    with pytest.raises(OSError, match="injected history"):
        train_division.train_one_model(
            _loaded_dataset(),
            model_name="scalar",
            output_dir=output,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            settings=_smoke_settings(2),
            device=torch.device("cpu"),
            resume=False,
        )

    paths = train_division.model_artifact_paths(output, "scalar")
    assert paths.state.is_file()
    assert not paths.history.exists()
    monkeypatch.setattr(train_division, "atomic_json_write", original_write)
    result = train_division.train_one_model(
        _loaded_dataset(),
        model_name="scalar",
        output_dir=output,
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        settings=_smoke_settings(2),
        device=torch.device("cpu"),
        resume=True,
    )
    assert result.completed_epochs == 2
    assert json.loads(paths.history.read_text())["completed_epochs"] == 2


def test_model_lock_rejects_contention_reuses_stale_file_and_releases_on_failure(
    tmp_path: Path,
) -> None:
    paths = train_division.model_artifact_paths(tmp_path, "scalar")

    with train_division.exclusive_model_lock(paths):
        with pytest.raises(RuntimeError, match="already active"):
            with train_division.exclusive_model_lock(paths):
                pass

    paths.lock.write_text("stale but unlocked")
    with pytest.raises(RuntimeError, match="injected"):
        with train_division.exclusive_model_lock(paths):
            raise RuntimeError("injected")
    with train_division.exclusive_model_lock(paths):
        pass


def test_nonfinite_logits_gradients_and_weights_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded_dataset()
    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS["scalar"])
    optimizer = train_division.build_optimizer(model)
    with torch.no_grad():
        next(model.head.parameters()).fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="logits|weights"):
        train_division.run_epoch(
            model,
            optimizer,
            loaded,
            epoch=1,
            settings=_smoke_settings(1),
            device=torch.device("cpu"),
        )

    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS["scalar"])
    optimizer = train_division.build_optimizer(model)
    original_backward = torch.Tensor.backward

    def nonfinite_backward(self, *args, **kwargs):
        original_backward(self, *args, **kwargs)
        next(model.head.parameters()).grad.fill_(float("nan"))

    monkeypatch.setattr(torch.Tensor, "backward", nonfinite_backward)
    with pytest.raises(FloatingPointError, match="gradients"):
        train_division.run_epoch(
            model,
            optimizer,
            loaded,
            epoch=1,
            settings=_smoke_settings(1),
            device=torch.device("cpu"),
        )


def test_reader_modes_are_separate_and_production_rejects_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke = _loaded_dataset()
    calls: list[str] = []

    def production_reader(*args, **kwargs):
        calls.append("production")
        return smoke

    def smoke_reader(*args, **kwargs):
        calls.append("smoke")
        return smoke

    monkeypatch.setattr(train_division, "read_division_dataset", production_reader)
    monkeypatch.setattr(train_division, "read_smoke_division_dataset", smoke_reader)
    inputs = train_division.ReaderInputs.synthetic(tmp_path)

    with pytest.raises(ValueError, match="fit_complete"):
        train_division.read_training_dataset(inputs, nonproduction=False)
    assert calls == ["production"]
    assert train_division.read_training_dataset(inputs, nonproduction=True) is smoke
    assert calls == ["production", "smoke"]

    fit = _loaded_dataset(build_mode=BUILD_MODE_FIT_COMPLETE)
    monkeypatch.setattr(train_division, "read_smoke_division_dataset", lambda *a, **k: fit)
    with pytest.raises(ValueError, match="smoke"):
        train_division.read_training_dataset(inputs, nonproduction=True)


def test_smoke_checkpoint_is_head_only_marked_nonproduction_and_not_loadable_as_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = _accepted_base_checkpoint(monkeypatch, tmp_path)
    output = tmp_path / "smoke-output"
    result = train_division.train_one_model(
        _loaded_dataset(),
        model_name="scalar",
        output_dir=output,
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        settings=_smoke_settings(1),
        device=torch.device("cpu"),
        resume=False,
    )

    payload = torch.load(result.paths.checkpoint, map_location="cpu", weights_only=True)
    assert payload["division_checkpoint_schema_version"] == DIVISION_CHECKPOINT_SCHEMA_VERSION
    assert payload["nonproduction"] is True
    assert payload["production_consumable"] is False
    assert payload["build_mode"] == BUILD_MODE_SMOKE
    assert payload["epoch"] == 1
    assert "head_state_dict" in payload
    assert "model_state_dict" not in payload
    assert set(payload["model_config"]) == {
        "model_name",
        "embedding_dim",
        "context_dim",
        "hidden_dim",
        "dropout",
    }
    history = json.loads(result.paths.history.read_text())
    state = torch.load(result.paths.state, map_location="cpu", weights_only=True)
    assert history["completed_epochs"] == state["last_epoch"] == 1
    assert history["nonproduction"] is state["nonproduction"] is True

    with pytest.raises(ValueError, match="wrong fields|unexpected"):
        load_division_checkpoint(
            result.paths.checkpoint,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            feature_schema=_loaded_dataset().feature_schema,
            expected_dataset_manifest_sha256="a" * 64,
            expected_model_inputs_sha256="c" * 64,
            expected_trainer_source_sha256=result.trainer_source_sha256,
            expected_model_config=FIXED_DIVISION_MODEL_CONFIGS["scalar"],
        )


def test_resume_rejects_signature_change_and_missing_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = _accepted_base_checkpoint(monkeypatch, tmp_path)
    output = tmp_path / "interrupted"

    def interrupt(epoch: int) -> None:
        raise RuntimeError(f"interrupt after {epoch}")

    with pytest.raises(RuntimeError, match="interrupt after 1"):
        train_division.train_one_model(
            _loaded_dataset(),
            model_name="embedding_linear",
            output_dir=output,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            settings=_smoke_settings(3),
            device=torch.device("cpu"),
            resume=False,
            after_epoch=interrupt,
        )

    paths = train_division.model_artifact_paths(output, "embedding_linear")
    assert paths.history.is_file()
    assert paths.state.is_file()
    assert not paths.checkpoint.exists()
    monkeypatch.setattr(train_division, "trainer_source_sha256", lambda *args: "f" * 64)
    with pytest.raises(ValueError, match="resume signature"):
        train_division.train_one_model(
            _loaded_dataset(),
            model_name="embedding_linear",
            output_dir=output,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            settings=_smoke_settings(3),
            device=torch.device("cpu"),
            resume=True,
        )

    paths.state.unlink()
    with pytest.raises(FileNotFoundError, match="resume state"):
        train_division.train_one_model(
            _loaded_dataset(),
            model_name="embedding_linear",
            output_dir=output,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            settings=_smoke_settings(3),
            device=torch.device("cpu"),
            resume=True,
        )


def test_interrupted_and_continuous_cpu_training_are_exactly_equivalent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_checkpoint = _accepted_base_checkpoint(monkeypatch, tmp_path)
    loaded = _loaded_dataset()
    settings = _smoke_settings(4)
    continuous = train_division.train_one_model(
        loaded,
        model_name="embedding_mlp32",
        output_dir=tmp_path / "continuous",
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        settings=settings,
        device=torch.device("cpu"),
        resume=False,
    )

    class SimulatedInterruption(RuntimeError):
        pass

    def interrupt_after_two(epoch: int) -> None:
        if epoch == 2:
            raise SimulatedInterruption

    interrupted_dir = tmp_path / "resumed"
    with pytest.raises(SimulatedInterruption):
        train_division.train_one_model(
            loaded,
            model_name="embedding_mlp32",
            output_dir=interrupted_dir,
            base_checkpoint_path=base_checkpoint,
            config_dir=ROOT / "configs",
            settings=settings,
            device=torch.device("cpu"),
            resume=False,
            after_epoch=interrupt_after_two,
        )
    resumed = train_division.train_one_model(
        loaded,
        model_name="embedding_mlp32",
        output_dir=interrupted_dir,
        base_checkpoint_path=base_checkpoint,
        config_dir=ROOT / "configs",
        settings=settings,
        device=torch.device("cpu"),
        resume=True,
    )

    assert continuous.checkpoint_sha256 == resumed.checkpoint_sha256
    continuous_checkpoint = torch.load(
        continuous.paths.checkpoint, map_location="cpu", weights_only=True
    )
    resumed_checkpoint = torch.load(
        resumed.paths.checkpoint, map_location="cpu", weights_only=True
    )
    _assert_nested_equal(
        continuous_checkpoint["head_state_dict"], resumed_checkpoint["head_state_dict"]
    )
    assert continuous_checkpoint["metrics"] == resumed_checkpoint["metrics"]

    continuous_history = json.loads(continuous.paths.history.read_text())
    resumed_history = json.loads(resumed.paths.history.read_text())
    assert continuous_history["history"] == resumed_history["history"]
    continuous_state = torch.load(continuous.paths.state, map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed.paths.state, map_location="cpu", weights_only=True)
    _assert_nested_equal(
        continuous_state["head_state_dict"], resumed_state["head_state_dict"]
    )
    _assert_nested_equal(
        continuous_state["optimizer_state_dict"], resumed_state["optimizer_state_dict"]
    )
    assert continuous_state["torch_cpu_rng_state"].equal(
        resumed_state["torch_cpu_rng_state"]
    )
    assert continuous_state["numpy_generator_state"] == resumed_state[
        "numpy_generator_state"
    ]


def test_production_final_checkpoint_uses_schema_v2_api_at_epoch_100(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = _loaded_dataset(build_mode=BUILD_MODE_FIT_COMPLETE)
    model = DivisionModel(FIXED_DIVISION_MODEL_CONFIGS["scalar"])
    paths = train_division.model_artifact_paths(tmp_path, "scalar")
    captured: dict[str, object] = {}

    def capture(path, passed_model, **kwargs):
        captured.update(kwargs)
        captured["path"] = path
        captured["model"] = passed_model
        Path(path).write_bytes(b"schema-v2")
        return hashlib.sha256(b"schema-v2").hexdigest()

    monkeypatch.setattr(train_division, "save_division_checkpoint", capture)
    digest = train_division.save_final_checkpoint(
        paths,
        model,
        loaded,
        base_checkpoint_path=tmp_path / "base.pt",
        config_dir=ROOT / "configs",
        settings=train_division.TrainingSettings.for_mode(
            nonproduction=False,
            epochs=None,
        ),
        trainer_source_hash="d" * 64,
        metrics={"loss": 0.25},
    )

    assert digest == hashlib.sha256(b"schema-v2").hexdigest()
    assert captured["path"] == paths.checkpoint
    assert captured["model"] is model
    assert captured["epoch"] == 100
    assert captured["dataset_manifest_sha256"] == loaded.dataset_manifest_sha256
    assert captured["model_inputs_sha256"] == loaded.model_inputs_sha256
    assert captured["feature_schema"] == loaded.feature_schema
    assert captured["trainer_source_sha256"] == "d" * 64


def test_cli_help_is_clean_clone_safe_without_optional_torch() -> None:
    script = ROOT / "scripts" / "train_division.py"
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
    assert "--smoke-dataset" in result.stdout
    assert "--all" in result.stdout
    assert "evaluation" not in result.stdout
