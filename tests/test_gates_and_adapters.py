"""Gates and adapters that guard expensive or irreversible steps. None of these tests needs a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stair_engine.dataset.preflight import DatasetGateError, check_dataset_gates
from stair_engine.synthetic.llm import LLMConfigurationError, _is_retryable, client_from_environment
from stair_engine.training.upstream import TrainingOverrides, checkpoint_dirs, run_upstream_training


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(("status", "expected"), [(429, True), (503, True), (408, True), (400, False), (404, False)])
def test_retry_only_transient_statuses(status, expected):
    assert _is_retryable(_StatusError(status)) is expected


def test_connection_errors_are_retryable_by_name():
    class APIConnectionError(Exception):
        pass

    assert _is_retryable(APIConnectionError("refused"))
    assert not _is_retryable(ValueError("bad input"))


def test_missing_endpoint_is_a_configuration_error(monkeypatch):
    for name in ("STAIR_LLM_API_BASE", "STAIR_LLM_API_KEY", "STAIR_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMConfigurationError, match="STAIR_LLM_API_BASE"):
        client_from_environment()


def test_endpoint_configuration_comes_from_environment_only(monkeypatch):
    monkeypatch.setenv("STAIR_LLM_API_BASE", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("STAIR_LLM_API_KEY", "local")
    monkeypatch.setenv("STAIR_LLM_MODEL", "example-model")
    client = client_from_environment(extra_body={"reasoning_effort": "none"})
    assert client.base_url == "http://127.0.0.1:11434/v1"
    assert client.model == "example-model"
    assert client.extra_body == {"reasoning_effort": "none"}


def test_training_refuses_a_run_directory_that_already_has_checkpoints(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-8").mkdir(parents=True)
    assert checkpoint_dirs(run)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.jsonl", "val.jsonl", "toc.json"):
        (data / name).write_text("", encoding="utf-8")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh run directory"):
        run_upstream_training(upstream_root=upstream, data_dir=data, run_dir=run, overrides=TrainingOverrides())


def test_training_requires_the_upstream_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="setup_upstream"):
        run_upstream_training(
            upstream_root=tmp_path / "missing", data_dir=tmp_path, run_dir=tmp_path / "run", overrides=TrainingOverrides()
        )


def test_overrides_serialise_to_the_upstream_keys():
    overrides = TrainingOverrides(num_train_epochs=5, lora_rank=8).to_override_yaml()
    assert overrides["training"]["num_train_epochs"] == 5
    assert overrides["model"]["lora"]["rank"] == 8
    assert set(overrides) == {"model", "training"}


def _dataset_manifest(tmp_path: Path, *, passed: bool, covered: bool) -> Path:
    manifest = {
        "fingerprint_sha256": "abc",
        "structure": {"validation_passed": passed},
        "coverage_passed": covered,
        "uncovered_leaves": [] if covered else [{"leaf_id": "n0001", "section": "1.1 Alpha"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_gate_refuses_failed_structure(tmp_path):
    _dataset_manifest(tmp_path, passed=False, covered=True)
    with pytest.raises(DatasetGateError, match="validation errors"):
        check_dataset_gates(tmp_path)


def test_gate_refuses_uncovered_leaves(tmp_path):
    _dataset_manifest(tmp_path, passed=True, covered=False)
    with pytest.raises(DatasetGateError, match="no training query"):
        check_dataset_gates(tmp_path)


def test_gate_passes_a_clean_dataset(tmp_path):
    _dataset_manifest(tmp_path, passed=True, covered=True)
    assert check_dataset_gates(tmp_path)["fingerprint_sha256"] == "abc"
