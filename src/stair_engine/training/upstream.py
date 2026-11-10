"""Run the pinned upstream trainer (`stair train`) unmodified, and record what it did.

The upstream repository is never copied into this project. It lives in a separate checkout at
the commit pinned in ``UPSTREAM_COMMIT``, and it runs in its own uv environment. This module
only writes an override YAML, starts the upstream CLI, samples GPU memory and writes a manifest.

Upstream behaviour that this module guards against:

* `stair train` returns success without training when the run directory already contains a
  ``checkpoint-*`` directory. Every run therefore gets a fresh directory, which is checked first.
* Training needs a CUDA GPU. The vendored trainer moves the model to "cuda" unconditionally.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

UPSTREAM_REPOSITORY = "https://github.com/meghanadhpulivarthi/stair-clean.git"
UPSTREAM_COMMIT = "3729c54f54e6e28aa3a4d714fc7ff8a3af72c0f3"


@dataclass
class TrainingOverrides:
    """The subset of configs/base.yaml this project varies. Everything else keeps upstream defaults."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 1.0e-4
    num_train_epochs: int = 3
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    eval_interval: int = 30
    save_interval: int = 30
    early_stopping_patience: int = 5
    warmup_steps: int = 20

    def to_override_yaml(self) -> dict:
        return {
            "model": {
                "name": self.model_name,
                "lora": {"rank": self.lora_rank, "alpha": self.lora_alpha, "dropout": self.lora_dropout},
            },
            "training": {
                "learning_rate": self.learning_rate,
                "num_train_epochs": self.num_train_epochs,
                "batch_size_per_gpu": self.batch_size_per_gpu,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_seq_length": self.max_seq_length,
                "eval_interval": self.eval_interval,
                "save_interval": self.save_interval,
                "early_stopping_patience": self.early_stopping_patience,
                "warmup_steps": self.warmup_steps,
            },
        }


@dataclass
class GpuSampler:
    interval_seconds: float = 2.0
    samples: list[tuple[str, int]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            used = query_gpu_memory_used_mib()
            if used is not None:
                self.samples.append((dt.datetime.now(dt.timezone.utc).isoformat(), used))
            self._stop.wait(self.interval_seconds)

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        peaks = [used for _, used in self.samples]
        return max(peaks) if peaks else None

    def write_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_utc", "gpu_memory_used_mib"])
            writer.writerows(self.samples)


def query_gpu_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    return int(line) if line.isdigit() else None


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.glob("checkpoint-*") if path.is_dir())


def run_upstream_training(
    *,
    upstream_root: str | Path,
    data_dir: str | Path,
    run_dir: str | Path,
    overrides: TrainingOverrides,
    uv_binary: str = "uv",
    dataset_fingerprint: str = "",
    prompt_contract_sha256: str = "",
) -> dict:
    upstream = Path(upstream_root)
    data = Path(data_dir).resolve()
    run = Path(run_dir).resolve()

    if not (upstream / "pyproject.toml").exists():
        raise FileNotFoundError(f"upstream checkout not found at {upstream}; run scripts/setup_upstream.sh")
    for required in ("train.jsonl", "val.jsonl", "toc.json"):
        if not (data / required).exists():
            raise FileNotFoundError(f"{data} is missing {required}")

    if run.exists() and checkpoint_dirs(run):
        raise FileExistsError(
            f"{run} already contains checkpoints. Upstream would silently skip training there, "
            "so choose a fresh run directory."
        )
    run.mkdir(parents=True, exist_ok=True)

    override_path = run / "override.yaml"
    override_path.write_text(yaml.safe_dump(overrides.to_override_yaml(), sort_keys=False), encoding="utf-8")

    command = [
        uv_binary,
        "run",
        "--project",
        str(upstream),
        "stair",
        "train",
        "--data",
        str(data),
        "--out",
        str(run),
        "--config",
        str(override_path),
    ]
    # The upstream project must run in its own environment. If the caller's shell exports
    # UV_PROJECT_ENVIRONMENT (this project does), uv would install upstream's dependencies into
    # this project's venv, and both projects would break. Pin the upstream .venv explicitly.
    upstream_env = dict(os.environ)
    upstream_env["UV_PROJECT_ENVIRONMENT"] = str(upstream / ".venv")
    started = time.time()
    sampler = GpuSampler()
    sampler.start()
    try:
        with open(run / "train.log", "w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False, env=upstream_env
            )
    finally:
        peak_mib = sampler.stop()
        sampler.write_csv(run / "gpu_memory.csv")
    duration = time.time() - started

    log_text = (run / "train.log").read_text(encoding="utf-8", errors="replace")
    best = run / "checkpoint-best"
    adapter_ok = (best / "adapter_config.json").exists() and any(
        best.glob("adapter_model*.safetensors")
    )
    manifest = {
        "run_id": run.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT, "checkout": str(upstream)},
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 1),
        "peak_gpu_memory_mib": peak_mib,
        "gpu_total_mib": None,
        "overrides": asdict(overrides),
        "dataset_fingerprint": dataset_fingerprint,
        "prompt_contract_sha256": prompt_contract_sha256,
        "checkpoints": [path.name for path in checkpoint_dirs(run)],
        "checkpoint_best_present": best.exists(),
        "adapter_saved": adapter_ok,
        "boundary_mismatch_warnings": log_text.count("Mismatch between tokenized prompt"),
        "success": completed.returncode == 0 and adapter_ok,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def remove_run_directory(run_dir: str | Path) -> None:
    """Only used by tests on directories they created themselves."""

    shutil.rmtree(run_dir, ignore_errors=True)
