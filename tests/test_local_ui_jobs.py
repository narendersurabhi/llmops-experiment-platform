from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from components.ui.local_jobs import (
    LocalJobSpec,
    REPO_ROOT,
    build_local_eval_job,
    build_local_training_job,
    default_eval_model_uri,
    poll_local_job,
    read_job_output,
    start_local_job,
    stop_local_job,
)


def test_default_eval_model_uri_for_local_path():
    assert default_eval_model_uri(None, "artifacts/model") == f"file://{(REPO_ROOT / 'artifacts/model').resolve()}"


def test_build_local_training_job_sets_expected_command_and_env():
    job = build_local_training_job(
        run_config_path="configs/training/smoke_qwen2_5_0_5b.cpt.json",
        dataset_metadata_path="dataset/metadata.smoke_qwen2_5_0_5b.json",
        output_dir="artifacts/local-train/ui-test",
        hf_home=".cache/hf",
    )

    assert job.command[1:3] == ["-m", "components.trainer.run_pretraining"]
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in job.env
    assert job.env["TOKENIZERS_PARALLELISM"] == "false"
    assert job.cwd == str(REPO_ROOT)


def test_build_local_eval_job_for_local_model():
    job = build_local_eval_job(
        eval_config_path="configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json",
        output_dir="artifacts/local-eval/ui-test",
        hf_home=".cache/hf",
        local_model_path="artifacts/local-train/ui-test/model",
        device="mps",
    )

    assert str((REPO_ROOT / "scripts" / "local" / "run_single_model_eval_local.py").resolve()) in job.command
    assert "--local-model-path" in job.command
    assert "--device" in job.command
    assert "mps" in job.command
    assert job.cwd == str(REPO_ROOT)


def test_build_local_eval_job_rejects_ambiguous_model_source():
    try:
        build_local_eval_job(
            eval_config_path="configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json",
            output_dir="artifacts/local-eval/ui-test",
            hf_home=".cache/hf",
            model_id="Qwen/Qwen2.5-0.5B",
            local_model_path="artifacts/local-train/ui-test/model",
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
        return
    raise AssertionError("Expected ValueError for ambiguous model source")


def test_start_local_job_writes_log_and_completes(tmp_path: Path):
    log_path = tmp_path / "job.log"
    job = LocalJobSpec(
        label="Echo",
        command=[sys.executable, "-c", "print('hello from ui job')"],
        env=os.environ.copy(),
        cwd=str(REPO_ROOT),
    )

    running = start_local_job(job, str(log_path))
    running.process.wait(timeout=5)
    status = poll_local_job(running)
    output = read_job_output(str(log_path))

    assert status.state == "completed"
    assert "hello from ui job" in output


def test_stop_local_job_terminates_running_process(tmp_path: Path):
    log_path = tmp_path / "sleep.log"
    job = LocalJobSpec(
        label="Sleep",
        command=[sys.executable, "-c", "import time; print('start'); time.sleep(30)"],
        env=os.environ.copy(),
        cwd=str(REPO_ROOT),
    )

    running = start_local_job(job, str(log_path))
    time.sleep(0.2)
    status = stop_local_job(running)
    output = read_job_output(str(log_path))

    assert status.state in {"stopped", "failed"}
    assert "Stop requested" in output


@pytest.mark.skipif(os.name != "posix", reason="process group test is POSIX-specific")
def test_stop_local_job_terminates_child_processes(tmp_path: Path):
    log_path = tmp_path / "tree.log"
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time\n"
        f"child_pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "child_pid_path.write_text(str(child.pid), encoding='utf-8')\n"
        "print('child-started', child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    job = LocalJobSpec(
        label="ProcessTree",
        command=[sys.executable, "-c", code],
        env=os.environ.copy(),
        cwd=str(REPO_ROOT),
    )

    running = start_local_job(job, str(log_path))
    deadline = time.time() + 5
    while time.time() < deadline and not child_pid_path.exists():
        time.sleep(0.05)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    status = stop_local_job(running)
    assert status.state in {"stopped", "failed"}

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    else:
        raise AssertionError(f"child process {child_pid} was still running after stop")
