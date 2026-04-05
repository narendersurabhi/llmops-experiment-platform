from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_TRAINING_OUTPUT = "artifacts/local-train/ui_smoke_qwen2_5_0_5b"
DEFAULT_LOCAL_EVAL_OUTPUT = "artifacts/local-eval/ui_smoke_qwen2_5_0_5b"


@dataclass
class LocalJobSpec:
    label: str
    command: list[str]
    env: dict[str, str]
    cwd: str
    output_dir: str | None = None


@dataclass
class LocalJobResult:
    returncode: int
    output: str
    duration_seconds: float


@dataclass
class LocalRunningJob:
    label: str
    process: subprocess.Popen[str]
    log_path: str
    started_at: float
    command_preview: str
    output_dir: str | None = None


@dataclass
class LocalJobStatus:
    state: str
    returncode: int | None
    duration_seconds: float


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def default_eval_model_uri(model_id: str | None, local_model_path: str | None) -> str:
    if local_model_path:
        return f"file://{resolve_repo_path(local_model_path)}"
    if model_id:
        return f"hf://{model_id}"
    return "hf://Qwen/Qwen2.5-0.5B"


def sanitize_job_label(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-") or "job"


def build_local_training_job(
    *,
    run_config_path: str,
    dataset_metadata_path: str,
    output_dir: str,
    hf_home: str,
    mlflow_tracking_uri: str | None = None,
    python_executable: str | None = None,
) -> LocalJobSpec:
    resolved_output_dir = resolve_repo_path(output_dir)
    resolved_hf_home = resolve_repo_path(hf_home)
    resolved_run_config = resolve_repo_path(run_config_path)
    resolved_dataset_metadata = resolve_repo_path(dataset_metadata_path)
    tracking_uri = mlflow_tracking_uri or f"sqlite:///{(resolved_output_dir / 'mlflow.db').resolve()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HF_HOME"] = str(resolved_hf_home)
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    command = [
        python_executable or sys.executable,
        "-m",
        "components.trainer.run_pretraining",
        "--config-path",
        str(resolved_run_config),
        "--dataset-metadata-path",
        str(resolved_dataset_metadata),
        "--output-dir",
        str(resolved_output_dir),
    ]
    return LocalJobSpec(
        label="Local Training",
        command=command,
        env=env,
        cwd=str(REPO_ROOT),
        output_dir=str(resolved_output_dir),
    )


def build_local_eval_job(
    *,
    eval_config_path: str,
    output_dir: str,
    hf_home: str,
    device: str = "auto",
    torch_threads: int = 2,
    host: str = "127.0.0.1",
    port: int = 8000,
    server_start_timeout: int = 900,
    model_id: str | None = None,
    local_model_path: str | None = None,
    model_uri: str | None = None,
    mlflow_tracking_uri: str | None = None,
    python_executable: str | None = None,
) -> LocalJobSpec:
    if bool(model_id) == bool(local_model_path):
        raise ValueError("Provide exactly one of model_id or local_model_path for local evaluation.")

    resolved_output_dir = resolve_repo_path(output_dir)
    resolved_hf_home = resolve_repo_path(hf_home)
    resolved_eval_config = resolve_repo_path(eval_config_path)
    tracking_uri = mlflow_tracking_uri or f"sqlite:///{(resolved_output_dir / 'mlflow.db').resolve()}"

    command = [
        python_executable or sys.executable,
        str((REPO_ROOT / "scripts" / "local" / "run_single_model_eval_local.py").resolve()),
        "--eval-config-path",
        str(resolved_eval_config),
        "--output-dir",
        str(resolved_output_dir),
        "--hf-home",
        str(resolved_hf_home),
        "--host",
        host,
        "--port",
        str(port),
        "--server-start-timeout",
        str(server_start_timeout),
        "--torch-threads",
        str(torch_threads),
        "--device",
        device,
        "--mlflow-tracking-uri",
        tracking_uri,
        "--model-uri",
        model_uri or default_eval_model_uri(model_id, local_model_path),
    ]
    if model_id:
        command.extend(["--model-id", model_id])
    if local_model_path:
        command.extend(["--local-model-path", str(resolve_repo_path(local_model_path))])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return LocalJobSpec(
        label="Local Evaluation",
        command=command,
        env=env,
        cwd=str(REPO_ROOT),
        output_dir=str(resolved_output_dir),
    )


def render_job_command(job: LocalJobSpec) -> str:
    return " ".join(shlex.quote(part) for part in job.command)


def default_job_log_path(job: LocalJobSpec) -> Path:
    log_dir = REPO_ROOT / "artifacts" / "ui-logs"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{sanitize_job_label(job.label)}-{timestamp}.log"


def start_local_job(job: LocalJobSpec, log_path: str | None = None) -> LocalRunningJob:
    resolved_log_path = Path(log_path).resolve() if log_path else default_job_log_path(job)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path.write_text(f"$ {render_job_command(job)}\n\n", encoding="utf-8")
    log_handle = resolved_log_path.open("a", encoding="utf-8")
    try:
        popen_kwargs = {
            "cwd": job.cwd,
            "env": job.env,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            job.command,
            **popen_kwargs,
        )
    finally:
        log_handle.close()

    return LocalRunningJob(
        label=job.label,
        process=process,
        log_path=str(resolved_log_path),
        started_at=time.time(),
        command_preview=render_job_command(job),
        output_dir=job.output_dir,
    )


def poll_local_job(job: LocalRunningJob) -> LocalJobStatus:
    returncode = job.process.poll()
    duration_seconds = round(time.time() - job.started_at, 2)
    if returncode is None:
        return LocalJobStatus(state="running", returncode=None, duration_seconds=duration_seconds)
    if returncode == 0:
        return LocalJobStatus(state="completed", returncode=0, duration_seconds=duration_seconds)
    if returncode < 0:
        return LocalJobStatus(state="stopped", returncode=returncode, duration_seconds=duration_seconds)
    return LocalJobStatus(state="failed", returncode=returncode, duration_seconds=duration_seconds)


def stop_local_job(job: LocalRunningJob, timeout_seconds: int = 5) -> LocalJobStatus:
    status = poll_local_job(job)
    if status.state != "running":
        return status
    Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(job.log_path, "a", encoding="utf-8") as handle:
        handle.write("\n[llmops-ui] Stop requested\n")
    if os.name == "posix":
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return poll_local_job(job)
    else:
        job.process.terminate()
    try:
        job.process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            job.process.kill()
        job.process.wait(timeout=5)
    return poll_local_job(job)


def read_job_output(log_path: str, max_chars: int = 40000) -> str:
    path = Path(log_path)
    if not path.exists():
        return "<log file not found>"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_local_job(job: LocalJobSpec, on_output: Callable[[str], None] | None = None) -> LocalJobResult:
    start = time.time()
    process = subprocess.Popen(
        job.command,
        cwd=job.cwd,
        env=job.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        if on_output is not None:
            on_output(line)
    process.wait()
    return LocalJobResult(
        returncode=int(process.returncode),
        output="".join(output_lines),
        duration_seconds=round(time.time() - start, 2),
    )
