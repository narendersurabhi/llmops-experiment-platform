from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.ui.file_dialogs import choose_directory_path, choose_file_path
from components.ui.presets import (
    DEFAULT_HF_HOME,
    DEFAULT_TRAINING_PRESET_KEY,
    TRAINING_PRESETS,
    training_preset_values,
)
from components.ui.local_jobs import (
    DEFAULT_LOCAL_EVAL_OUTPUT,
    DEFAULT_LOCAL_TRAINING_OUTPUT,
    REPO_ROOT,
    build_local_eval_job,
    build_local_training_job,
    poll_local_job,
    read_job_output,
    render_job_command,
    start_local_job,
    stop_local_job,
)


st.set_page_config(page_title="LLMOps Local UI", layout="wide")


DEFAULT_TRAIN_VALUES = training_preset_values(DEFAULT_TRAINING_PRESET_KEY)
DEFAULT_EVAL_CONFIG = "configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json"
DEFAULT_EVAL_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_EVAL_LOCAL_MODEL = f"{DEFAULT_LOCAL_TRAINING_OUTPUT}/model"
DEFAULTS = {
    "training_preset": DEFAULT_TRAINING_PRESET_KEY,
    "training_run_config_path": DEFAULT_TRAIN_VALUES["training_run_config_path"],
    "training_dataset_metadata_path": DEFAULT_TRAIN_VALUES["training_dataset_metadata_path"],
    "training_output_dir": DEFAULT_TRAIN_VALUES["training_output_dir"],
    "training_hf_home": DEFAULT_TRAIN_VALUES["training_hf_home"],
    "training_mlflow_tracking_uri": "",
    "evaluation_eval_config_path": DEFAULT_EVAL_CONFIG,
    "evaluation_model_source": "Saved local model",
    "evaluation_local_model_path": DEFAULT_EVAL_LOCAL_MODEL,
    "evaluation_model_id": DEFAULT_EVAL_MODEL_ID,
    "evaluation_model_uri": "",
    "evaluation_output_dir": DEFAULT_LOCAL_EVAL_OUTPUT,
    "evaluation_hf_home": DEFAULT_HF_HOME,
    "evaluation_device": "auto",
    "evaluation_torch_threads": 2,
    "evaluation_port": 8000,
    "evaluation_server_timeout": 900,
    "evaluation_mlflow_tracking_uri": "",
}


def ensure_ui_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_training_preset(preset_key: str) -> None:
    for key, value in training_preset_values(preset_key).items():
        st.session_state[key] = value


def handle_training_preset_change() -> None:
    apply_training_preset(str(st.session_state["training_preset"]))


def browse_and_set_path(*, key: str, title: str, directory: bool) -> None:
    chooser = choose_directory_path if directory else choose_file_path
    selected = chooser(title=title, current_value=st.session_state.get(key))
    if selected:
        st.session_state[key] = selected
        st.rerun()


def render_path_picker(label: str, key: str, *, directory: bool = False) -> None:
    field_col, button_col = st.columns([6, 1])
    with field_col:
        st.text_input(label, key=key)
    with button_col:
        st.write("")
        st.write("")
        if st.button("Browse...", key=f"{key}-browse", use_container_width=True):
            browse_and_set_path(key=key, title=label, directory=directory)


ensure_ui_defaults()


def repo_relative(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def show_expected_artifacts(label: str, output_dir: str | None) -> None:
    if not output_dir:
        return
    output_path = Path(output_dir)
    if label == "Local Training":
        st.info(
            "Expected artifacts:\n"
            f"- summary: `{repo_relative(str(output_path / 'training_summary.json'))}`\n"
            f"- model dir: `{repo_relative(str(output_path / 'model'))}`"
        )
        return
    st.info(
        "Expected artifacts:\n"
        f"- scorecard: `{repo_relative(str(output_path / 'scorecard.json'))}`\n"
        f"- model card: `{repo_relative(str(output_path / 'model_cards' / 'model_card.md'))}`"
    )


def render_job_status(session_key: str) -> None:
    running_job = st.session_state.get(session_key)
    if not running_job:
        return

    status = poll_local_job(running_job)
    if status.state == "running":
        st.warning(f"{running_job.label} is running for {status.duration_seconds:.2f}s.")
    elif status.state == "completed":
        st.success(f"{running_job.label} finished successfully in {status.duration_seconds:.2f}s.")
    elif status.state == "stopped":
        st.warning(f"{running_job.label} was stopped after {status.duration_seconds:.2f}s.")
    else:
        st.error(
            f"{running_job.label} failed with exit code {status.returncode} after {status.duration_seconds:.2f}s."
        )

    control_columns = st.columns(3)
    if status.state == "running":
        if control_columns[0].button("Stop", key=f"{session_key}-stop", use_container_width=True):
            stop_local_job(running_job)
            st.rerun()
    if control_columns[1].button("Refresh", key=f"{session_key}-refresh", use_container_width=True):
        st.rerun()
    if status.state != "running":
        if control_columns[2].button("Clear", key=f"{session_key}-clear", use_container_width=True):
            del st.session_state[session_key]
            st.rerun()

    with st.expander("Executed Command", expanded=False):
        st.code(running_job.command_preview, language="bash")
    st.code(read_job_output(running_job.log_path), language="text")
    show_expected_artifacts(running_job.label, running_job.output_dir)


st.title("Local LLMOps UI")
st.caption("Configure and run local training and evaluation jobs on this MacBook. This UI reuses the existing local CLI entrypoints.")

with st.sidebar:
    st.subheader("Repo")
    st.code(str(REPO_ROOT), language="text")
    st.markdown(
        "Install UI + local runtime dependencies with:\n"
        "```bash\nuv sync --extra dev --extra ui --extra trainer --extra inference\n```"
    )


training_tab, evaluation_tab = st.tabs(["Training", "Evaluation"])

with training_tab:
    st.subheader("Local Training")
    st.caption("Choose a preset for a fast smoke run or the full 3000-record local Qwen training run.")
    render_job_status("training_job")
    st.selectbox(
        "Training preset",
        options=list(TRAINING_PRESETS),
        key="training_preset",
        on_change=handle_training_preset_change,
    )
    st.info(TRAINING_PRESETS[str(st.session_state["training_preset"])].description)
    render_path_picker("Run config path", "training_run_config_path")
    render_path_picker("Dataset metadata path", "training_dataset_metadata_path")
    render_path_picker("Output dir", "training_output_dir", directory=True)
    render_path_picker("HF cache dir", "training_hf_home", directory=True)
    st.text_input("MLflow tracking URI", key="training_mlflow_tracking_uri")
    submitted_training = st.button("Run Local Training", key="run-local-training", use_container_width=True)

    if submitted_training:
        try:
            active_job = st.session_state.get("training_job")
            if active_job and poll_local_job(active_job).state == "running":
                st.error("A local training job is already running. Stop it first or wait for it to finish.")
            else:
                training_job = build_local_training_job(
                    run_config_path=str(st.session_state["training_run_config_path"]),
                    dataset_metadata_path=str(st.session_state["training_dataset_metadata_path"]),
                    output_dir=str(st.session_state["training_output_dir"]),
                    hf_home=str(st.session_state["training_hf_home"]),
                    mlflow_tracking_uri=str(st.session_state["training_mlflow_tracking_uri"]) or None,
                )
                st.session_state["training_job"] = start_local_job(training_job)
                st.rerun()
        except Exception as exc:  # pragma: no cover - streamlit presentation path
            st.exception(exc)


with evaluation_tab:
    st.subheader("Local Evaluation")
    st.caption("Runs the existing local single-model eval flow, including the local inference server.")
    render_job_status("evaluation_job")
    render_path_picker("Eval config path", "evaluation_eval_config_path")
    st.radio(
        "Model source",
        options=["Saved local model", "Hugging Face model ID"],
        horizontal=True,
        key="evaluation_model_source",
    )
    if st.session_state["evaluation_model_source"] == "Saved local model":
        render_path_picker("Local model path", "evaluation_local_model_path", directory=True)
    else:
        st.text_input("Model ID", key="evaluation_model_id")
    st.text_input("Model URI", key="evaluation_model_uri")
    render_path_picker("Output dir", "evaluation_output_dir", directory=True)
    render_path_picker("HF cache dir", "evaluation_hf_home", directory=True)
    st.selectbox("Device", options=["auto", "mps", "cpu"], key="evaluation_device")
    st.number_input("Torch threads", min_value=1, max_value=16, step=1, key="evaluation_torch_threads")
    st.number_input("Port", min_value=1024, max_value=65535, step=1, key="evaluation_port")
    st.number_input(
        "Server start timeout (seconds)",
        min_value=30,
        max_value=3600,
        step=30,
        key="evaluation_server_timeout",
    )
    st.text_input("MLflow tracking URI", key="evaluation_mlflow_tracking_uri")
    submitted_eval = st.button("Run Local Evaluation", key="run-local-evaluation", use_container_width=True)

    if submitted_eval:
        try:
            active_job = st.session_state.get("evaluation_job")
            if active_job and poll_local_job(active_job).state == "running":
                st.error("A local evaluation job is already running. Stop it first or wait for it to finish.")
            else:
                evaluation_job = build_local_eval_job(
                    eval_config_path=str(st.session_state["evaluation_eval_config_path"]),
                    output_dir=str(st.session_state["evaluation_output_dir"]),
                    hf_home=str(st.session_state["evaluation_hf_home"]),
                    device=str(st.session_state["evaluation_device"]),
                    torch_threads=int(st.session_state["evaluation_torch_threads"]),
                    port=int(st.session_state["evaluation_port"]),
                    server_start_timeout=int(st.session_state["evaluation_server_timeout"]),
                    model_id=(
                        str(st.session_state["evaluation_model_id"])
                        if st.session_state["evaluation_model_source"] == "Hugging Face model ID"
                        else None
                    ),
                    local_model_path=(
                        str(st.session_state["evaluation_local_model_path"])
                        if st.session_state["evaluation_model_source"] == "Saved local model"
                        else None
                    ),
                    model_uri=str(st.session_state["evaluation_model_uri"]) or None,
                    mlflow_tracking_uri=str(st.session_state["evaluation_mlflow_tracking_uri"]) or None,
                )
                st.session_state["evaluation_job"] = start_local_job(evaluation_job)
                st.rerun()
        except Exception as exc:  # pragma: no cover - streamlit presentation path
            st.exception(exc)
