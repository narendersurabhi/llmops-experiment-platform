from __future__ import annotations

from dataclasses import dataclass


DEFAULT_HF_HOME = ".cache/hf"


@dataclass(frozen=True)
class TrainingPreset:
    key: str
    description: str
    run_config_path: str
    dataset_metadata_path: str
    output_dir: str
    hf_home: str = DEFAULT_HF_HOME


QWEN_SMOKE_TRAINING_PRESET = TrainingPreset(
    key="Qwen Smoke (5 records)",
    description="Fast sanity check for the local Qwen continued pretraining flow using the 5-record smoke dataset.",
    run_config_path="configs/training/smoke_qwen2_5_0_5b.cpt.json",
    dataset_metadata_path="dataset/metadata.smoke_qwen2_5_0_5b.json",
    output_dir="artifacts/local-train/ui_smoke_qwen2_5_0_5b",
)

QWEN_FULL_PUBLIC_TRAINING_PRESET = TrainingPreset(
    key="Qwen Full Public (3000 records)",
    description="Full local Qwen continued pretraining on the 3000-record public FineWeb-Edu + arXiv dataset.",
    run_config_path="configs/training/local_qwen2_5_0_5b_public.cpt.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.public.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_public",
)

TRAINING_PRESETS = {
    QWEN_SMOKE_TRAINING_PRESET.key: QWEN_SMOKE_TRAINING_PRESET,
    QWEN_FULL_PUBLIC_TRAINING_PRESET.key: QWEN_FULL_PUBLIC_TRAINING_PRESET,
}

DEFAULT_TRAINING_PRESET_KEY = QWEN_SMOKE_TRAINING_PRESET.key


def training_preset_values(key: str) -> dict[str, str]:
    preset = TRAINING_PRESETS[key]
    return {
        "training_run_config_path": preset.run_config_path,
        "training_dataset_metadata_path": preset.dataset_metadata_path,
        "training_output_dir": preset.output_dir,
        "training_hf_home": preset.hf_home,
    }
