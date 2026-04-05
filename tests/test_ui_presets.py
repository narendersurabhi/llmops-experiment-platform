from __future__ import annotations

from components.ui.presets import (
    DEFAULT_TRAINING_PRESET_KEY,
    QWEN_FULL_PUBLIC_TRAINING_PRESET,
    QWEN_SMOKE_TRAINING_PRESET,
    training_preset_values,
)


def test_default_training_preset_is_smoke():
    assert DEFAULT_TRAINING_PRESET_KEY == QWEN_SMOKE_TRAINING_PRESET.key


def test_full_training_preset_points_to_full_local_qwen_assets():
    values = training_preset_values(QWEN_FULL_PUBLIC_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/local_qwen2_5_0_5b_public.cpt.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.public.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_public"


def test_smoke_training_preset_points_to_smoke_assets():
    values = training_preset_values(QWEN_SMOKE_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/smoke_qwen2_5_0_5b.cpt.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.smoke_qwen2_5_0_5b.json"
