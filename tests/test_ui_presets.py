from __future__ import annotations

from components.ui.presets import (
    DEFAULT_EVALUATION_PRESET_KEY,
    DEFAULT_TRAINING_PRESET_KEY,
    QWEN_BASE_PUBLIC_CPT_EVAL_PRESET,
    QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET,
    QWEN_BASE_HEALTHCARE_PUBMED_EXPANDED_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_TRAINING_PRESET,
    QWEN_FULL_PUBLIC_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_TRAINING_PRESET,
    QWEN_SMOKE_TRAINING_PRESET,
    evaluation_preset_values,
    training_preset_values,
)


def test_default_training_preset_is_smoke():
    assert DEFAULT_TRAINING_PRESET_KEY == QWEN_SMOKE_TRAINING_PRESET.key


def test_default_evaluation_preset_is_base_public_golden():
    assert DEFAULT_EVALUATION_PRESET_KEY == QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET.key


def test_full_training_preset_points_to_full_local_qwen_assets():
    values = training_preset_values(QWEN_FULL_PUBLIC_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/local_qwen2_5_0_5b_public.cpt.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.public.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_public"


def test_smoke_training_preset_points_to_smoke_assets():
    values = training_preset_values(QWEN_SMOKE_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/smoke_qwen2_5_0_5b.cpt.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.smoke_qwen2_5_0_5b.json"


def test_healthcare_pubmed_training_preset_points_to_latest_research_assets():
    values = training_preset_values(QWEN_HEALTHCARE_PUBMED_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.cpt.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500"


def test_healthcare_pubmed_lora_training_preset_points_to_parameter_efficient_assets():
    values = training_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_TRAINING_PRESET.key)

    assert values["training_run_config_path"] == "configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora.json"
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora"


def test_healthcare_pubmed_lora_qkvo_training_preset_points_to_ablation_assets():
    values = training_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_TRAINING_PRESET.key)

    assert (
        values["training_run_config_path"]
        == "configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora_qkvo.json"
    )
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo"


def test_healthcare_pubmed_lora_qkvo_r16_training_preset_points_to_rank16_assets():
    values = training_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_TRAINING_PRESET.key)

    assert (
        values["training_run_config_path"]
        == "configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora_qkvo_r16.json"
    )
    assert values["training_dataset_metadata_path"] == "dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json"
    assert values["training_output_dir"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16"


def test_healthcare_pubmed_lora_qkvo_r16_expanded_training_preset_points_to_expanded_assets():
    values = training_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_TRAINING_PRESET.key)

    assert (
        values["training_run_config_path"]
        == "configs/training/local_qwen2_5_0_5b_healthcare_pubmed_2000_500_500.lora_qkvo_r16.json"
    )
    assert (
        values["training_dataset_metadata_path"]
        == "dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.2000_500_500.json"
    )
    assert (
        values["training_output_dir"]
        == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16"
    )


def test_base_public_cpt_evaluation_preset_points_to_cpt_eval_assets():
    values = evaluation_preset_values(QWEN_BASE_PUBLIC_CPT_EVAL_PRESET.key)

    assert values["evaluation_eval_config_path"] == "configs/evaluation/qwen2_5_0_5b.public.cpt.sample.json"
    assert values["evaluation_model_source"] == "Hugging Face model ID"
    assert values["evaluation_model_id"] == "Qwen/Qwen2.5-0.5B"
    assert values["evaluation_output_dir"] == "artifacts/local-eval/base_qwen2_5_0_5b_public_cpt"


def test_base_healthcare_pubmed_expanded_cpt_evaluation_preset_points_to_expanded_assets():
    values = evaluation_preset_values(QWEN_BASE_HEALTHCARE_PUBMED_EXPANDED_CPT_EVAL_PRESET.key)

    assert (
        values["evaluation_eval_config_path"]
        == "configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.2000_500_500.cpt.sample.json"
    )
    assert values["evaluation_model_source"] == "Hugging Face model ID"
    assert values["evaluation_model_id"] == "Qwen/Qwen2.5-0.5B"
    assert (
        values["evaluation_output_dir"]
        == "artifacts/local-eval/base_qwen2_5_0_5b_healthcare_pubmed_2000_500_500_cpt"
    )


def test_healthcare_pubmed_lora_cpt_evaluation_preset_points_to_saved_model_assets():
    values = evaluation_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_CPT_EVAL_PRESET.key)

    assert (
        values["evaluation_eval_config_path"]
        == "configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json"
    )
    assert values["evaluation_model_source"] == "Saved local model"
    assert values["evaluation_local_model_path"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora/model"
    assert values["evaluation_output_dir"] == "artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_cpt"


def test_healthcare_pubmed_lora_qkvo_cpt_evaluation_preset_points_to_saved_model_assets():
    values = evaluation_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_CPT_EVAL_PRESET.key)

    assert values["evaluation_eval_config_path"] == "configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json"
    assert values["evaluation_model_source"] == "Saved local model"
    assert values["evaluation_local_model_path"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo/model"
    assert values["evaluation_output_dir"] == "artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_cpt"


def test_healthcare_pubmed_lora_qkvo_r16_cpt_evaluation_preset_points_to_saved_model_assets():
    values = evaluation_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_CPT_EVAL_PRESET.key)

    assert values["evaluation_eval_config_path"] == "configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json"
    assert values["evaluation_model_source"] == "Saved local model"
    assert values["evaluation_local_model_path"] == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16/model"
    assert values["evaluation_output_dir"] == "artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16_cpt"


def test_healthcare_pubmed_lora_qkvo_r16_expanded_cpt_evaluation_preset_points_to_saved_model_assets():
    values = evaluation_preset_values(QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_CPT_EVAL_PRESET.key)

    assert (
        values["evaluation_eval_config_path"]
        == "configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.2000_500_500.cpt.sample.json"
    )
    assert values["evaluation_model_source"] == "Saved local model"
    assert (
        values["evaluation_local_model_path"]
        == "artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16/model"
    )
    assert (
        values["evaluation_output_dir"]
        == "artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16_cpt"
    )
