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


@dataclass(frozen=True)
class EvaluationPreset:
    key: str
    description: str
    eval_config_path: str
    model_source: str
    output_dir: str
    hf_home: str = DEFAULT_HF_HOME
    local_model_path: str = ""
    model_id: str = ""
    model_uri: str = ""


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

QWEN_HEALTHCARE_PUBMED_TRAINING_PRESET = TrainingPreset(
    key="Qwen Healthcare PubMed (500/500/500)",
    description="Local Qwen continued pretraining on 1500 recent PubMed/PMC Open Access healthcare abstracts split into 500 train, 500 eval, and 500 test records.",
    run_config_path="configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.cpt.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500",
)

QWEN_HEALTHCARE_PUBMED_LORA_TRAINING_PRESET = TrainingPreset(
    key="Qwen Healthcare PubMed LoRA (500/500/500)",
    description="Local Qwen parameter-efficient continued pretraining with LoRA on 1500 recent PubMed/PMC Open Access healthcare abstracts.",
    run_config_path="configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_TRAINING_PRESET = TrainingPreset(
    key="Qwen Healthcare PubMed LoRA qkvo (500/500/500)",
    description="Local Qwen LoRA continued pretraining on PubMed healthcare abstracts using attention q/k/v/o target modules.",
    run_config_path="configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora_qkvo.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_TRAINING_PRESET = TrainingPreset(
    key="Qwen Healthcare PubMed LoRA qkvo r16 (500/500/500)",
    description="Local Qwen LoRA continued pretraining on PubMed healthcare abstracts using q/k/v/o target modules with rank 16.",
    run_config_path="configs/training/local_qwen2_5_0_5b_healthcare_pubmed_500.lora_qkvo_r16.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.500.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_TRAINING_PRESET = TrainingPreset(
    key="Qwen Healthcare PubMed LoRA qkvo r16 Expanded (2000/500/500)",
    description="Local Qwen LoRA continued pretraining on an expanded recent PubMed healthcare corpus using q/k/v/o target modules with rank 16.",
    run_config_path="configs/training/local_qwen2_5_0_5b_healthcare_pubmed_2000_500_500.lora_qkvo_r16.json",
    dataset_metadata_path="dataset/metadata.qwen2_5_0_5b.healthcare.pubmed.2000_500_500.json",
    output_dir="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16",
)

TRAINING_PRESETS = {
    QWEN_SMOKE_TRAINING_PRESET.key: QWEN_SMOKE_TRAINING_PRESET,
    QWEN_FULL_PUBLIC_TRAINING_PRESET.key: QWEN_FULL_PUBLIC_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_TRAINING_PRESET.key: QWEN_HEALTHCARE_PUBMED_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_TRAINING_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_TRAINING_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_TRAINING_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_TRAINING_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_TRAINING_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_TRAINING_PRESET,
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


QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET = EvaluationPreset(
    key="Qwen Base Public Golden Eval",
    description="Chat-style single-model evaluation of the base Qwen model against the public golden-set prompts.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.public.single_model.sample.json",
    model_source="Hugging Face model ID",
    model_id="Qwen/Qwen2.5-0.5B",
    model_uri="hf://Qwen/Qwen2.5-0.5B",
    output_dir="artifacts/local-eval/base_qwen2_5_0_5b",
)

QWEN_BASE_PUBLIC_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Base Public CPT Eval",
    description="Held-out CPT evaluation of the base Qwen model on the public continued-pretraining corpus.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.public.cpt.sample.json",
    model_source="Hugging Face model ID",
    model_id="Qwen/Qwen2.5-0.5B",
    model_uri="hf://Qwen/Qwen2.5-0.5B",
    output_dir="artifacts/local-eval/base_qwen2_5_0_5b_public_cpt",
)

QWEN_HEALTHCARE_PUBMED_LORA_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Healthcare PubMed LoRA CPT Eval",
    description="Held-out CPT evaluation of the saved LoRA healthcare model on the PubMed healthcare eval and test splits.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json",
    model_source="Saved local model",
    local_model_path="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora/model",
    model_uri="",
    output_dir="artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_cpt",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Healthcare PubMed LoRA qkvo CPT Eval",
    description="Held-out CPT evaluation of the saved healthcare LoRA q/k/v/o model on the PubMed healthcare eval and test splits.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json",
    model_source="Saved local model",
    local_model_path="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo/model",
    model_uri="",
    output_dir="artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_cpt",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Healthcare PubMed LoRA qkvo r16 CPT Eval",
    description="Held-out CPT evaluation of the saved healthcare LoRA q/k/v/o rank-16 model on the PubMed healthcare eval and test splits.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.cpt.sample.json",
    model_source="Saved local model",
    local_model_path="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16/model",
    model_uri="",
    output_dir="artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_500_lora_qkvo_r16_cpt",
)

QWEN_BASE_HEALTHCARE_PUBMED_EXPANDED_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Base Healthcare PubMed Expanded CPT Eval",
    description="Held-out CPT evaluation of the base Qwen model on the expanded PubMed healthcare eval and test splits.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.2000_500_500.cpt.sample.json",
    model_source="Hugging Face model ID",
    model_id="Qwen/Qwen2.5-0.5B",
    model_uri="hf://Qwen/Qwen2.5-0.5B",
    output_dir="artifacts/local-eval/base_qwen2_5_0_5b_healthcare_pubmed_2000_500_500_cpt",
)

QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_CPT_EVAL_PRESET = EvaluationPreset(
    key="Qwen Healthcare PubMed LoRA qkvo r16 Expanded CPT Eval",
    description="Held-out CPT evaluation of the saved healthcare LoRA q/k/v/o rank-16 model on the expanded PubMed healthcare eval and test splits.",
    eval_config_path="configs/evaluation/qwen2_5_0_5b.healthcare_pubmed.2000_500_500.cpt.sample.json",
    model_source="Saved local model",
    local_model_path="artifacts/local-train/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16/model",
    model_uri="",
    output_dir="artifacts/local-eval/qwen2_5_0_5b_healthcare_pubmed_2000_500_500_lora_qkvo_r16_cpt",
)

EVALUATION_PRESETS = {
    QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET.key: QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET,
    QWEN_BASE_PUBLIC_CPT_EVAL_PRESET.key: QWEN_BASE_PUBLIC_CPT_EVAL_PRESET,
    QWEN_BASE_HEALTHCARE_PUBMED_EXPANDED_CPT_EVAL_PRESET.key: QWEN_BASE_HEALTHCARE_PUBMED_EXPANDED_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_CPT_EVAL_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_CPT_EVAL_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_CPT_EVAL_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_CPT_EVAL_PRESET,
    QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_CPT_EVAL_PRESET.key: QWEN_HEALTHCARE_PUBMED_LORA_QKVO_R16_EXPANDED_CPT_EVAL_PRESET,
}

DEFAULT_EVALUATION_PRESET_KEY = QWEN_BASE_PUBLIC_GOLDEN_EVAL_PRESET.key


def evaluation_preset_values(key: str) -> dict[str, str]:
    preset = EVALUATION_PRESETS[key]
    return {
        "evaluation_eval_config_path": preset.eval_config_path,
        "evaluation_model_source": preset.model_source,
        "evaluation_local_model_path": preset.local_model_path,
        "evaluation_model_id": preset.model_id,
        "evaluation_model_uri": preset.model_uri,
        "evaluation_output_dir": preset.output_dir,
        "evaluation_hf_home": preset.hf_home,
    }
