.PHONY: help venv-install venv-install-all test validate-local-data build-public-data pipelines-compile docker-build run-local-ui run-local-training run-local-eval run-local-base-eval base-model-apply baseline-apply candidate-apply promote-candidate inference-apply infra-apply minio-bootstrap infra-apply-kubeflow k8s-status kfp-status mlflow-port-forward kfp-port-forward submit-training submit-evaluation submit-single-evaluation

UV ?= uv
PYTHON ?= $(CURDIR)/.venv/bin/python
KUBECTL ?= kubectl
DOCKER ?= docker
KFP_HOST ?= http://localhost:3000
KFP_VERSION ?= 2.15.0
KFP_DISABLE_PROXY_AGENT ?= true
KFP_DISABLE_METADATA_WRITER ?= true
KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000 ?= true
BASE_IMAGE ?= llmops/base:dev
TRAINER_IMAGE ?= llmops/trainer:dev
EVALUATOR_IMAGE ?= llmops/evaluator:dev
MLFLOW_IMAGE ?= llmops/mlflow:dev
QWEN_SERVICE_IMAGE ?= llmops/qwen-service:dev
BASELINE_MODEL_ID ?= Qwen/Qwen2.5-0.5B
CANDIDATE_MODEL_ID ?= Qwen/Qwen2.5-0.5B-Instruct

help:
	@echo "LLMOps local-first commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv-install           Install base + dev dependencies with uv"
	@echo "  make venv-install-all       Install all optional dependency sets with uv"
	@echo ""
	@echo "Validation:"
	@echo "  make test                   Run unit tests"
	@echo "  make validate-local-data DATASET_METADATA_PATH=... [GOLDEN_SET_PATH=...]"
	@echo "  make build-public-data      Build a real public-source CPT corpus from FineWeb-Edu + arXiv"
	@echo "  make pipelines-compile      Compile training/evaluation pipelines"
	@echo "  make run-local-ui           Launch the local Streamlit UI for training/eval on this Mac"
	@echo "  make run-local-training [RUN_CONFIG_PATH=...] [DATASET_METADATA_PATH=...] [OUTPUT_DIR=...]"
	@echo "  make run-local-eval MODEL_ID=... [LOCAL_MODEL_PATH=...] [EVAL_CONFIG_PATH=...] [MODEL_URI=...]"
	@echo "  make run-local-base-eval    Run base-model eval locally against the public golden set"
	@echo ""
	@echo "Build:"
	@echo "  make docker-build           Build base/trainer/evaluator/mlflow/qwen images"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make infra-apply            Apply namespace + postgres + minio + mlflow, then bootstrap the MLflow bucket"
	@echo "  make base-model-apply MODEL_ID=... [LOCAL_MODEL_PATH=...] Deploy the stable base-model endpoint"
	@echo "  make baseline-apply MODEL_ID=... [LOCAL_MODEL_PATH=...]   Deploy or update the stable baseline inference endpoint"
	@echo "  make candidate-apply MODEL_ID=... [LOCAL_MODEL_PATH=...]  Deploy or update the candidate inference endpoint"
	@echo "  make promote-candidate      Promote the current candidate deployment into baseline"
	@echo "  make inference-apply        Deploy default baseline + candidate inference endpoints"
	@echo "  make minio-bootstrap        Recreate/verify the mlflow-artifacts bucket inside MinIO"
	@echo "  make infra-apply-kubeflow   Install standalone Kubeflow Pipelines (KFP_VERSION=$(KFP_VERSION))"
	@echo "                             Default local mode disables proxy-agent and metadata-writer and patches SeaweedFS port 9000"
	@echo "  make k8s-status             Show pods/services in llmops namespace"
	@echo "  make kfp-status             Show pods/services in kubeflow namespace"
	@echo "  make mlflow-port-forward    Port-forward MLflow UI to localhost:5000"
	@echo "  make kfp-port-forward       Port-forward KFP UI/API to localhost:3000"
	@echo ""
	@echo "Kubeflow Pipelines:"
	@echo "  make submit-training RUN_CONFIG_PATH=... DATASET_METADATA_PATH=... [KFP_HOST=...]"
	@echo "  make submit-evaluation EVAL_CONFIG_PATH=... CANDIDATE_MODEL_URI=... BASELINE_MODEL_URI=... [KFP_HOST=...]"
	@echo "  make submit-single-evaluation EVAL_CONFIG_PATH=... MODEL_URI=... [KFP_HOST=...]"

venv-install:
	$(UV) sync --extra dev

venv-install-all:
	$(UV) sync --all-extras

test:
	$(UV) run pytest -q

validate-local-data:
	@if [ -z "$(DATASET_METADATA_PATH)" ]; then \
		echo "DATASET_METADATA_PATH is required"; \
		exit 1; \
	fi
	PYTHONPATH=. $(UV) run python scripts/data/validate_local_data.py \
		--dataset-metadata-path "$(DATASET_METADATA_PATH)" \
		$(if $(GOLDEN_SET_PATH),--golden-set-path "$(GOLDEN_SET_PATH)",)

build-public-data:
	PYTHONPATH=. $(UV) run python scripts/data/build_public_cpt_corpus.py

pipelines-compile:
	PYTHON="$(PYTHON)" bash scripts/dev/compile_pipelines.sh

run-local-ui:
	PYTHONPATH=. $(UV) run --extra ui --extra trainer --extra inference streamlit run apps/local_experiment_ui.py

run-local-training:
	PYTHONPATH=. HF_HOME="$(or $(HF_HOME),$(CURDIR)/.cache/hf)" MLFLOW_TRACKING_URI="$(or $(MLFLOW_TRACKING_URI),sqlite:///$(CURDIR)/$(or $(OUTPUT_DIR),artifacts/local-train/qwen2_5_0_5b)/mlflow.db)" PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false $(UV) run python -m components.trainer.run_pretraining \
		--config-path "$(or $(RUN_CONFIG_PATH),configs/training/qwen2_5_0_5b.cpt.sample.json)" \
		--dataset-metadata-path "$(or $(DATASET_METADATA_PATH),dataset/metadata.qwen2_5_0_5b.sample.json)" \
		--output-dir "$(or $(OUTPUT_DIR),artifacts/local-train/qwen2_5_0_5b)"

run-local-eval:
	PYTHONPATH=. $(UV) run python scripts/local/run_single_model_eval_local.py \
		--eval-config-path "$(or $(EVAL_CONFIG_PATH),configs/evaluation/single_model.sample.json)" \
		$(if $(LOCAL_MODEL_PATH),--local-model-path "$(LOCAL_MODEL_PATH)",$(if $(MODEL_ID),--model-id "$(MODEL_ID)",)) \
		$(if $(MODEL_URI),--model-uri "$(MODEL_URI)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",) \
		$(if $(MLFLOW_TRACKING_URI),--mlflow-tracking-uri "$(MLFLOW_TRACKING_URI)",) \
		$(if $(HF_HOME),--hf-home "$(HF_HOME)",) \
		$(if $(HOST),--host "$(HOST)",) \
		$(if $(PORT),--port "$(PORT)",) \
		$(if $(SERVER_START_TIMEOUT),--server-start-timeout "$(SERVER_START_TIMEOUT)",) \
		$(if $(TORCH_THREADS),--torch-threads "$(TORCH_THREADS)",) \
		$(if $(DEVICE),--device "$(DEVICE)",)

run-local-base-eval:
	./scripts/local/run_base_model_eval_local.sh

docker-build:
	$(DOCKER) build -f docker/base/Dockerfile -t $(BASE_IMAGE) .
	$(DOCKER) build --build-arg BASE_IMAGE=$(BASE_IMAGE) -f docker/trainer/Dockerfile -t $(TRAINER_IMAGE) .
	$(DOCKER) build --build-arg BASE_IMAGE=$(BASE_IMAGE) -f docker/evaluator/Dockerfile -t $(EVALUATOR_IMAGE) .
	$(DOCKER) build -f docker/mlflow/Dockerfile -t $(MLFLOW_IMAGE) .
	$(DOCKER) build -f docker/qwen-service/Dockerfile -t $(QWEN_SERVICE_IMAGE) .

baseline-apply:
	@if [ -z "$(MODEL_ID)" ] && [ -z "$(LOCAL_MODEL_PATH)" ]; then \
		echo "MODEL_ID or LOCAL_MODEL_PATH is required"; \
		exit 1; \
	fi
	PYTHON="$(PYTHON)" KUBECTL="$(KUBECTL)" QWEN_SERVICE_IMAGE="$(QWEN_SERVICE_IMAGE)" LOCAL_MODEL_PATH="$(LOCAL_MODEL_PATH)" LOCAL_MODEL_MOUNT_PATH="$(LOCAL_MODEL_MOUNT_PATH)" bash scripts/deploy/apply_qwen_inference.sh --role baseline $(if $(MODEL_ID),--model-id "$(MODEL_ID)",) $(if $(MODEL_URI),--model-uri "$(MODEL_URI)",)

base-model-apply: baseline-apply

candidate-apply:
	@if [ -z "$(MODEL_ID)" ] && [ -z "$(LOCAL_MODEL_PATH)" ]; then \
		echo "MODEL_ID or LOCAL_MODEL_PATH is required"; \
		exit 1; \
	fi
	PYTHON="$(PYTHON)" KUBECTL="$(KUBECTL)" QWEN_SERVICE_IMAGE="$(QWEN_SERVICE_IMAGE)" LOCAL_MODEL_PATH="$(LOCAL_MODEL_PATH)" LOCAL_MODEL_MOUNT_PATH="$(LOCAL_MODEL_MOUNT_PATH)" bash scripts/deploy/apply_qwen_inference.sh --role candidate $(if $(MODEL_ID),--model-id "$(MODEL_ID)",) $(if $(MODEL_URI),--model-uri "$(MODEL_URI)",)

promote-candidate:
	PYTHON="$(PYTHON)" KUBECTL="$(KUBECTL)" QWEN_SERVICE_IMAGE="$(QWEN_SERVICE_IMAGE)" bash scripts/deploy/promote_candidate.sh

inference-apply:
	PYTHON="$(PYTHON)" KUBECTL="$(KUBECTL)" QWEN_SERVICE_IMAGE="$(QWEN_SERVICE_IMAGE)" bash scripts/deploy/apply_qwen_inference.sh --role baseline --model-id "$(BASELINE_MODEL_ID)"
	PYTHON="$(PYTHON)" KUBECTL="$(KUBECTL)" QWEN_SERVICE_IMAGE="$(QWEN_SERVICE_IMAGE)" bash scripts/deploy/apply_qwen_inference.sh --role candidate --model-id "$(CANDIDATE_MODEL_ID)"

infra-apply:
	bash scripts/deploy/apply_infra.sh

minio-bootstrap:
	bash scripts/deploy/bootstrap_minio_bucket.sh

infra-apply-kubeflow:
	KFP_VERSION="$(KFP_VERSION)" \
	KFP_DISABLE_PROXY_AGENT="$(KFP_DISABLE_PROXY_AGENT)" \
	KFP_DISABLE_METADATA_WRITER="$(KFP_DISABLE_METADATA_WRITER)" \
	KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000="$(KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000)" \
	bash scripts/deploy/install_kubeflow_pipelines.sh

k8s-status:
	$(KUBECTL) get pods,svc -n llmops

kfp-status:
	$(KUBECTL) get pods,svc -n kubeflow

mlflow-port-forward:
	$(KUBECTL) port-forward -n llmops svc/mlflow 5000:5000

kfp-port-forward:
	$(KUBECTL) port-forward -n kubeflow svc/ml-pipeline-ui 3000:80

submit-training:
	@if [ -z "$(RUN_CONFIG_PATH)" ] || [ -z "$(DATASET_METADATA_PATH)" ]; then \
		echo "RUN_CONFIG_PATH and DATASET_METADATA_PATH are required"; \
		exit 1; \
	fi
	KFP_HOST="$(KFP_HOST)" $(UV) run python scripts/pipelines/submit_training_run.py \
		--host "$(KFP_HOST)" \
		--run-config-path "$(RUN_CONFIG_PATH)" \
		--dataset-metadata-path "$(DATASET_METADATA_PATH)"

submit-evaluation:
	@if [ -z "$(EVAL_CONFIG_PATH)" ] || [ -z "$(CANDIDATE_MODEL_URI)" ] || [ -z "$(BASELINE_MODEL_URI)" ]; then \
		echo "EVAL_CONFIG_PATH, CANDIDATE_MODEL_URI, and BASELINE_MODEL_URI are required"; \
		exit 1; \
	fi
	KFP_HOST="$(KFP_HOST)" $(UV) run python scripts/pipelines/submit_evaluation_run.py \
		--host "$(KFP_HOST)" \
		--eval-config-path "$(EVAL_CONFIG_PATH)" \
		--candidate-model-uri "$(CANDIDATE_MODEL_URI)" \
		--baseline-model-uri "$(BASELINE_MODEL_URI)"

submit-single-evaluation:
	@if [ -z "$(EVAL_CONFIG_PATH)" ] || [ -z "$(MODEL_URI)" ]; then \
		echo "EVAL_CONFIG_PATH and MODEL_URI are required"; \
		exit 1; \
	fi
	KFP_HOST="$(KFP_HOST)" $(UV) run python scripts/pipelines/submit_single_model_evaluation_run.py \
		--host "$(KFP_HOST)" \
		--eval-config-path "$(EVAL_CONFIG_PATH)" \
		--model-uri "$(MODEL_URI)"
