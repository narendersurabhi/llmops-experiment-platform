.PHONY: help venv-install test pipelines-compile docker-build infra-apply infra-apply-kubeflow k8s-status mlflow-port-forward submit-training submit-evaluation

PYTHON ?= python
KUBECTL ?= kubectl
DOCKER ?= docker
KFP_HOST ?= http://localhost:8080

help:
	@echo "LLMOps local-first commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv-install           Install Python dependencies"
	@echo ""
	@echo "Validation:"
	@echo "  make test                   Run unit tests"
	@echo "  make pipelines-compile      Compile training/evaluation pipelines"
	@echo ""
	@echo "Build:"
	@echo "  make docker-build           Build base/trainer/evaluator images"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make infra-apply            Apply namespace + postgres + minio + mlflow"
	@echo "  make infra-apply-kubeflow   Apply optional Kubeflow manifests"
	@echo "  make k8s-status             Show pods/services in llmops namespace"
	@echo "  make mlflow-port-forward    Port-forward MLflow UI to localhost:5000"
	@echo ""
	@echo "Kubeflow Pipelines:"
	@echo "  make submit-training RUN_CONFIG_PATH=... DATASET_METADATA_PATH=... [KFP_HOST=...]"
	@echo "  make submit-evaluation EVAL_CONFIG_PATH=... CANDIDATE_MODEL_URI=... BASELINE_MODEL_URI=... [KFP_HOST=...]"

venv-install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

pipelines-compile:
	bash scripts/dev/compile_pipelines.sh

docker-build:
	$(DOCKER) build -f docker/base/Dockerfile -t llmops/base:latest .
	$(DOCKER) build -f docker/trainer/Dockerfile -t llmops/trainer:latest .
	$(DOCKER) build -f docker/evaluator/Dockerfile -t llmops/evaluator:latest .

infra-apply:
	bash scripts/deploy/apply_infra.sh

infra-apply-kubeflow:
	$(KUBECTL) apply -f infra/kubeflow/

k8s-status:
	$(KUBECTL) get pods,svc -n llmops

mlflow-port-forward:
	$(KUBECTL) port-forward -n llmops svc/mlflow 5000:5000

submit-training:
	@if [ -z "$(RUN_CONFIG_PATH)" ] || [ -z "$(DATASET_METADATA_PATH)" ]; then \
		echo "RUN_CONFIG_PATH and DATASET_METADATA_PATH are required"; \
		exit 1; \
	fi
	KFP_HOST="$(KFP_HOST)" $(PYTHON) scripts/pipelines/submit_training_run.py \
		--host "$(KFP_HOST)" \
		--run-config-path "$(RUN_CONFIG_PATH)" \
		--dataset-metadata-path "$(DATASET_METADATA_PATH)"

submit-evaluation:
	@if [ -z "$(EVAL_CONFIG_PATH)" ] || [ -z "$(CANDIDATE_MODEL_URI)" ] || [ -z "$(BASELINE_MODEL_URI)" ]; then \
		echo "EVAL_CONFIG_PATH, CANDIDATE_MODEL_URI, and BASELINE_MODEL_URI are required"; \
		exit 1; \
	fi
	KFP_HOST="$(KFP_HOST)" $(PYTHON) scripts/pipelines/submit_evaluation_run.py \
		--host "$(KFP_HOST)" \
		--eval-config-path "$(EVAL_CONFIG_PATH)" \
		--candidate-model-uri "$(CANDIDATE_MODEL_URI)" \
		--baseline-model-uri "$(BASELINE_MODEL_URI)"
