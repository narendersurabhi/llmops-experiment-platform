import yaml

from components.inference.manifest import (
    DEFAULT_LOCAL_MODEL_MOUNT_PATH,
    build_qwen_inference_manifests,
    dump_qwen_inference_manifests,
)


def test_build_baseline_manifests_include_stable_service_alias():
    documents = build_qwen_inference_manifests(
        role="baseline",
        model_id="hf://org/base",
        model_uri="models:/baseline/1",
        image="llmops/qwen-service:test",
    )

    assert len(documents) == 3
    stable_service, baseline_service, deployment = documents
    assert stable_service["metadata"]["name"] == "qwen-inference"
    assert stable_service["spec"]["selector"]["app"] == "qwen-baseline"
    assert baseline_service["metadata"]["name"] == "qwen-baseline"
    assert deployment["metadata"]["annotations"]["llmops/model-id"] == "hf://org/base"
    assert deployment["metadata"]["annotations"]["llmops/model-uri"] == "models:/baseline/1"
    assert deployment["spec"]["template"]["spec"]["containers"][0]["image"] == "llmops/qwen-service:test"


def test_dump_candidate_manifests_omits_stable_service_alias():
    rendered = dump_qwen_inference_manifests(role="candidate", model_id="hf://org/candidate")
    documents = list(yaml.safe_load_all(rendered))

    assert len(documents) == 2
    service, deployment = documents
    assert service["metadata"]["name"] == "qwen-candidate"
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    model_id = next(item["value"] for item in env if item["name"] == "QWEN_MODEL_ID")
    assert model_id == "hf://org/candidate"


def test_build_candidate_manifests_with_local_model_path():
    documents = build_qwen_inference_manifests(
        role="candidate",
        model_id="ignored-when-local-path-is-set",
        local_model_source_path="/var/models/base-model",
        local_model_mount_path=DEFAULT_LOCAL_MODEL_MOUNT_PATH,
    )

    assert len(documents) == 2
    _, deployment = documents
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = container["env"]
    model_id = next(item["value"] for item in env if item["name"] == "QWEN_MODEL_ID")
    assert model_id == DEFAULT_LOCAL_MODEL_MOUNT_PATH
    mount_paths = {mount["mountPath"] for mount in container["volumeMounts"]}
    assert DEFAULT_LOCAL_MODEL_MOUNT_PATH in mount_paths
    volumes = {volume["name"]: volume for volume in deployment["spec"]["template"]["spec"]["volumes"]}
    assert volumes["qwen-local-model"]["hostPath"]["path"] == "/var/models/base-model"
    assert volumes["qwen-local-model"]["hostPath"]["type"] == "Directory"
