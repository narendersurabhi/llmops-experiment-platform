from __future__ import annotations

from typing import Any

import yaml


DEFAULT_NAMESPACE = "llmops"
DEFAULT_IMAGE = "llmops/qwen-service:dev"
DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TORCH_THREADS = 2
DEFAULT_REPLICAS = 1
DEFAULT_CPU_REQUEST = "1000m"
DEFAULT_MEMORY_REQUEST = "3Gi"
DEFAULT_MEMORY_LIMIT = "8Gi"
DEFAULT_LOCAL_MODEL_MOUNT_PATH = "/models/local-model"
DEFAULT_MODEL_IDS = {
    "candidate": "Qwen/Qwen2.5-0.5B-Instruct",
    "baseline": "Qwen/Qwen2.5-0.5B",
}


def role_defaults(role: str) -> dict[str, str]:
    if role == "candidate":
        return {
            "deployment_name": "qwen-candidate",
            "service_name": "qwen-candidate",
            "app_name": "qwen-candidate",
        }
    if role == "baseline":
        return {
            "deployment_name": "qwen-baseline",
            "service_name": "qwen-baseline",
            "app_name": "qwen-baseline",
        }
    raise ValueError(f"Unsupported inference role: {role}")


def build_service(name: str, namespace: str, selector_app: str, role: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app": name,
                "llmops.component": "inference",
                "llmops.role": role,
            },
        },
        "spec": {
            "selector": {"app": selector_app},
            "ports": [{"name": "http", "port": 8000, "targetPort": 8000}],
        },
    }


def build_deployment(
    *,
    namespace: str,
    deployment_name: str,
    app_name: str,
    role: str,
    image: str,
    model_id: str,
    model_uri: str | None,
    replicas: int,
    max_new_tokens: int,
    temperature: float,
    torch_threads: int,
    cpu_request: str,
    memory_request: str,
    memory_limit: str,
    local_model_source_path: str | None,
    local_model_mount_path: str,
) -> dict[str, Any]:
    annotations = {"llmops/model-id": model_id}
    if model_uri:
        annotations["llmops/model-uri"] = model_uri
    if local_model_source_path:
        annotations["llmops/local-model-source-path"] = local_model_source_path
        annotations["llmops/local-model-mount-path"] = local_model_mount_path

    effective_model_id = local_model_mount_path if local_model_source_path else model_id
    volume_mounts = [{"name": "qwen-model-cache", "mountPath": "/models"}]
    volumes: list[dict[str, Any]] = [{"name": "qwen-model-cache", "emptyDir": {}}]

    if local_model_source_path:
        volume_mounts.append(
            {
                "name": "qwen-local-model",
                "mountPath": local_model_mount_path,
                "readOnly": True,
            }
        )
        volumes.append(
            {
                "name": "qwen-local-model",
                "hostPath": {
                    "path": local_model_source_path,
                    "type": "Directory",
                },
            }
        )

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deployment_name,
            "namespace": namespace,
            "labels": {
                "app": app_name,
                "llmops.component": "inference",
                "llmops.role": role,
            },
            "annotations": annotations,
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": app_name}},
            "template": {
                "metadata": {
                    "labels": {
                        "app": app_name,
                        "llmops.component": "inference",
                        "llmops.role": role,
                    },
                    "annotations": annotations,
                },
                "spec": {
                    "containers": [
                        {
                            "name": deployment_name,
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "QWEN_MODEL_ID", "value": effective_model_id},
                                {"name": "QWEN_DEFAULT_MAX_NEW_TOKENS", "value": str(max_new_tokens)},
                                {"name": "QWEN_DEFAULT_TEMPERATURE", "value": str(temperature)},
                                {"name": "QWEN_TORCH_THREADS", "value": str(torch_threads)},
                            ],
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": 8000},
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "startupProbe": {
                                "httpGet": {"path": "/healthz", "port": 8000},
                                "periodSeconds": 10,
                                "failureThreshold": 180,
                                "timeoutSeconds": 5,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": 8000},
                                "periodSeconds": 30,
                                "timeoutSeconds": 5,
                            },
                            "resources": {
                                "requests": {"cpu": cpu_request, "memory": memory_request},
                                "limits": {"memory": memory_limit},
                            },
                            "volumeMounts": volume_mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


def build_qwen_inference_manifests(
    *,
    role: str,
    model_id: str,
    namespace: str = DEFAULT_NAMESPACE,
    image: str = DEFAULT_IMAGE,
    model_uri: str | None = None,
    replicas: int = DEFAULT_REPLICAS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    torch_threads: int = DEFAULT_TORCH_THREADS,
    cpu_request: str = DEFAULT_CPU_REQUEST,
    memory_request: str = DEFAULT_MEMORY_REQUEST,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    local_model_source_path: str | None = None,
    local_model_mount_path: str = DEFAULT_LOCAL_MODEL_MOUNT_PATH,
) -> list[dict[str, Any]]:
    defaults = role_defaults(role)
    documents: list[dict[str, Any]] = []

    if role == "baseline":
        documents.append(build_service("qwen-inference", namespace, defaults["app_name"], role))

    documents.append(build_service(defaults["service_name"], namespace, defaults["app_name"], role))
    documents.append(
        build_deployment(
            namespace=namespace,
            deployment_name=defaults["deployment_name"],
            app_name=defaults["app_name"],
            role=role,
            image=image,
            model_id=model_id,
            model_uri=model_uri,
            replicas=replicas,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_threads=torch_threads,
            cpu_request=cpu_request,
            memory_request=memory_request,
            memory_limit=memory_limit,
            local_model_source_path=local_model_source_path,
            local_model_mount_path=local_model_mount_path,
        )
    )
    return documents


def dump_qwen_inference_manifests(**kwargs: Any) -> str:
    return yaml.safe_dump_all(build_qwen_inference_manifests(**kwargs), sort_keys=False)
