from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any


def is_enabled(env_var: str, default: str = "true") -> bool:
    return os.getenv(env_var, default).lower() in {"1", "true", "yes", "on"}


def run_kubectl(args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )


def build_seaweedfs_service_manifest(namespace: str) -> str:
    return f"""apiVersion: v1
kind: Service
metadata:
  name: seaweedfs
  namespace: {namespace}
spec:
  selector:
    app: seaweedfs
    application-crd-id: kubeflow-pipelines
  ports:
    - name: http-iam
      port: 8111
      protocol: TCP
      targetPort: 8111
    - name: http-s3
      port: 8333
      protocol: TCP
      targetPort: 8333
    - name: http-s3-compat
      port: 9000
      protocol: TCP
      targetPort: 8333
    - name: http-master
      port: 9333
      protocol: TCP
      targetPort: 9333
    - name: grpc-master
      port: 19333
      protocol: TCP
      targetPort: 19333
    - name: grpc-filer
      port: 18888
      protocol: TCP
      targetPort: 18888
    - name: http-filer
      port: 8888
      protocol: TCP
      targetPort: 8888
"""


def ensure_seaweedfs_service_port(namespace: str) -> None:
    if not is_enabled("KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000", "true"):
        return

    run_kubectl(["apply", "-f", "-"], input_text=build_seaweedfs_service_manifest(namespace))


def strip_run_as_non_root(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("runAsNonRoot", None)
        for value in node.values():
            strip_run_as_non_root(value)
        return

    if isinstance(node, list):
        for item in node:
            strip_run_as_non_root(item)


def find_workflow_name(run_id: str, namespace: str, timeout_seconds: int = 60) -> str | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run_kubectl(
            [
                "get",
                "workflows.argoproj.io",
                "-n",
                namespace,
                "-l",
                f"pipeline/runid={run_id}",
                "-o",
                "json",
            ]
        )
        payload = json.loads(result.stdout)
        items = payload.get("items", [])
        if items:
            return items[0]["metadata"]["name"]
        time.sleep(2)
    return None


def patch_workflow_security_context(workflow_name: str, namespace: str) -> bool:
    result = run_kubectl(["get", "workflow", workflow_name, "-n", namespace, "-o", "json"])
    workflow = json.loads(result.stdout)
    workflow.pop("status", None)

    metadata = workflow.get("metadata", {})
    for key in ["managedFields", "creationTimestamp", "generation", "uid"]:
        metadata.pop(key, None)

    spec = workflow.get("spec", {})
    before = json.dumps(spec, sort_keys=True)
    strip_run_as_non_root(spec)
    after = json.dumps(spec, sort_keys=True)

    if before == after:
        return False

    run_kubectl(["replace", "-f", "-"], input_text=json.dumps(workflow))
    run_kubectl(
        [
            "delete",
            "pod",
            "-n",
            namespace,
            "-l",
            f"workflows.argoproj.io/workflow={workflow_name}",
            "--ignore-not-found",
        ]
    )
    return True


def apply_local_k8s_workaround(run_id: str) -> str | None:
    if not is_enabled("KFP_LOCAL_COMPAT_WORKAROUND", "true"):
        return None

    namespace = os.getenv("KFP_RUN_NAMESPACE", "kubeflow")
    ensure_seaweedfs_service_port(namespace)
    workflow_name = find_workflow_name(run_id, namespace)
    if not workflow_name:
        return None

    patched = patch_workflow_security_context(workflow_name, namespace)
    if not patched:
        return workflow_name
    return workflow_name


def local_ui_metadata_note() -> str:
    return (
        "Local Docker Desktop mode disables metadata-writer, so the KFP Run Details page may show "
        "\"Cannot find context ...\" even while the Workflow is running. Use kubectl workflow/pod status "
        "to inspect progress."
    )
