# Kubeflow Pipelines Installation

This repository uses **standalone Kubeflow Pipelines**, not the full Kubeflow AI reference platform.

## Install

Use the repository helper:

```bash
make infra-apply-kubeflow
```

This installs the official standalone Kubeflow Pipelines manifests from:

- https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/
- https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/connect-api/

The helper script applies:

- `github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources`
- `github.com/kubeflow/pipelines/manifests/kustomize/env/dev`

with `KFP_VERSION` defaulting to `2.15.0`.

For Docker Desktop on Apple Silicon, the repository defaults to:

- `KFP_DISABLE_PROXY_AGENT=true`
- `KFP_DISABLE_METADATA_WRITER=true`
- `KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000=true`

Reason:

- `proxy-agent` is optional for local use because the UI/API is accessed via `kubectl port-forward`
- some upstream images used by these components may not provide a usable arm64 variant for local Docker Desktop nodes

The local submit scripts also patch newly created Workflow objects to remove `runAsNonRoot` from Argo-managed pod specs. This avoids a Docker Desktop compatibility issue where the `argoexec` init container is rejected before pipeline tasks start.

They also ensure the `seaweedfs` service exposes port `9000` targeting the SeaweedFS S3 endpoint on `8333`, because KFP launcher artifact uploads may still target `seaweedfs.kubeflow:9000` in local standalone installs.

## Access

```bash
make kfp-port-forward
export KFP_HOST="http://localhost:3000"
```

Then open `http://localhost:3000` or use the SDK submit scripts against `KFP_HOST`.
