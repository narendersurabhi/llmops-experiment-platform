#!/usr/bin/env bash
set -euo pipefail

PIPELINE_VERSION="${KFP_VERSION:-2.15.0}"
KFP_NAMESPACE="${KFP_NAMESPACE:-kubeflow}"
CRD_TIMEOUT="${KFP_CRD_TIMEOUT:-120s}"
DEPLOY_TIMEOUT="${KFP_DEPLOY_TIMEOUT:-600s}"
DISABLE_PROXY_AGENT="${KFP_DISABLE_PROXY_AGENT:-false}"
DISABLE_METADATA_WRITER="${KFP_DISABLE_METADATA_WRITER:-false}"
PATCH_SEAWEEDFS_SERVICE="${KFP_PATCH_SEAWEEDFS_SERVICE_PORT_9000:-false}"
CLUSTER_SCOPED_MANIFEST="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
ENV_MANIFEST="github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${PIPELINE_VERSION}"

kubectl apply -k "$CLUSTER_SCOPED_MANIFEST"
kubectl wait --for=condition=established --timeout="$CRD_TIMEOUT" crd/applications.app.k8s.io
kubectl apply -k "$ENV_MANIFEST"

if [[ "$PATCH_SEAWEEDFS_SERVICE" == "true" ]]; then
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: seaweedfs
  namespace: ${KFP_NAMESPACE}
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
EOF
fi

if [[ "$DISABLE_PROXY_AGENT" == "true" ]]; then
    kubectl scale deployment/proxy-agent -n "$KFP_NAMESPACE" --replicas=0
fi

if [[ "$DISABLE_METADATA_WRITER" == "true" ]]; then
    kubectl scale deployment/metadata-writer -n "$KFP_NAMESPACE" --replicas=0
fi

deployments="$(kubectl get deployments -n "$KFP_NAMESPACE" -o name)"
if [[ -z "$deployments" ]]; then
    echo "No deployments found in namespace ${KFP_NAMESPACE} after applying Kubeflow Pipelines manifests." >&2
    exit 1
fi

while IFS= read -r deployment; do
    [[ -z "$deployment" ]] && continue
    if [[ "$DISABLE_PROXY_AGENT" == "true" && "$deployment" == "deployment.apps/proxy-agent" ]]; then
        continue
    fi
    if [[ "$DISABLE_METADATA_WRITER" == "true" && "$deployment" == "deployment.apps/metadata-writer" ]]; then
        continue
    fi
    kubectl rollout status -n "$KFP_NAMESPACE" "$deployment" --timeout="$DEPLOY_TIMEOUT"
done <<EOF
${deployments}
EOF

cat <<EOF
Kubeflow Pipelines installed in namespace ${KFP_NAMESPACE}.
Port-forward the UI/API with:
  kubectl port-forward -n ${KFP_NAMESPACE} svc/ml-pipeline-ui 3000:80
Then use:
  export KFP_HOST=http://localhost:3000
EOF

if [[ "$DISABLE_PROXY_AGENT" == "true" || "$DISABLE_METADATA_WRITER" == "true" ]]; then
cat <<EOF

Local compatibility adjustments applied:
  disable proxy-agent: ${DISABLE_PROXY_AGENT}
  disable metadata-writer: ${DISABLE_METADATA_WRITER}
  patch seaweedfs service port 9000: ${PATCH_SEAWEEDFS_SERVICE}
EOF
fi
