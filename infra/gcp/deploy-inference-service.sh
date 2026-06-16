#!/usr/bin/env bash
# Provision the governed first-party inference service on private GCP GPU Cloud Run.
# Usage:
#   CANDIDATE_ID=<approved-id> ./infra/gcp/deploy-inference-service.sh
#
# Prerequisites:
#   - Approved registry candidate with matching artifact digest
#   - Inference image built from infra/gcp/Dockerfile.inference
#   - TALISMAN_API_KEY and TALISMAN_BASE_URL secrets populated via setup-secrets.sh
#
# Tunables (override via environment or infra/gcp/config.sh):
#   INFERENCE_SERVICE=talisman-inference-nonprod
#   INFERENCE_IMAGE_NAME=inference
#   INFERENCE_GPU_TYPE=nvidia-l4
#   INFERENCE_GPU_COUNT=1
#   INFERENCE_CPU=4
#   INFERENCE_MEMORY=16Gi
#   INFERENCE_MIN_INSTANCES=0
#   INFERENCE_MAX_INSTANCES=1
#   INFERENCE_COMBINATION_ID=qwen-managed-gpu
#   INFERENCE_ENVIRONMENT=nonprod

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_var GCS_STATE_BUCKET
require_active_project

PYTHON_BIN="$(python_bin)"
_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)"

INFERENCE_SERVICE="${INFERENCE_SERVICE:-talisman-inference-nonprod}"
INFERENCE_IMAGE_NAME="${INFERENCE_IMAGE_NAME:-inference}"
INFERENCE_IMAGE_TAG="${INFERENCE_IMAGE_TAG:-${IMAGE_TAG:-latest}}"
INFERENCE_SA="${INFERENCE_SA:-${WORKER_SA:-}}"
INFERENCE_ENVIRONMENT="${INFERENCE_ENVIRONMENT:-nonprod}"
INFERENCE_COMBINATION_ID="${INFERENCE_COMBINATION_ID:-qwen-managed-gpu}"
CANDIDATE_ID="${CANDIDATE_ID:-}"

if [[ -z "${INFERENCE_SA}" ]]; then
  echo "INFERENCE_SA or WORKER_SA must be set in infra/gcp/config.sh." >&2
  exit 1
fi

log() { printf '\n[inference-deploy] %s\n' "$*"; }

inference_image_uri() {
  echo "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${INFERENCE_IMAGE_NAME}:${INFERENCE_IMAGE_TAG}"
}

require_inference_image_exists() {
  if [[ "${SKIP_IMAGE_CHECK:-0}" == "1" ]]; then
    return 0
  fi
  if ! gcloud artifacts docker images describe "$(inference_image_uri)" \
        --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Inference image $(inference_image_uri) not found in Artifact Registry." >&2
    echo "Build it first, e.g.:" >&2
    echo "  docker build -f infra/gcp/Dockerfile.inference -t $(inference_image_uri) ." >&2
    echo "  docker push $(inference_image_uri)" >&2
    exit 1
  fi
}

log "Validate deployment eligibility and build manifest"
_manifest_args=(
  -m decision_quality.agent_inference_deployment build-manifest
  --environment "${INFERENCE_ENVIRONMENT}"
  --combination-id "${INFERENCE_COMBINATION_ID}"
)
if [[ -n "${CANDIDATE_ID}" ]]; then
  _manifest_args+=(--candidate-id "${CANDIDATE_ID}")
fi

cd "${_repo_root}"
_manifest_json="$("${PYTHON_BIN}" "${_manifest_args[@]}")"
_manifest_path="$(printf '%s' "${_manifest_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["manifest_path"])')"
_candidate_id="$(printf '%s' "${_manifest_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["manifest"]["candidate_id"])')"
_served_model="$(printf '%s' "${_manifest_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["manifest"]["served_model_name"])')"

log "Upload deployment manifest to GCS"
_manifest_object="inference/deployments/${INFERENCE_ENVIRONMENT}/${_candidate_id}.json"
gsutil cp "${_manifest_path}" "gs://${GCS_STATE_BUCKET}/${_manifest_object}"

require_inference_image_exists

INFERENCE_ENV_VARS=(
  "ENVIRONMENT=${INFERENCE_ENVIRONMENT}"
  "TALISMAN_DEPLOYMENT_MANIFEST=/runtime/deployment_manifest.json"
  "TALISMAN_REGISTRY_PATH=/runtime/registry.json"
  "INFERENCE_ALLOW_SERVE=1"
  "INFERENCE_HOST=0.0.0.0"
  "INFERENCE_PORT=8080"
  "TALISMAN_MODEL_LOW=${_served_model}"
  "TALISMAN_MODEL_MID=${_served_model}"
  "TALISMAN_MODEL_HIGH=${_served_model}"
)

if [[ ${#INFERENCE_SECRETS[@]:-0} -eq 0 ]]; then
  INFERENCE_DEPLOY_SECRETS=("TALISMAN_API_KEY=TALISMAN_API_KEY:latest")
else
  INFERENCE_DEPLOY_SECRETS=("${INFERENCE_SECRETS[@]}")
fi

log "Deploy private GPU Cloud Run service ${INFERENCE_SERVICE}"
gcloud run deploy "${INFERENCE_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(inference_image_uri)" \
  --service-account="${INFERENCE_SA}" \
  --set-env-vars="$(join_kv "${INFERENCE_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${INFERENCE_DEPLOY_SECRETS[@]}")" \
  --cpu="${INFERENCE_CPU:-4}" \
  --memory="${INFERENCE_MEMORY:-16Gi}" \
  --gpu="${INFERENCE_GPU_COUNT:-1}" \
  --gpu-type="${INFERENCE_GPU_TYPE:-nvidia-l4}" \
  --min-instances="${INFERENCE_MIN_INSTANCES:-0}" \
  --max-instances="${INFERENCE_MAX_INSTANCES:-1}" \
  --timeout="${INFERENCE_TIMEOUT:-900}" \
  --port=8080 \
  --no-allow-unauthenticated \
  --ingress=internal-and-cloud-load-balancing

log "Record service URL for TALISMAN_BASE_URL secret rotation"
_service_url="$(gcloud run services describe "${INFERENCE_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)')"
if [[ -n "${_service_url}" ]]; then
  printf '%s/v1' "${_service_url}" | gcloud secrets versions add TALISMAN_BASE_URL \
    --project="${PROJECT_ID}" \
    --data-file=- >/dev/null 2>&1 || true
  echo "  OpenAI-compatible base URL: ${_service_url}/v1"
  echo "  Rotate TALISMAN_BASE_URL in Secret Manager if this deploy should become active."
fi

log "Done. Run contract smoke with TALISMAN_INFERENCE_SMOKE=1 after wiring secrets."
