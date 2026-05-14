#!/usr/bin/env bash
# Deploy the FastAPI service to Cloud Run.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-api.sh
# IMAGE_TAG defaults to the current short git SHA (see lib.sh).
#
# Tunables (override via environment):
#   API_CPU=1  API_MEMORY=1Gi  API_CONCURRENCY=20
#   API_MIN_INSTANCES=0  API_MAX_INSTANCES=10  API_TIMEOUT=300
#   ASYNC_JOB_RUNNER_JOB=talisman-async-job

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE
require_var API_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

ASYNC_JOB_RUNNER_JOB="${ASYNC_JOB_RUNNER_JOB:-talisman-async-job}"

# The API service is now primarily request routing, short interactive reads, and
# async job dispatch/polling; heavier analysis runs in Cloud Run Jobs. Default to
# one vCPU and 1Gi with moderate concurrency so a single instance is not
# overcommitted, while max instances can still absorb interactive bursts.
API_CPU="${API_CPU:-1}"
API_MEMORY="${API_MEMORY:-1Gi}"
API_CONCURRENCY="${API_CONCURRENCY:-20}"

mapfile -t COMMON_ENV < <(common_env_vars)

# Resolve full git SHA for release identity env vars
_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
_full_sha="$(git -C "${_repo_root:-.}" rev-parse HEAD 2>/dev/null || echo "unknown")"

API_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "TALISMAN_RELEASE_GIT_SHA=${_full_sha}"
  "TALISMAN_RELEASE_GIT_SHA_SHORT=${IMAGE_TAG}"
  "TALISMAN_RELEASE_IMAGE_TAG=${IMAGE_TAG}"
  "TALISMAN_RELEASE_ENVIRONMENT=production"
  "CLOUD_RUN_JOBS_ENABLED=true"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "AGENT_CHAT_DISPATCH_BACKEND=warm_worker"
  "ASYNC_DISPATCH_BACKEND_ANALYZER=cloud_run_jobs"
  "ASYNC_DISPATCH_BACKEND_SIZER=inline"
  "ASYNC_DISPATCH_BACKEND_ONTOLOGY=cloud_run_jobs"
  "ASYNC_QUEUE_ANALYZER=analyzer"
  "ASYNC_QUEUE_SIZER=sizer"
  "ASYNC_QUEUE_ONTOLOGY=ontology"
  "ASYNC_CLOUD_RUN_JOB=${ASYNC_JOB_RUNNER_JOB}"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "POSTGRES_POOL_MAX_SIZE=${POSTGRES_POOL_MAX_SIZE:-4}"
  "ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED=${ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "ONTOLOGY_JOB_SUCCESS_READ_AUDIT_ENABLED=${ONTOLOGY_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "AGENT_DELTA_FLUSH_INTERVAL_MS=${AGENT_DELTA_FLUSH_INTERVAL_MS:-500}"
  "AGENT_DELTA_FLUSH_BYTES=${AGENT_DELTA_FLUSH_BYTES:-1024}"
  "CORS_ORIGINS=${CORS_ORIGINS}"
)

API_DEPLOY_SECRETS=()
for secret in "${API_SECRETS[@]}"; do
  [[ "${secret}" == REDIS_URL=* ]] && continue
  API_DEPLOY_SECRETS+=("${secret}")
done

gcloud run deploy "${API_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${API_SA}" \
  --add-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --set-env-vars="$(join_kv "${API_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${API_DEPLOY_SECRETS[@]}")" \
  --cpu="${API_CPU}" \
  --memory="${API_MEMORY}" \
  --concurrency="${API_CONCURRENCY}" \
  --min-instances="${API_MIN_INSTANCES:-0}" \
  --max-instances="${API_MAX_INSTANCES:-10}" \
  --timeout="${API_TIMEOUT:-300}" \
  --port=8080 \
  --allow-unauthenticated
