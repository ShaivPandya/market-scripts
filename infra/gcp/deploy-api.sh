#!/usr/bin/env bash
# Deploy the FastAPI service to Cloud Run.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-api.sh
# IMAGE_TAG defaults to the current short git SHA (see lib.sh).
#
# Tunables (override via environment):
#   API_CPU=2  API_MEMORY=2Gi  API_CONCURRENCY=40
#   API_MIN_INSTANCES=0  API_MAX_INSTANCES=10  API_TIMEOUT=300

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE
require_var API_SA
require_var CLOUDSQL_INSTANCE
require_var VPC_NETWORK
require_var VPC_SUBNET
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

mapfile -t COMMON_ENV < <(common_env_vars)

API_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "CLOUD_RUN_JOBS_ENABLED=true"
  "ASYNC_JOB_BACKEND=rq"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "CORS_ORIGINS=${CORS_ORIGINS}"
)

gcloud run deploy "${API_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${API_SA}" \
  --add-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --network="${VPC_NETWORK}" \
  --subnet="${VPC_SUBNET}" \
  --vpc-egress=private-ranges-only \
  --set-env-vars="$(join_kv "${API_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${API_SECRETS[@]}")" \
  --cpu="${API_CPU:-2}" \
  --memory="${API_MEMORY:-2Gi}" \
  --concurrency="${API_CONCURRENCY:-40}" \
  --min-instances="${API_MIN_INSTANCES:-0}" \
  --max-instances="${API_MAX_INSTANCES:-10}" \
  --timeout="${API_TIMEOUT:-300}" \
  --port=8080 \
  --allow-unauthenticated
