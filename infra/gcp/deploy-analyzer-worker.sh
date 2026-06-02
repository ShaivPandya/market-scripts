#!/usr/bin/env bash
# Deploy the warm portfolio analyzer worker pool.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-analyzer-worker.sh
#
# Tunables:
#   ANALYZER_WORKER_POOL=talisman-analyzer-worker
#   ANALYZER_WORKER_CPU=1  ANALYZER_WORKER_MEMORY=1Gi
#   ANALYZER_WORKER_INSTANCES=1
#   ANALYZER_WORKER_POLL_INTERVAL_SECONDS=0.25

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

ANALYZER_WORKER_POOL="${ANALYZER_WORKER_POOL:-talisman-analyzer-worker}"

if [[ -n "${ANALYZER_WORKER_MAX_INSTANCES:-}" && -z "${ANALYZER_WORKER_INSTANCES+x}" ]]; then
  echo "ANALYZER_WORKER_MAX_INSTANCES is ignored; Cloud Run worker pools use fixed --instances." >&2
  echo "Set ANALYZER_WORKER_INSTANCES to change the worker pool capacity." >&2
fi

ANALYZER_WORKER_INSTANCES="${ANALYZER_WORKER_INSTANCES:-${ANALYZER_WORKER_MIN_INSTANCES:-1}}"

mapfile -t COMMON_ENV < <(common_env_vars)

ANALYZER_WORKER_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "ASYNC_DISPATCH_BACKEND_ANALYZER=warm_worker"
  "ASYNC_QUEUE_ANALYZER=analyzer"
  "JOB_WORKER_JOB_TYPE=analyzer"
  "JOB_WORKER_QUEUE=analyzer"
  "ASYNC_ANALYZER_COMPLETED_TTL_SECONDS=300"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "POSTGRES_POOL_MAX_SIZE=${POSTGRES_POOL_MAX_SIZE:-2}"
  "ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED=${ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "ANALYZER_WORKER_POLL_INTERVAL_SECONDS=${ANALYZER_WORKER_POLL_INTERVAL_SECONDS:-0.25}"
  "SENTRY_ENVIRONMENT=production"
  "SENTRY_TRACES_SAMPLE_RATE=${SENTRY_TRACES_SAMPLE_RATE:-0.05}"
  "SENTRY_PROFILES_SAMPLE_RATE=${SENTRY_PROFILES_SAMPLE_RATE:-0.0}"
)

gcloud run worker-pools deploy "${ANALYZER_WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.job_worker_loop,run \
  --set-env-vars="$(join_kv "${ANALYZER_WORKER_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${WORKER_SECRETS[@]}")" \
  --cpu="${ANALYZER_WORKER_CPU:-1}" \
  --memory="${ANALYZER_WORKER_MEMORY:-1Gi}" \
  --instances="${ANALYZER_WORKER_INSTANCES}"
