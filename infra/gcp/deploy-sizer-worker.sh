#!/usr/bin/env bash
# Deploy the warm portfolio sizer worker pool.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-sizer-worker.sh
#
# Tunables:
#   SIZER_WORKER_POOL=talisman-sizer-worker
#   SIZER_WORKER_CPU=1  SIZER_WORKER_MEMORY=512Mi
#   SIZER_WORKER_INSTANCES=1
#   SIZER_WORKER_POLL_INTERVAL_SECONDS=0.25

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

SIZER_WORKER_POOL="${SIZER_WORKER_POOL:-talisman-sizer-worker}"

if [[ -n "${SIZER_WORKER_MAX_INSTANCES:-}" && -z "${SIZER_WORKER_INSTANCES+x}" ]]; then
  echo "SIZER_WORKER_MAX_INSTANCES is ignored; Cloud Run worker pools use fixed --instances." >&2
  echo "Set SIZER_WORKER_INSTANCES to change the worker pool capacity." >&2
fi

SIZER_WORKER_INSTANCES="${SIZER_WORKER_INSTANCES:-${SIZER_WORKER_MIN_INSTANCES:-1}}"

mapfile -t COMMON_ENV < <(common_env_vars)

SIZER_WORKER_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "ASYNC_DISPATCH_BACKEND_SIZER=warm_worker"
  "ASYNC_QUEUE_SIZER=sizer"
  "JOB_WORKER_JOB_TYPE=sizer"
  "JOB_WORKER_QUEUE=sizer"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "POSTGRES_POOL_MAX_SIZE=${POSTGRES_POOL_MAX_SIZE:-2}"
  "ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED=${ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "SIZER_WORKER_POLL_INTERVAL_SECONDS=${SIZER_WORKER_POLL_INTERVAL_SECONDS:-0.25}"
)

gcloud run worker-pools deploy "${SIZER_WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.job_worker_loop,run \
  --set-env-vars="$(join_kv "${SIZER_WORKER_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${WORKER_SECRETS[@]}")" \
  --cpu="${SIZER_WORKER_CPU:-1}" \
  --memory="${SIZER_WORKER_MEMORY:-512Mi}" \
  --instances="${SIZER_WORKER_INSTANCES}"
