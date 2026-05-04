#!/usr/bin/env bash
# Deploy the generic on-demand async executor as a Cloud Run Job.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-async-job.sh
# IMAGE_TAG defaults to the current short git SHA (see lib.sh).
#
# Tunables:
#   ASYNC_JOB_RUNNER_JOB=talisman-async-job
#   ASYNC_JOB_CPU=2  ASYNC_JOB_MEMORY=2Gi
#   ASYNC_JOB_TIMEOUT=3600  ASYNC_JOB_MAX_RETRIES=0

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

ASYNC_JOB_RUNNER_JOB="${ASYNC_JOB_RUNNER_JOB:-talisman-async-job}"

mapfile -t COMMON_ENV < <(common_env_vars)

ASYNC_JOB_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "ASYNC_CLOUD_RUN_JOB=${ASYNC_JOB_RUNNER_JOB}"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "GOVERNANCE_OUTBOX_BATCH_SIZE=${GOVERNANCE_OUTBOX_BATCH_SIZE:-50}"
  "GOVERNANCE_OUTBOX_LEASE_SECONDS=${GOVERNANCE_OUTBOX_LEASE_SECONDS:-300}"
  "GOVERNANCE_OUTBOX_MAX_ATTEMPTS=${GOVERNANCE_OUTBOX_MAX_ATTEMPTS:-8}"
  "GOVERNANCE_OUTBOX_RETRY_BASE_SECONDS=${GOVERNANCE_OUTBOX_RETRY_BASE_SECONDS:-30}"
  "GOVERNANCE_OUTBOX_RETRY_MAX_SECONDS=${GOVERNANCE_OUTBOX_RETRY_MAX_SECONDS:-3600}"
)

ASYNC_JOB_SECRETS=()
for secret in "${WORKER_SECRETS[@]}"; do
  [[ "${secret}" == REDIS_URL=* ]] && continue
  ASYNC_JOB_SECRETS+=("${secret}")
done

gcloud run jobs deploy "${ASYNC_JOB_RUNNER_JOB}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.async_job_runner,run \
  --set-env-vars="$(join_kv "${ASYNC_JOB_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${ASYNC_JOB_SECRETS[@]}")" \
  --cpu="${ASYNC_JOB_CPU:-2}" \
  --memory="${ASYNC_JOB_MEMORY:-2Gi}" \
  --max-retries="${ASYNC_JOB_MAX_RETRIES:-0}" \
  --task-timeout="${ASYNC_JOB_TIMEOUT:-3600}"
