#!/usr/bin/env bash
# Deploy the warm ontology query worker pool.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-ontology-worker.sh
#
# Tunables:
#   ONTOLOGY_WORKER_POOL=talisman-ontology-worker
#   ONTOLOGY_WORKER_CPU=1  ONTOLOGY_WORKER_MEMORY=512Mi
#   ONTOLOGY_WORKER_INSTANCES=1
#   ONTOLOGY_WORKER_POLL_INTERVAL_SECONDS=0.25

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

ONTOLOGY_WORKER_POOL="${ONTOLOGY_WORKER_POOL:-talisman-ontology-worker}"

if [[ -n "${ONTOLOGY_WORKER_MAX_INSTANCES:-}" && -z "${ONTOLOGY_WORKER_INSTANCES+x}" ]]; then
  echo "ONTOLOGY_WORKER_MAX_INSTANCES is ignored; Cloud Run worker pools use fixed --instances." >&2
  echo "Set ONTOLOGY_WORKER_INSTANCES to change the worker pool capacity." >&2
fi

ONTOLOGY_WORKER_INSTANCES="${ONTOLOGY_WORKER_INSTANCES:-${ONTOLOGY_WORKER_MIN_INSTANCES:-1}}"

mapfile -t COMMON_ENV < <(common_env_vars)

ONTOLOGY_WORKER_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "ASYNC_DISPATCH_BACKEND_ONTOLOGY=warm_worker"
  "ASYNC_QUEUE_ONTOLOGY=ontology"
  "JOB_WORKER_JOB_TYPE=ontology"
  "JOB_WORKER_QUEUE=ontology"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "POSTGRES_POOL_MAX_SIZE=${POSTGRES_POOL_MAX_SIZE:-2}"
  "ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED=${ASYNC_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "ONTOLOGY_JOB_SUCCESS_READ_AUDIT_ENABLED=${ONTOLOGY_JOB_SUCCESS_READ_AUDIT_ENABLED:-false}"
  "ONTOLOGY_WORKER_POLL_INTERVAL_SECONDS=${ONTOLOGY_WORKER_POLL_INTERVAL_SECONDS:-0.25}"
  "SENTRY_ENVIRONMENT=production"
  "SENTRY_TRACES_SAMPLE_RATE=${SENTRY_TRACES_SAMPLE_RATE:-0.05}"
  "SENTRY_PROFILES_SAMPLE_RATE=${SENTRY_PROFILES_SAMPLE_RATE:-0.0}"
)

gcloud run worker-pools deploy "${ONTOLOGY_WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.job_worker_loop,run \
  --set-env-vars="$(join_kv "${ONTOLOGY_WORKER_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${WORKER_SECRETS[@]}")" \
  --cpu="${ONTOLOGY_WORKER_CPU:-1}" \
  --memory="${ONTOLOGY_WORKER_MEMORY:-512Mi}" \
  --instances="${ONTOLOGY_WORKER_INSTANCES}"
