#!/usr/bin/env bash
# Deploy the warm agent chat worker pool.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-agent-worker.sh
#
# Tunables:
#   AGENT_WORKER_POOL=talisman-agent-worker
#   AGENT_WORKER_CPU=2  AGENT_WORKER_MEMORY=2Gi
#   AGENT_WORKER_MIN_INSTANCES=1  AGENT_WORKER_MAX_INSTANCES=3
#   AGENT_WORKER_POLL_INTERVAL_SECONDS=0.25

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

AGENT_WORKER_POOL="${AGENT_WORKER_POOL:-talisman-agent-worker}"

mapfile -t COMMON_ENV < <(common_env_vars)

AGENT_WORKER_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=cloud_run_jobs"
  "AGENT_CHAT_DISPATCH_BACKEND=warm_worker"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "AGENT_WORKER_POLL_INTERVAL_SECONDS=${AGENT_WORKER_POLL_INTERVAL_SECONDS:-0.25}"
)

gcloud beta run worker-pools deploy "${AGENT_WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.agent_worker_loop,run \
  --set-env-vars="$(join_kv "${AGENT_WORKER_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${WORKER_SECRETS[@]}")" \
  --cpu="${AGENT_WORKER_CPU:-2}" \
  --memory="${AGENT_WORKER_MEMORY:-2Gi}" \
  --min-instances="${AGENT_WORKER_MIN_INSTANCES:-1}" \
  --max-instances="${AGENT_WORKER_MAX_INSTANCES:-3}"
