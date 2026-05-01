#!/usr/bin/env bash
# Deploy the RQ worker as a Cloud Run worker pool (no inbound HTTP traffic).
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-worker.sh
# IMAGE_TAG defaults to the current short git SHA (see lib.sh).
#
# Worker pools are not request-driven, so they keep instances running
# regardless of traffic. The same image is reused; we override the entrypoint.
#
# Tunables (override via environment):
#   WORKER_CPU=2  WORKER_MEMORY=2Gi  WORKER_INSTANCES=1
#   WORKER_QUEUES=default,screens,reports

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_POOL
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var VPC_NETWORK
require_var VPC_SUBNET
require_var GCS_STATE_BUCKET
require_active_project
require_image_exists

mapfile -t COMMON_ENV < <(common_env_vars)

WORKER_QUEUES="${WORKER_QUEUES:-default,screens,reports}"
WORKER_ENV_VARS=(
  "${COMMON_ENV[@]}"
  "ASYNC_JOB_BACKEND=rq"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
  "ASYNC_WORKER_QUEUES=${WORKER_QUEUES}"
)

# `gcloud run worker-pools` is in beta on most projects today; drop the
# `beta` token if your project has it GA.
gcloud beta run worker-pools deploy "${WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --add-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --network="${VPC_NETWORK}" \
  --subnet="${VPC_SUBNET}" \
  --vpc-egress=private-ranges-only \
  --command=python \
  --args="-m,api.rq_worker,${WORKER_QUEUES}" \
  --set-env-vars="$(join_kv "${WORKER_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${WORKER_SECRETS[@]}")" \
  --cpu="${WORKER_CPU:-2}" \
  --memory="${WORKER_MEMORY:-2Gi}" \
  --instances="${WORKER_INSTANCES:-1}"
