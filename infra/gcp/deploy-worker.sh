#!/usr/bin/env bash
# Deploy the RQ worker as a Cloud Run worker pool (no inbound HTTP traffic).
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-worker.sh
#
# Worker pools are not request-driven, so they keep instances running
# regardless of traffic. The same image is reused; we override the entrypoint.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var WORKER_POOL
require_var WORKER_SA
require_var CLOUDSQL_INSTANCE
require_var VPC_CONNECTOR
require_var GCS_STATE_BUCKET

WORKER_ENV_VARS=(
  "ENVIRONMENT=production"
  "STATE_STORAGE_BACKEND=gcs"
  "STATE_DB_BACKEND=postgres"
  "GCS_STATE_BUCKET=${GCS_STATE_BUCKET}"
  "CLOUD_RUN_REGION=${REGION}"
  "ASYNC_JOB_BACKEND=rq"
  "ASYNC_WORKER_QUEUES=default,screens,reports"
  "ASYNC_JOB_COMPLETED_TTL_SECONDS=86400"
  "ASYNC_JOB_FAILED_TTL_SECONDS=604800"
)

# `gcloud run worker-pools` is in beta on most projects today; drop the
# `beta` token if your project has it GA.
gcloud beta run worker-pools deploy "${WORKER_POOL}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${WORKER_SA}" \
  --add-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --vpc-connector="${VPC_CONNECTOR}" \
  --vpc-egress=private-ranges-only \
  --command=python \
  --args=-m,api.rq_worker,default,screens,reports \
  --set-env-vars="$(join_csv WORKER_ENV_VARS)" \
  --set-secrets="$(join_csv WORKER_SECRETS)" \
  --cpu=2 \
  --memory=2Gi \
  --min-instances=1 \
  --max-instances=3
