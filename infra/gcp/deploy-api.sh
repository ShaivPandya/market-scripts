#!/usr/bin/env bash
# Deploy the FastAPI service to Cloud Run.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-api.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE
require_var API_SA
require_var CLOUDSQL_INSTANCE
require_var VPC_CONNECTOR
require_var GCS_STATE_BUCKET

API_ENV_VARS=(
  "ENVIRONMENT=production"
  "STATE_STORAGE_BACKEND=gcs"
  "STATE_DB_BACKEND=postgres"
  "GCS_STATE_BUCKET=${GCS_STATE_BUCKET}"
  "CLOUD_RUN_REGION=${REGION}"
  "CLOUD_RUN_JOBS_ENABLED=true"
  "ASYNC_JOB_BACKEND=rq"
  "ASYNC_WORKER_QUEUES=default,screens,reports"
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
  --vpc-connector="${VPC_CONNECTOR}" \
  --vpc-egress=private-ranges-only \
  --set-env-vars="$(join_csv API_ENV_VARS)" \
  --set-secrets="$(join_csv API_SECRETS)" \
  --cpu=2 \
  --memory=2Gi \
  --concurrency=40 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=300 \
  --port=8080 \
  --no-allow-unauthenticated
