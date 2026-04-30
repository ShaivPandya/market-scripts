#!/usr/bin/env bash
# Create or update the Cloud Run Job that runs api.gcp_state_migration.
# Usage: IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-migration-job.sh
# Execute (after deploy):
#   MIGRATION_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
#   gcloud run jobs execute "${MIGRATION_JOB}" \
#     --project="${PROJECT_ID}" --region="${REGION}" \
#     --update-env-vars="MIGRATION_RUN_ID=${MIGRATION_RUN_ID}"

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var MIGRATION_JOB
require_var MIGRATOR_SA
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET

MIGRATION_ENV_VARS=(
  "ENVIRONMENT=production"
  "STATE_STORAGE_BACKEND=gcs"
  "GCS_STATE_BUCKET=${GCS_STATE_BUCKET}"
)

gcloud run jobs deploy "${MIGRATION_JOB}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${MIGRATOR_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,api.gcp_state_migration,migrate \
  --set-env-vars="$(join_kv "${MIGRATION_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${MIGRATION_SECRETS[@]}")" \
  --cpu=2 \
  --memory=4Gi \
  --max-retries=0 \
  --task-timeout=3600
