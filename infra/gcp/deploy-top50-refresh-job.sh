#!/usr/bin/env bash
# Create or update the Cloud Run Job that refreshes the cached top-50 S&P 500
# leadership list (table: sp500_top50_tickers). The list barely moves day-to-day,
# so a daily Cloud Scheduler trigger keeps it fresh while the API reads from the
# table at request time.
#
# Usage:
#   IMAGE_TAG=<sha-or-tag> ./infra/gcp/deploy-top50-refresh-job.sh
#
# Execute on demand (after deploy):
#   gcloud run jobs execute "${TOP50_REFRESH_JOB}" \
#     --project="${PROJECT_ID}" --region="${REGION}"
#
# Schedule daily (Cloud Scheduler → Cloud Run Jobs):
#   gcloud scheduler jobs create http top50-refresh-daily \
#     --location="${REGION}" \
#     --schedule="0 23 * * 1-5" \
#     --time-zone="UTC" \
#     --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${TOP50_REFRESH_JOB}:run" \
#     --http-method=POST \
#     --oauth-service-account-email="${MIGRATOR_SA}"

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var TOP50_REFRESH_JOB
require_var MIGRATOR_SA
require_var CLOUDSQL_INSTANCE

TOP50_REFRESH_ENV_VARS=(
  "ENVIRONMENT=production"
  "STATE_DB_BACKEND=postgres"
)

# Reuses the migrator-scoped DATABASE_URL — the refresh writes to a single table
# and otherwise needs only outbound internet for yfinance/Wikipedia.
TOP50_REFRESH_SECRETS=(
  "DATABASE_URL=DATABASE_URL_MIGRATION:latest"
)

gcloud run jobs deploy "${TOP50_REFRESH_JOB}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="$(image_uri)" \
  --service-account="${MIGRATOR_SA}" \
  --set-cloudsql-instances="${CLOUDSQL_INSTANCE}" \
  --command=python \
  --args=-m,equities.market_technicals.get_top50 \
  --set-env-vars="$(join_kv "${TOP50_REFRESH_ENV_VARS[@]}")" \
  --set-secrets="$(join_kv "${TOP50_REFRESH_SECRETS[@]}")" \
  --cpu=1 \
  --memory=1Gi \
  --max-retries=1 \
  --task-timeout=600
