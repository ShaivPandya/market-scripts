#!/usr/bin/env bash
# Idempotently create/update the Cloud Scheduler jobs that drive the API and
# Cloud Run Jobs. Re-run anytime; existing jobs are updated in place.
#
# Jobs:
#   enqueue-cache-warm        */5 * * * *   POST  /api/v1/admin/jobs/enqueue-cache-warm
#   enqueue-async-job-sweep   0 * * * *     POST  /api/v1/admin/jobs/enqueue-async-job-sweep
#   top50-refresh-daily       0 23 * * 1-5  POST  Cloud Run Jobs run -> ${TOP50_REFRESH_JOB}
#
# Prereqs: deploy-api.sh has run (so the API URL is resolvable), deploy-top50-
# refresh-job.sh has run (so the job exists), and iam.sh has bound run.invoker
# for migrator-sa on the job.
#
# The X-Scheduler-Secret header is pulled from Secret Manager (SCHEDULER_SECRET)
# so the value never lives in this repo.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE
require_var API_SA
require_var MIGRATOR_SA
require_var TOP50_REFRESH_JOB
require_active_project

log() { printf '\n[scheduler] %s\n' "$*"; }

API_URL="$(gcloud run services describe "${API_SERVICE}" \
  --project="${PROJECT_ID}" --region="${REGION}" \
  --format='value(status.url)')"
if [[ -z "${API_URL}" ]]; then
  echo "Could not resolve URL for Cloud Run service ${API_SERVICE}." >&2
  echo "Run ./infra/gcp/deploy-api.sh first." >&2
  exit 1
fi
log "API URL: ${API_URL}"

SCHEDULER_SECRET_VALUE="$(gcloud secrets versions access latest \
  --secret=SCHEDULER_SECRET --project="${PROJECT_ID}" 2>/dev/null || true)"
if [[ -z "${SCHEDULER_SECRET_VALUE}" ]]; then
  echo "SCHEDULER_SECRET is not populated in Secret Manager." >&2
  echo "Run ./infra/gcp/setup-secrets.sh first." >&2
  exit 1
fi

# Create-or-update an HTTP scheduler job that POSTs to the API with the
# X-Scheduler-Secret header and OIDC auth as api-sa.
upsert_api_job() {
  local name="$1" schedule="$2" path="$3"
  local uri="${API_URL}${path}"
  local action=create
  if gcloud scheduler jobs describe "${name}" \
        --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    action=update
  fi
  log "${action} ${name} (${schedule})"
  gcloud scheduler jobs "${action}" http "${name}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="${schedule}" \
    --time-zone=UTC \
    --uri="${uri}" \
    --http-method=POST \
    --headers="X-Scheduler-Secret=${SCHEDULER_SECRET_VALUE}" \
    --oidc-service-account-email="${API_SA}" \
    --oidc-token-audience="${API_URL}" \
    --quiet >/dev/null
}

# Create-or-update an HTTP scheduler job that runs a Cloud Run Job via the
# admin API, authenticated as migrator-sa (which needs run.invoker on the job;
# iam.sh provides that binding).
upsert_run_job_trigger() {
  local name="$1" schedule="$2" job="$3"
  local uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${job}:run"
  local action=create
  if gcloud scheduler jobs describe "${name}" \
        --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    action=update
  fi
  log "${action} ${name} (${schedule}) -> jobs/${job}"
  gcloud scheduler jobs "${action}" http "${name}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="${schedule}" \
    --time-zone=UTC \
    --uri="${uri}" \
    --http-method=POST \
    --oauth-service-account-email="${MIGRATOR_SA}" \
    --quiet >/dev/null
}

upsert_api_job enqueue-cache-warm      "*/5 * * * *" /api/v1/admin/jobs/enqueue-cache-warm
upsert_api_job enqueue-async-job-sweep "0 * * * *"   /api/v1/admin/jobs/enqueue-async-job-sweep
upsert_run_job_trigger top50-refresh-daily "0 23 * * 1-5" "${TOP50_REFRESH_JOB}"

log "Scheduler sync complete."
