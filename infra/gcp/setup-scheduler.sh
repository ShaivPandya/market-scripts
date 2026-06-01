#!/usr/bin/env bash
# Idempotently create/update the Cloud Scheduler jobs that drive the API and
# Cloud Run Jobs. Re-run anytime; existing jobs are updated in place.
#
# Jobs:
#   enqueue-async-job-sweep   0 * * * *     POST  /api/admin/jobs/enqueue-async-job-sweep
#   top50-refresh-daily       0 23 * * 1-5  POST  Cloud Run Jobs run -> ${TOP50_REFRESH_JOB}
#   market-snapshot-refresh   15 23 * * 1-5 POST  /api/admin/jobs/enqueue-market-snapshot-refresh
#   macro-snapshot-refresh    30 23 * * 1-5 POST  /api/admin/jobs/enqueue-macro-snapshot-refresh
#   workspace-source-refresh  45 23 * * 1-5 POST  /api/admin/jobs/enqueue-workspace-source-refresh
#   continuous-optimizer      15 10 * * 1-5 POST /api/admin/jobs/enqueue-continuous-optimizer
#
# Optional:
#   watch-trigger-monitor     30 14-22 * * 1-5 POST /api/admin/jobs/enqueue-watch-trigger-monitor
#   monitor-mission-runner    35 14-22 * * 1-5 POST /api/admin/jobs/enqueue-monitor-mission-runner
#   governance-outbox-drain   */5 * * * *   POST  /api/admin/jobs/enqueue-governance-outbox-drain
#   enqueue-cache-warm        0 * * * *     POST  /api/admin/jobs/enqueue-cache-warm
#
# The governance outbox drain is disabled by default because the runtime
# job is now a no-op. Set SCHEDULE_GOVERNANCE_OUTBOX_DRAIN=1 to recreate it.
#
# The watch trigger monitor is disabled by default. It remains available as a
# manual/admin job, but routine scheduling should stay off until its ontology-ID
# handling is fixed. Set SCHEDULE_WATCH_TRIGGER_MONITOR=1 to recreate it.
#
# Scheduled cache warming is disabled by default. The warm job runs in the
# generic async Cloud Run Job, which does not share the API service's in-memory
# cache or local filesystem cache, so a 5-minute warm cadence adds many API/job
# executions without materially improving interactive cold-start behavior. Set
# SCHEDULE_CACHE_WARM=1 to recreate it at the lower CACHE_WARM_SCHEDULE cadence.
#
# Prereqs: deploy-api.sh has run (so the API URL is resolvable), deploy-top50-
# refresh-job.sh has run (so the job exists), and iam.sh has bound
# roles/run.jobsExecutor for migrator-sa on the job.
#
# The X-Scheduler-Secret and X-Api-Proxy-Secret headers are pulled from Secret
# Manager so the values never live in this repo.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE
require_var API_SA
require_var MIGRATOR_SA
require_var TOP50_REFRESH_JOB
require_active_project

log() { printf '\n[scheduler] %s\n' "$*"; }

is_truthy() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

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

API_PROXY_SECRET_VALUE="$(gcloud secrets versions access latest \
  --secret=API_PROXY_SECRET --project="${PROJECT_ID}" 2>/dev/null || true)"
if [[ -z "${API_PROXY_SECRET_VALUE}" ]]; then
  echo "API_PROXY_SECRET is not populated in Secret Manager." >&2
  echo "Run ./infra/gcp/setup-secrets.sh first." >&2
  exit 1
fi

# Create-or-update an HTTP scheduler job that POSTs to the API with the
# scheduler and proxy-secret headers plus OIDC auth as api-sa.
upsert_api_job() {
  local name="$1" schedule="$2" path="$3"
  local timezone="${4:-UTC}"
  local uri="${API_URL}${path}"
  local action=create
  local headers_flag=--headers
  if gcloud scheduler jobs describe "${name}" \
        --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    action=update
    headers_flag=--update-headers
  fi
  log "${action} ${name} (${schedule}, ${timezone})"
  gcloud scheduler jobs "${action}" http "${name}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --schedule="${schedule}" \
    --time-zone="${timezone}" \
    --uri="${uri}" \
    --http-method=POST \
    "${headers_flag}=X-Scheduler-Secret=${SCHEDULER_SECRET_VALUE},X-Api-Proxy-Secret=${API_PROXY_SECRET_VALUE}" \
    --oidc-service-account-email="${API_SA}" \
    --oidc-token-audience="${API_URL}" \
    --quiet >/dev/null
}

delete_scheduler_job_if_present() {
  local name="$1"
  if gcloud scheduler jobs describe "${name}" \
        --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    log "delete ${name} (disabled)"
    gcloud scheduler jobs delete "${name}" \
      --project="${PROJECT_ID}" \
      --location="${REGION}" \
      --quiet >/dev/null
  else
    log "skip ${name} (not configured)"
  fi
}

# Create-or-update an HTTP scheduler job that runs a Cloud Run Job via the
# admin API, authenticated as migrator-sa (which needs roles/run.jobsExecutor on
# the job; iam.sh provides that binding).
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

upsert_api_job enqueue-async-job-sweep "0 * * * *"   /api/admin/jobs/enqueue-async-job-sweep
upsert_run_job_trigger top50-refresh-daily "0 23 * * 1-5" "${TOP50_REFRESH_JOB}"
upsert_api_job market-snapshot-refresh "${MARKET_SNAPSHOT_SCHEDULE:-15 23 * * 1-5}" /api/admin/jobs/enqueue-market-snapshot-refresh
upsert_api_job macro-snapshot-refresh "${MACRO_SNAPSHOT_SCHEDULE:-30 23 * * 1-5}" /api/admin/jobs/enqueue-macro-snapshot-refresh
upsert_api_job workspace-source-refresh "${WORKSPACE_SOURCE_REFRESH_SCHEDULE:-45 23 * * 1-5}" /api/admin/jobs/enqueue-workspace-source-refresh
upsert_api_job continuous-optimizer "${CONTINUOUS_OPTIMIZER_SCHEDULE:-15 10 * * 1-5}" /api/admin/jobs/enqueue-continuous-optimizer "${CONTINUOUS_OPTIMIZER_TIME_ZONE:-America/New_York}"

if is_truthy "${SCHEDULE_WATCH_TRIGGER_MONITOR:-0}"; then
  upsert_api_job watch-trigger-monitor "${WATCH_TRIGGER_MONITOR_SCHEDULE:-30 14-22 * * 1-5}" /api/admin/jobs/enqueue-watch-trigger-monitor
else
  delete_scheduler_job_if_present watch-trigger-monitor
fi

if is_truthy "${SCHEDULE_MONITOR_MISSION_RUNNER:-0}"; then
  upsert_api_job monitor-mission-runner "${MONITOR_MISSION_RUNNER_SCHEDULE:-35 14-22 * * 1-5}" /api/admin/jobs/enqueue-monitor-mission-runner
else
  delete_scheduler_job_if_present monitor-mission-runner
fi

if is_truthy "${SCHEDULE_CATALYST_KILL_MONITOR:-0}"; then
  upsert_api_job catalyst-kill-monitor "${CATALYST_KILL_MONITOR_SCHEDULE:-0 15-21 * * 1-5}" /api/admin/jobs/enqueue-catalyst-kill-monitor
else
  delete_scheduler_job_if_present catalyst-kill-monitor
fi

if is_truthy "${SCHEDULE_GOVERNANCE_OUTBOX_DRAIN:-0}"; then
  upsert_api_job governance-outbox-drain "${GOVERNANCE_OUTBOX_DRAIN_SCHEDULE:-*/5 * * * *}" /api/admin/jobs/enqueue-governance-outbox-drain
else
  delete_scheduler_job_if_present governance-outbox-drain
fi

if is_truthy "${SCHEDULE_CACHE_WARM:-0}"; then
  upsert_api_job enqueue-cache-warm "${CACHE_WARM_SCHEDULE:-0 * * * *}" /api/admin/jobs/enqueue-cache-warm
else
  delete_scheduler_job_if_present enqueue-cache-warm
fi

log "Scheduler sync complete."
