#!/usr/bin/env bash
# Idempotently grant the project-, bucket-, and Cloud Run-level IAM bindings
# the deploy SAs need. Per-secret accessor IAM is owned by setup-secrets.sh.
#
# Bindings applied:
#   project:
#     api-sa, worker-sa, migrator-sa  -> roles/cloudsql.client
#     api-sa, worker-sa, migrator-sa  -> roles/logging.logWriter
#   bucket gs://${GCS_STATE_BUCKET}:
#     api-sa, worker-sa, migrator-sa  -> roles/storage.objectAdmin
#   Cloud Run job ${MIGRATION_JOB}:
#     migrator-sa  -> roles/run.invoker  (so Cloud Scheduler can run it)
#   Cloud Run job ${TOP50_REFRESH_JOB}:
#     migrator-sa  -> roles/run.invoker
#
# Re-running is safe — gcloud `add-iam-policy-binding` is idempotent.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var GCS_STATE_BUCKET
require_var API_SA
require_var WORKER_SA
require_var MIGRATOR_SA
require_active_project

log() { printf '\n[iam] %s\n' "$*"; }

bind_project() {
  local member="$1" role="$2"
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${member}" \
    --role="${role}" \
    --condition=None \
    --quiet >/dev/null
  echo "  project ${role} -> ${member}"
}

bind_bucket() {
  local member="$1" role="$2"
  gcloud storage buckets add-iam-policy-binding "gs://${GCS_STATE_BUCKET}" \
    --member="serviceAccount:${member}" \
    --role="${role}" >/dev/null
  echo "  gs://${GCS_STATE_BUCKET} ${role} -> ${member}"
}

bind_run_job_invoker() {
  local job="$1" member="$2"
  if ! gcloud run jobs describe "${job}" \
        --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
    echo "  (skip) job ${job} not deployed yet — re-run iam.sh after deploy-${job##*-}-job.sh"
    return
  fi
  gcloud run jobs add-iam-policy-binding "${job}" \
    --project="${PROJECT_ID}" --region="${REGION}" \
    --member="serviceAccount:${member}" \
    --role=roles/run.invoker >/dev/null
  echo "  run jobs/${job} run.invoker -> ${member}"
}

###############################################################################
# Project IAM
###############################################################################
log "Project bindings"
for sa in "${API_SA}" "${WORKER_SA}" "${MIGRATOR_SA}"; do
  bind_project "${sa}" roles/cloudsql.client
  bind_project "${sa}" roles/logging.logWriter
done

###############################################################################
# Bucket IAM
###############################################################################
log "Bucket bindings on gs://${GCS_STATE_BUCKET}"
for sa in "${API_SA}" "${WORKER_SA}" "${MIGRATOR_SA}"; do
  bind_bucket "${sa}" roles/storage.objectAdmin
done

###############################################################################
# Cloud Run Job invoker (for Cloud Scheduler -> Cloud Run Jobs)
###############################################################################
if [[ -n "${MIGRATION_JOB:-}" ]]; then
  log "Cloud Run job invoker: ${MIGRATION_JOB}"
  bind_run_job_invoker "${MIGRATION_JOB}" "${MIGRATOR_SA}"
fi
if [[ -n "${TOP50_REFRESH_JOB:-}" ]]; then
  log "Cloud Run job invoker: ${TOP50_REFRESH_JOB}"
  bind_run_job_invoker "${TOP50_REFRESH_JOB}" "${MIGRATOR_SA}"
fi

log "IAM sync complete."
