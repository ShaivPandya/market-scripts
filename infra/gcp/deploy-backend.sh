#!/usr/bin/env bash
# Build the image at the current short git SHA and complete a backend
# production rollout: run DB migrations, roll the API service, Cloud Run jobs,
# IAM bindings, and Scheduler jobs to the current config.
#
# Usage:
#   ./infra/gcp/deploy-backend.sh                 # build + deploy at git short SHA
#   IMAGE_TAG=<sha> ./infra/gcp/deploy-backend.sh # build + deploy at the given tag
#   SKIP_BUILD=1 ./infra/gcp/deploy-backend.sh    # skip cloudbuild (image must exist)
#
# Refuses to run on a dirty working tree unless ALLOW_DIRTY=1 (for hotfixes).
#
# Tunables:
#   RUN_DB_MIGRATIONS=0  skip Alembic upgrade head
#   SYNC_IAM=0           skip iam.sh
#   SYNC_SCHEDULER=0     skip setup-scheduler.sh

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_active_project

log() { printf '\n[deploy-backend] %s\n' "$*"; }

_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

run_job_and_wait() {
  local job="$1"
  shift
  gcloud run jobs execute "${job}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --wait \
    "$@"
}

if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
  if ! git -C "${_repo_root}" diff --quiet || ! git -C "${_repo_root}" diff --cached --quiet; then
    cat >&2 <<EOF
Working tree is dirty. Refusing to deploy a tag that doesn't match the repo state.
Commit or stash, or run:  ALLOW_DIRTY=1 ./infra/gcp/deploy-backend.sh
EOF
    exit 1
  fi
fi

log "Image tag: ${IMAGE_TAG}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  log "Building image via Cloud Build"
  gcloud builds submit "${_repo_root}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --default-buckets-behavior=regional-user-owned-bucket \
    --config="${_repo_root}/infra/gcp/cloudbuild.yaml" \
    --substitutions="_TAG=${IMAGE_TAG}"
else
  log "SKIP_BUILD=1; using existing image"
  require_image_exists
fi

log "Deploying migration job"
"${_repo_root}/infra/gcp/deploy-migration-job.sh"

log "Deploying top50 refresh job"
"${_repo_root}/infra/gcp/deploy-top50-refresh-job.sh"

log "Deploying async job runner"
"${_repo_root}/infra/gcp/deploy-async-job.sh"

if [[ "${SYNC_IAM:-1}" == "1" ]]; then
  log "Syncing IAM bindings"
  "${_repo_root}/infra/gcp/iam.sh"
else
  log "SYNC_IAM=0; skipping IAM sync"
fi

if [[ "${RUN_DB_MIGRATIONS:-1}" == "1" ]]; then
  log "Running Alembic migrations"
  # The migration job's default args run the one-time state migration tool.
  # Override args for this execution so routine deploys only apply schema
  # migrations against the production Cloud SQL database.
  run_job_and_wait "${MIGRATION_JOB}" --args=-m,alembic,upgrade,head
else
  log "RUN_DB_MIGRATIONS=0; skipping Alembic migrations"
fi

log "Deploying API service"
"${_repo_root}/infra/gcp/deploy-api.sh"

log "Skipping legacy worker pool deploy; async work now runs via Cloud Run Jobs"

if [[ "${SYNC_SCHEDULER:-1}" == "1" ]]; then
  log "Syncing Cloud Scheduler jobs"
  "${_repo_root}/infra/gcp/setup-scheduler.sh"
else
  log "SYNC_SCHEDULER=0; skipping Scheduler sync"
fi

log "Backend deploy complete at ${IMAGE_TAG}."
