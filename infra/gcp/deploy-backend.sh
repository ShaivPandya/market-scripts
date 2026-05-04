#!/usr/bin/env bash
# Build the image at the current short git SHA and complete a backend
# production rollout: run DB migrations and roll the API service + Cloud Run
# jobs. Optional full sync mode also reconciles IAM, Scheduler, and monitoring.
#
# Usage:
#   ./infra/gcp/deploy-backend.sh                 # build + deploy at git short SHA
#   IMAGE_TAG=<sha> ./infra/gcp/deploy-backend.sh # build + deploy at the given tag
#   SKIP_BUILD=1 ./infra/gcp/deploy-backend.sh    # skip cloudbuild (image must exist)
#   FULL_SYNC=1 ./infra/gcp/deploy-backend.sh     # also sync IAM, Scheduler, monitoring
#
# Refuses to run on a dirty working tree unless ALLOW_DIRTY=1 (for hotfixes).
#
# Tunables:
#   RUN_DB_MIGRATIONS=0   skip Alembic upgrade head
#   PARALLEL_JOB_DEPLOYS=0 deploy Cloud Run jobs sequentially
#   SYNC_IAM=1            run iam.sh (default: only when FULL_SYNC=1)
#   SYNC_SCHEDULER=1      run setup-scheduler.sh (default: only when FULL_SYNC=1)
#   SYNC_MONITORING=1     run setup-governance-monitoring.sh (default: only when FULL_SYNC=1)
#   SHOW_PARALLEL_LOGS=1  print successful parallel job deploy logs

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_active_project

log() { printf '\n[deploy-backend] %s\n' "$*"; }

_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
_parallel_tmp_dir=""
_parallel_pids=()
_parallel_labels=()
_parallel_logs=()

if [[ "${FULL_SYNC:-0}" == "1" ]]; then
  SYNC_IAM="${SYNC_IAM:-1}"
  SYNC_SCHEDULER="${SYNC_SCHEDULER:-1}"
  SYNC_MONITORING="${SYNC_MONITORING:-1}"
else
  SYNC_IAM="${SYNC_IAM:-0}"
  SYNC_SCHEDULER="${SYNC_SCHEDULER:-0}"
  SYNC_MONITORING="${SYNC_MONITORING:-0}"
fi
PARALLEL_JOB_DEPLOYS="${PARALLEL_JOB_DEPLOYS:-1}"

cleanup_parallel_logs() {
  local pid
  for pid in "${_parallel_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${_parallel_tmp_dir}" ]]; then
    rm -rf "${_parallel_tmp_dir}"
  fi
}
trap cleanup_parallel_logs EXIT

run_job_and_wait() {
  local job="$1"
  shift
  gcloud run jobs execute "${job}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --wait \
    "$@"
}

start_parallel_step() {
  local label="$1"
  shift

  if [[ -z "${_parallel_tmp_dir}" ]]; then
    _parallel_tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/deploy-backend.XXXXXX")"
  fi

  local index="${#_parallel_pids[@]}"
  local log_file="${_parallel_tmp_dir}/${index}.log"
  log "Starting ${label}"
  ( "$@" ) >"${log_file}" 2>&1 &
  _parallel_pids+=("$!")
  _parallel_labels+=("${label}")
  _parallel_logs+=("${log_file}")
}

wait_parallel_steps() {
  if [[ "${#_parallel_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  local failed=0
  local index
  local status
  for index in "${!_parallel_pids[@]}"; do
    if wait "${_parallel_pids[${index}]}"; then
      log "${_parallel_labels[${index}]} complete"
      if [[ "${SHOW_PARALLEL_LOGS:-0}" == "1" ]]; then
        sed 's/^/  /' "${_parallel_logs[${index}]}"
      fi
    else
      status="$?"
      failed=1
      printf '\n[deploy-backend] %s failed (exit %s); output follows:\n' \
        "${_parallel_labels[${index}]}" "${status}" >&2
      sed 's/^/  /' "${_parallel_logs[${index}]}" >&2
    fi
  done

  _parallel_pids=()
  _parallel_labels=()
  _parallel_logs=()

  if [[ "${failed}" != "0" ]]; then
    exit 1
  fi
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

# deploy-backend has either just built the image or checked it once above.
export SKIP_IMAGE_CHECK=1

log "Deploying migration job"
"${_repo_root}/infra/gcp/deploy-migration-job.sh"

if [[ "${PARALLEL_JOB_DEPLOYS}" == "1" ]]; then
  log "Deploying non-migration Cloud Run jobs in parallel"
  start_parallel_step "top50 refresh job deploy" \
    "${_repo_root}/infra/gcp/deploy-top50-refresh-job.sh"
  start_parallel_step "async job runner deploy" \
    "${_repo_root}/infra/gcp/deploy-async-job.sh"
else
  log "Deploying top50 refresh job"
  "${_repo_root}/infra/gcp/deploy-top50-refresh-job.sh"

  log "Deploying async job runner"
  "${_repo_root}/infra/gcp/deploy-async-job.sh"
fi

if [[ "${SYNC_IAM}" == "1" ]]; then
  wait_parallel_steps
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

if [[ "${SYNC_IAM}" != "1" ]]; then
  wait_parallel_steps
fi

log "Deploying API service"
"${_repo_root}/infra/gcp/deploy-api.sh"

log "Skipping legacy worker pool deploy; async work now runs via Cloud Run Jobs"

if [[ "${SYNC_SCHEDULER}" == "1" ]]; then
  log "Syncing Cloud Scheduler jobs"
  "${_repo_root}/infra/gcp/setup-scheduler.sh"
else
  log "SYNC_SCHEDULER=0; skipping Scheduler sync"
fi

if [[ "${SYNC_MONITORING}" == "1" ]]; then
  log "Syncing governance monitoring"
  "${_repo_root}/infra/gcp/setup-governance-monitoring.sh"
else
  log "SYNC_MONITORING=0; skipping governance monitoring sync"
fi

log "Backend deploy complete at ${IMAGE_TAG}."
