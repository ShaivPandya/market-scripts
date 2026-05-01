#!/usr/bin/env bash
# Build the image at the current short git SHA and roll the API service, the
# worker pool, the migration job, and the top50-refresh job to that SHA.
#
# Usage:
#   ./infra/gcp/deploy-all.sh                 # build + deploy at git short SHA
#   IMAGE_TAG=<sha> ./infra/gcp/deploy-all.sh # build + deploy at the given tag
#   SKIP_BUILD=1 ./infra/gcp/deploy-all.sh    # skip cloudbuild (image must exist)
#
# Refuses to run on a dirty working tree unless ALLOW_DIRTY=1 (for hotfixes).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_active_project

log() { printf '\n[deploy-all] %s\n' "$*"; }

_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
  if ! git -C "${_repo_root}" diff --quiet || ! git -C "${_repo_root}" diff --cached --quiet; then
    cat >&2 <<EOF
Working tree is dirty. Refusing to deploy a tag that doesn't match the repo state.
Commit or stash, or run:  ALLOW_DIRTY=1 ./infra/gcp/deploy-all.sh
EOF
    exit 1
  fi
fi

log "Image tag: ${IMAGE_TAG}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  log "Building image via Cloud Build"
  gcloud builds submit "${_repo_root}" \
    --project="${PROJECT_ID}" \
    --config="${_repo_root}/infra/gcp/cloudbuild.yaml" \
    --substitutions="_TAG=${IMAGE_TAG}"
else
  log "SKIP_BUILD=1; using existing image"
  require_image_exists
fi

log "Deploying API service"
"${_repo_root}/infra/gcp/deploy-api.sh"

log "Deploying worker pool"
"${_repo_root}/infra/gcp/deploy-worker.sh"

log "Deploying migration job"
"${_repo_root}/infra/gcp/deploy-migration-job.sh"

log "Deploying top50 refresh job"
"${_repo_root}/infra/gcp/deploy-top50-refresh-job.sh"

log "All services deployed at ${IMAGE_TAG}."
