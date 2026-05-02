#!/usr/bin/env bash
# Deploy the full production stack in dependency order.
#
# Usage:
#   ./infra/gcp/deploy-all.sh
#   IMAGE_TAG=<sha> ./infra/gcp/deploy-all.sh
#   SKIP_BUILD=1 ./infra/gcp/deploy-all.sh          # skip backend image build
#   SKIP_FRONTEND_BUILD=1 ./infra/gcp/deploy-all.sh # deploy existing frontend/dist
#
# Backend deploy runs first so Firebase Hosting rewrites target the freshly
# rolled Cloud Run service. Frontend deploy runs second so static assets are
# published after the API and scheduled backend jobs are in place.

set -euo pipefail

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

log() { printf '\n[deploy-all] %s\n' "$*"; }

log "Deploying backend"
"${repo_root}/infra/gcp/deploy-backend.sh"

log "Deploying frontend"
if [[ "${SKIP_FRONTEND_BUILD:-0}" == "1" ]]; then
  SKIP_BUILD=1 "${repo_root}/infra/gcp/deploy-frontend.sh"
else
  SKIP_BUILD=0 "${repo_root}/infra/gcp/deploy-frontend.sh"
fi

log "Full deploy complete."
