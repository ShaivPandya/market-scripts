#!/usr/bin/env bash
# Build and deploy the Firebase Hosting frontend.
#
# Usage:
#   ./infra/gcp/deploy-frontend.sh
#   SKIP_BUILD=1 ./infra/gcp/deploy-frontend.sh
#
# Uses PROJECT_ID from infra/gcp/config.sh and deploys firebase.json hosting.
# Run after deploy-backend.sh when deploying components separately.
# Refuses to run on a dirty working tree unless ALLOW_DIRTY=1.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_active_project

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
frontend_dir="${repo_root}/frontend"

if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
  if ! git -C "${repo_root}" diff --quiet || ! git -C "${repo_root}" diff --cached --quiet; then
    cat >&2 <<EOF
Working tree is dirty. Refusing to deploy frontend assets that don't match the repo state.
Commit or stash, or run:  ALLOW_DIRTY=1 ./infra/gcp/deploy-frontend.sh
EOF
    exit 1
  fi
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js >=20.19.0 and retry." >&2
  exit 1
fi

if ! command -v firebase >/dev/null 2>&1; then
  echo "firebase CLI not found. Install it with: npm install -g firebase-tools" >&2
  exit 1
fi

if [[ ! -d "${frontend_dir}/node_modules" ]]; then
  echo "frontend/node_modules not found; installing dependencies with npm ci."
  (cd "${frontend_dir}" && npm ci)
elif [[ "${frontend_dir}/package-lock.json" -nt "${frontend_dir}/node_modules/.package-lock.json" ]]; then
  echo "frontend/package-lock.json is newer than node_modules; refreshing dependencies with npm ci."
  (cd "${frontend_dir}" && npm ci)
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  _full_sha="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
  _short_sha="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || true)"
  export VITE_SENTRY_ENVIRONMENT="${VITE_SENTRY_ENVIRONMENT:-production}"
  export VITE_SENTRY_RELEASE="${VITE_SENTRY_RELEASE:-${_full_sha}}"
  export VITE_TALISMAN_RELEASE_GIT_SHA_SHORT="${VITE_TALISMAN_RELEASE_GIT_SHA_SHORT:-${_short_sha}}"
  (cd "${frontend_dir}" && npm run build)
else
  echo "SKIP_BUILD=1; deploying existing frontend/dist."
fi

firebase deploy --only hosting --project="${PROJECT_ID}" --config="${repo_root}/firebase.json"
