#!/usr/bin/env bash
# Build and deploy the Firebase Hosting frontend.
#
# Usage:
#   ./infra/gcp/deploy-frontend.sh
#   SKIP_BUILD=1 ./infra/gcp/deploy-frontend.sh
#
# Uses PROJECT_ID from infra/gcp/config.sh and deploys firebase.json hosting.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_active_project

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
frontend_dir="${repo_root}/frontend"

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
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  (cd "${frontend_dir}" && npm run build)
else
  echo "SKIP_BUILD=1; deploying existing frontend/dist."
fi

firebase deploy --only hosting --project="${PROJECT_ID}" --config="${repo_root}/firebase.json"
