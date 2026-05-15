#!/usr/bin/env bash
# SHA-34: Run backend smoke tests against the live Cloud Run API.
#
# Usage (automatic — called by deploy-backend.sh after deploy-api.sh):
#   EXPECTED_IMAGE_TAG=abc1234 ./infra/gcp/run-backend-smoke.sh
#
# Usage (manual — post-rollback):
#   SMOKE_MODE=post-rollback EXPECTED_IMAGE_TAG=<tag> ./infra/gcp/run-backend-smoke.sh
#
# Environment:
#   API_SERVICE           Cloud Run service name (from config.sh)
#   SMOKE_MODE            "post-deploy" (default) or "post-rollback"
#   EXPECTED_IMAGE_TAG    Optional tag to assert against /api/health release
#   API_PROXY_SECRET      Fetched from Secret Manager (never printed)
#   AUTH_SMOKE_PASSWORD   Fetched from Secret Manager (never printed)

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var API_SERVICE

log() { printf '\n[smoke] %s\n' "$*"; }

SMOKE_MODE="${SMOKE_MODE:-post-deploy}"

# ---------------------------------------------------------------------------
# Resolve Cloud Run service URL
# ---------------------------------------------------------------------------
log "Resolving service URL for ${API_SERVICE}"
SERVICE_URL="$(gcloud run services describe "${API_SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)' 2>/dev/null)"

if [[ -z "${SERVICE_URL}" ]]; then
  echo "ERROR: Could not resolve service URL for ${API_SERVICE}" >&2
  exit 1
fi
log "Service URL: ${SERVICE_URL}"

# ---------------------------------------------------------------------------
# Fetch secrets from Secret Manager (values never printed)
# ---------------------------------------------------------------------------
log "Fetching smoke secrets from Secret Manager"

API_PROXY_SECRET="$(gcloud secrets versions access latest \
  --secret=API_PROXY_SECRET \
  --project="${PROJECT_ID}" 2>/dev/null || true)"

AUTH_SMOKE_PASSWORD="$(gcloud secrets versions access latest \
  --secret=AUTH_SMOKE_PASSWORD \
  --project="${PROJECT_ID}" 2>/dev/null || true)"

if [[ -z "${AUTH_SMOKE_PASSWORD}" ]]; then
  echo "ERROR: AUTH_SMOKE_PASSWORD secret not found. Run setup-secrets.sh first." >&2
  exit 1
fi

export API_PROXY_SECRET
export AUTH_SMOKE_PASSWORD

# ---------------------------------------------------------------------------
# Run the Python smoke CLI
# ---------------------------------------------------------------------------
_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"

SMOKE_ARGS=(
  -m infra.gcp.deploy_smoke
  --service-url "${SERVICE_URL}"
  --mode "${SMOKE_MODE}"
)

if [[ -n "${EXPECTED_IMAGE_TAG:-}" ]]; then
  SMOKE_ARGS+=(--expected-image-tag "${EXPECTED_IMAGE_TAG}")
fi

log "Running smoke checks (mode=${SMOKE_MODE})"
python "${SMOKE_ARGS[@]}"
