# Shared helpers for the deploy scripts. Sourced, not executed.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${_here}/config.sh" ]]; then
  echo "infra/gcp/config.sh not found. Copy config.example.sh to config.sh and fill it in." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${_here}/config.sh"

# Default IMAGE_TAG to the current short git SHA when the caller didn't set one.
# config.sh's own default of "latest" is preserved as a fallback for environments
# without git (e.g. running from a tarball). Setting IMAGE_TAG=<value> beforehand
# always wins.
if [[ "${IMAGE_TAG:-latest}" == "latest" ]] && command -v git >/dev/null 2>&1; then
  if _sha="$(git -C "${_here}" rev-parse --short HEAD 2>/dev/null)"; then
    export IMAGE_TAG="${_sha}"
  fi
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable ${name} is unset (set it in infra/gcp/config.sh)." >&2
    exit 1
  fi
}

# Fail fast if the user's active gcloud project doesn't match config.sh's
# PROJECT_ID. Catches the common "deployed to the wrong project" footgun.
require_active_project() {
  local active
  active="$(gcloud config get-value project 2>/dev/null || true)"
  if [[ -n "${active}" && "${active}" != "${PROJECT_ID}" ]]; then
    cat >&2 <<EOF
Active gcloud project (${active}) does not match config.sh PROJECT_ID (${PROJECT_ID}).
Run:  gcloud config set project ${PROJECT_ID}
Or unset the active project to silence this check:  gcloud config unset project
EOF
    exit 1
  fi
}

image_uri() {
  echo "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
}

# Verify the resolved image tag actually exists in Artifact Registry before
# we ask Cloud Run to pull it. Saves a round-trip-and-fail when someone forgets
# to push or types the wrong SHA.
require_image_exists() {
  if [[ "${SKIP_IMAGE_CHECK:-0}" == "1" ]]; then
    return 0
  fi

  if ! gcloud artifacts docker images describe "$(image_uri)" \
        --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Image $(image_uri) not found in Artifact Registry." >&2
    echo "Build it first: gcloud builds submit --region=${REGION} --default-buckets-behavior=regional-user-owned-bucket --config=infra/gcp/cloudbuild.yaml --substitutions=_TAG=${IMAGE_TAG} ." >&2
    exit 1
  fi
}

# Format an array of KEY=VALUE pairs for gcloud --set-env-vars / --set-secrets.
# Uses gcloud's "^|^" alternate-delimiter syntax so values may contain commas
# (e.g. CORS_ORIGINS=https://a,https://b).
# Callers pass the array expanded: join_kv "${MY_ARRAY[@]}"
# Bash 3.2 compatible (no `local -n`).
join_kv() {
  local IFS='|'
  echo "^|^$*"
}

# Env vars shared by every Cloud Run service / job in this stack. Keeps the
# per-script env arrays focused on what's actually role-specific.
common_env_vars() {
  cat <<EOF
ENVIRONMENT=production
STATE_STORAGE_BACKEND=gcs
STATE_DB_BACKEND=postgres
GCS_STATE_BUCKET=${GCS_STATE_BUCKET}
CLOUD_RUN_REGION=${REGION}
LLM_PROVIDER=${LLM_PROVIDER:-openai}
EOF
}

# Returns the common env vars as a bash array on stdout, one per line.
# Usage:  mapfile -t COMMON < <(common_env_vars)
