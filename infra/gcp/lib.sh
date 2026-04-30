# Shared helpers for the deploy scripts. Sourced, not executed.

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${_here}/config.sh" ]]; then
  echo "infra/gcp/config.sh not found. Copy config.example.sh to config.sh and fill it in." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${_here}/config.sh"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable ${name} is unset (set it in infra/gcp/config.sh)." >&2
    exit 1
  fi
}

image_uri() {
  echo "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
}

# Comma-join an array passed by name. Used for --set-secrets and --set-env-vars.
join_csv() {
  local -n _arr="$1"
  local IFS=','
  echo "${_arr[*]}"
}
