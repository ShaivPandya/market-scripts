#!/usr/bin/env bash
# Idempotently create/update log-based metrics and alert policies for the
# governed first-party inference service.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_active_project

INFERENCE_SERVICE="${INFERENCE_SERVICE:-talisman-inference-nonprod}"

log() { printf '\n[inference-monitoring] %s\n' "$*"; }

_repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

metric_exists() {
  gcloud logging metrics describe "$1" --project="${PROJECT_ID}" >/dev/null 2>&1
}

metric_config_file() {
  local description="$1"
  local filter="$2"
  local value_type="$3"
  local unit="$4"
  local value_extractor="${5:-}"
  local file
  file="$(mktemp)"

  {
    printf 'description: "%s"\n' "${description//\"/\\\"}"
    printf 'filter: |-\n'
    printf '  %s\n' "${filter}"
    if [[ -n "${value_extractor}" ]]; then
      printf 'valueExtractor: |-\n'
      printf '  %s\n' "${value_extractor}"
    fi
    printf 'metricDescriptor:\n'
    printf '  metricKind: DELTA\n'
    printf '  valueType: %s\n' "${value_type}"
    printf '  unit: "%s"\n' "${unit}"
    if [[ "${value_type}" == "DISTRIBUTION" ]]; then
      printf 'bucketOptions:\n'
      printf '  exponentialBuckets:\n'
      printf '    numFiniteBuckets: 12\n'
      printf '    growthFactor: 2\n'
      printf '    scale: 1\n'
    fi
  } >"${file}"

  echo "${file}"
}

upsert_counter_metric() {
  local name="$1"
  local description="$2"
  local filter="$3"
  local action="create"
  if metric_exists "${name}"; then
    action="update"
  fi
  log "${action} log metric ${name}"
  local config_file
  config_file="$(metric_config_file "${description}" "${filter}" "INT64" "1")"
  gcloud logging metrics "${action}" "${name}" \
    --project="${PROJECT_ID}" \
    --config-from-file="${config_file}"
  rm -f "${config_file}"
}

upsert_distribution_metric() {
  local name="$1"
  local description="$2"
  local filter="$3"
  local value_extractor="$4"
  local action="create"
  if metric_exists "${name}"; then
    action="update"
  fi
  log "${action} log metric ${name}"
  local config_file
  config_file="$(metric_config_file "${description}" "${filter}" "DISTRIBUTION" "s" "${value_extractor}")"
  gcloud logging metrics "${action}" "${name}" \
    --project="${PROJECT_ID}" \
    --config-from-file="${config_file}"
  rm -f "${config_file}"
}

_service_filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${INFERENCE_SERVICE}\""

log "Inference error and refusal metrics"
upsert_counter_metric \
  "inference_startup_refused_count" \
  "Inference startup refused due to registry or digest gate failures" \
  "${_service_filter} AND textPayload:\"inference_startup\" AND severity>=ERROR"

upsert_counter_metric \
  "inference_request_error_count" \
  "Inference service request errors (5xx or vLLM failures)" \
  "${_service_filter} AND (httpRequest.status>=500 OR textPayload:\"error\")"

upsert_distribution_metric \
  "inference_generation_latency_seconds" \
  "Inference generation latency in seconds" \
  "${_service_filter} AND jsonPayload.event=\"inference_generation_complete\"" \
  "EXTRACT(jsonPayload.latency_seconds)"

log "Alert policy"
_policy_file="${_repo_root}/infra/gcp/monitoring-inference-alerts.json"
if [[ -f "${_policy_file}" ]]; then
  existing="$(gcloud alpha monitoring policies list \
    --project="${PROJECT_ID}" \
    --filter="displayName=\"Talisman inference service health\"" \
    --format='value(name)' 2>/dev/null || true)"
  if [[ -n "${existing}" ]]; then
    log "update alert policy"
    gcloud alpha monitoring policies update "${existing}" \
      --project="${PROJECT_ID}" \
      --policy-from-file="${_policy_file}" >/dev/null
  else
    log "create alert policy"
    gcloud alpha monitoring policies create \
      --project="${PROJECT_ID}" \
      --policy-from-file="${_policy_file}" >/dev/null
  fi
else
  echo "  monitoring-inference-alerts.json not found; skipping alert policy" >&2
fi

log "Done."
