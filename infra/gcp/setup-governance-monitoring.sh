#!/usr/bin/env bash
# Idempotently create/update log-based metrics and alert policies for
# audit/provenance durability.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_active_project

log() { printf '\n[governance-monitoring] %s\n' "$*"; }

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
      printf '    scale: 60\n'
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

upsert_counter_metric \
  governance_outbox_dead_letter_count \
  "Governance outbox drain dead-letter results." \
  'resource.type=("cloud_run_job" OR "cloud_run_revision") AND textPayload:"governance_outbox_drain" AND textPayload=~"dead_lettered=[1-9][0-9]*"'

upsert_counter_metric \
  governance_outbox_failed_count \
  "Governance outbox drain failures or retry scheduling." \
  'resource.type=("cloud_run_job" OR "cloud_run_revision") AND textPayload:"governance_outbox_drain" AND textPayload=~"failed=[1-9][0-9]*"'

upsert_distribution_metric \
  governance_outbox_oldest_pending_age_seconds \
  "Age in seconds of the oldest pending or failed governance outbox item." \
  'resource.type=("cloud_run_job" OR "cloud_run_revision") AND textPayload:"governance_outbox_drain" AND textPayload:"oldest_pending_age_seconds="' \
  'REGEXP_EXTRACT(textPayload, "oldest_pending_age_seconds=([0-9]+(?:\\.[0-9]+)?)")'

upsert_counter_metric \
  governance_critical_write_failure_count \
  "Mandatory audit/provenance write failures." \
  'resource.type=("cloud_run_revision" OR "cloud_run_job") AND (textPayload:"GovernanceWriteError" OR textPayload:"mandatory governance" OR textPayload:"mandatory provenance" OR textPayload:"failed_closed")'

upsert_counter_metric \
  governance_lineage_completeness_warning_count \
  "Lineage completeness warnings returned by governance reports." \
  'resource.type=("cloud_run_revision" OR "cloud_run_job") AND (textPayload:"lineage_completeness" OR textPayload:"Lineage completeness") AND (textPayload:"retry_pending" OR textPayload:"dead_letter" OR textPayload:"legacy_partial" OR textPayload:"failed_closed")'

upsert_counter_metric \
  governance_redaction_violation_count \
  "Redaction violation scan failures." \
  'resource.type=("cloud_run_revision" OR "cloud_run_job") AND (textPayload:"redaction violation" OR textPayload:"raw sensitive" OR textPayload:"sensitive payload")'

policy_file="${_repo_root}/infra/gcp/monitoring-governance-alerts.json"
policy_name="$(
  gcloud alpha monitoring policies list \
    --project="${PROJECT_ID}" \
    --filter='displayName="Governance audit and provenance health"' \
    --format='value(name)' \
    | head -n 1
)"
if [[ -n "${policy_name}" ]]; then
  log "Updating alert policy ${policy_name}"
  gcloud alpha monitoring policies update "${policy_name}" \
    --project="${PROJECT_ID}" \
    --policy-from-file="${policy_file}"
else
  log "Creating alert policy"
  gcloud alpha monitoring policies create \
    --project="${PROJECT_ID}" \
    --policy-from-file="${policy_file}"
fi

log "Governance monitoring sync complete."
