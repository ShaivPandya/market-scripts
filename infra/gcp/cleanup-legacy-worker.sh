#!/usr/bin/env bash
# Dry-run or delete only the deprecated Cloud Run worker pool `talisman-worker`.
#
# Usage:
#   ./infra/gcp/cleanup-legacy-worker.sh           # dry-run
#   ./infra/gcp/cleanup-legacy-worker.sh --apply   # actually delete
#
# This is intentionally narrower than cleanup-stale.sh so the legacy worker can
# be removed without touching Redis, secrets, or other stale resources.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_active_project

APPLY=0
case "${1:-}" in
  --apply) APPLY=1 ;;
  ""|--dry-run) APPLY=0 ;;
  *) echo "Unknown arg: $1 (expected --apply or --dry-run)" >&2; exit 1 ;;
esac

WORKER_POOL="talisman-worker"

if [[ "${APPLY}" == "1" ]]; then
  echo "[legacy-worker cleanup] APPLY mode: ${WORKER_POOL} will be DELETED if it exists."
else
  echo "[legacy-worker cleanup] dry-run mode. Re-run with --apply to actually delete."
fi
echo

if gcloud run worker-pools describe "${WORKER_POOL}" \
      --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
  echo "[cloud-run worker-pool] ${WORKER_POOL} exists"
  if [[ "${APPLY}" == "1" ]]; then
    echo "  delete: cloud run worker pool ${WORKER_POOL}"
    gcloud run worker-pools delete "${WORKER_POOL}" \
      --project="${PROJECT_ID}" --region="${REGION}" --quiet
  else
    echo "  would delete: cloud run worker pool ${WORKER_POOL}"
    echo "    cmd: gcloud run worker-pools delete ${WORKER_POOL} --project=${PROJECT_ID} --region=${REGION} --quiet"
  fi
else
  echo "[cloud-run worker-pool] ${WORKER_POOL} not found; nothing to do."
fi
