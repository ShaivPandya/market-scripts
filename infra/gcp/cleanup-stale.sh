#!/usr/bin/env bash
# List (default) or delete (--apply) GCP resources that pre-date the current
# infra/gcp scripts and are not referenced by any of them.
#
# Stale resources targeted:
#   Cloud Run service:        talisman-service        (uses old app-sa, image talisman:*)
#   Cloud Run job:            talisman-job            (uses old app-sa)
#   Service account:          app-sa@…                (only used by the two above)
#   Memorystore:              market-scripts-redis    (replaced by redis instance "talisman")
#   Artifact Registry repo:   my-docker-repo          (empty leftover)
#   Secret:                   db-password             (replaced by DATABASE_URL_*)
#   Secret:                   scheduler-secret        (lowercase; replaced by SCHEDULER_SECRET)
#
# Usage:
#   ./infra/gcp/cleanup-stale.sh           # dry-run; just print what would be deleted
#   ./infra/gcp/cleanup-stale.sh --apply   # actually delete
#
# Each delete is gated by an existence check, so re-running is safe and
# resources you've already removed are skipped.

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

if [[ "${APPLY}" == "1" ]]; then
  echo "[cleanup] APPLY mode: resources below will be DELETED."
else
  echo "[cleanup] dry-run mode. Re-run with --apply to actually delete."
fi
echo

run() {
  local label="$1"; shift
  if [[ "${APPLY}" == "1" ]]; then
    echo "  delete: ${label}"
    "$@"
  else
    echo "  would delete: ${label}"
    echo "    cmd: $*"
  fi
}

# --- Cloud Run service: talisman-service -----------------------------------
if gcloud run services describe talisman-service \
      --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
  echo "[cloud-run service] talisman-service exists"
  run "cloud run service talisman-service" \
    gcloud run services delete talisman-service \
      --project="${PROJECT_ID}" --region="${REGION}" --quiet
fi

# --- Cloud Run job: talisman-job -------------------------------------------
if gcloud run jobs describe talisman-job \
      --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
  echo "[cloud-run job] talisman-job exists"
  run "cloud run job talisman-job" \
    gcloud run jobs delete talisman-job \
      --project="${PROJECT_ID}" --region="${REGION}" --quiet
fi

# --- Service account: app-sa -----------------------------------------------
APP_SA_EMAIL="app-sa@${PROJECT_ID}.iam.gserviceaccount.com"
if gcloud iam service-accounts describe "${APP_SA_EMAIL}" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[iam] service account ${APP_SA_EMAIL} exists"
  run "service account ${APP_SA_EMAIL}" \
    gcloud iam service-accounts delete "${APP_SA_EMAIL}" \
      --project="${PROJECT_ID}" --quiet
fi

# --- Memorystore: market-scripts-redis -------------------------------------
if gcloud redis instances describe market-scripts-redis \
      --project="${PROJECT_ID}" --region="${REGION}" >/dev/null 2>&1; then
  echo "[memorystore] market-scripts-redis exists"
  run "redis instance market-scripts-redis" \
    gcloud redis instances delete market-scripts-redis \
      --project="${PROJECT_ID}" --region="${REGION}" --quiet
fi

# --- Artifact Registry: my-docker-repo -------------------------------------
if gcloud artifacts repositories describe my-docker-repo \
      --project="${PROJECT_ID}" --location="${REGION}" >/dev/null 2>&1; then
  echo "[artifact registry] my-docker-repo exists"
  run "artifact registry my-docker-repo" \
    gcloud artifacts repositories delete my-docker-repo \
      --project="${PROJECT_ID}" --location="${REGION}" --quiet
fi

# --- Secrets: db-password, scheduler-secret --------------------------------
for s in db-password scheduler-secret; do
  if gcloud secrets describe "${s}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "[secret] ${s} exists"
    run "secret ${s}" \
      gcloud secrets delete "${s}" --project="${PROJECT_ID}" --quiet
  fi
done

echo
if [[ "${APPLY}" == "1" ]]; then
  echo "[cleanup] done."
else
  echo "[cleanup] dry-run complete. Re-run with --apply to actually delete."
fi
