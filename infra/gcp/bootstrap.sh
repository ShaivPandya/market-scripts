#!/usr/bin/env bash
# One-shot, idempotent provisioning of the foundation GCP resources.
#
# Covers:
#   1. Enable APIs
#   2. Artifact Registry repo
#   3. Service accounts (api-sa, worker-sa, migrator-sa)
#   4. Cloud SQL instance + database (users are created separately;
#      passwords belong in Secret Manager). New instances are created with
#      backups + PITR + deletion protection + require-SSL.
#   5. Cloud Storage state bucket
#
# Re-running is safe: every step skips if the resource already exists. Hardening
# flags only apply at create time — to upgrade an existing instance, see the
# follow-up notes printed at the end.
#
# Cloud SQL creation can take several minutes; the script waits on it.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_active_project

# Cloud SQL instance name is the third segment of CLOUDSQL_INSTANCE.
SQL_INSTANCE="${CLOUDSQL_INSTANCE##*:}"
SQL_DATABASE="talisman"
SQL_TIER="${SQL_TIER:-db-custom-2-7680}"

log() { printf '\n[bootstrap] %s\n' "$*"; }

###############################################################################
# 1. Enable APIs
###############################################################################
log "Enabling required APIs (idempotent)…"
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  cloudscheduler.googleapis.com \
  storage.googleapis.com \
  firebasehosting.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}"

###############################################################################
# 2. Artifact Registry
###############################################################################
log "Artifact Registry repo: ${ARTIFACT_REPO}"
if gcloud artifacts repositories describe "${ARTIFACT_REPO}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
fi

###############################################################################
# 3. Service accounts
###############################################################################
for sa in api-sa worker-sa migrator-sa; do
  log "Service account: ${sa}"
  if gcloud iam service-accounts describe \
        "${sa}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  exists"
  else
    gcloud iam service-accounts create "${sa}" --project="${PROJECT_ID}"
  fi
done

###############################################################################
# 4. Cloud SQL instance + database
###############################################################################
log "Cloud SQL instance: ${SQL_INSTANCE}"
if gcloud sql instances describe "${SQL_INSTANCE}" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud sql instances create "${SQL_INSTANCE}" \
    --project="${PROJECT_ID}" \
    --database-version=POSTGRES_16 \
    --edition=enterprise \
    --region="${REGION}" \
    --tier="${SQL_TIER}" \
    --database-flags=cloudsql.enable_pgvector=on \
    --storage-auto-increase \
    --backup \
    --backup-start-time="07:00" \
    --enable-point-in-time-recovery \
    --retained-backups-count=14 \
    --retained-transaction-log-days=7 \
    --deletion-protection \
    --require-ssl
fi

log "Cloud SQL database: ${SQL_DATABASE}"
if gcloud sql databases describe "${SQL_DATABASE}" \
      --instance="${SQL_INSTANCE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud sql databases create "${SQL_DATABASE}" \
    --instance="${SQL_INSTANCE}" --project="${PROJECT_ID}"
fi

###############################################################################
# 5. Cloud Storage state bucket
###############################################################################
log "GCS bucket: ${GCS_STATE_BUCKET}"
if gcloud storage buckets describe "gs://${GCS_STATE_BUCKET}" \
      --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud storage buckets create "gs://${GCS_STATE_BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

###############################################################################
# Done. Print follow-up steps.
###############################################################################
log "Bootstrap complete. Still to do (not handled by this script):"
cat <<EOF
  - Cloud SQL users + secrets:  ./infra/gcp/setup-secrets.sh
      Generates passwords, creates talisman_app / talisman_worker /
      talisman_migrator users, writes DATABASE_URL_* and the rest into
      Secret Manager, and binds least-privilege accessor IAM.

  - Project / bucket / Cloud Run IAM bindings:  ./infra/gcp/iam.sh
      cloudsql.client + logging.logWriter for the SAs, bucket objectAdmin on
      \${GCS_STATE_BUCKET}, and Cloud Run Jobs executor roles for Scheduler
      and API dispatch.

  - CREATE EXTENSION vector;  (run as the migrator user via cloud-sql-proxy)

  - alembic upgrade head as the migrator user.

  - Full deploy:     ./infra/gcp/deploy-all.sh
      (or run cloudbuild + deploy-{api,async-job,migration-job}.sh manually)

  - Cloud Scheduler jobs:  ./infra/gcp/setup-scheduler.sh
      async-job-sweep (hourly), top50-refresh (weekday 23z), optional cache-warm.

  - If you already have an existing Cloud SQL instance that pre-dates the
    hardening flags, apply them in place:
      gcloud sql instances patch ${SQL_INSTANCE} \\
        --backup --backup-start-time=07:00 \\
        --enable-point-in-time-recovery \\
        --retained-backups-count=14 --retained-transaction-log-days=7 \\
        --deletion-protection --require-ssl
EOF
