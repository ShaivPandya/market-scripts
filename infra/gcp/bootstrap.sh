#!/usr/bin/env bash
# One-shot, idempotent provisioning of the foundation GCP resources.
#
# Covers steps 1-7 from the GCP setup list:
#   1. Enable APIs
#   2. Artifact Registry repo
#   3. Service accounts (api-sa, worker-sa, migrator-sa)
#   4. Cloud SQL instance + database (users are created separately;
#      passwords belong in Secret Manager)
#   5. Cloud Storage state bucket
#   6. Memorystore for Valkey
#   7. Serverless VPC Access connector
#
# Re-running is safe: every step skips if the resource already exists.
# Cloud SQL + Memorystore creation each take several minutes; the script
# waits on them.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var ARTIFACT_REPO
require_var CLOUDSQL_INSTANCE
require_var VPC_CONNECTOR
require_var GCS_STATE_BUCKET

# Cloud SQL instance name is the third segment of CLOUDSQL_INSTANCE.
SQL_INSTANCE="${CLOUDSQL_INSTANCE##*:}"
SQL_DATABASE="talisman"
VPC_CONNECTOR_NAME="${VPC_CONNECTOR##*/}"
VPC_NETWORK="${VPC_NETWORK:-default}"
VPC_CONNECTOR_RANGE="${VPC_CONNECTOR_RANGE:-10.8.0.0/28}"
SQL_TIER="${SQL_TIER:-db-custom-2-7680}"
REDIS_INSTANCE="${REDIS_INSTANCE:-talisman}"
REDIS_SIZE_GB="${REDIS_SIZE_GB:-1}"
REDIS_TIER="${REDIS_TIER:-basic}"
REDIS_VERSION="${REDIS_VERSION:-valkey_7_2}"

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
  redis.googleapis.com \
  vpcaccess.googleapis.com \
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
    --region="${REGION}" \
    --tier="${SQL_TIER}" \
    --database-flags=cloudsql.enable_pgvector=on \
    --storage-auto-increase
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
# 6. Memorystore for Valkey
###############################################################################
log "Memorystore (Valkey): ${REDIS_INSTANCE}"
if gcloud redis instances describe "${REDIS_INSTANCE}" \
      --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud redis instances create "${REDIS_INSTANCE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --tier="${REDIS_TIER}" \
    --size="${REDIS_SIZE_GB}" \
    --redis-version="${REDIS_VERSION}" \
    --network="${VPC_NETWORK}"
fi

###############################################################################
# 7. Serverless VPC Access connector
###############################################################################
log "VPC connector: ${VPC_CONNECTOR_NAME}"
if gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR_NAME}" \
      --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "  exists"
else
  gcloud compute networks vpc-access connectors create "${VPC_CONNECTOR_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --network="${VPC_NETWORK}" \
    --range="${VPC_CONNECTOR_RANGE}"
fi

###############################################################################
# Done. Print follow-up steps.
###############################################################################
log "Bootstrap complete. Still to do (not handled by this script):"
cat <<EOF
  - Cloud SQL users (talisman_app, talisman_worker, talisman_migrator):
      gcloud sql users create talisman_app       --instance=${SQL_INSTANCE} --password=...
      gcloud sql users create talisman_worker    --instance=${SQL_INSTANCE} --password=...
      gcloud sql users create talisman_migrator  --instance=${SQL_INSTANCE} --password=...
    Store each password in Secret Manager (DATABASE_URL_API/_WORKER/_MIGRATION)
    using the URL format from infra/gcp/README.md.

  - CREATE EXTENSION vector;  (run as the migrator user via cloud-sql-proxy)

  - Populate the remaining Secret Manager entries listed in
    infra/gcp/config.example.sh (REDIS_URL, AUTH_PASSWORD_HASH, JWT_SECRET,
    API_PROXY_SECRET, SCHEDULER_SECRET, ANTHROPIC_API_KEY, FRED_API_KEY,
    ESTAT_APP_ID, SODA_APP_TOKEN). Memorystore IP is shown by:
      gcloud redis instances describe ${REDIS_INSTANCE} --region=${REGION} \\
        --format='value(host)'

  - Grant per-resource IAM bindings to api-sa / worker-sa / migrator-sa
    (bucket access, secret accessor on the right secrets, cloudsql.client,
    redis.editor, run.invoker for api-sa).

  - Run alembic upgrade head as the migrator user.

  - Build the image and run the deploy scripts:
      gcloud builds submit --config=infra/gcp/cloudbuild.yaml .
      IMAGE_TAG=\$(git rev-parse --short HEAD) ./infra/gcp/deploy-api.sh
      IMAGE_TAG=\$(git rev-parse --short HEAD) ./infra/gcp/deploy-worker.sh
      IMAGE_TAG=\$(git rev-parse --short HEAD) ./infra/gcp/deploy-migration-job.sh
EOF
