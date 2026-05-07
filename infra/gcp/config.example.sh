# Copy to infra/gcp/config.sh (gitignored) and fill in.
# Every deploy script sources this file.

# Project + location
export PROJECT_ID="project-a6b8946d-eba6-4f39-b83"
export REGION="us-central1"

# Artifact Registry image (built by infra/gcp/cloudbuild.yaml)
export ARTIFACT_REPO="talisman"
export IMAGE_NAME="api"
# Tag deployed by the scripts. Override per-deploy with IMAGE_TAG=<sha> ./deploy-api.sh
export IMAGE_TAG="${IMAGE_TAG:-latest}"

# Cloud SQL (Postgres). Format: PROJECT_ID:REGION:INSTANCE_ID
export CLOUDSQL_INSTANCE="${PROJECT_ID}:${REGION}:talisman"

# Cloud Storage bucket holding production state (theses, overviews, backups).
export GCS_STATE_BUCKET="talisman-state-prod"

# Service accounts (least-privilege per role; see infra/gcp/README.md).
export API_SA="api-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export WORKER_SA="worker-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export MIGRATOR_SA="migrator-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Cloud Run service / job names.
export API_SERVICE="talisman-api"      # must match firebase.json rewrite
export ASYNC_JOB_RUNNER_JOB="talisman-async-job"
export AGENT_WORKER_POOL="talisman-agent-worker"
export AGENT_WORKER_INSTANCES="1"      # fixed Cloud Run worker-pool capacity
export AGENT_WORKER_CPU="1"
export AGENT_WORKER_MEMORY="512Mi"
export ANALYZER_WORKER_POOL="talisman-analyzer-worker"
export ANALYZER_WORKER_INSTANCES="1"   # fixed Cloud Run worker-pool capacity
export ANALYZER_WORKER_CPU="1"
export ANALYZER_WORKER_MEMORY="1Gi"
export SIZER_WORKER_POOL="talisman-sizer-worker"
export SIZER_WORKER_INSTANCES="1"      # fixed Cloud Run worker-pool capacity
export SIZER_WORKER_CPU="1"
export SIZER_WORKER_MEMORY="512Mi"
export ONTOLOGY_WORKER_POOL="talisman-ontology-worker"
export ONTOLOGY_WORKER_INSTANCES="1"   # fixed Cloud Run worker-pool capacity
export ONTOLOGY_WORKER_CPU="1"
export ONTOLOGY_WORKER_MEMORY="512Mi"
export MIGRATION_JOB="talisman-migrate"
export TOP50_REFRESH_JOB="talisman-top50-refresh"

# Frontend origins allowed by API CORS. Include Firebase Hosting defaults plus
# every production custom domain that can serve the frontend.
export CORS_ORIGINS="https://${PROJECT_ID}.web.app,https://${PROJECT_ID}.firebaseapp.com,https://shaivpandya.com,https://www.shaivpandya.com"

# Secret Manager names (left of "=") map to env vars consumed by api/* (right).
# These names must already exist in Secret Manager.
export API_SECRETS=(
  "DATABASE_URL=DATABASE_URL_API:latest"
  "AUTH_PASSWORD_HASH=AUTH_PASSWORD_HASH:latest"
  "JWT_SECRET=JWT_SECRET:latest"
  "API_PROXY_SECRET=API_PROXY_SECRET:latest"
  "SCHEDULER_SECRET=SCHEDULER_SECRET:latest"
  "REPORT_SYNC_SECRET=REPORT_SYNC_SECRET:latest"
  "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
  "OPENAI_API_KEY=OPENAI_API_KEY:latest"
  "FRED_API_KEY=FRED_API_KEY:latest"
  "ESTAT_APP_ID=ESTAT_APP_ID:latest"
  "EIA_API_KEY=EIA_API_KEY:latest"
)

export WORKER_SECRETS=(
  "DATABASE_URL=DATABASE_URL_WORKER:latest"
  "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
  "OPENAI_API_KEY=OPENAI_API_KEY:latest"
  "FRED_API_KEY=FRED_API_KEY:latest"
  "ESTAT_APP_ID=ESTAT_APP_ID:latest"
  "EIA_API_KEY=EIA_API_KEY:latest"
)

# Migration job runs with the migrator user only — no LLM/data-vendor secrets.
export MIGRATION_SECRETS=(
  "DATABASE_URL=DATABASE_URL_MIGRATION:latest"
)
