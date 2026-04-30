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

# Serverless VPC connector for Memorystore (Valkey) access.
# projects/<project>/locations/<region>/connectors/<name>
export VPC_CONNECTOR="projects/${PROJECT_ID}/locations/${REGION}/connectors/talisman"

# Cloud Storage bucket holding production state (theses, overviews, backups).
export GCS_STATE_BUCKET="talisman-state-prod"

# Service accounts (least-privilege per role; see infra/gcp/README.md).
export API_SA="api-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export WORKER_SA="worker-sa@${PROJECT_ID}.iam.gserviceaccount.com"
export MIGRATOR_SA="migrator-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Cloud Run service / job names.
export API_SERVICE="talisman-api"      # must match firebase.json rewrite
export WORKER_POOL="talisman-worker"
export MIGRATION_JOB="talisman-migrate"

# Frontend origin allowed by API CORS (Firebase Hosting URL).
export CORS_ORIGINS="https://${PROJECT_ID}.web.app,https://${PROJECT_ID}.firebaseapp.com"

# Secret Manager names (left of "=") map to env vars consumed by api/* (right).
# These names must already exist in Secret Manager.
export API_SECRETS=(
  "DATABASE_URL=DATABASE_URL_API:latest"
  "REDIS_URL=REDIS_URL:latest"
  "AUTH_PASSWORD_HASH=AUTH_PASSWORD_HASH:latest"
  "JWT_SECRET=JWT_SECRET:latest"
  "API_PROXY_SECRET=API_PROXY_SECRET:latest"
  "SCHEDULER_SECRET=SCHEDULER_SECRET:latest"
  "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
  "FRED_API_KEY=FRED_API_KEY:latest"
  "ESTAT_APP_ID=ESTAT_APP_ID:latest"
  "SODA_APP_TOKEN=SODA_APP_TOKEN:latest"
)

export WORKER_SECRETS=(
  "DATABASE_URL=DATABASE_URL_WORKER:latest"
  "REDIS_URL=REDIS_URL:latest"
  "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
  "FRED_API_KEY=FRED_API_KEY:latest"
  "ESTAT_APP_ID=ESTAT_APP_ID:latest"
  "SODA_APP_TOKEN=SODA_APP_TOKEN:latest"
)

# Migration job runs with the migrator user only — no LLM/data-vendor secrets.
export MIGRATION_SECRETS=(
  "DATABASE_URL=DATABASE_URL_MIGRATION:latest"
)
