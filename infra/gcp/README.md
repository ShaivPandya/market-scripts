# GCP Production State Runbook

This repository now has the code-level migration pieces for the GCP state move:

- Alembic schema in `migrations/versions/20260429_0001_gcp_state_schema.py`.
- Snapshot and migration CLI in `api/gcp_state_migration.py`.
- Cloud Storage-backed document writes via `api/state_storage.py`.
- Production local-write guard via `api/local_write_guard.py`.
- Firebase Hosting rewrite config in `firebase.json`.

## Deploy automation (this directory)

- `cloudbuild.yaml` — builds the API image and pushes to Artifact Registry.
- `config.example.sh` — copy to `config.sh` (gitignored) and fill in project / SA / bucket values.
- `lib.sh` — shared helpers sourced by every other script. Defaults `IMAGE_TAG` to the current short git SHA, verifies the image exists in Artifact Registry, and refuses to run if the active gcloud project differs from `PROJECT_ID`.
- `bootstrap.sh` — idempotent provisioning of APIs, Artifact Registry, service accounts, Cloud SQL (with backups + PITR + deletion protection + require-SSL), and the GCS state bucket.
- `setup-secrets.sh` — generates random secrets, prompts for the rest, creates Cloud SQL users with their generated passwords, writes everything to Secret Manager, and binds per-secret accessor IAM to each SA.
- `iam.sh` — idempotently grants the project-, bucket-, and Cloud Run-job-level IAM bindings the deploy SAs need (cloudsql.client, logging.logWriter, bucket objectAdmin, job executor on scheduled jobs, and job executor with overrides on the async runner).
- `deploy-api.sh` — Cloud Run service `${API_SERVICE}` (matches the `firebase.json` rewrite). Tunables: `API_CPU`, `API_MEMORY`, `API_CONCURRENCY`, `API_MIN_INSTANCES`, `API_MAX_INSTANCES`, `API_TIMEOUT`. Defaults are `1` vCPU, `1Gi`, and concurrency `20` because long-running analysis is offloaded to Cloud Run Jobs; raise these if synchronous endpoints show memory pressure or CPU saturation.
- `deploy-async-job.sh` — generic Cloud Run Job running `python -m api.async_job_runner run`. Tunables: `ASYNC_JOB_CPU`, `ASYNC_JOB_MEMORY`, `ASYNC_JOB_TIMEOUT`, `ASYNC_JOB_MAX_RETRIES`.
- `deploy-worker.sh` — deprecated stub; do not redeploy the legacy worker pool.
- `deploy-migration-job.sh` — Cloud Run Job that runs `python -m api.gcp_state_migration migrate`.
- `deploy-top50-refresh-job.sh` — Cloud Run Job that refreshes the cached S&P 500 top-50.
- `deploy-backend.sh` — build via Cloud Build at the current short git SHA, then roll API + Cloud Run Jobs to that SHA. Refuses to run on a dirty tree (override with `ALLOW_DIRTY=1`); skip the build with `SKIP_BUILD=1`.
- `deploy-frontend.sh` — builds `frontend/dist` and deploys Firebase Hosting for the configured `PROJECT_ID`.
- `deploy-all.sh` — deploys the full production stack by running `deploy-backend.sh` first and `deploy-frontend.sh` second. `SKIP_BUILD=1` skips the backend container build; `SKIP_FRONTEND_BUILD=1` deploys the existing `frontend/dist`.
- `setup-scheduler.sh` — idempotently create/update the required Cloud Scheduler jobs (async-job-sweep hourly, top50-refresh weekday 23z UTC, market-snapshot-refresh weekday 23:15z UTC) and delete the old high-frequency cache-warm job unless `SCHEDULE_CACHE_WARM=1` is set. Pulls `X-Scheduler-Secret` and `X-Api-Proxy-Secret` from Secret Manager so the values never live in this repo.
- `setup-governance-monitoring.sh` — idempotently creates/updates governance audit/provenance log-based metrics and the alert policy in `monitoring-governance-alerts.json`.
- `cleanup-stale.sh` — dry-runs (or `--apply` deletes) GCP resources that pre-date the current scripts and are no longer referenced.

First-time setup:

```bash
cp infra/gcp/config.example.sh infra/gcp/config.sh   # then edit
./infra/gcp/bootstrap.sh           # provision foundation infra
./infra/gcp/setup-secrets.sh       # SQL users + Secret Manager + per-secret IAM
./infra/gcp/iam.sh                 # project + bucket + run-job IAM
# (still need: CREATE EXTENSION vector + alembic upgrade head as the migrator)
./infra/gcp/deploy-all.sh          # deploy backend, then frontend, at the current SHA
./infra/gcp/iam.sh                 # re-run to bind job executor roles on the now-deployed jobs
./infra/gcp/setup-scheduler.sh     # wire up Cloud Scheduler
./infra/gcp/setup-governance-monitoring.sh
```

Routine deploys:

```bash
# full stack
./infra/gcp/deploy-all.sh

# or deploy components separately:
./infra/gcp/deploy-backend.sh
./infra/gcp/deploy-frontend.sh

# or roll a single component:
./infra/gcp/deploy-api.sh
./infra/gcp/deploy-async-job.sh
```

## Cloud SQL

Use Unix socket connectivity:

```bash
INSTANCE_CONNECTION_NAME="PROJECT_ID:REGION:INSTANCE_ID"
DATABASE_URL_API="postgresql+psycopg://talisman_app:${DB_PASSWORD}@/talisman?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
DATABASE_URL_WORKER="postgresql+psycopg://talisman_worker:${DB_PASSWORD}@/talisman?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
DATABASE_URL_MIGRATION="postgresql+psycopg://talisman_migrator:${DB_PASSWORD}@/talisman?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
```

Cloud Run services and jobs must include:

```bash
--add-cloudsql-instances="${INSTANCE_CONNECTION_NAME}"
```

## Async Jobs

Production async work uses one generic Cloud Run Job for on-demand execution and
the `async_jobs` Postgres table for durable status, progress, results, and
dedupe.

Required services:

- Cloud Run service `api`: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}`
- Cloud Run Job `talisman-async-job`: `python -m api.async_job_runner run`
- Cloud Scheduler jobs:
  - hourly: `POST /api/v1/admin/jobs/enqueue-async-job-sweep`
  - weekdays at 23:00 UTC: run `${TOP50_REFRESH_JOB}`
  - weekdays at 23:15 UTC: `POST /api/v1/admin/jobs/enqueue-market-snapshot-refresh`

Scheduled cache warming is disabled by default. The cache-warm endpoint remains
available for manual/admin use, and can be scheduled with
`SCHEDULE_CACHE_WARM=1 CACHE_WARM_SCHEDULE="0 * * * *" ./infra/gcp/setup-scheduler.sh`,
but the default deployment does not schedule it because the warmer executes in a
separate Cloud Run Job and does not share the API service's process cache.

Cloud Scheduler should send both `X-Scheduler-Secret: $SCHEDULER_SECRET` and
`X-Api-Proxy-Secret: $API_PROXY_SECRET` when it calls the API service directly.
The setup script pulls both values from Secret Manager.

```bash
--add-cloudsql-instances="${INSTANCE_CONNECTION_NAME}"
```

Required async env/secrets:

```bash
ASYNC_JOB_BACKEND="cloud_run_jobs"
ASYNC_CLOUD_RUN_JOB="talisman-async-job"
ASYNC_JOB_COMPLETED_TTL_SECONDS="86400"
ASYNC_JOB_FAILED_TTL_SECONDS="604800"
ASYNC_JOB_STALE_GRACE_SECONDS="300"
SCHEDULER_SECRET="..."
```

The API service account needs `roles/run.jobsExecutorWithOverrides` on the async
Cloud Run Job because dispatch passes `ASYNC_JOB_ID` as a per-execution env
override. `iam.sh` applies that binding after the job exists.

Legacy cleanup after cutover:

```bash
gcloud beta run worker-pools delete talisman-worker --region="${REGION}" --project="${PROJECT_ID}"
gcloud redis instances delete talisman --region="${REGION}" --project="${PROJECT_ID}"
gcloud secrets delete REDIS_URL --project="${PROJECT_ID}"
```

Only run those deletes after a deployed API revision with
`ASYNC_JOB_BACKEND=cloud_run_jobs` is receiving traffic and async polling shows
no old active jobs.

## Database Migrations

Run Alembic with the migration URL:

```bash
DATABASE_URL="${DATABASE_URL_MIGRATION}" alembic upgrade head
```

The retrieval schema is pinned to all-MiniLM-L6-v2:

- `retrieval_chunks.embedding vector(384) not null`
- HNSW cosine index: `USING hnsw (embedding vector_cosine_ops)`

## Source Snapshot

After freezing writes, create the source tarball locally or on the old runtime:

```bash
python3 -m api.gcp_state_migration snapshot \
  --project-root . \
  --output /tmp/source.tar.zst
```

Upload it to:

```bash
gs://$GCS_STATE_BUCKET/backups/pre-migration/$MIGRATION_RUN_ID/source.tar.zst
```

The snapshot command copies each SQLite DB plus `-wal` and `-shm` siblings, runs
`PRAGMA wal_checkpoint(TRUNCATE)` on the copy, and then reads from a `VACUUM INTO`
compact snapshot.

## Migration Job

The Cloud Run Job entrypoint should be:

```bash
python3 -m api.gcp_state_migration migrate
```

Required env/secrets for the migration job:

```bash
DATABASE_URL="$DATABASE_URL_MIGRATION"
GCS_STATE_BUCKET="talisman-state-prod"
MIGRATION_RUN_ID="YYYYMMDDTHHMMSSZ"
ENVIRONMENT="production"
STATE_STORAGE_BACKEND="gcs"
```

The migration service account should have only:

- `roles/cloudsql.client`
- Secret accessor for `DATABASE_URL_MIGRATION`
- GCS access to `backups/pre-migration/**` and the live destination prefixes

It should not have LLM or data-vendor secrets.

## Cutover Checks

Before migration:

- Deploy old API with `WRITE_FREEZE=true`.
- Disable scheduled report workflows.
- Verify `GET /api/v1/admin/quiescence` returns `write_freeze=true`, `active_jobs=0`, `pending_writes=0`.

After migration:

- Cloud Run `/api/health` is healthy.
- Firebase frontend loads and authenticates.
- Thesis and overview writes land in `live/theses/**` and `live/overviews/**`.
- Async jobs persist status in `async_jobs`.
- `ENVIRONMENT=production` direct writes under project root raise `ProductionLocalWriteError`.
