# GCP Production State Runbook

This repository now has the code-level migration pieces for the GCP state move:

- Alembic schema in `migrations/versions/20260429_0001_gcp_state_schema.py`.
- Snapshot and migration CLI in `api/gcp_state_migration.py`.
- Cloud Storage-backed document writes via `api/state_storage.py`.
- Production local-write guard via `api/local_write_guard.py`.
- Firebase Hosting rewrite config in `firebase.json`.

## Deploy automation (this directory)

- `cloudbuild.yaml` — builds the API image and pushes to Artifact Registry.
- `config.example.sh` — copy to `config.sh` (gitignored) and fill in project / SA / bucket / VPC values.
- `lib.sh` — shared helpers sourced by every other script. Defaults `IMAGE_TAG` to the current short git SHA, verifies the image exists in Artifact Registry, and refuses to run if the active gcloud project differs from `PROJECT_ID`.
- `bootstrap.sh` — idempotent provisioning of APIs, Artifact Registry, service accounts, Cloud SQL (with backups + PITR + deletion protection + require-SSL), the GCS state bucket, and Memorystore. Direct VPC egress is configured per-service in the deploy scripts; no connector is provisioned.
- `setup-secrets.sh` — generates random secrets, prompts for the rest, creates Cloud SQL users with their generated passwords, writes everything to Secret Manager, and binds per-secret accessor IAM to each SA.
- `iam.sh` — idempotently grants the project-, bucket-, and Cloud Run-job-level IAM bindings the deploy SAs need (cloudsql.client, logging.logWriter, bucket objectAdmin, run.invoker on the migrator's jobs).
- `deploy-api.sh` — Cloud Run service `${API_SERVICE}` (matches the `firebase.json` rewrite). Tunables: `API_CPU`, `API_MEMORY`, `API_CONCURRENCY`, `API_MIN_INSTANCES`, `API_MAX_INSTANCES`, `API_TIMEOUT`.
- `deploy-worker.sh` — Cloud Run worker pool running `python -m api.rq_worker`. Tunables: `WORKER_CPU`, `WORKER_MEMORY`, `WORKER_INSTANCES`, `WORKER_QUEUES`.
- `deploy-migration-job.sh` — Cloud Run Job that runs `python -m api.gcp_state_migration migrate`.
- `deploy-top50-refresh-job.sh` — Cloud Run Job that refreshes the cached S&P 500 top-50.
- `deploy-frontend.sh` — builds `frontend/dist` and deploys Firebase Hosting for the configured `PROJECT_ID`.
- `deploy-all.sh` — build via Cloud Build at the current short git SHA, then roll API + worker + jobs to that SHA. Refuses to run on a dirty tree (override with `ALLOW_DIRTY=1`); skip the build with `SKIP_BUILD=1`.
- `setup-scheduler.sh` — idempotently create/update the three Cloud Scheduler jobs (cache-warm every 5min, async-job-sweep hourly, top50-refresh weekday 23z UTC). Pulls `X-Scheduler-Secret` from Secret Manager so the value never lives in this repo.
- `cleanup-stale.sh` — dry-runs (or `--apply` deletes) GCP resources that pre-date the current scripts and are no longer referenced.

First-time setup:

```bash
cp infra/gcp/config.example.sh infra/gcp/config.sh   # then edit
./infra/gcp/bootstrap.sh           # provision foundation infra
./infra/gcp/setup-secrets.sh       # SQL users + Secret Manager + per-secret IAM
./infra/gcp/iam.sh                 # project + bucket + run-job IAM
# (still need: CREATE EXTENSION vector + alembic upgrade head as the migrator)
./infra/gcp/deploy-all.sh          # build + deploy everything at the current SHA
./infra/gcp/iam.sh                 # re-run to bind run.invoker on the now-deployed jobs
./infra/gcp/setup-scheduler.sh     # wire up Cloud Scheduler
```

Routine deploys:

```bash
# backend stack
./infra/gcp/deploy-all.sh

# frontend hosting
./infra/gcp/deploy-frontend.sh

# or roll a single component:
./infra/gcp/deploy-api.sh
./infra/gcp/deploy-worker.sh
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

Production async work uses RQ workers, Memorystore for Valkey, and the
`async_jobs` Postgres table.

Required services:

- Cloud Run service `api`: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}`
- Cloud Run worker pool `worker`: `python -m api.rq_worker default screens reports`
- Cloud Scheduler jobs:
  - every 5 minutes: `POST /api/v1/admin/jobs/enqueue-cache-warm`
  - hourly: `POST /api/v1/admin/jobs/enqueue-async-job-sweep`

Cloud Scheduler should send `X-Scheduler-Secret: $SCHEDULER_SECRET`. If
`API_PROXY_SECRET` is enabled on the API service, the scheduler request must
also include `X-Api-Proxy-Secret` or route through the same proxy that injects it.

Memorystore connectivity:

- Create Memorystore in the same region as Cloud Run on the `default` VPC.
- Use Direct VPC egress on Cloud Run (no Serverless VPC Access connector
  required). The deploy scripts pass `--network=default --subnet=default
  --vpc-egress=private-ranges-only`, which gives the service a network
  interface in the default subnet that can reach Memorystore's private IP.

```bash
--network="${VPC_NETWORK}" \
--subnet="${VPC_SUBNET}" \
--vpc-egress=private-ranges-only \
--add-cloudsql-instances="${INSTANCE_CONNECTION_NAME}"
```

Required async env/secrets:

```bash
REDIS_URL="redis://MEMORYSTORE_PRIVATE_IP:6379/0"
ASYNC_JOB_BACKEND="rq"
ASYNC_WORKER_QUEUES="default,screens,reports"
ASYNC_JOB_COMPLETED_TTL_SECONDS="86400"
ASYNC_JOB_FAILED_TTL_SECONDS="604800"
SCHEDULER_SECRET="..."
```

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
