# GCP Production State Runbook

This repository now has the code-level migration pieces for the GCP state move:

- Alembic schema in `migrations/versions/20260429_0001_gcp_state_schema.py`.
- Snapshot and migration CLI in `api/gcp_state_migration.py`.
- Cloud Storage-backed document writes via `api/state_storage.py`.
- Production local-write guard via `api/local_write_guard.py`.
- Firebase Hosting rewrite config in `firebase.json`.

## Deploy automation (this directory)

- `cloudbuild.yaml` — builds the API image and pushes to Artifact Registry. It pulls the previous `latest` image first so unchanged dependency layers can be reused.
- `config.example.sh` — copy to `config.sh` (gitignored) and fill in project / SA / bucket values.
- `lib.sh` — shared helpers sourced by every other script. Defaults `IMAGE_TAG` to the current short git SHA, verifies the image exists in Artifact Registry, and refuses to run if the active gcloud project differs from `PROJECT_ID`.
- `bootstrap.sh` — idempotent provisioning of APIs, Artifact Registry, service accounts, Cloud SQL (with backups + PITR + deletion protection + require-SSL), and the GCS state bucket.
- `setup-secrets.sh` — generates random secrets, prompts for the rest, creates Cloud SQL users with their generated passwords, writes everything to Secret Manager, and binds per-secret accessor IAM to each SA.
- `iam.sh` — idempotently grants the project-, bucket-, and Cloud Run-job-level IAM bindings the deploy SAs need (cloudsql.client, logging.logWriter, bucket objectAdmin, job executor on scheduled jobs, and job executor with overrides on the async runner).
- `deploy-api.sh` — Cloud Run service `${API_SERVICE}` (matches the `firebase.json` rewrite). Tunables: `API_CPU`, `API_MEMORY`, `API_CONCURRENCY`, `API_MIN_INSTANCES`, `API_MAX_INSTANCES`, `API_TIMEOUT`. Defaults are `1` vCPU, `1Gi`, and concurrency `20` because long-running analysis is offloaded to Cloud Run Jobs; raise these if synchronous endpoints show memory pressure or CPU saturation.
- `deploy-async-job.sh` — generic Cloud Run Job running `python -m api.async_job_runner run`. Tunables: `ASYNC_JOB_CPU`, `ASYNC_JOB_MEMORY`, `ASYNC_JOB_TIMEOUT`, `ASYNC_JOB_MAX_RETRIES`.
- `deploy-agent-worker.sh` — warm Cloud Run worker pool running `python -m api.agent_worker_loop run` for durable agent workflow turns. Defaults to `1` vCPU and `512Mi`.
- `deploy-analyzer-worker.sh` — optional warm Cloud Run worker pool running `python -m api.job_worker_loop run` with analyzer job/queue defaults from env for low-latency portfolio analyzer jobs. Defaults to disabled (`0` instances), `1` vCPU, and `1Gi`.
- `deploy-ontology-worker.sh` — optional warm Cloud Run worker pool running `python -m api.job_worker_loop run` with ontology job/queue defaults from env for low-latency ontology query jobs. Defaults to disabled (`0` instances), `1` vCPU, and `512Mi`.
- `deploy-migration-job.sh` — Cloud Run Job that runs `python -m api.gcp_state_migration migrate`.
- `deploy-top50-refresh-job.sh` — Cloud Run Job that refreshes the cached S&P 500 top-50.
- `deploy-backend.sh` — build via Cloud Build at the current short git SHA, run Alembic migrations, then roll API + Cloud Run Jobs to that SHA. Refuses to run on a dirty tree (override with `ALLOW_DIRTY=1`); skip the build with `SKIP_BUILD=1`. Routine deploys skip IAM, Scheduler, and monitoring reconciliation by default; use `FULL_SYNC=1` after infrastructure/config changes.
- `deploy-frontend.sh` — builds `frontend/dist` and deploys Firebase Hosting for the configured `PROJECT_ID`.
- `deploy-all.sh` — deploys the full production stack by running `deploy-backend.sh` first and `deploy-frontend.sh` second. `SKIP_BUILD=1` skips the backend container build; `SKIP_FRONTEND_BUILD=1` deploys the existing `frontend/dist`.
- `setup-scheduler.sh` — idempotently create/update the required Cloud Scheduler jobs (async-job-sweep hourly, top50-refresh weekday 23:00 UTC, market-snapshot-refresh weekday 23:15 UTC, macro-snapshot-refresh weekday 23:30 UTC, and continuous-optimizer weekday 10:15 America/New_York) and delete optional/deprecated jobs unless explicitly enabled. Pulls `X-Scheduler-Secret` and `X-Api-Proxy-Secret` from Secret Manager so the values never live in this repo.
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
./infra/gcp/deploy-agent-worker.sh
./infra/gcp/deploy-ontology-worker.sh
```

Routine backend deploys are optimized for the common code-rollout path:

- `deploy-backend.sh` deploys the migration job first, starts the non-migration Cloud Run Job and warm worker updates in parallel, runs Alembic migrations, then deploys the API service.
- IAM, Scheduler, and monitoring syncs are intentionally skipped by default because those resources rarely change and each sync performs multiple GCP control-plane calls.
- Run `FULL_SYNC=1 ./infra/gcp/deploy-backend.sh` when service accounts, Scheduler definitions, monitoring policy files, or other infra config changed.
- Set `PARALLEL_JOB_DEPLOYS=0` if you need the older sequential Cloud Run Job deploy behavior for debugging.

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

Production async work uses the `async_jobs` Postgres table for durable status,
progress, results, and dedupe. Batch jobs, portfolio analyzer jobs, portfolio
sizer jobs, and ontology query jobs run through the generic Cloud Run Job by
default. Agent chat workflow turns run through a warm Cloud Run worker pool so
interactive agent paths do not pay per-request Cloud Run Job startup latency.

Required services:

- Cloud Run service `${API_SERVICE}`: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}`
- Cloud Run Job `${ASYNC_JOB_RUNNER_JOB}`: `python -m api.async_job_runner run`
- Cloud Run worker pool `${AGENT_WORKER_POOL}`: `python -m api.agent_worker_loop run`
- Optional Cloud Run worker pool `${ANALYZER_WORKER_POOL}`: `python -m api.job_worker_loop run` with `JOB_WORKER_JOB_TYPE=analyzer` and `JOB_WORKER_QUEUE=analyzer`; keep at `0` instances unless low-latency analyzer jobs are needed
- Optional Cloud Run worker pool `${ONTOLOGY_WORKER_POOL}`: `python -m api.job_worker_loop run` with `JOB_WORKER_JOB_TYPE=ontology` and `JOB_WORKER_QUEUE=ontology`; keep at `0` instances unless low-latency ontology queries are needed
- Cloud Scheduler jobs:
  - hourly: `POST /api/admin/jobs/enqueue-async-job-sweep`
  - weekdays at 23:00 UTC: run `${TOP50_REFRESH_JOB}`
  - weekdays at 23:15 UTC: `POST /api/admin/jobs/enqueue-market-snapshot-refresh`
  - weekdays at 23:30 UTC: `POST /api/admin/jobs/enqueue-macro-snapshot-refresh`
  - weekdays at 10:15 America/New_York: `POST /api/admin/jobs/enqueue-continuous-optimizer`

Cloud Run worker pools run a fixed number of instances, not min/max autoscaling.
Set `AGENT_WORKER_INSTANCES`, `ANALYZER_WORKER_INSTANCES`, and
`ONTOLOGY_WORKER_INSTANCES` in `infra/gcp/config.sh` to control warm worker
capacity.

Optional analyzer and ontology warm worker pools default to `0` instances. If
Cloud Monitoring shows p95 memory approaching roughly 80% of a configured worker
limit, raise the affected `*_WORKER_MEMORY` setting before the next deploy.

The governance outbox drain scheduler is disabled by default because the
runtime job is now a no-op. To recreate it for a temporary safety check, run
`SCHEDULE_GOVERNANCE_OUTBOX_DRAIN=1 ./infra/gcp/setup-scheduler.sh`; override
the cadence with `GOVERNANCE_OUTBOX_DRAIN_SCHEDULE` if needed.

The watch trigger monitor scheduler is disabled by default while its ontology-ID
handling is repaired. The endpoint remains available for manual/admin use, and
can be scheduled with
`SCHEDULE_WATCH_TRIGGER_MONITOR=1 ./infra/gcp/setup-scheduler.sh`; override the
cadence with `WATCH_TRIGGER_MONITOR_SCHEDULE` if needed.

Scheduled cache warming is disabled by default. The cache-warm endpoint remains
available for manual/admin use, and can be scheduled with
`SCHEDULE_CACHE_WARM=1 CACHE_WARM_SCHEDULE="0 * * * *" ./infra/gcp/setup-scheduler.sh`,
but the default deployment does not schedule it because the warmer executes in a
separate Cloud Run Job and does not share the API service's process cache.

Cloud Run services and jobs also set `API_DISK_CACHE_DISABLE=true` through the
shared deployment environment. This disables only `api.cache`'s generic JSON
disk cache for route-level `short_cache` / `long_cache` entries; it does not
disable module-specific durable caches, GCS/Postgres state, or market snapshot
storage. Local development keeps the generic disk cache enabled unless that env
var is set explicitly.

Cloud Scheduler should send both `X-Scheduler-Secret: $SCHEDULER_SECRET` and
`X-Api-Proxy-Secret: $API_PROXY_SECRET` when it calls the API service directly.
The setup script pulls both values from Secret Manager.

Required async env/secrets:

```bash
ASYNC_JOB_BACKEND="cloud_run_jobs"
AGENT_CHAT_DISPATCH_BACKEND="warm_worker"
ASYNC_DISPATCH_BACKEND_ANALYZER="cloud_run_jobs"
ASYNC_DISPATCH_BACKEND_SIZER="cloud_run_jobs"
ASYNC_DISPATCH_BACKEND_ONTOLOGY="cloud_run_jobs"
ASYNC_QUEUE_ANALYZER="analyzer"
ASYNC_QUEUE_SIZER="sizer"
ASYNC_QUEUE_ONTOLOGY="ontology"
ASYNC_CLOUD_RUN_JOB="talisman-async-job"
ASYNC_JOB_COMPLETED_TTL_SECONDS="86400"
ASYNC_JOB_FAILED_TTL_SECONDS="604800"
ASYNC_JOB_STALE_GRACE_SECONDS="300"
SCHEDULER_SECRET="..."
```

The API service account needs `roles/run.jobsExecutorWithOverrides` on the async
Cloud Run Job because dispatch passes `ASYNC_JOB_ID` as a per-execution env
override. `iam.sh` applies that binding after the job exists.

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
- Verify `GET /api/admin/quiescence` returns `write_freeze=true`, `active_jobs=0`, `pending_writes=0`.

After migration:

- Cloud Run `/api/health` is healthy.
- Firebase frontend loads and authenticates.
- Thesis and overview writes land in `live/theses/**` and `live/overviews/**`.
- Async jobs persist status in `async_jobs`.
- `ENVIRONMENT=production` direct writes under project root raise `ProductionLocalWriteError`.
