# GCP Production State Runbook

This repository now has the code-level migration pieces for the GCP state move:

- Alembic schema in `migrations/versions/20260429_0001_gcp_state_schema.py`.
- Snapshot and migration CLI in `api/gcp_state_migration.py`.
- Cloud Storage-backed document writes via `api/state_storage.py`.
- Production local-write guard via `api/local_write_guard.py`.
- Firebase Hosting rewrite config in `firebase.json`.

## Cloud SQL

Use Unix socket connectivity:

```bash
INSTANCE_CONNECTION_NAME="PROJECT_ID:REGION:INSTANCE_ID"
DATABASE_URL_API="postgresql+psycopg://market_scripts_app:${DB_PASSWORD}@/market_scripts?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
DATABASE_URL_WORKER="postgresql+psycopg://market_scripts_worker:${DB_PASSWORD}@/market_scripts?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
DATABASE_URL_MIGRATION="postgresql+psycopg://market_scripts_migrator:${DB_PASSWORD}@/market_scripts?host=/cloudsql/${INSTANCE_CONNECTION_NAME}"
```

Cloud Run services and jobs must include:

```bash
--add-cloudsql-instances="${INSTANCE_CONNECTION_NAME}"
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
GCS_STATE_BUCKET="market-scripts-state-prod"
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
