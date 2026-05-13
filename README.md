# Talisman

Talisman is a personal investment research and market-monitoring application. It combines a React dashboard, a FastAPI backend, and a collection of Python research modules for portfolio analytics, macro data, equities screens, FX, commodities, fixed income, investment theses, research workflows, and LLM-assisted analysis.

The repository is intentionally mixed-mode:

- Most market and portfolio analysis lives in plain Python modules under topical folders.
- The FastAPI app exposes those modules through authenticated JSON APIs.
- The React frontend is the main operating surface for dashboards, screeners, thesis management, portfolio workflows, and agent-assisted research.
- Local development uses Postgres for runtime state and local files for document/blob state by default. Production runs on Google Cloud with Cloud Run, Cloud SQL, Cloud Storage, Cloud Scheduler, and Firebase Hosting.

Nothing in this repository is investment advice.

## Quickstart

### Prerequisites

- Python 3.12
- Node 20.19 or newer
- A local `.env` file copied from `.env.example`

### Install Python dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -c requirements-lock.txt
```

For a smaller runtime-only environment, use:

```bash
pip install -r requirements.txt -c requirements-lock.txt
```

Optional extras:

```bash
# Semantic retrieval / embeddings
pip install -r requirements-embeddings.txt -c requirements-lock.txt

# FX model helpers
pip install -r fx/model/requirements.txt

# Government bonds standalone tools
pip install -r government_bonds/requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Common local settings:

- `FRED_API_KEY` for FRED-backed macro and rates modules.
- `AUTH_PASSWORD_HASH` and `JWT_SECRET` for password-mode UI login.
- `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and/or `GEMINI_API_KEY` for agent, thesis, overview, and report workflows.
- `ESTAT_APP_ID`, `SODA_APP_TOKEN`, and `EIA_API_KEY` for modules that use those data sources.

Generate a bcrypt login hash with:

```bash
python3 -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt(12)).decode())"
```

Local development uses Postgres for runtime state. To make the defaults explicit:

```bash
ENVIRONMENT=development
STATE_DB_BACKEND=postgres
DATABASE_URL=postgresql://localhost/talisman_dev
STATE_STORAGE_BACKEND=local
ASYNC_JOB_BACKEND=local
AUTH_MODE=password
```

Create the local database and apply the state schema before starting the API:

```bash
createdb talisman_dev
DATABASE_URL=postgresql://localhost/talisman_dev alembic upgrade head
```

### Run the backend

From the repository root:

```bash
ENVIRONMENT=development \
STATE_DB_BACKEND=postgres \
DATABASE_URL=postgresql://localhost/talisman_dev \
STATE_STORAGE_BACKEND=local \
ASYNC_JOB_BACKEND=local \
uvicorn api.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Interactive API docs are available in development at `http://localhost:8000/api/docs`.

### Run the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The UI runs at `http://localhost:5173`. Vite proxies `/api/*` to `http://localhost:8000`, and the frontend API client defaults to `/api`.

You can run the API and UI together from `frontend/`:

```bash
npm run dev:all
```

## What Is In The App

Core portfolio and research workflows:

- Portfolio dashboard, position editing, hedge positions, and performance analytics.
- Portfolio analyzer, optimizer, hedging tool, and portfolio sizer with async execution paths for heavier jobs.
- Thesis and overview generation from uploaded PDFs, editable thesis/overview storage, thesis status tracking, and position dossiers.
- Workspace entities: catalysts, kill conditions, approvals, actions, triggers, recommendations, and workflow runs.
- Agent chat and memory for cross-cutting research questions over portfolio data, theses, market data, materialized ontology risk snapshots, and workflow artifacts.

Market and macro modules:

- Equity screeners, fundamental momentum, financials, DCF, market technicals, sector metrics, index dashboard, and sentiment.
- Macro dashboards for liquidity, economic growth, labor, housing, country data, central banks, positioning, and signal aggregation.
- FX dashboard and FX macro model.
- Commodities dashboard, futures curve tools, commodity research, and aluminum model/backtests.
- Yield curve and bond dashboard.
- Weekly, daily, recommendation, and portfolio news report workflows.

## Repository Layout

- `api/` - FastAPI application, routers, serializers, async jobs, state adapters, retrieval, agent tools, and GCP migration utilities.
- `frontend/` - React 19, TypeScript, Vite, Tailwind CSS 4, Radix UI, React Query, Recharts, and route-level pages.
- `portfolio/` - Portfolio database helpers, analytics, thesis storage, news digests, optimizer, sizer, hedging, momentum, and technical analysis.
- `macro/` - Macro dashboards and monitors.
- `equities/` - Equity screens, index dashboard, market technicals, financial metrics, and sector tools.
- `fx/` - FX dashboard and multi-currency macro model.
- `commodities/` - Commodities dashboard, curve/research tools, and aluminum research.
- `government_bonds/` - Fixed-income and yield-curve tooling.
- `ontology/` - Materialized semantic/risk graph ingestion, parsing, repository, risk, and sector mapping over portfolio, thesis, process, and market data. See [Ontology Materialization Boundary](docs/architecture/ontology.md).
- `auto_report/` - Daily, weekly, and recommendation report generation.
- `investment_theses/` and `investment_overviews/` - Markdown research documents.
- `migrations/` - Alembic migrations for the production Postgres state schema.
- `infra/gcp/` - GCP provisioning, deployment, scheduler, IAM, and migration runbooks.
- `tests/` - Pytest coverage for API behavior, state adapters, reports, async jobs, ontology, security controls, and market modules.

## API Shape

The backend exposes application routes under `/api`. The `/api/health` endpoint is public for service health checks.

Important route groups include these paths under `/api` unless the path already
starts with `/api`:

- Auth: `/api/auth/login`, `/api/auth/logout`, `/api/auth/me`
- Portfolio: `/portfolio`, `/portfolio-positions`, `/hedge-positions`, `/portfolio-analyzer`, `/portfolio-sizer`, `/hedging-tool`
- Research documents: `/thesis/*`, `/overview/*`, `/management-quality/*`, `/document-generation/*`, `/dossier/{ticker}`, `/portfolio-news`
- Agent and retrieval: `/agent/chat`, `/agent/chat/async`, `/agent/workflows`, `/memory/sessions`, `/ontology/*`
- Investing OS state: `/workspace`, `/actions`, `/approvals`, `/triggers`, `/catalysts`, `/kill-conditions`, `/recommendations`, `/workflow-runs`
- Markets: `/momentum`, `/chart`, `/quality-screen`, `/short-screen`, `/long-screen`, `/fundamental-momentum`, `/financials`, `/dcf/*`
- Macro and cross-asset: `/economic-growth`, `/labor-market`, `/housing`, `/liquidity`, `/country-dashboard`, `/central-banks`, `/positioning/*`, `/sentiment/*`, `/signal-aggregator`, `/industry-monitor`
- FX, commodities, and rates: `/fx-dashboard`, `/fx-model`, `/commodities`, `/commodities-curve`, `/commodity-research`, `/yield-curve`, `/bond-dashboard`
- Governance and provenance: `/domain-actions/*`, `/policy-gate/*`, `/risk/*`, `/provenance/*`, `/source-ingestion/*`, `/report-sync/*`
- Admin/settings: `/cache`, `/admin/health`, `/admin/quiescence`, `/admin/jobs/*`, `/settings/*`

Optional routers are imported with graceful degradation. If a heavy dependency or data module fails to import, the API can still start and report degraded modules from admin health.

## Authentication And State

Local password mode is the default:

- Backend `AUTH_MODE=password`
- Frontend `VITE_AUTH_MODE=password` or unset
- Login sets an HTTP-only JWT cookie.
- Frontend requests use `withCredentials: true`.
- Login is protected by `AUTH_LOGIN_RATE_LIMIT` plus failed-attempt lockout:
  `AUTH_LOGIN_FAILURE_LIMIT` failures within `AUTH_LOGIN_FAILURE_WINDOW_SECONDS`
  blocks that client for `AUTH_LOGIN_LOCKOUT_SECONDS`.

Cloudflare Access mode is also supported:

- Backend `AUTH_MODE=cloudflare`
- Frontend `VITE_AUTH_MODE=cloudflare`
- The edge layer must inject `X-Api-Proxy-Secret`.
- The backend requires `API_PROXY_SECRET` in this mode to reject direct API traffic.

State backends:

- Normal development, CI, and production runtime paths use `STATE_DB_BACKEND=postgres` and `DATABASE_URL`; a few isolated tests still exercise explicit SQLite compatibility paths.
- Production additionally uses `STATE_STORAGE_BACKEND=gcs` and `GCS_STATE_BUCKET`.
- Async work runs locally unless `ASYNC_JOB_BACKEND=cloud_run_jobs` or `CLOUD_RUN_JOBS_ENABLED=true` opts into dispatching the generic Cloud Run Job configured by `ASYNC_CLOUD_RUN_JOB=talisman-async-job`.

## Development Commands

Python checks:

```bash
ruff check .
ruff format --check .
mypy api/ --ignore-missing-imports
pytest --tb=short -q
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

Security audit:

```bash
pip-audit -r requirements-audit.txt
```

CI runs Ruff, pytest with coverage over `api`, `auto_report`, and `portfolio`, mypy over `api/`, and `pip-audit`.

## Deployment

Production is deployed through the scripts in `infra/gcp/`:

- Cloud Run service `talisman-api` serves FastAPI.
- A generic Cloud Run Job runs async work with `python -m api.async_job_runner run`.
- Cloud SQL stores application state with Alembic migrations.
- Cloud Storage stores generated and migrated research documents.
- Cloud Scheduler enqueues async sweeps, top-50 refreshes, market/macro snapshot refreshes, and continuous optimizer runs.
- Firebase Hosting serves `frontend/dist` and rewrites `/api/**` to Cloud Run.

First-time setup and routine deploy commands are documented in `infra/gcp/README.md`. The common entry points are:

```bash
./infra/gcp/bootstrap.sh
./infra/gcp/setup-secrets.sh
./infra/gcp/iam.sh
./infra/gcp/deploy-all.sh
./infra/gcp/setup-scheduler.sh
```

Routine full-stack deploy:

```bash
./infra/gcp/deploy-all.sh
```

Routine backend deploys use the fast path: they build the API image with Cloud
Build, deploy non-migration Cloud Run Jobs in parallel, and skip IAM,
Scheduler, and monitoring reconciliation unless requested. After infra/config
changes, run:

```bash
FULL_SYNC=1 ./infra/gcp/deploy-backend.sh
```

## Useful Module Docs

- `frontend/README.md`
- `infra/gcp/README.md`
- `ontology/README.md`
- `fx/model/README.md`
- `government_bonds/README.md`
- `government_bonds/data/README.md`
- `commodities/aluminum/README.md`
- `macro/economic_growth/README.md`
- `macro/liquidity/README.md`
- `macro/positioning/README.md`
- `portfolio/momentum/price_momentum/README.md`
- `equities/market_technicals/README.md`

## Troubleshooting

- `401 Not authenticated`: verify `AUTH_PASSWORD_HASH` and `JWT_SECRET`, then log in at `/login`.
- API starts but a page returns a dependency/degraded error: check `/api/admin/health` and the optional router import logs.
- FRED-backed pages fail: verify `FRED_API_KEY` is present in `.env`.
- Direct script imports fail: run from the repository root so top-level package imports resolve.
- Production writes fail locally: production mode intentionally blocks direct project-root writes unless the configured state storage backend is safe for production.

## Security Notes

- Do not commit `.env` or secrets.
- Production secrets belong in Google Secret Manager.
- Use `API_PROXY_SECRET` with `REQUIRE_API_PROXY_SECRET=true` only when a trusted proxy can inject `X-Api-Proxy-Secret`; Firebase Hosting rewrites do not inject that header.
- Production disables FastAPI docs and schema routes.
