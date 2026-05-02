# Market Scripts (Market Analysis Dashboard)

A collection of Python “market dashboards” (macro, equities, FX, commodities, portfolio analytics) that can be run in three ways:

1. As standalone Python scripts (terminal output, CSV/PNG outputs for some modules)
2. Via a FastAPI backend (`api/`) + React frontend (`frontend/`) web dashboard

This repo is intentionally “flat”: most analysis modules are plain Python files inside topical folders, and the GUI/API layers import them by adding those folders to `sys.path`.

## Quickstart (local development)

### Prereqs

- Python **3.11+** (the Docker image uses Python 3.12)
- Node **20+** (see `frontend/package.json` engines)

### 1) Install Python dependencies

```bash
pip install -r requirements.txt -c requirements-lock.txt
```

Optional extras (needed for some web pages / endpoints):

```bash
# FX Model (installs statsmodels + pandaSDMX)
pip install -r fx/model/requirements.txt

# Portfolio sizing modules (convex optimization)
pip install cvxpy
```

### 2) Create `.env`

```bash
cp .env.example .env
```

At minimum, you’ll usually want:
- `FRED_API_KEY` (required for most FRED-backed macro modules)
- API auth settings (required to use the web UI in password mode):
  - `AUTH_PASSWORD_HASH`
  - `JWT_SECRET`

Generate a bcrypt password hash:

```bash
python3 -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt(12)).decode())"
```

Then set `AUTH_PASSWORD_HASH=...` and choose any random `JWT_SECRET`.

Optional but supported:
- `LLM_PROVIDER` (`anthropic` or `openai`; defaults to `anthropic`)
- `ANTHROPIC_API_KEY` (required when `LLM_PROVIDER=anthropic`)
- `OPENAI_API_KEY` (required when `LLM_PROVIDER=openai`)
- `ANTHROPIC_MODEL_LOW|MID|HIGH` and `OPENAI_MODEL_LOW|MID|HIGH` (optional tier overrides)
- `ESTAT_APP_ID` (Japan CPI via e-Stat; used by the Country Dashboard and FX model helpers)
- `SODA_APP_TOKEN` (CFTC positioning API throttling reduction)

### 3) Run the FastAPI backend

From the repo root:

```bash
uvicorn api.main:app --reload --port 8000
```

Health check: `GET /api/health` → `{"status":"ok"}`

### 4) Run the React frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI on `http://localhost:5173` and proxies `/api/*` to `http://localhost:8000` (see `frontend/vite.config.ts`).

One command (from `frontend/`):

```bash
npm run dev:all
```

## How the codebase fits together

### The core pattern: “data modules”

Most analysis modules expose a small, UI-friendly entry point like:

- `get_data(...) -> dict` (common)
- `get_snapshot(...) -> dict` (liquidity)

Those functions typically return a Python `dict` containing a mix of:
- `pandas.DataFrame` / `pandas.Series` (tables + time series)
- plain Python scalars / lists / dicts
- `datetime` objects

Each module also often has a `__main__` / CLI path for terminal usage.

### UI layers

**FastAPI (`api/`)**
- Wraps the analysis modules under `/api/*` routes.
- Converts pandas objects to JSON-safe structures with `api/serializers.py`.
- Adds small in-memory TTL caches in `api/cache.py` to avoid refetching expensive data on every request.
- Secures routes with cookie-based JWT auth (`api/routers/auth.py`).

**React frontend (`frontend/`)**
- Calls `/api/*` via Axios with `withCredentials: true` so the HTTP-only auth cookie is included.
- Pages live in `frontend/src/pages/*` and are wired in `frontend/src/App.tsx`.

### Dev/prod API routing

**Local dev**
- The frontend uses Vite’s dev proxy: `/api/*` → `http://localhost:8000` (`frontend/vite.config.ts`).

**Production**
- The frontend is served by Firebase Hosting and `/api/**` is rewritten to the Cloud Run service `talisman-api` in `us-central1` (see `firebase.json`).
- `API_PROXY_SECRET` is required for `AUTH_MODE=cloudflare`, where an edge proxy must inject `X-Api-Proxy-Secret` on `/api/*` requests. For password-mode Firebase Hosting rewrites, leave `REQUIRE_API_PROXY_SECRET=false` because Firebase does not inject that custom header.

## Repository structure (high level)

- `api/` — FastAPI server (`api/main.py`) + route adapters in `api/routers/`
- `frontend/` — React + TypeScript + Vite UI (deployed via Firebase Hosting)
- `macro/` — macro monitors (liquidity, country dashboard, positioning, breakouts, central banks, industry transcripts)
- `equities/` — equity screens + dashboards (index dashboard, market technicals, quality, sector metrics, short screen, universes)
- `portfolio/` — portfolio dashboard, analyzer/sizer modules, momentum modules, technical analysis chart module
- `fx/` — FX dashboard + multi-currency FX macro model (`fx/model/`)
- `commodities/` — commodities dashboard
- `government_bonds/` — standalone bond yield tracker (FRED + local CSVs)
- `data_cache/` — on-disk caches for some modules (e.g. FRED/IMF pulls)

## Web API: route map (overview)

All routes below are under the `/api` prefix (see `api/main.py`).

- Auth (public):
  - `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- Dashboards:
  - `GET /portfolio` → `portfolio/portfolio_dashboard.py`
  - `POST /portfolio-analyzer` (alias: `POST /portfolio-optimizer`) → `portfolio/portfolio_optimizer/portfolio_analyzer.py`
  - `POST /hedging-tool`, `POST /hedging-tool/async`, `GET /hedging-tool/async/{job_id}` → `portfolio/portfolio_optimizer/hedging_tool.py`
  - `GET /momentum` → `portfolio/momentum/price_momentum/momentum.py`
  - `POST /chart` → `portfolio/technical_analysis/technical_analysis.py`
  - `POST /quality-screen` → `equities/quality/quality.py`
  - `POST /short-screen` → `equities/short_screen/short_screen.py`
  - `POST /fundamental-momentum` → `portfolio/momentum/fundamental_momentum/*`
  - `GET /index-dashboard` → `equities/index_dashboard/index_dashboard.py`
  - `GET /fx-dashboard` → `fx/fx_dashboard/fx_dashboard.py`
  - `GET /commodities` → `commodities/commodities_dashboard.py`
  - `GET /market-breadth`, `GET /top50-breadth`, `GET /price-volume-signals`, `GET /vix-term-structure` → `equities/market_technicals/*`
  - `GET /sector-metrics` → `equities/sector_metrics/sector_metrics.py`
  - `GET /positioning/*` → `macro/positioning/positioning.py`
  - `GET /breakout` → `macro/breakout/breakout.py`
  - `POST /fx-model`, `GET /fx-model/pairs` → `fx/model/`
  - `GET /economic-growth` → `macro/economic_growth/economic_growth.py`
  - `GET /liquidity` → `macro/liquidity/liquidity.py`
  - `GET /country-dashboard` → `macro/country_dashboard/country_dashboard.py`
  - `GET /yield-curve` → `government_bonds/yield_curve.py`
  - `GET /central-banks` → `macro/central_banks/central_bank.py`
  - `GET /industry-monitor` → `macro/industry/industry_monitor.py`

## Module READMEs

Some modules have their own deeper docs:

- Frontend: `frontend/README.md`
- FX macro model: `fx/model/README.md`
- Government bonds: `government_bonds/README.md`
- Economic growth: `macro/economic_growth/README.md`
- Liquidity: `macro/liquidity/README.md`
- Breakouts: `macro/breakout/README.md`
- CFTC positioning: `macro/positioning/README.md`
- Momentum (price ROC): `portfolio/momentum/price_momentum/README.md`
- Market technicals: `equities/market_technicals/README.md`

## Troubleshooting

- `401 Not authenticated` in the UI:
  - Ensure `AUTH_PASSWORD_HASH` and `JWT_SECRET` are set in `.env`
  - Log in at `/login` (the cookie is HTTP-only; you won’t see it in JS)
- “Module not found” errors when running a script directly:
  - Run from the repo root, or use the documented command for that module (many scripts assume project-root imports)
- FRED errors:
  - Verify `FRED_API_KEY` is set (run `python3 load_env.py`)

## Security notes

- `.env` is ignored by git; don’t commit secrets
- In production, store secrets in Google Secret Manager. Set `API_PROXY_SECRET` and `REQUIRE_API_PROXY_SECRET=true` only when traffic reaches Cloud Run through an edge proxy that injects the matching `X-Api-Proxy-Secret` header.

## Deployment

The production stack runs on Google Cloud:

- **Cloud Run** — `talisman-api` (FastAPI) and a worker service running `python -m api.rq_worker`
- **Cloud SQL (Postgres + pgvector)** — application state
- **Memorystore for Valkey** — async job queue (RQ)
- **Cloud Storage** — generated documents (theses, overviews) and pre-migration backups
- **Cloud Run Jobs** — long-running batch work and the state migration entrypoint
- **Cloud Scheduler** — periodic cache warm + async-job sweep
- **Firebase Hosting** — serves the built frontend and rewrites `/api/**` to Cloud Run

See `infra/gcp/README.md` for the full runbook (env vars, service accounts, cutover checks).

## Disclaimer

This is a personal research toolkit. Nothing here is investment advice.
