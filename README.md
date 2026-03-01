# Market Scripts (Market Analysis Dashboard)

A collection of Python “market dashboards” (macro, equities, FX, commodities, portfolio analytics) that can be run in three ways:

1. As standalone Python scripts (terminal output, CSV/PNG outputs for some modules)
2. Via a Streamlit GUI (`gui/app.py`)
3. Via a FastAPI backend (`api/`) + React frontend (`frontend/`) web dashboard

This repo is intentionally “flat”: most analysis modules are plain Python files inside topical folders, and the GUI/API layers import them by adding those folders to `sys.path`.

## Quickstart (local development)

### Prereqs

- Python **3.11+** (the Docker image uses Python 3.12)
- Node **20+** (see `frontend/package.json` engines)

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

Optional extras (needed for some web pages / endpoints):

```bash
# FX Model (installs statsmodels + pandaSDMX)
pip install -r fx/model/requirements.txt

# Portfolio Optimizer (convex optimization)
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
- `OPENAI_API_KEY` (central bank + industry transcript summarization)
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

### 5) (Optional) Run the Streamlit GUI

```bash
streamlit run gui/app.py
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

**Streamlit UI (`gui/`)**
- Directly imports the analysis modules and renders them.

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
- The repo includes a Cloudflare Pages Function proxy at `frontend/functions/api/[[path]].ts`.
- That proxy forwards `/api/*` to your backend origin and injects an `X-Api-Proxy-Secret` header.
- The backend can enforce this by setting `API_PROXY_SECRET` (see `api/main.py` middleware and `.env.example`).

## Repository structure (high level)

- `api/` — FastAPI server (`api/main.py`) + route adapters in `api/routers/`
- `frontend/` — React + TypeScript + Vite UI, plus Cloudflare Pages Functions under `frontend/functions/`
- `gui/` — Streamlit dashboard (`gui/app.py`)
- `macro/` — macro monitors (liquidity, country dashboard, positioning, breakouts, central banks, industry transcripts)
- `equities/` — equity screens + dashboards (index dashboard, market technicals, quality, sector metrics, short screen, universes)
- `portfolio/` — portfolio dashboard, optimizer, momentum modules, technical analysis chart module
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
  - `POST /portfolio-optimizer` → `portfolio/portfolio_optimizer/portfolio_optimizer.py`
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
- Prefer production deployments behind:
  - Cloudflare Access (optional) and/or
  - `API_PROXY_SECRET` + the Pages function proxy

## Disclaimer

This is a personal research toolkit. Nothing here is investment advice.
