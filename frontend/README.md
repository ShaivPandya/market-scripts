# Frontend (Market Analysis Dashboard UI)

React + TypeScript + Vite web UI for the Market Scripts project.

The UI is a thin client:
- It renders charts/tables
- It fetches all data from the backend via `/api/*` (see `frontend/src/lib/api.ts`)

For the full project overview, see the repo root `README.md`.

## Local development

### Prereqs

- Node 20+ (see `engines.node` in `frontend/package.json`)
- A running backend API (FastAPI) on `http://localhost:8000`:
  - `uvicorn api.main:app --reload --port 8000` (run from repo root)

### Install & run

```bash
npm install
npm run dev
```

Vite runs on `http://localhost:5173` and proxies `/api/*` to `http://localhost:8000` via `frontend/vite.config.ts`.

Run UI + API together (from `frontend/`):

```bash
npm run dev:all
```

## Auth modes (UI behavior)

The frontend supports two auth strategies:

### 1) Password mode (default)

- Set `VITE_AUTH_MODE=password` (or omit it; password is the default)
- The UI shows a login form that calls:
  - `POST /api/auth/login` (sets an HTTP-only `access_token` cookie)
  - `GET /api/auth/me` (session validation)
  - `POST /api/auth/logout` (clears cookie)
- The Axios client is created with `withCredentials: true` so cookies are sent on API calls.

Implementation:
- `frontend/src/contexts/AuthContext.tsx`
- `frontend/src/components/auth/ProtectedRoute.tsx`

Note: password sessions are intentionally “tab-scoped” — the UI uses `sessionStorage` as a simple “this tab is logged in” flag.

### 2) Cloudflare Access mode

- Build with `VITE_AUTH_MODE=cloudflare`
- The UI probes `GET /cdn-cgi/access/get-identity`:
  - If Access isn’t configured on that hostname, it falls back to password mode automatically.
  - If Access is configured, it treats `200 OK` as authenticated.
- Login/logout actions redirect to Cloudflare endpoints instead of calling `/api/auth/*`.

This mode is meant for deployments where Cloudflare Access gates the app at the edge, while the backend API remains protected by the API proxy secret (next section).

## Production API proxy (Cloudflare Pages Functions)

In production the UI expects `/api/*` to exist on the same origin as the frontend.

This repo includes a Cloudflare Pages Function proxy:
- `frontend/functions/api/[[path]].ts`

It forwards requests to a configured origin and injects an `X-Api-Proxy-Secret` header so the backend can reject direct requests.

Cloudflare Pages env vars (runtime for the Function):
- `API_ORIGIN` — base URL of the backend, e.g. `https://your-api.example.com`
- `API_PROXY_SECRET` — must match the backend’s `API_PROXY_SECRET`

Backend enforcement is implemented in `api/main.py`.

## Code organization

- `frontend/src/App.tsx` — route table (pages)
- `frontend/src/pages/*` — top-level screens (Portfolio, Liquidity, FX Model, etc.)
- `frontend/src/lib/api.ts` — API client + typed helpers (GET/POST wrappers)
- `frontend/src/components/*` — shared UI components (tables, charts, layout)
- `frontend/src/contexts/AuthContext.tsx` — auth state + mode detection

## Build

```bash
npm run build
```

Outputs static assets to `frontend/dist/`.
