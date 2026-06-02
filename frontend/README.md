# Talisman Frontend

React + TypeScript + Vite web UI for Talisman.

The UI is a thin client:
- It renders charts/tables
- It fetches all data from the backend via `/api/*` (see `frontend/src/lib/api.ts`)

For the full project overview, see the repo root `README.md`.

## Local development

### Prereqs

- Node 20.19 or newer (see `engines.node` in `frontend/package.json`)
- A running backend API (FastAPI) on `http://localhost:8000`, started from the repo root:

```bash
ENVIRONMENT=development \
STATE_DB_BACKEND=postgres \
DATABASE_URL=postgresql://localhost/talisman_dev \
STATE_STORAGE_BACKEND=local \
ASYNC_JOB_BACKEND=local \
uvicorn api.main:app --reload --port 8000
```

### Install & run

```bash
npm install
npm run dev
```

Vite runs on `http://localhost:5173` and proxies `/api/*` to `http://localhost:8000` via `frontend/vite.config.ts`.

The Axios API client defaults to `VITE_API_BASE_URL=/api`; set
`VITE_API_BASE_URL` only when the API should be called at another origin or
base path.

Run UI + API together (from `frontend/`):

```bash
npm run dev:all
```

## Auth modes (UI behavior)

The frontend supports two auth strategies:

### 1) Password mode (default)

- Set `VITE_AUTH_MODE=password` (or omit it; password is the default)
- The UI shows a login form that calls:
  - `POST /api/auth/login` (sets HTTP-only `__session` cookie; returns `csrfToken`)
  - `GET /api/auth/me` (session validation; refreshes `csrfToken`)
  - `POST /api/auth/logout` (revokes session and clears cookie)
- The Axios client uses `withCredentials: true` and sends `X-CSRF-Token` on mutating requests.

Implementation:
- `frontend/src/contexts/AuthContext.tsx`
- `frontend/src/components/auth/ProtectedRoute.tsx`

Note: password sessions are server-side and shared across tabs; the UI calls `/api/auth/me` on load to confirm the cookie is still valid.

### 2) Cloudflare Access mode

- Build with `VITE_AUTH_MODE=cloudflare`
- The UI probes `GET /cdn-cgi/access/get-identity`:
  - If Access isn’t configured on that hostname, it falls back to password mode automatically.
  - If Access is configured, it treats `200 OK` as authenticated.
- Login/logout actions redirect to Cloudflare endpoints instead of calling `/api/auth/*`.

This mode is meant for deployments where Cloudflare Access gates the app at the edge and a trusted API proxy can inject `X-Api-Proxy-Secret`. The checked-in Firebase Hosting rewrite cannot inject that header, so do not combine `AUTH_MODE=cloudflare` with plain Firebase rewrites unless another proxy layer adds the secret.

## Observability (optional Sentry)

Frontend Sentry is opt-in via build-time env vars. When `VITE_SENTRY_DSN` is unset, `npm run dev` and `npm run build` behave as before.

Suggested production build vars:

```bash
VITE_SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
VITE_SENTRY_ENVIRONMENT=production
VITE_SENTRY_RELEASE=<git-sha>
VITE_SENTRY_TRACES_SAMPLE_RATE=0.05
```

Route render errors are captured from `RouteErrorBoundary`. Chunk-load recovery reloads are excluded. Auth headers, CSRF tokens, API bodies, portfolio data, and agent prompts are scrubbed before events leave the browser.

## Production hosting

In production the UI expects `/api/*` to exist on the same origin as the frontend.

This repo deploys the built Vite app with Firebase Hosting:
- `firebase.json` serves `frontend/dist`
- `/api/**` is rewritten to the Cloud Run service `talisman-api`
- Application routes are rewritten to `/index.html`

Firebase Hosting rewrites do not add custom headers. Backend proxy-secret
enforcement in `api/main.py` is therefore enabled only for Cloudflare auth mode
or when `REQUIRE_API_PROXY_SECRET=true`.

## Code organization

- `frontend/src/App.tsx` — route table (pages)
- `frontend/src/pages/*` — top-level screens (Portfolio, Liquidity, FX Model, etc.)
- `frontend/src/lib/api.ts` — API client + typed helpers (GET/POST wrappers)
- `frontend/src/components/*` — shared UI components (tables, charts, layout)
- `frontend/src/contexts/AuthContext.tsx` — auth state + mode detection
- `frontend/src/hooks/*` — React Query and feature-specific data hooks

## Build

```bash
npm run build
```

Outputs static assets to `frontend/dist/`.
