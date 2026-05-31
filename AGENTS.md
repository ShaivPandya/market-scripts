## Cursor Cloud specific instructions

Talisman is a **FastAPI + React (Vite)** app. Full local dev needs **PostgreSQL 16 with pgvector**, Python 3.12, and Node ≥20.19. Standard commands live in `README.md` and `frontend/package.json`.

### PostgreSQL (first-time VM)

Cloud VMs need Postgres installed and running before the API:

- `sudo apt-get install postgresql postgresql-16-pgvector`
- `sudo service postgresql start`
- Create a DB role matching the VM user (peer auth): `sudo -u postgres createuser -s $USER`
- `createdb talisman_dev`
- `DATABASE_URL=postgresql:///talisman_dev alembic upgrade head` (from repo root with `.venv` active)

Use the **unix-socket** URL `postgresql:///talisman_dev` in `.env` unless you configure `pg_hba` for passwordless TCP to `localhost`. TCP `postgresql://localhost/talisman_dev` often fails with `fe_sendauth: no password supplied` on fresh Ubuntu Postgres.

### `.env` and auth for local login

Copy `.env.example` → `.env`. For quick local login, use the pytest hash (password **`testpass`**) from `tests/conftest.py`:

- `AUTH_PASSWORD_HASH="$2b$12$43F.9axQmqL0Owf7Hsp4tub0wukaMzCmz8JlTz.UJD8emjTZUVy0C"` (quotes required so `$` is not mangled)
- `JWT_SECRET=test-secret-for-dev`
- `ENVIRONMENT=development`, `STATE_DB_BACKEND=postgres`, `DATABASE_URL=postgresql:///talisman_dev`, `STATE_STORAGE_BACKEND=local`, `ASYNC_JOB_BACKEND=local`

**Do not `source .env` in bash** before starting Uvicorn: unquoted `$` in the bcrypt hash gets expanded and breaks login (`Invalid salt`). Uvicorn loads `.env` via `load_dotenv()` in `api/main.py`. If you already sourced a broken `.env`, restart with `env -u AUTH_PASSWORD_HASH -u JWT_SECRET uvicorn api.main:app --reload --port 8000`.

### Running services (two processes)

Run API and UI in **separate** terminals/tmux sessions (one blocks the other in a single pane):

| Service | Command (repo root unless noted) |
|---------|----------------------------------|
| API | `source .venv/bin/activate && uvicorn api.main:app --reload --port 8000` |
| UI | `cd frontend && npm run dev` |

Health: `curl http://localhost:8000/api/health`. UI: http://localhost:5173 (Vite proxies `/api` to port 8000). Convenience: `cd frontend && npm run dev:all` (needs DB + `.env`).

### Lint / test (see CI)

- Python: `ruff check .`, `pytest` (uses in-memory SQLite; a few async-job tests may fail if `DATABASE_URL` is set to Postgres in the shell)
- Frontend: `cd frontend && npm run lint`, `npm run test:smoke` (Playwright mocks API; no Postgres required)

### Alembic note

Migration `20260503_0006` tries to create `schema_definitions` again (table already exists from `20260429_0001`). On a clean DB, `alembic upgrade head` may fail at that revision; stamp past it or apply the non-duplicate statements manually, then `alembic upgrade head` (documented here so future agents are not blocked).

### Hello-world check

Sign in at http://localhost:5173/login with **`testpass`**, confirm **Portfolio Dashboard**, then open **Workspace**. Empty portfolio / vendor API warnings are normal without real API keys; routing and auth prove the stack is up.
