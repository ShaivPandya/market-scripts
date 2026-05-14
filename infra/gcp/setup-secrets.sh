#!/usr/bin/env bash
# Populate Secret Manager and bind per-secret IAM. Run after bootstrap.sh.
#
# Behavior:
#   - Random secrets (JWT_SECRET, API_PROXY_SECRET, SCHEDULER_SECRET) are
#     generated locally and never displayed.
#   - Cloud SQL passwords are generated locally; the corresponding SQL user is
#     created (or its password reset), and the full DATABASE_URL_* string is
#     written to Secret Manager. The plaintext password is never echoed and
#     never lands on disk.
#   - User-provided secrets (ANTHROPIC_API_KEY, GEMINI_API_KEY, FRED_API_KEY,
#     AUTH_PASSWORD_HASH source, optional vendor tokens) are read silently.
#   - Each secret is created on first run and skipped on re-run; delete a
#     secret with `gcloud secrets delete` to force regeneration.
#   - IAM bindings are added with --condition=None and are no-ops on re-run.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_var PROJECT_ID
require_var REGION
require_var CLOUDSQL_INSTANCE
require_var GCS_STATE_BUCKET
require_var API_SA
require_var WORKER_SA
require_var MIGRATOR_SA

SQL_INSTANCE="${CLOUDSQL_INSTANCE##*:}"

log() { printf '\n[secrets] %s\n' "$*"; }

###############################################################################
# Helpers
###############################################################################
secret_exists() {
  gcloud secrets describe "$1" --project="${PROJECT_ID}" >/dev/null 2>&1
}

# Create the secret with the given value if it doesn't exist yet.
# Value is read from stdin so it never appears in argv (and never in shell history).
create_if_missing() {
  local name="$1"
  if secret_exists "${name}"; then
    echo "  ${name}: exists, leaving alone"
    cat >/dev/null   # discard piped value
    return
  fi
  gcloud secrets create "${name}" \
    --project="${PROJECT_ID}" \
    --replication-policy=automatic \
    --data-file=- >/dev/null
  echo "  ${name}: created"
}

# Add accessor binding for a service account on a secret. Idempotent.
bind_accessor() {
  local secret="$1" sa="$2"
  gcloud secrets add-iam-policy-binding "${secret}" \
    --project="${PROJECT_ID}" \
    --role=roles/secretmanager.secretAccessor \
    --member="serviceAccount:${sa}" \
    --condition=None >/dev/null
}

random_url_safe() { openssl rand -hex 32; }            # 64 hex chars, safe in URLs
random_token()    { openssl rand -base64 48 | tr -d '\n'; }

prompt_secret() {
  local label="$1" var
  read -rsp "${label}: " var
  echo >&2   # newline to stderr — must NOT mix with the captured stdout value
  printf '%s' "${var}"
}

prompt_optional() {
  local label="$1" var
  read -rsp "${label} (leave blank to skip): " var
  echo >&2   # see prompt_secret
  printf '%s' "${var}"
}

###############################################################################
# 1. Cloud SQL users + DATABASE_URL_* secrets
###############################################################################
# For each role, if the DATABASE_URL_* secret doesn't yet exist we generate a
# fresh password, ensure the SQL user exists with that password, and write the
# URL into Secret Manager. If the secret already exists we leave it alone.
create_db_url_secret() {
  local secret="$1" sql_user="$2"
  if secret_exists "${secret}"; then
    echo "  ${secret}: exists, leaving alone"
    return
  fi
  local pw
  pw="$(random_url_safe)"
  if gcloud sql users list --instance="${SQL_INSTANCE}" --project="${PROJECT_ID}" \
        --format='value(name)' | grep -qx "${sql_user}"; then
    gcloud sql users set-password "${sql_user}" \
      --instance="${SQL_INSTANCE}" --project="${PROJECT_ID}" \
      --password="${pw}" >/dev/null
  else
    gcloud sql users create "${sql_user}" \
      --instance="${SQL_INSTANCE}" --project="${PROJECT_ID}" \
      --password="${pw}" >/dev/null
  fi
  printf 'postgresql+psycopg://%s:%s@/talisman?host=/cloudsql/%s' \
    "${sql_user}" "${pw}" "${CLOUDSQL_INSTANCE}" \
    | create_if_missing "${secret}"
}

log "Cloud SQL users + DATABASE_URL_* secrets"
create_db_url_secret DATABASE_URL_API       talisman_app
create_db_url_secret DATABASE_URL_WORKER    talisman_worker
create_db_url_secret DATABASE_URL_MIGRATION talisman_migrator

###############################################################################
# 2. Random tokens
###############################################################################
log "Random tokens"
random_token | create_if_missing JWT_SECRET
random_token | create_if_missing API_PROXY_SECRET
random_token | create_if_missing SCHEDULER_SECRET

###############################################################################
# 3. AUTH_PASSWORD_HASH (bcrypt of an admin password the user picks)
###############################################################################
log "AUTH_PASSWORD_HASH"
if secret_exists AUTH_PASSWORD_HASH; then
  echo "  AUTH_PASSWORD_HASH: exists, leaving alone"
else
  if ! python3 -c 'import bcrypt' 2>/dev/null; then
    echo "  Python 'bcrypt' not installed locally. Run: pip install bcrypt" >&2
    echo "  Skipping AUTH_PASSWORD_HASH; re-run after installing." >&2
  else
    pw="$(prompt_secret 'Admin login password (will be bcrypted)')"
    HASH="$(python3 -c \
      'import bcrypt,sys; print(bcrypt.hashpw(sys.argv[1].encode(),bcrypt.gensalt(12)).decode())' \
      "${pw}")"
    unset pw
    printf '%s' "${HASH}" | create_if_missing AUTH_PASSWORD_HASH
  fi
fi

###############################################################################
# 4. User-provided API keys
###############################################################################
log "User-provided API keys"
if ! secret_exists ANTHROPIC_API_KEY; then
  v="$(prompt_secret 'ANTHROPIC_API_KEY (sk-ant-...)')"
  [[ "${v}" == sk-ant-* ]] || echo "  warning: value does not start with sk-ant-" >&2
  printf '%s' "${v}" | create_if_missing ANTHROPIC_API_KEY
  unset v
else
  echo "  ANTHROPIC_API_KEY: exists, leaving alone"
fi

if ! secret_exists GEMINI_API_KEY; then
  v="$(prompt_optional 'GEMINI_API_KEY (AIza...)')"
  if [[ -n "${v}" ]]; then
    [[ "${v}" == AIza* ]] || echo "  warning: value does not start with AIza" >&2
    printf '%s' "${v}" | create_if_missing GEMINI_API_KEY
  else
    echo "  GEMINI_API_KEY: skipped (remove from API_SECRETS/WORKER_SECRETS in config.sh until added)"
  fi
  unset v
else
  echo "  GEMINI_API_KEY: exists, leaving alone"
fi

if ! secret_exists FRED_API_KEY; then
  v="$(prompt_secret 'FRED_API_KEY')"
  printf '%s' "${v}" | create_if_missing FRED_API_KEY
  unset v
else
  echo "  FRED_API_KEY: exists, leaving alone"
fi

if ! secret_exists ESTAT_APP_ID; then
  v="$(prompt_optional 'ESTAT_APP_ID')"
  if [[ -n "${v}" ]]; then
    printf '%s' "${v}" | create_if_missing ESTAT_APP_ID
  else
    echo "  ESTAT_APP_ID: skipped (remove from API_SECRETS/WORKER_SECRETS in config.sh until added)"
  fi
  unset v
else
  echo "  ESTAT_APP_ID: exists, leaving alone"
fi

if ! secret_exists SODA_APP_TOKEN; then
  v="$(prompt_optional 'SODA_APP_TOKEN')"
  if [[ -n "${v}" ]]; then
    printf '%s' "${v}" | create_if_missing SODA_APP_TOKEN
  else
    echo "  SODA_APP_TOKEN: skipped (remove from API_SECRETS/WORKER_SECRETS in config.sh until added)"
  fi
  unset v
else
  echo "  SODA_APP_TOKEN: exists, leaving alone"
fi

if ! secret_exists EIA_API_KEY; then
  v="$(prompt_optional 'EIA_API_KEY')"
  if [[ -n "${v}" ]]; then
    printf '%s' "${v}" | create_if_missing EIA_API_KEY
  else
    echo "  EIA_API_KEY: skipped (remove from API_SECRETS/WORKER_SECRETS in config.sh until added)"
  fi
  unset v
else
  echo "  EIA_API_KEY: exists, leaving alone"
fi

###############################################################################
# 5. AUTH_SMOKE_PASSWORD + AUTH_SMOKE_PASSWORD_HASH (SHA-34)
###############################################################################
log "Smoke auth secrets"
if secret_exists AUTH_SMOKE_PASSWORD && secret_exists AUTH_SMOKE_PASSWORD_HASH; then
  echo "  AUTH_SMOKE_PASSWORD: exists, leaving alone"
  echo "  AUTH_SMOKE_PASSWORD_HASH: exists, leaving alone"
else
  if ! python3 -c 'import bcrypt' 2>/dev/null; then
    echo "  Python 'bcrypt' not installed locally. Run: pip install bcrypt" >&2
    echo "  Skipping smoke auth secrets; re-run after installing." >&2
  else
    smoke_pw="$(random_url_safe)"
    SMOKE_HASH="$(python3 -c \
      'import bcrypt,sys; print(bcrypt.hashpw(sys.argv[1].encode(),bcrypt.gensalt(12)).decode())' \
      "${smoke_pw}")"
    printf '%s' "${smoke_pw}" | create_if_missing AUTH_SMOKE_PASSWORD
    printf '%s' "${SMOKE_HASH}" | create_if_missing AUTH_SMOKE_PASSWORD_HASH
    unset smoke_pw SMOKE_HASH
  fi
fi

###############################################################################
# 6. IAM bindings — least-privilege per service account
###############################################################################
API_ALLOWED=(
  DATABASE_URL_API AUTH_PASSWORD_HASH AUTH_SMOKE_PASSWORD_HASH JWT_SECRET API_PROXY_SECRET
  SCHEDULER_SECRET ANTHROPIC_API_KEY GEMINI_API_KEY FRED_API_KEY
  ESTAT_APP_ID SODA_APP_TOKEN EIA_API_KEY
)
WORKER_ALLOWED=(
  DATABASE_URL_WORKER ANTHROPIC_API_KEY GEMINI_API_KEY FRED_API_KEY
  ESTAT_APP_ID SODA_APP_TOKEN EIA_API_KEY
)
MIGRATOR_ALLOWED=( DATABASE_URL_MIGRATION )

log "IAM bindings: api-sa"
for s in "${API_ALLOWED[@]}"; do
  if secret_exists "${s}"; then
    bind_accessor "${s}" "${API_SA}"
    echo "  ${s} -> ${API_SA}"
  fi
done

log "IAM bindings: worker-sa"
for s in "${WORKER_ALLOWED[@]}"; do
  if secret_exists "${s}"; then
    bind_accessor "${s}" "${WORKER_SA}"
    echo "  ${s} -> ${WORKER_SA}"
  fi
done

log "IAM bindings: migrator-sa"
for s in "${MIGRATOR_ALLOWED[@]}"; do
  if secret_exists "${s}"; then
    bind_accessor "${s}" "${MIGRATOR_SA}"
    echo "  ${s} -> ${MIGRATOR_SA}"
  fi
done

log "Done."
echo "To rotate any value, delete the secret and re-run, e.g.:"
echo "  gcloud secrets delete JWT_SECRET --project=${PROJECT_ID}"
