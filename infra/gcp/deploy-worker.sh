#!/usr/bin/env bash
# Deprecated. Async work no longer uses an always-on Cloud Run worker pool.

set -euo pipefail

cat >&2 <<'EOF'
deploy-worker.sh is deprecated.

Async work now runs on demand through:
  ./infra/gcp/deploy-async-job.sh

Do not redeploy talisman-worker. After the new async path is active and no old
jobs remain, delete the legacy worker pool manually.
EOF
exit 2
