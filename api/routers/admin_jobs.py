from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.routers.auth import require_auth

router = APIRouter()


class GovernanceOutboxRequeueRequest(BaseModel):
    idempotency_key: str | None = None
    next_attempt_at: str | None = None


def require_scheduler_or_job_admin(
    access_token: str | None = Cookie(default=None, alias="__session"),
    scheduler_secret: str | None = Header(default=None, alias="X-Scheduler-Secret"),
) -> str:
    expected = (os.getenv("SCHEDULER_SECRET") or "").strip()
    if expected and scheduler_secret and hmac.compare_digest(scheduler_secret, expected):
        return "scheduler"
    try:
        return require_auth(access_token)
    except HTTPException:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")  # noqa: B904


def require_job_admin(
    access_token: str | None = Cookie(default=None, alias="__session"),
) -> str:
    try:
        return require_auth(access_token)
    except HTTPException:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")  # noqa: B904


@router.post("/admin/jobs/enqueue-cache-warm")
def enqueue_cache_warm(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "cache_warm",
        {"source": "scheduler"},
        cache_key="maintenance:cache_warm:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-async-job-sweep")
def enqueue_async_job_sweep(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "async_job_sweep",
        {"source": "scheduler"},
        cache_key="maintenance:async_job_sweep:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-governance-outbox-drain")
def enqueue_governance_outbox_drain(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "governance_outbox_drain",
        {"source": "scheduler"},
        cache_key="maintenance:governance_outbox_drain:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.get("/admin/governance-outbox")
def list_governance_outbox(
    status_filter: str | None = None,
    lineage_root_id: str | None = None,
    limit: int = 100,
    _sub: str = Depends(require_job_admin),
):
    return {
        "items": [],
        "count": 0,
        "metrics": {"pending": 0, "retry": 0, "failed": 0},
        "lineage_state": "ontology",
        "message": "Governance outbox has been removed from runtime.",
    }


@router.post("/admin/governance-outbox/{outbox_id}/requeue")
def requeue_governance_outbox(
    outbox_id: int,
    body: GovernanceOutboxRequeueRequest | None = None,
    _sub: str = Depends(require_job_admin),
):
    raise HTTPException(status_code=410, detail="Governance outbox has been removed from runtime.")


@router.post("/admin/jobs/enqueue-market-snapshot-refresh")
def enqueue_market_snapshot_refresh(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "market_snapshot_refresh",
        {"source": "scheduler"},
        cache_key="maintenance:market_snapshot_refresh:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-macro-snapshot-refresh")
def enqueue_macro_snapshot_refresh(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "macro_snapshot_refresh",
        {"source": "scheduler"},
        cache_key="maintenance:macro_snapshot_refresh:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-watch-trigger-monitor")
def enqueue_watch_trigger_monitor(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "watch_trigger_monitor",
        {"source": "scheduler"},
        cache_key="maintenance:watch_trigger_monitor:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-continuous-optimizer")
def enqueue_continuous_optimizer(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "continuous_optimizer",
        {"source": "scheduler"},
        cache_key="maintenance:continuous_optimizer:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")


@router.get("/admin/jobs/{job_id}")
def get_admin_job(job_id: str, _sub: str = Depends(require_job_admin)):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904


# ---------------------------------------------------------------------------
# SHA-34: Deploy smoke endpoint
# ---------------------------------------------------------------------------


def _check_postgres() -> tuple[bool, str]:
    """Verify Postgres connectivity."""
    try:
        from api.postgres import connect

        with connect() as conn:
            conn.execute("SELECT 1")
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _check_migration_head() -> tuple[bool, str]:
    """Compare deployed TALISMAN_RELEASE_MIGRATION_HEAD with actual DB head."""
    expected = (os.environ.get("TALISMAN_RELEASE_MIGRATION_HEAD") or "").strip()
    if not expected:
        return True, "not_configured"
    try:
        from api.postgres import connect

        with connect() as conn:
            row = conn.execute("SELECT version_num FROM alembic_version LIMIT 1").fetchone()
            if row is None:
                return False, "no alembic_version row"
            actual = row["version_num"] if isinstance(row, dict) else row[0]
            if actual == expected:
                return True, f"head={actual}"
            return False, f"mismatch: deployed={expected} db={actual}"
    except Exception as exc:
        return False, str(exc)


def _check_read_model() -> tuple[bool, str]:
    """Verify the temporal read model (ontology) is queryable."""
    try:
        from ontology.runtime_read_service import OntologyRuntimeReadService

        reads = OntologyRuntimeReadService()
        bundle = reads.workspace_bundle()
        if isinstance(bundle, dict):
            return True, "ok"
        return False, "unexpected bundle type"
    except Exception as exc:
        return False, str(exc)


def _check_action_approval_safety() -> tuple[bool, str]:
    """Light invariant check on action/approval registry availability."""
    try:
        from ontology.command_service import OntologyCommandService
        from ontology.policy import admin_actor

        service = OntologyCommandService()
        actor = admin_actor(source="deploy-smoke")
        approvals = service.list_approvals(status="pending", actor=actor)
        if not isinstance(approvals, list):
            return False, "approvals list not a list"
        return True, f"pending_count={len(approvals)}"
    except Exception as exc:
        return False, str(exc)


@router.get("/admin/deploy-smoke")
def deploy_smoke(_sub: str = Depends(require_job_admin)):
    """Run backend invariant checks for post-deploy / post-rollback smoke tests."""
    checks: dict[str, dict[str, object]] = {}
    failed_checks: list[str] = []

    for name, fn in [
        ("postgres", _check_postgres),
        ("migration_head", _check_migration_head),
        ("read_model", _check_read_model),
        ("action_approval_safety", _check_action_approval_safety),
    ]:
        passed, detail = fn()
        checks[name] = {"passed": passed, "detail": detail}
        if not passed:
            failed_checks.append(name)

    # Include release metadata for debugging
    migration_head = (os.environ.get("TALISMAN_RELEASE_MIGRATION_HEAD") or "").strip()
    release_info: dict[str, str] = {}
    if migration_head:
        release_info["migration_head"] = migration_head
    image_tag = (os.environ.get("TALISMAN_RELEASE_IMAGE_TAG") or "").strip()
    if image_tag:
        release_info["image_tag"] = image_tag

    body: dict[str, object] = {"checks": checks}
    if release_info:
        body["release"] = release_info

    if failed_checks:
        body["failed_checks"] = failed_checks
        return JSONResponse(body, status_code=503)

    return body
