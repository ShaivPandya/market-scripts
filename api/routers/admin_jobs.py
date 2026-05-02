from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, status

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.routers.auth import require_auth

router = APIRouter()


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
    return enqueue_response(row, "/api/v1/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-async-job-sweep")
def enqueue_async_job_sweep(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "async_job_sweep",
        {"source": "scheduler"},
        cache_key="maintenance:async_job_sweep:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/v1/admin/jobs/{job_id}")


@router.post("/admin/jobs/enqueue-market-snapshot-refresh")
def enqueue_market_snapshot_refresh(_sub: str = Depends(require_scheduler_or_job_admin)):
    row, _disposition = enqueue_registered_job(
        "market_snapshot_refresh",
        {"source": "scheduler"},
        cache_key="maintenance:market_snapshot_refresh:v1",
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/v1/admin/jobs/{job_id}")


@router.get("/admin/jobs/{job_id}")
def get_admin_job(job_id: str, _sub: str = Depends(require_job_admin)):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904
