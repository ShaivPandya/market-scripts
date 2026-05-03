"""Authenticated report artifact sync endpoint for GitHub Actions."""

from __future__ import annotations

import hmac
import os
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from api.report_sync import persist_report_sync

router = APIRouter()


class ReportSyncPayload(BaseModel):
    report_id: str | None = None
    as_of: str | None = None
    report_md: str | None = None
    commentary_md: str | None = None
    recommendations_md: str | None = None
    recommendations: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    bundle: dict[str, Any] = Field(default_factory=dict)
    thesis_claims: list[dict[str, Any]] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    report_hash: str | None = None
    input_hash: str | None = None


def require_report_sync_secret(
    report_sync_secret: str | None = Header(default=None, alias="X-Report-Sync-Secret"),
) -> None:
    expected = (os.getenv("REPORT_SYNC_SECRET") or "").strip()
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Report sync is not configured")
    if not report_sync_secret or not hmac.compare_digest(report_sync_secret, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid report sync secret")


@router.post("/report-sync/{report_type}")
def sync_report(
    report_type: Literal["daily", "weekly"],
    body: ReportSyncPayload,
    _auth: str | None = Header(default=None, alias="X-Report-Sync-Secret"),
):
    require_report_sync_secret(_auth)
    try:
        return persist_report_sync(report_type, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
