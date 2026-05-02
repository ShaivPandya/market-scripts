"""Helpers for dispatching long-running work to Cloud Run Jobs."""

from __future__ import annotations

import os

from api.job_queue import cloud_run_job_name, set_cloud_run_job_name


class CloudRunJobConfigError(RuntimeError):
    """Raised when Cloud Run Job dispatch is requested but not configured."""


def cloud_run_jobs_enabled() -> bool:
    value = os.getenv("CLOUD_RUN_JOBS_ENABLED", "").strip().lower()
    if value:
        return value in ("1", "true", "yes")
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def _project_id() -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or ""
    if not project:
        raise CloudRunJobConfigError("GOOGLE_CLOUD_PROJECT or GCP_PROJECT is required for Cloud Run Job dispatch.")
    return project


def _region() -> str:
    region = os.getenv("CLOUD_RUN_REGION") or os.getenv("GCP_REGION") or ""
    if not region:
        raise CloudRunJobConfigError("CLOUD_RUN_REGION or GCP_REGION is required for Cloud Run Job dispatch.")
    return region


def dispatch_cloud_run_job(job_type: str, job_id: str) -> str:
    """Invoke the generic async Cloud Run Job for an existing async_jobs row."""
    job_name = cloud_run_job_name(job_type)
    set_cloud_run_job_name(job_id, job_name)

    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
    except ImportError as exc:
        raise CloudRunJobConfigError("google-auth is required for Cloud Run Job dispatch.") from exc

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = f"https://run.googleapis.com/v2/projects/{_project_id()}/locations/{_region()}/jobs/{job_name}:run"
    response = session.post(
        url,
        json={
            "overrides": {
                "containerOverrides": [
                    {
                        "env": [
                            {"name": "ASYNC_JOB_ID", "value": job_id},
                            {"name": "ASYNC_JOB_TYPE", "value": job_type},
                        ]
                    }
                ]
            }
        },
        timeout=30,
    )
    response.raise_for_status()
    return job_name
