from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.async_job_runner import poll_registered_job

router = APIRouter()


@router.get("/document-generation/async/{job_id}")
def get_document_generation_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id") from None
