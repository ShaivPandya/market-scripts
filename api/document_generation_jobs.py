"""Async document upload generation helpers."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field

from api.async_job_runner import enqueue_registered_job, enqueue_response
from api.exceptions import AppError, DataFetchError, ValidationError
from api.request_limits import read_upload_file_bytes
from api.state_storage import delete_file, read_bytes, write_bytes
from paths import PROJECT_ROOT

logger = logging.getLogger("api.document_generation_jobs")

DocumentGenerationKind = Literal["thesis", "overview", "management_quality"]
UploadDocumentType = Literal["pdf", "markdown"]

MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024
TEMP_UPLOAD_PREFIX = "tmp/document_generation_uploads"
TEMP_UPLOAD_DIR = PROJECT_ROOT / TEMP_UPLOAD_PREFIX
POLL_PATH = "/api/v1/document-generation/async/{job_id}"
_MARKDOWN_CONTENT_TYPES = {"text/markdown", "text/x-markdown"}


class DocumentGenerationJobRequest(BaseModel):
    kind: DocumentGenerationKind
    ticker: str = Field(..., min_length=1)
    filename: str = ""
    content_type: str = ""
    storage_key: str = Field(..., min_length=1)


def classify_upload_document(upload_bytes: bytes, *, content_type: str, filename: str) -> UploadDocumentType:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    filename = (filename or "").lower()
    has_pdf_type = content_type == "application/pdf" or filename.endswith(".pdf")
    has_pdf_signature = upload_bytes.startswith(b"%PDF-")
    has_markdown_type = content_type in _MARKDOWN_CONTENT_TYPES or filename.endswith(".md")

    if has_pdf_type or has_pdf_signature:
        if not (has_pdf_type and has_pdf_signature):
            raise ValidationError("File must be a valid PDF or Markdown (.md) file.")
        return "pdf"
    if has_markdown_type:
        return "markdown"
    raise ValidationError("File must be a valid PDF or Markdown (.md) file.")


async def enqueue_document_generation_upload(
    *,
    kind: DocumentGenerationKind,
    ticker: str,
    file: UploadFile,
    max_upload_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> object:
    upload_bytes = await read_upload_file_bytes(file, limit_bytes=max_upload_size_bytes, limit_label="30 MiB")
    if not upload_bytes:
        raise ValidationError("Uploaded file is empty.")

    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    filename = file.filename or ""
    classify_upload_document(upload_bytes, content_type=content_type, filename=filename)

    upload_id = uuid.uuid4().hex
    storage_key = f"{TEMP_UPLOAD_PREFIX}/{upload_id}.bin"
    local_path = _local_path_for_storage_key(storage_key)
    try:
        write_bytes(
            local_path,
            storage_key,
            upload_bytes,
            content_type=content_type or "application/octet-stream",
            metadata={
                "kind": kind,
                "ticker": ticker,
                "filename": filename,
                "content_type": content_type,
            },
        )
        payload = DocumentGenerationJobRequest(
            kind=kind,
            ticker=ticker,
            filename=filename,
            content_type=content_type,
            storage_key=storage_key,
        ).model_dump()
        row, _disposition = enqueue_registered_job(
            "document_generation",
            payload,
            cache_key=None,
            reuse_completed=False,
        )
    except Exception:
        _delete_temp_upload(storage_key)
        raise
    return enqueue_response(row, POLL_PATH)


def run_document_generation_job(req: DocumentGenerationJobRequest) -> dict:
    try:
        upload_bytes = read_bytes(_local_path_for_storage_key(req.storage_key), req.storage_key)
        if req.kind == "thesis":
            result = _run_thesis_generation(req, upload_bytes)
        elif req.kind == "overview":
            result = _run_overview_generation(req, upload_bytes)
        elif req.kind == "management_quality":
            result = _run_management_quality_generation(req, upload_bytes)
        else:
            raise ValidationError(f"Unsupported document generation kind: {req.kind}")
        return result
    except Exception as exc:
        raise RuntimeError(_job_error_message(req.kind, exc)) from exc
    finally:
        _delete_temp_upload(req.storage_key)


def _run_thesis_generation(req: DocumentGenerationJobRequest, upload_bytes: bytes) -> dict:
    from api.routers import thesis

    return thesis.generate_thesis_from_upload_bytes(
        req.ticker,
        upload_bytes,
        content_type=req.content_type,
        filename=req.filename,
    )


def _run_overview_generation(req: DocumentGenerationJobRequest, upload_bytes: bytes) -> dict:
    from api.routers import overview

    return overview.generate_overview_from_upload_bytes(
        req.ticker,
        upload_bytes,
        content_type=req.content_type,
        filename=req.filename,
    )


def _run_management_quality_generation(req: DocumentGenerationJobRequest, upload_bytes: bytes) -> dict:
    from api.routers import management_quality

    return management_quality.generate_management_quality_from_upload_bytes(
        req.ticker,
        upload_bytes,
        content_type=req.content_type,
        filename=req.filename,
    )


def _local_path_for_storage_key(storage_key: str) -> Path:
    normalized = storage_key.strip().lstrip("/")
    expected = f"{TEMP_UPLOAD_PREFIX}/"
    if not normalized.startswith(expected) or "/" in normalized.removeprefix(expected):
        raise ValidationError("Invalid document generation upload storage key.")
    return PROJECT_ROOT / normalized


def _delete_temp_upload(storage_key: str) -> None:
    try:
        delete_file(_local_path_for_storage_key(storage_key), storage_key)
    except Exception:
        logger.warning("failed to delete temporary document generation upload", exc_info=True)


def _job_error_message(kind: str, exc: Exception) -> str:
    if isinstance(exc, DataFetchError):
        detail = str(exc.detail or "").strip()
        if detail:
            return detail
        return exc.message
    if isinstance(exc, AppError):
        return exc.message
    return str(exc) or f"{kind.replace('_', ' ').title()} generation failed"
