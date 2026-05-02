from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from api.exceptions import AppError, NotFoundError, ValidationError
from api.request_limits import read_upload_file_bytes
from api.serializers import serialize_response

router = APIRouter()
log = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
_MARKDOWN_CONTENT_TYPES = {"text/markdown", "text/x-markdown"}


def _is_markdown_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    filename = (file.filename or "").lower()
    return filename.endswith(".md") or content_type in _MARKDOWN_CONTENT_TYPES


def _decode_markdown_upload(payload: bytes) -> str:
    try:
        content = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("Markdown file must be UTF-8 encoded.") from exc
    if not content.strip():
        raise ValidationError("Markdown file is empty.")
    return content


def _index_digest_best_effort(detail: dict) -> None:
    digest_id = str(detail.get("id") or "").strip()
    content = str(detail.get("content") or "")
    if not digest_id or not content.strip():
        return
    try:
        from api.retrieval import index_document

        index_document(
            doc_type="news_digest",
            content=content,
            ticker=None,
            source_path=f"news_digests/{digest_id}.md",
            doc_id=f"news_digest-{digest_id}",
        )
    except Exception:
        log.debug("Failed to index news digest %s", digest_id, exc_info=True)


def _delete_digest_index_best_effort(digest_id: str) -> None:
    try:
        from api.retrieval import delete_document

        delete_document(f"news_digest-{digest_id}")
    except Exception:
        log.debug("Failed to delete news digest index %s", digest_id, exc_info=True)


@router.get("/portfolio-news")
def list_portfolio_news(refresh: bool = False):
    """List uploaded news digests and parsed story metadata."""
    del refresh
    from portfolio.news_digests import list_digests

    return serialize_response(list_digests())


@router.post("/portfolio-news")
async def upload_portfolio_news_digest(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    """Upload a user-curated markdown digest."""
    if not _is_markdown_upload(file):
        raise ValidationError("File must be a Markdown (.md) file.")

    payload = await read_upload_file_bytes(file, limit_bytes=MAX_UPLOAD_SIZE_BYTES, limit_label="10 MiB")
    if not payload:
        raise ValidationError("Uploaded file is empty.")

    content = _decode_markdown_upload(payload)
    filename = Path(file.filename or "digest.md").name

    try:
        from portfolio.news_digests import save_digest

        detail = save_digest(content, filename=filename)
    except Exception as exc:
        raise AppError(f"Failed to store news digest: {exc}") from exc

    _index_digest_best_effort(detail)
    return serialize_response({"status": "ok", "digest": detail})


@router.get("/portfolio-news/{digest_id}")
def get_portfolio_news_digest(digest_id: str):
    from portfolio.news_digests import get_digest, validate_digest_id

    try:
        return serialize_response(get_digest(validate_digest_id(digest_id)))
    except ValueError as exc:
        raise ValidationError("Invalid news digest id.") from exc
    except FileNotFoundError as exc:
        raise NotFoundError("news digest", digest_id) from exc


@router.delete("/portfolio-news/{digest_id}")
def delete_portfolio_news_digest(digest_id: str):
    from portfolio.news_digests import delete_digest, validate_digest_id

    try:
        normalized_digest_id = validate_digest_id(digest_id)
    except ValueError as exc:
        raise ValidationError("Invalid news digest id.") from exc

    deleted = delete_digest(normalized_digest_id)
    if not deleted:
        raise NotFoundError("news digest", normalized_digest_id)
    _delete_digest_index_best_effort(normalized_digest_id)
    return {"status": "ok", "deleted": True, "id": normalized_digest_id}
