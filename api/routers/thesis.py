from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from api.action_execution import stage_api_action
from api.document_generation_jobs import classify_upload_document, enqueue_document_generation_upload
from api.exceptions import DataFetchError, NotFoundError, ValidationError
from api.routers.portfolio_edit import _TICKER_RE
from llm_utils import MODEL_MID, call_llm_pdf_text
from paths import PROJECT_ROOT
from portfolio import thesis_content
from portfolio.action_registry import (
    ActionNotFoundError,
    ActionValidationError,
)

router = APIRouter()

THESES_DIR = PROJECT_ROOT / "investment_theses"
THESES_GCS_PREFIX = "live/theses"
MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024

_REQ_SECTIONS = ("## Thesis", "## Key Catalysts", "## Risk Factors")

_SYSTEM_PROMPT = """You are a buy-side investment analyst.
Generate a concise investment thesis in markdown.

Output only markdown and follow this structure exactly:
# {ticker}
## Thesis
## Key Catalysts
## Risk Factors

Use short bullets under each section.
Focus on company-specific and financially material points.
Do not include disclaimers or any text outside the markdown."""

_USER_PROMPT = """Use the attached PDF to write the investment thesis markdown.
If critical data is missing, make conservative assumptions and state uncertainty briefly in Risk Factors.
Keep it concise and decision-useful."""


def _normalize_ticker(raw_ticker: str) -> str:
    return raw_ticker.strip().upper()


def _validate_ticker(ticker: str) -> None:
    if not ticker:
        raise ValidationError("Ticker cannot be empty.")
    if not _TICKER_RE.match(ticker):
        raise ValidationError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")


def _thesis_path(ticker: str) -> Path:
    return THESES_DIR / f"{ticker}.md"


def _thesis_gcs_key(ticker: str) -> str:
    return f"{THESES_GCS_PREFIX}/{ticker}.md"


def _configure_thesis_content_storage() -> None:
    thesis_content.THESES_DIR = THESES_DIR
    thesis_content.THESES_GCS_PREFIX = THESES_GCS_PREFIX


def _thesis_exists(ticker: str) -> bool:
    _configure_thesis_content_storage()
    return thesis_content.thesis_exists(ticker)


def _read_thesis(ticker: str) -> str:
    _configure_thesis_content_storage()
    return thesis_content.read_thesis(ticker)


def _write_thesis(ticker: str, content: str) -> str:
    _configure_thesis_content_storage()
    return thesis_content.write_thesis(ticker, content)


def _strip_outer_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _normalize_output_markdown(ticker: str, content: str) -> str:
    cleaned = _strip_outer_markdown_fence(content)
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {ticker}"
        cleaned = "\n".join(lines).strip()
    elif cleaned:
        cleaned = f"# {ticker}\n\n{cleaned}"
    else:
        cleaned = f"# {ticker}"

    for section in _REQ_SECTIONS:
        if section not in cleaned:
            cleaned += f"\n\n{section}\n- TBD"
    return cleaned.strip() + "\n"


def _decode_markdown_upload(markdown_bytes: bytes) -> str:
    try:
        content = markdown_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValidationError("Markdown file must be UTF-8 encoded.") from e
    if not content.strip():
        raise ValidationError("Markdown file is empty.")
    return content


def _llm_error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        message = body.get("message")
        if isinstance(message, str):
            return message

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return message
            message = data.get("message")
            if isinstance(message, str):
                return message

    text = str(exc)
    match = re.search(r"'message': '([^']+)'", text)
    if match:
        return match.group(1)
    return text


def _call_llm_pdf(*, ticker: str, pdf_bytes: bytes) -> str:
    generated, _citations, _response = call_llm_pdf_text(
        pdf_bytes=pdf_bytes,
        prompt=_USER_PROMPT,
        model=MODEL_MID,
        api_key=None,
        max_tokens=4096,
        system=_SYSTEM_PROMPT.format(ticker=ticker),
        filename=f"{ticker}.pdf",
    )
    if not generated:
        raise DataFetchError(source="llm", detail="LLM returned empty thesis output.")
    return _normalize_output_markdown(ticker, generated)


def generate_thesis_from_upload_bytes(
    ticker: str,
    upload_bytes: bytes,
    *,
    content_type: str,
    filename: str,
) -> dict:
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)
    if not upload_bytes:
        raise ValidationError("Uploaded file is empty.")

    upload_type = classify_upload_document(upload_bytes, content_type=content_type, filename=filename)
    if upload_type == "pdf":
        try:
            content = _call_llm_pdf(ticker=normalized_ticker, pdf_bytes=upload_bytes)
        except (ValidationError, DataFetchError):
            raise
        except Exception as e:
            raise DataFetchError(
                source="llm",
                detail=f"Failed to generate thesis: {_llm_error_message(e)}",
            ) from e
    else:
        content = _normalize_output_markdown(normalized_ticker, _decode_markdown_upload(upload_bytes))

    _configure_thesis_content_storage()
    return stage_api_action(
        "save_thesis_content",
        {"ticker": normalized_ticker, "content": content, "preserve_exact_content": True},
        source_id="thesis.generate_thesis",
        reason=f"Generate thesis for {normalized_ticker} from uploaded document",
    )


@router.post("/thesis/generate")
async def generate_thesis(
    ticker: str = Form(...),  # noqa: B008 - FastAPI parameter declaration
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)
    return await enqueue_document_generation_upload(
        kind="thesis",
        ticker=normalized_ticker,
        file=file,
        max_upload_size_bytes=MAX_UPLOAD_SIZE_BYTES,
    )


@router.get("/thesis/meta")
def get_thesis_meta_all():
    from portfolio.portfolio_db import get_positions
    from portfolio.thesis_db import get_all_thesis_meta, get_latest_evaluations

    positions = {}
    for row in get_positions():
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if ticker and ticker not in positions:
            positions[ticker] = row

    meta_by_ticker = {}
    for row in get_all_thesis_meta():
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if ticker in positions and ticker not in meta_by_ticker:
            meta_by_ticker[ticker] = dict(row)

    meta = []
    for ticker in sorted(positions):
        row = dict(
            meta_by_ticker.get(
                ticker,
                {
                    "ticker": ticker,
                    "status": "missing",
                    "created_at": None,
                    "updated_at": None,
                },
            )
        )
        row["ticker"] = ticker
        pos = positions[ticker]
        row["direction"] = pos.get("direction")
        row["asset"] = pos.get("asset")
        row["conviction"] = pos.get("conviction")
        meta.append(row)

    latest = {_normalize_ticker(str(e.get("ticker", ""))): e for e in get_latest_evaluations()}
    for m in meta:
        ticker = _normalize_ticker(str(m["ticker"]))
        ev = latest.get(ticker)
        if ev:
            ev = dict(ev)
            ev["ticker"] = ticker
        m["latest_evaluation"] = ev
        m["last_evaluated"] = ev.get("evaluated_at") if ev else None
    return meta


@router.get("/thesis/evaluations/latest")
def get_latest_evaluations_endpoint():
    from portfolio.thesis_db import get_latest_evaluations

    return get_latest_evaluations()


@router.get("/thesis/status")
def get_thesis_status() -> dict[str, Literal["populated", "empty", "missing"]]:
    from portfolio.portfolio_db import get_positions

    statuses: dict[str, Literal["populated", "empty", "missing"]] = {}
    for row in get_positions():
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if not ticker or ticker in statuses:
            continue
        if not _TICKER_RE.match(ticker):
            statuses[ticker] = "missing"
            continue
        if not _thesis_exists(ticker):
            statuses[ticker] = "missing"
            continue
        try:
            content = _read_thesis(ticker).strip()
            statuses[ticker] = "populated" if content else "empty"
        except Exception:
            statuses[ticker] = "missing"
    return statuses


@router.get("/thesis/{ticker}/detail")
def get_thesis_detail(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    from portfolio.thesis_db import get_evaluations, get_status_history, get_thesis_meta

    meta = get_thesis_meta(normalized_ticker)
    if not meta:
        raise NotFoundError("Thesis metadata", normalized_ticker)

    content = None
    if _thesis_exists(normalized_ticker):
        try:
            content = _read_thesis(normalized_ticker)
        except Exception:
            content = None

    return {
        "meta": meta,
        "content": content,
        "status_history": get_status_history(normalized_ticker),
        "evaluations": get_evaluations(normalized_ticker, limit=52),
    }


class StatusChangeRequest(BaseModel):
    status: str
    reason: str = ""
    apply: bool = False
    approval_note: str | None = None


@router.put("/thesis/{ticker}/status")
def change_thesis_status(ticker: str, body: StatusChangeRequest):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    try:
        return stage_api_action(
            "change_thesis_status",
            {"ticker": normalized_ticker, "status": body.status, "reason": body.reason},
            source_id="thesis.change_thesis_status",
            reason=body.reason or f"Change thesis status for {normalized_ticker}",
            apply=body.apply,
            approval_note=body.approval_note,
        )
    except ActionValidationError as e:
        raise ValidationError(e.message) from e
    except ActionNotFoundError as e:
        raise NotFoundError("Thesis", normalized_ticker) from e


class SaveThesisRequest(BaseModel):
    content: str = Field(..., max_length=MAX_UPLOAD_SIZE_BYTES)
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.put("/thesis/{ticker}")
def save_thesis(ticker: str, body: SaveThesisRequest):
    """Save thesis markdown content directly."""
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    content = body.content.strip()
    if not content:
        raise ValidationError("Thesis content cannot be empty.")

    _configure_thesis_content_storage()
    return stage_api_action(
        "save_thesis_content",
        {"ticker": normalized_ticker, "content": content},
        source_id="thesis.save_thesis",
        reason=body.reason or f"Update thesis content for {normalized_ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.get("/thesis/{ticker}")
def get_thesis(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    if not _thesis_exists(normalized_ticker):
        raise NotFoundError("Thesis", normalized_ticker)
    try:
        content = _read_thesis(normalized_ticker)
    except Exception as e:
        from api.exceptions import AppError

        raise AppError(f"Failed to read thesis file: {e}") from e
    return {"status": "ok", "ticker": normalized_ticker, "content": content}
