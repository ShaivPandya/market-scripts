from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.routers.portfolio_edit import _TICKER_RE
from llm_utils import MODEL_SONNET, extract_text
from paths import PROJECT_ROOT

router = APIRouter()

THESES_DIR = PROJECT_ROOT / "investment_theses"
MAX_PDF_SIZE_BYTES = 30 * 1024 * 1024

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
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")
    if not _TICKER_RE.match(ticker):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.",
        )


def _extract_stop_reason(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("stop_reason")
    else:
        value = getattr(response, "stop_reason", None)
    return value if isinstance(value, str) else None


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


def _call_claude_pdf(*, ticker: str, pdf_bytes: bytes) -> str:
    import anthropic

    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip() or None
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": _USER_PROMPT},
            ],
        }
    ]
    kwargs: dict[str, Any] = {
        "model": MODEL_SONNET,
        "max_tokens": 4096,
        "system": _SYSTEM_PROMPT.format(ticker=ticker),
        "messages": messages,
    }
    response = client.messages.create(**kwargs)
    while _extract_stop_reason(response) == "pause_turn":
        assistant_content = (
            response.get("content", []) if isinstance(response, dict) else getattr(response, "content", [])
        )
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
        kwargs["messages"] = messages
        response = client.messages.create(**kwargs)

    generated = extract_text(response)
    if not generated:
        raise HTTPException(status_code=502, detail="Claude returned empty thesis output.")
    return _normalize_output_markdown(ticker, generated)


@router.post("/thesis/generate")
async def generate_thesis(
    ticker: str = Form(...),  # noqa: B008 - FastAPI parameter declaration
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 30MB limit.")

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    has_pdf_type = content_type == "application/pdf" or filename.endswith(".pdf")
    has_pdf_signature = pdf_bytes.startswith(b"%PDF-")
    if not (has_pdf_type and has_pdf_signature):
        raise HTTPException(status_code=400, detail="File must be a valid PDF.")

    try:
        content = _call_claude_pdf(ticker=normalized_ticker, pdf_bytes=pdf_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate thesis via Claude: {e}") from e

    THESES_DIR.mkdir(parents=True, exist_ok=True)
    thesis_path = THESES_DIR / f"{normalized_ticker}.md"
    try:
        thesis_path.write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write thesis file: {e}") from e

    from portfolio.thesis_db import upsert_thesis_meta

    upsert_thesis_meta(normalized_ticker, status="active")

    return {"status": "ok", "ticker": normalized_ticker, "content": content}


@router.get("/thesis/meta")
def get_thesis_meta_all():
    from portfolio.thesis_db import get_all_thesis_meta, get_latest_evaluations

    meta = get_all_thesis_meta()
    latest = {e["ticker"]: e for e in get_latest_evaluations()}
    for m in meta:
        m["latest_evaluation"] = latest.get(m["ticker"])
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
        thesis_path: Path = THESES_DIR / f"{ticker}.md"
        if not thesis_path.exists():
            statuses[ticker] = "missing"
            continue
        try:
            content = thesis_path.read_text(encoding="utf-8").strip()
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
        raise HTTPException(status_code=404, detail=f"No thesis metadata for '{normalized_ticker}'.")

    content = None
    thesis_path = THESES_DIR / f"{normalized_ticker}.md"
    if thesis_path.exists():
        try:
            content = thesis_path.read_text(encoding="utf-8")
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


@router.put("/thesis/{ticker}/status")
def change_thesis_status(ticker: str, body: StatusChangeRequest):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    new_status = body.status.strip().lower()
    if new_status not in ("active", "under_review", "invalidated"):
        raise HTTPException(
            status_code=400, detail=f"Invalid status: '{new_status}'. Must be active, under_review, or invalidated."
        )

    from portfolio.thesis_db import update_thesis_status

    try:
        return update_thesis_status(normalized_ticker, new_status, body.reason.strip())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/thesis/{ticker}")
def get_thesis(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    thesis_path = THESES_DIR / f"{normalized_ticker}.md"
    if not thesis_path.exists():
        raise HTTPException(status_code=404, detail=f"No thesis found for ticker '{normalized_ticker}'.")
    try:
        content = thesis_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read thesis file: {e}") from e
    return {"status": "ok", "ticker": normalized_ticker, "content": content}
