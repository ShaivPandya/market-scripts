"""Equity overview API — generate, save, and retrieve overview markdown."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from api.exceptions import DataFetchError, NotFoundError, ValidationError
from api.routers.portfolio_edit import _TICKER_RE
from llm_utils import MODEL_SONNET, extract_text
from paths import PROJECT_ROOT

router = APIRouter()

OVERVIEWS_DIR = PROJECT_ROOT / "investment_overviews"
MAX_PDF_SIZE_BYTES = 30 * 1024 * 1024

_REQ_SECTIONS = ("## Financials", "## Sensitivity to Extrinsic Factors", "## Industry")

_SYSTEM_PROMPT = """You are a buy-side equity research analyst.
Extract a structured equity overview from the attached PDF and output it as markdown.

Output only markdown and follow this structure exactly:

# {ticker} Overview

## Financials
- **3-Year Avg. YoY Revenue Growth**: percentage with brief context
- **3-Year Avg. YoY EPS Growth**: percentage with brief context
- **Debt**: total debt, maturity schedule, key details. Use a markdown table if a maturity schedule is available.
- **Reinvestment Costs**: capex, R&D, or other reinvestment requirements

## Sensitivity to Extrinsic Factors
Present as a markdown table with three columns:

| Factor | Sensitivity | Capacity to Deal |
|--------|------------|-----------------|

Include rows for all applicable factors from this list: commodity prices, interest rates, currency/FX, tariffs/trade policy, war/geopolitical disruption, regulatory changes, inflation, labor costs.
Rate sensitivity as Low, Low-medium, Medium, Medium-high, or High. Describe capacity to deal briefly.
Only include factors relevant to this company. If the PDF does not cover a factor, omit it.

## Industry
- **Porter's Five Forces**: Summarize each force (threat of new entrants, bargaining power of suppliers, bargaining power of buyers, threat of substitutes, competitive rivalry) in one bullet each with a Low/Medium/High rating
- **Supply Outlook**: current and forward-looking supply dynamics
- **Demand Outlook**: current and forward-looking demand dynamics

Use short bullets. Be concise and decision-useful.
Focus on company-specific and financially material points.
Do not include disclaimers or any text outside the markdown."""

_USER_PROMPT = """Use the attached PDF to write the equity overview markdown.
Extract specific data points (revenue growth percentages, EPS figures, debt amounts, sensitivity ratings) directly from the document.
If a section's data is not present in the PDF, write "Data not available in source document" under that subsection.
Keep it concise and decision-useful."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_ticker(raw_ticker: str) -> str:
    return raw_ticker.strip().upper()


def _validate_ticker(ticker: str) -> None:
    if not ticker:
        raise ValidationError("Ticker cannot be empty.")
    if not _TICKER_RE.match(ticker):
        raise ValidationError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")


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


def _normalize_overview_markdown(ticker: str, content: str) -> str:
    cleaned = _strip_outer_markdown_fence(content)
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {ticker} Overview"
        cleaned = "\n".join(lines).strip()
    elif cleaned:
        cleaned = f"# {ticker} Overview\n\n{cleaned}"
    else:
        cleaned = f"# {ticker} Overview"

    for section in _REQ_SECTIONS:
        if section not in cleaned:
            cleaned += f"\n\n{section}\n- TBD"
    return cleaned.strip() + "\n"


def _call_claude_overview_pdf(*, ticker: str, pdf_bytes: bytes) -> str:
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
        raise DataFetchError(source="claude", detail="Claude returned empty overview output.")
    return _normalize_overview_markdown(ticker, generated)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/overview/generate")
async def generate_overview(
    ticker: str = Form(...),  # noqa: B008 - FastAPI parameter declaration
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise ValidationError("Uploaded file is empty.")
    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise ValidationError("PDF exceeds 30MB limit.")

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    has_pdf_type = content_type == "application/pdf" or filename.endswith(".pdf")
    has_pdf_signature = pdf_bytes.startswith(b"%PDF-")
    if not (has_pdf_type and has_pdf_signature):
        raise ValidationError("File must be a valid PDF.")

    try:
        content = _call_claude_overview_pdf(ticker=normalized_ticker, pdf_bytes=pdf_bytes)
    except (ValidationError, DataFetchError):
        raise
    except Exception as e:
        raise DataFetchError(source="claude", detail=f"Failed to generate overview: {e}") from e

    OVERVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    overview_path = OVERVIEWS_DIR / f"{normalized_ticker}.md"
    try:
        overview_path.write_text(content, encoding="utf-8")
    except Exception as e:
        from api.exceptions import AppError

        raise AppError(f"Failed to write overview file: {e}") from e

    # Index overview for semantic search (best-effort)
    try:
        from api.retrieval import index_document

        index_document(
            doc_type="thesis",
            content=content,
            ticker=normalized_ticker,
            source_path=str(overview_path),
            doc_id=f"overview-{normalized_ticker}",
        )
    except Exception:
        pass

    return {"status": "ok", "ticker": normalized_ticker, "content": content}


class SaveOverviewRequest(BaseModel):
    content: str


@router.put("/overview/{ticker}")
def save_overview(ticker: str, body: SaveOverviewRequest):
    """Save overview markdown content directly."""
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    content = body.content.strip()
    if not content:
        raise ValidationError("Overview content cannot be empty.")

    OVERVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    overview_path = OVERVIEWS_DIR / f"{normalized_ticker}.md"
    try:
        overview_path.write_text(content + "\n", encoding="utf-8")
    except Exception as e:
        from api.exceptions import AppError

        raise AppError(f"Failed to write overview file: {e}") from e

    # Index overview for semantic search (best-effort)
    try:
        from api.retrieval import index_document

        index_document(
            doc_type="thesis",
            content=content,
            ticker=normalized_ticker,
            source_path=str(overview_path),
            doc_id=f"overview-{normalized_ticker}",
        )
    except Exception:
        pass

    return {"status": "ok", "ticker": normalized_ticker, "content": content}


@router.get("/overview/{ticker}")
def get_overview(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    overview_path = OVERVIEWS_DIR / f"{normalized_ticker}.md"
    if not overview_path.exists():
        raise NotFoundError("Overview", normalized_ticker)
    try:
        content = overview_path.read_text(encoding="utf-8")
    except Exception as e:
        from api.exceptions import AppError

        raise AppError(f"Failed to read overview file: {e}") from e
    return {"status": "ok", "ticker": normalized_ticker, "content": content}
