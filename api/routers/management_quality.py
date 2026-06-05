"""Management quality API - generate, save, and retrieve management assessment markdown."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from api.action_execution import stage_api_action
from api.document_generation_jobs import classify_upload_document, enqueue_document_generation_upload
from api.exceptions import DataFetchError, NotFoundError, ValidationError
from api.routers.auth import ActorDep
from api.routers.portfolio_edit import _TICKER_RE
from llm_utils import MODEL_MID, call_llm_pdf_text, call_llm_text, strip_assistant_citation_tokens
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio import management_quality_content

router = APIRouter()

MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024

_REQ_SECTIONS = (
    "## Executive Summary",
    "## Management Scorecard",
    "## Most Impressive Accomplishments",
    "## Biggest Setbacks and Responses",
    "## Chronology / Detail",
    "## Evidence Notes",
)

_SYSTEM_PROMPT = """You are a buy-side investment analyst evaluating management quality.
Extract a structured management-quality assessment from the source document and output it as markdown.

Output only markdown and follow this structure exactly:

# {ticker} Management Quality

## Executive Summary
- **Overall Rating**: Strong, Mixed, Weak, or Insufficient evidence
- **Bottom Line**: 2-4 sentence assessment of management quality.
- **Owner Mindset**: Strong, Mixed, Weak, or Insufficient evidence - concise evidence-backed explanation.
- **Business Value Understanding**: Strong, Mixed, Weak, or Insufficient evidence - concise evidence-backed explanation.
- **Follow-through / Character**: Strong, Mixed, Weak, or Insufficient evidence - concise evidence-backed explanation.

## Management Scorecard
| Question | Rating | Evidence |
|----------|--------|----------|
| Do managers think and act like owners? | Strong/Mixed/Weak/Insufficient evidence | concise evidence |
| Do managers understand what drives business value? | Strong/Mixed/Weak/Insufficient evidence | concise evidence |
| Did they do what they said they would do? | Strong/Mixed/Weak/Insufficient evidence | concise evidence |

## Most Impressive Accomplishments
- **Accomplishment title (period)**: What management accomplished, why it mattered, and source/citation markers.

## Biggest Setbacks and Responses
- **Setback title (period)**: What went wrong. **Response**: Handled well, Mixed, Handled poorly, or Too early - how management dealt with it.

## Chronology / Detail
### Period or event
- **Said**: What management said it would do.
- **Did**: What later happened.
- **Assessment**: Whether management followed through and how well it responded.

## Evidence Notes
- Preserve compact citations, source markers, and source limitations where present.

Evaluation standard:
1. Owner mindset: capital allocation, acquisitions, buybacks, options, incentives, insider alignment, shareholder treatment.
2. Business value understanding: grasp of core business, economic drivers, reinvestment discipline, unit economics, competitive position.
3. Follow-through / character: whether management did what it said it would do, admitted misses, and acted promptly.

Use source-backed facts only. Do not invent missing evidence. Preserve compact citations/source markers from the source when useful. Remove navigation lists, boilerplate, and citation artifacts that do not help audit the claim."""

_PDF_USER_PROMPT = """Use the attached PDF to write the management-quality assessment markdown.
Keep the aggregate summary brief and put detailed chronology below it.
Preserve compact citations/source markers when present."""

_MARKDOWN_USER_PROMPT = """Use the uploaded markdown below to write the management-quality assessment markdown.
Restructure the source into the exact schema from the system prompt.
Keep the aggregate summary brief and put detailed chronology below it.
Preserve compact citations/source markers when present, but remove navigation lists and source metadata that are not useful to the dossier UI."""


def _normalize_ticker(raw_ticker: str) -> str:
    return raw_ticker.strip().upper()


def _validate_ticker(ticker: str) -> None:
    if not ticker:
        raise ValidationError("Ticker cannot be empty.")
    if not _TICKER_RE.match(ticker):
        raise ValidationError(f"Invalid ticker format: '{ticker}'. Only letters, digits, and dots are allowed.")


def _strip_outer_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _normalize_management_quality_markdown(ticker: str, content: str) -> str:
    cleaned = strip_assistant_citation_tokens(_strip_outer_markdown_fence(content))
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {ticker} Management Quality"
        cleaned = "\n".join(lines).strip()
    elif cleaned:
        cleaned = f"# {ticker} Management Quality\n\n{cleaned}"
    else:
        cleaned = f"# {ticker} Management Quality"

    for section in _REQ_SECTIONS:
        if section not in cleaned:
            cleaned += f"\n\n{section}\n- Insufficient evidence in source document."
    return cleaned.strip() + "\n"


def _decode_markdown_upload(markdown_bytes: bytes) -> str:
    try:
        content = markdown_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValidationError("Markdown file must be UTF-8 encoded.") from e
    if not content.strip():
        raise ValidationError("Markdown file is empty.")
    return strip_assistant_citation_tokens(content)


def _llm_error_message(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(body.get("message"), str):
            return str(body["message"])
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return str(error["message"])
            if isinstance(data.get("message"), str):
                return str(data["message"])
    return str(exc)


def _finish_llm_management_quality(ticker: str, generated: str) -> str:
    if not generated:
        raise DataFetchError(source="llm", detail="LLM returned empty management-quality output.")
    return _normalize_management_quality_markdown(ticker, generated)


def _call_llm_management_quality_pdf(*, ticker: str, pdf_bytes: bytes) -> str:
    generated, _citations, _response = call_llm_pdf_text(
        pdf_bytes=pdf_bytes,
        prompt=_PDF_USER_PROMPT,
        model=MODEL_MID,
        api_key=None,
        max_tokens=8192,
        system=_SYSTEM_PROMPT.format(ticker=ticker),
        filename=f"{ticker}-management-quality.pdf",
    )
    return _finish_llm_management_quality(ticker, generated)


def _call_llm_management_quality_markdown(*, ticker: str, markdown: str) -> str:
    prompt = f"{_MARKDOWN_USER_PROMPT}\n\n<uploaded_markdown>\n{markdown}\n</uploaded_markdown>"
    generated, _citations, _response = call_llm_text(
        prompt=prompt,
        model=MODEL_MID,
        api_key=None,
        max_tokens=8192,
        system=_SYSTEM_PROMPT.format(ticker=ticker),
    )
    return _finish_llm_management_quality(ticker, generated)


def _split_sections(content: str) -> dict[str, str]:
    expected_sections = {
        "executive summary",
        "management scorecard",
        "most impressive accomplishments",
        "biggest setbacks and responses",
        "chronology / detail",
        "chronology",
        "evidence notes",
    }
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for line in content.splitlines():
        heading = re.match(r"^\s*#{2,6}\s+(.+?)\s*$", line)
        heading_key = heading.group(1).strip().lower() if heading else None
        if heading_key in expected_sections:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = heading_key
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


_SUMMARY_RATINGS = {
    "strong": "Strong",
    "mixed": "Mixed",
    "weak": "Weak",
    "insufficient evidence": "Insufficient evidence",
}


def _strip_inline_emphasis(raw: str) -> str:
    return re.sub(r"[*_`~]+", "", raw).strip()


def _canonical_summary_rating(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = " ".join(_strip_inline_emphasis(str(raw)).lower().split())
    return _SUMMARY_RATINGS.get(key)


def _split_rating_text(raw: str) -> dict[str, str | None]:
    text = raw.strip()
    match = re.match(
        r"^\s*(?:[*_`~]+)?\s*(Strong|Mixed|Weak|Insufficient evidence)\b\s*(?:[*_`~]+)?\s*(?:(?:[:—–-]+)\s*(.+)|\s+(.+))?$",
        text,
        flags=re.I,
    )
    if not match:
        return {"rating": None, "text": text}
    rating = _canonical_summary_rating(match.group(1))
    return {"rating": rating, "text": (match.group(2) or match.group(3) or "").strip() or None}


def _parse_summary(text: str) -> dict | None:
    summary: dict[str, object] = {}
    question_map = {
        "owner mindset": "owner_mindset",
        "business value understanding": "business_value_understanding",
        "follow-through / character": "follow_through",
        "follow-through": "follow_through",
    }
    for line in text.splitlines():
        match = re.match(r"^\s*[-*]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*:\s*(.+)", line)
        if not match:
            continue
        label = match.group(1).strip()
        value = match.group(2).strip()
        label_key = label.lower()
        if label_key == "overall rating":
            split = _split_rating_text(value)
            summary["overall_rating"] = split["rating"] or _strip_inline_emphasis(value)
        elif label_key == "bottom line":
            summary["bottom_line"] = value
        elif label_key in question_map:
            summary[question_map[label_key]] = _split_rating_text(value)
    return summary if summary else None


def _parse_scorecard(text: str) -> list[dict] | None:
    rows: list[dict] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].lower() == "question" or re.match(r"^[-:]+$", cells[0]):
            continue
        split = _split_rating_text(cells[1])
        rows.append({"question": cells[0], "rating": split["rating"] or cells[1], "evidence": cells[2]})
    return rows if rows else None


def _parse_bullets(text: str) -> list[dict] | None:
    rows: list[dict] = []
    for line in text.splitlines():
        match = re.match(r"^\s*-\s*(?:\*\*(.+?)\*\*:\s*)?(.+)", line)
        if not match:
            continue
        title = (match.group(1) or "").strip()
        body = match.group(2).strip()
        if not title and body in {"-", "--", "—", "–"}:
            continue
        rows.append({"title": title or None, "text": body})
    return rows if rows else None


def _parse_setbacks(text: str) -> list[dict] | None:
    rows = _parse_bullets(text) or []
    for row in rows:
        body = str(row.get("text") or "")
        response = re.search(
            r"(?:\*\*)?Response(?:\*\*)?:\s*(?:[*_`~]+)?\s*(Handled well|Mixed|Handled poorly|Too early)\s*(?:[*_`~]+)?(?:\s*[—–-]+\s*(.+))?",
            body,
            flags=re.I,
        )
        if response:
            row["response_rating"] = response.group(1)
            row["response_text"] = (response.group(2) or "").strip() or None
            row["text"] = body[: response.start()].rstrip(" -—–")
    return rows if rows else None


def parse_management_quality_markdown(content: str) -> dict | None:
    if not content or not content.strip():
        return None
    sections = _split_sections(content)
    result: dict = {}
    try:
        result["summary"] = _parse_summary(sections.get("executive summary", ""))
    except Exception:
        result["summary"] = None
    try:
        result["scorecard"] = _parse_scorecard(sections.get("management scorecard", ""))
    except Exception:
        result["scorecard"] = None
    try:
        result["accomplishments"] = _parse_bullets(sections.get("most impressive accomplishments", ""))
    except Exception:
        result["accomplishments"] = None
    try:
        result["setbacks"] = _parse_setbacks(sections.get("biggest setbacks and responses", ""))
    except Exception:
        result["setbacks"] = None
    return result if any(v is not None for v in result.values()) else None


def _render_management_quality_markdown(ticker: str, assessment: dict) -> str:
    raw_parsed = assessment.get("parsed")
    parsed: dict[str, Any] = raw_parsed if isinstance(raw_parsed, dict) else {}
    raw_summary = parsed.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    raw_scorecard = parsed.get("scorecard")
    scorecard: list[Any] = raw_scorecard if isinstance(raw_scorecard, list) else []
    raw_accomplishments = parsed.get("accomplishments")
    accomplishments: list[Any] = raw_accomplishments if isinstance(raw_accomplishments, list) else []
    raw_setbacks = parsed.get("setbacks")
    setbacks: list[Any] = raw_setbacks if isinstance(raw_setbacks, list) else []

    lines = [
        f"# {ticker} Management Quality",
        "",
        "## Executive Summary",
        f"- **Overall Rating**: {summary.get('overall_rating') or 'Insufficient evidence'}",
        f"- **Bottom Line**: {summary.get('bottom_line') or assessment.get('bottom_line') or 'Insufficient evidence.'}",
    ]
    for label, key in (
        ("Owner Mindset", "owner_mindset"),
        ("Business Value Understanding", "business_value_understanding"),
        ("Follow-through / Character", "follow_through"),
    ):
        raw_item = summary.get(key)
        item: dict[str, Any] = raw_item if isinstance(raw_item, dict) else {}
        rating = item.get("rating") or "Insufficient evidence"
        text = item.get("text") or "Insufficient evidence."
        lines.append(f"- **{label}**: {rating} - {text}")

    lines.extend(
        ["", "## Management Scorecard", "| Question | Rating | Evidence |", "|----------|--------|----------|"]
    )
    if scorecard:
        for row in scorecard:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row.get('question') or 'Question'} | {row.get('rating') or 'Insufficient evidence'} | {row.get('evidence') or ''} |"
            )
    else:
        lines.append("| Insufficient evidence | Insufficient evidence | No scorecard rows parsed. |")

    lines.extend(["", "## Most Impressive Accomplishments"])
    if accomplishments:
        for row in accomplishments:
            if isinstance(row, dict):
                title = row.get("title") or "Accomplishment"
                lines.append(f"- **{title}**: {row.get('text') or ''}".rstrip())
    else:
        lines.append("- Insufficient evidence in source document.")

    lines.extend(["", "## Biggest Setbacks and Responses"])
    if setbacks:
        for row in setbacks:
            if not isinstance(row, dict):
                continue
            title = row.get("title") or "Setback"
            response = row.get("response_rating")
            response_text = row.get("response_text")
            suffix = f" **Response**: {response}{f' - {response_text}' if response_text else ''}" if response else ""
            lines.append(f"- **{title}**: {row.get('text') or ''}{suffix}".rstrip())
    else:
        lines.append("- Insufficient evidence in source document.")

    lines.extend(
        ["", "## Chronology / Detail", "- See source assessment.", "", "## Evidence Notes", "- See source assessment."]
    )
    return "\n".join(lines).strip() + "\n"


def _read_markdown_projection(ticker: str) -> str | None:
    try:
        if management_quality_content.management_quality_exists(ticker):
            return management_quality_content.read_management_quality(ticker)
    except Exception:
        return None
    return None


def generate_management_quality_from_upload_bytes(
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
            content = _call_llm_management_quality_pdf(ticker=normalized_ticker, pdf_bytes=upload_bytes)
        except (ValidationError, DataFetchError):
            raise
        except Exception as e:
            raise DataFetchError(
                source="llm",
                detail=f"Failed to generate management quality: {_llm_error_message(e)}",
            ) from e
    else:
        markdown = _decode_markdown_upload(upload_bytes)
        try:
            content = _call_llm_management_quality_markdown(ticker=normalized_ticker, markdown=markdown)
        except (ValidationError, DataFetchError):
            raise
        except Exception as e:
            raise DataFetchError(
                source="llm",
                detail=f"Failed to generate management quality: {_llm_error_message(e)}",
            ) from e

    return stage_api_action(
        "save_management_quality_content",
        {"ticker": normalized_ticker, "content": content, "preserve_exact_content": True},
        source_id="management_quality.generate_management_quality",
        reason=f"Generate management quality assessment for {normalized_ticker} from uploaded document",
    )


@router.post("/management-quality/generate")
async def generate_management_quality(
    ticker: str = Form(...),  # noqa: B008 - FastAPI parameter declaration
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)
    return await enqueue_document_generation_upload(
        kind="management_quality",
        ticker=normalized_ticker,
        file=file,
        max_upload_size_bytes=MAX_UPLOAD_SIZE_BYTES,
    )


class SaveManagementQualityRequest(BaseModel):
    content: str = Field(..., max_length=MAX_UPLOAD_SIZE_BYTES)
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.put("/management-quality/{ticker}")
def save_management_quality(ticker: str, body: SaveManagementQualityRequest, actor: ActorDep):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    content = body.content.strip()
    if not content:
        raise ValidationError("Management quality content cannot be empty.")

    content = _normalize_management_quality_markdown(normalized_ticker, content)
    return stage_api_action(
        "save_management_quality_content",
        {"ticker": normalized_ticker, "content": content, "preserve_exact_content": True},
        source_id="management_quality.save_management_quality",
        actor=actor,
        reason=body.reason or f"Update management quality assessment for {normalized_ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.get("/management-quality/{ticker}")
def get_management_quality(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    assessment = OntologyRuntimeReadService().management_quality_assessment(normalized_ticker)
    if not assessment:
        raise NotFoundError("Management quality", normalized_ticker)

    content = _read_markdown_projection(normalized_ticker) or _render_management_quality_markdown(
        normalized_ticker, assessment
    )
    parsed = assessment.get("parsed") if isinstance(assessment.get("parsed"), dict) else None
    if not parsed:
        parsed = parse_management_quality_markdown(content)
    return {
        "status": "ok",
        "ticker": normalized_ticker,
        "content": content,
        "parsed": parsed,
        "assessment": assessment,
    }
