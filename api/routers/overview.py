"""Equity overview API — generate, save, and retrieve overview markdown."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from api.action_execution import stage_api_action
from api.document_generation_jobs import classify_upload_document, enqueue_document_generation_upload
from api.exceptions import DataFetchError, NotFoundError, ValidationError
from api.routers.auth import ActorDep
from api.routers.portfolio_edit import _TICKER_RE
from llm_utils import MODEL_MID, call_llm_pdf_text, call_llm_text, strip_assistant_citation_tokens
from paths import PROJECT_ROOT
from portfolio import overview_content

router = APIRouter()

OVERVIEWS_DIR = PROJECT_ROOT / "investment_overviews"
OVERVIEWS_GCS_PREFIX = "live/overviews"
MAX_UPLOAD_SIZE_BYTES = 30 * 1024 * 1024

_REQ_SECTIONS = ("## Financials", "## Sensitivity to Extrinsic Factors", "## Industry")

_SYSTEM_PROMPT = """You are a buy-side equity research analyst.
Extract a structured equity overview from the source document and output it as markdown.

Output only markdown and follow this structure exactly:

# {ticker} Overview

## Financials
- **3-Year Avg. YoY Revenue Growth**: percentage with brief context
- **3-Year Avg. YoY EPS Growth**: percentage with brief context
- **Interest Coverage**: OI / interest expense ratio with brief context; flag if below 4x
- **Operating Margin**: percentage with brief context
- **Net Income Margin**: percentage with brief context
- **Debt**: total debt, maturity schedule, key details. Use a markdown table if a maturity schedule is available.
- **Reinvestment Costs**: capex, R&D, or other reinvestment requirements

## Sensitivity to Extrinsic Factors
Present as a markdown table with four columns:

| Factor | Sensitivity | Capacity to Deal | Rationale |
|--------|-------------|------------------|-----------|

Include rows for all applicable factors from this list: commodity prices, interest rates, currency/FX, tariffs/trade policy, war/geopolitical disruption, regulatory changes, inflation, labor costs.
Rate sensitivity and capacity to deal as Low, Low-medium, Medium, Medium-high, or High. Put only the capacity rating in Capacity to Deal, and put the brief source-backed explanation for both ratings in Rationale.
Only include factors relevant to this company. If the source document does not cover a factor, omit it.

## Industry

### Porter's Five Forces
- **Threat of New Entrants — Low/Medium/High**: concise explanation
- **Bargaining Power of Suppliers — Low/Medium/High**: concise explanation
- **Bargaining Power of Buyers — Low/Medium/High**: concise explanation
- **Threat of Substitutes — Low/Medium/High**: concise explanation
- **Competitive Rivalry — Low/Medium/High**: concise explanation

### Supply Chain
Only include this subsection if the source names specific suppliers or customers.
If included, use these tables and include only source-backed named entities:

#### Key Suppliers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|

#### Key Customers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|

If the source does not name specific suppliers or customers, omit the entire ### Supply Chain section.
Do not output empty Supply Chain tables, demo names, hallucinated counterparties, or "Data not available" for Supply Chain.

### Supply Outlook
- current and forward-looking supply dynamics

### Demand Outlook
- current and forward-looking demand dynamics

Use short bullets. Be concise and decision-useful.
Focus on company-specific and financially material points.
Do not include disclaimers or any text outside the markdown."""

_PDF_USER_PROMPT = """Use the attached PDF to write the equity overview markdown.
Extract specific data points (revenue growth percentages, EPS figures, debt amounts, sensitivity ratings) directly from the document.
If a section's data is not present in the source document, write "Data not available in source document" under that subsection.
Supply Chain is the only exception: omit the entire ### Supply Chain section unless the source names specific suppliers or customers.
Keep it concise and decision-useful."""

_MARKDOWN_USER_PROMPT = """Use the uploaded markdown below to write the equity overview markdown.
Restructure the source into the exact schema from the system prompt so the application can render the structured Overview UI.
Preserve source-backed company-specific facts, but remove citation artifacts, entity tags, nav lists, and source metadata that are not useful in the overview UI.
If a section's data is not present in the source document, write "Data not available in source document" under that subsection.
Supply Chain is the only exception: omit the entire ### Supply Chain section unless the source names specific suppliers or customers.
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


def _sync_overview_content_paths() -> None:
    overview_content.OVERVIEWS_DIR = OVERVIEWS_DIR
    overview_content.OVERVIEWS_GCS_PREFIX = OVERVIEWS_GCS_PREFIX


def _overview_path(ticker: str) -> Path:
    _sync_overview_content_paths()
    return overview_content.overview_path(ticker)


def _overview_gcs_key(ticker: str) -> str:
    _sync_overview_content_paths()
    return overview_content.overview_gcs_key(ticker)


def _overview_exists(ticker: str) -> bool:
    _sync_overview_content_paths()
    return overview_content.overview_exists(ticker)


def _read_overview(ticker: str) -> str:
    _sync_overview_content_paths()
    return overview_content.read_overview(ticker)


def _write_overview(ticker: str, content: str) -> str:
    _sync_overview_content_paths()
    return overview_content.write_overview(ticker, content)


def _strip_outer_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _split_sections(content: str) -> dict[str, str]:
    """Split markdown into named sections keyed by heading text."""
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line.lstrip("#").strip().lower()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _parse_financial_metric_line(lines: list[str], label_pattern: str) -> dict | None:
    for line in lines:
        m = re.match(rf"^\s*-\s*\*\*{label_pattern}\*\*:\s*(.+)", line)
        if not m:
            continue
        raw = m.group(1).strip()
        val_m = re.match(r"^([~+\-\d.%xX]+)", raw)
        return {
            "value": val_m.group(1) if val_m else None,
            "context": raw,
        }
    return None


def _parse_financials(text: str) -> dict | None:
    result: dict = {}
    lines = text.splitlines()

    # Revenue growth
    result["revenue_growth"] = _parse_financial_metric_line(lines, r"3-Year Avg\.?\s*YoY Revenue Growth")

    # EPS growth
    result["eps_growth"] = _parse_financial_metric_line(lines, r"3-Year Avg\.?\s*YoY EPS Growth")
    result["interest_coverage"] = _parse_financial_metric_line(lines, r"Interest Coverage")
    result["operating_margin"] = _parse_financial_metric_line(lines, r"Operating Margin")
    result["net_income_margin"] = _parse_financial_metric_line(lines, r"Net Income Margin")

    # Debt
    debt_started = False
    debt_summary = ""
    tranches: list[dict] = []
    in_table = False
    for line in lines:
        if not debt_started:
            m = re.match(r"^\s*-\s*\*\*Debt\*\*:\s*(.+)", line)
            if m:
                debt_started = True
                debt_summary = m.group(1).strip()
            continue
        # Stop at next bullet that isn't a table row
        if re.match(r"^\s*-\s*\*\*", line):
            break
        # Parse table rows
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if cells and re.match(r"^[-:]+$", cells[0]):
                in_table = True
                continue
            if in_table and len(cells) >= 3:
                tranches.append({"tranche": cells[0], "rate": cells[1], "maturity": cells[2]})
            elif not in_table and len(cells) >= 3:
                # header row
                in_table = False
                continue
    result["debt"] = {"summary": debt_summary, "tranches": tranches} if debt_started else None

    # Reinvestment
    for line in lines:
        m = re.match(r"^\s*-\s*\*\*Reinvestment Costs?\*\*:\s*(.+)", line)
        if m:
            result["reinvestment"] = m.group(1).strip()
            break
    if "reinvestment" not in result:
        result["reinvestment"] = None

    return result if any(v is not None for v in result.values()) else None


def _parse_sensitivity(text: str) -> list[dict] | None:
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if re.match(r"^[-:]+$", cells[0]):
            in_table = True
            continue
        if not in_table:
            continue
        if len(cells) >= 3:
            row = {
                "factor": cells[0],
                "sensitivity": cells[1],
                "capacity": cells[2],
            }
            if len(cells) >= 4:
                row["rationale"] = cells[3]
            rows.append(row)
    return rows if rows else None


def _markdown_table_cells(line: str) -> list[str]:
    raw = line.strip()
    if not raw or "|" not in raw:
        return []
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [cell.strip() for cell in raw.split("|")]


def _is_markdown_separator_row(cells: list[str]) -> bool:
    non_empty = [cell.replace(" ", "") for cell in cells if cell.strip()]
    return bool(non_empty) and all(re.match(r"^:?-{2,}:?$", cell) for cell in non_empty)


_PLACEHOLDER_COUNTERPARTY_RE = re.compile(
    r"^(?:entity|name|n/?a|na|none|unknown|not\s+(?:available|disclosed|named)|"
    r"data\s+not\s+available(?:\s+in\s+source\s+document)?|"
    r"no\s+specific\s+(?:suppliers?|customers?|buyers?)\s+(?:named|disclosed|identified))$",
    re.IGNORECASE,
)


def _optional_supply_chain_cell(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text or _PLACEHOLDER_COUNTERPARTY_RE.match(text):
        return None
    return text


def _parse_supply_chain_table(text: str) -> list[dict]:
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        cells = _markdown_table_cells(line)
        if not cells:
            continue
        if _is_markdown_separator_row(cells):
            in_table = True
            continue
        if not in_table:
            continue
        name = _optional_supply_chain_cell(cells[0] if len(cells) > 0 else None)
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "relationship": _optional_supply_chain_cell(cells[1] if len(cells) > 1 else None),
                "exposure": _optional_supply_chain_cell(cells[2] if len(cells) > 2 else None),
                "notes": _optional_supply_chain_cell(cells[3] if len(cells) > 3 else None),
            }
        )
    return rows


def _split_supply_chain_subsections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^\s*####\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(1).strip().lower()
            if "supplier" in title:
                current_key = "suppliers"
            elif "customer" in title or "buyer" in title:
                current_key = "customers"
            else:
                current_key = None
            if current_key is not None:
                sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _parse_supply_chain(text: str) -> dict | None:
    subsections = _split_supply_chain_subsections(text)
    suppliers = _parse_supply_chain_table(subsections.get("suppliers", ""))
    customers = _parse_supply_chain_table(subsections.get("customers", ""))
    if not suppliers and not customers:
        return None
    return {"suppliers": suppliers, "customers": customers}


def _parse_porters(text: str) -> list[dict] | None:
    forces: list[dict] = []
    for line in text.splitlines():
        # Pattern: - **Force Name — Rating**: Description
        m = re.match(
            r"^\s*-\s*\*\*(.+?)\s*[—–\-]+\s*(.+?)\*\*:\s*(.+)",
            line,
        )
        if m:
            forces.append(
                {
                    "force": m.group(1).strip(),
                    "rating": m.group(2).strip(),
                    "description": m.group(3).strip(),
                }
            )
    return forces if forces else None


def _infer_outlook_rating(text: str) -> str | None:
    lower = text.lower()
    strong_signals = ["strong demand", "strong growth", "robust demand", "robust growth", "elevated demand"]
    weak_signals = ["weak demand", "weak growth", "declining demand", "slowing demand", "contracting"]
    for s in strong_signals:
        if s in lower:
            return "Strong"
    for s in weak_signals:
        if s in lower:
            return "Weak"
    if "moderate" in lower or "mixed" in lower:
        return "Medium"
    return None


def _parse_outlook(text: str) -> dict | None:
    points: list[dict | str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*\*\*(.+?)\*\*:\s*(.+)", line)
        if m:
            points.append({"label": m.group(1).strip(), "text": m.group(2).strip()})
        elif re.match(r"^\s*-\s+\S", line):
            points.append(line.lstrip("- ").strip())
    if not points:
        return None
    return {
        "rating": _infer_outlook_rating(text),
        "points": points,
    }


def parse_overview_markdown(content: str) -> dict | None:
    """Parse overview markdown into structured JSON for frontend rendering."""
    if not content or not content.strip():
        return None

    sections = _split_sections(content)
    result: dict = {}

    # Financials
    try:
        result["financials"] = _parse_financials(sections.get("financials", ""))
    except Exception:
        result["financials"] = None

    # Sensitivity
    try:
        result["sensitivity"] = _parse_sensitivity(sections.get("sensitivity to extrinsic factors", ""))
    except Exception:
        result["sensitivity"] = None

    # Porter's Five Forces
    try:
        result["porters_five_forces"] = _parse_porters(
            sections.get("porter's five forces", sections.get("porters five forces", ""))
        )
    except Exception:
        result["porters_five_forces"] = None

    # Supply Outlook
    try:
        result["supply_outlook"] = _parse_outlook(sections.get("supply outlook", ""))
    except Exception:
        result["supply_outlook"] = None

    # Demand Outlook
    try:
        result["demand_outlook"] = _parse_outlook(sections.get("demand outlook", ""))
    except Exception:
        result["demand_outlook"] = None

    # Supply Chain
    try:
        result["supply_chain"] = _parse_supply_chain(sections.get("supply chain", ""))
    except Exception:
        result["supply_chain"] = None

    return result if any(v is not None for v in result.values()) else None


def _normalize_overview_markdown(ticker: str, content: str) -> str:
    cleaned = strip_assistant_citation_tokens(_strip_outer_markdown_fence(content))
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


def _decode_markdown_upload(markdown_bytes: bytes) -> str:
    try:
        content = markdown_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValidationError("Markdown file must be UTF-8 encoded.") from e
    if not content.strip():
        raise ValidationError("Markdown file is empty.")
    return strip_assistant_citation_tokens(content)


def _finish_llm_overview(ticker: str, generated: str) -> str:
    if not generated:
        raise DataFetchError(source="llm", detail="LLM returned empty overview output.")
    return _normalize_overview_markdown(ticker, generated)


def _call_llm_overview_pdf(*, ticker: str, pdf_bytes: bytes) -> str:
    generated, _citations, _response = call_llm_pdf_text(
        pdf_bytes=pdf_bytes,
        prompt=_PDF_USER_PROMPT,
        model=MODEL_MID,
        api_key=None,
        max_tokens=4096,
        system=_SYSTEM_PROMPT.format(ticker=ticker),
        filename=f"{ticker}-overview.pdf",
    )
    return _finish_llm_overview(ticker, generated)


def _call_llm_overview_markdown(*, ticker: str, markdown: str) -> str:
    prompt = f"{_MARKDOWN_USER_PROMPT}\n\n<uploaded_markdown>\n{markdown}\n</uploaded_markdown>"
    generated, _citations, _response = call_llm_text(
        prompt=prompt,
        model=MODEL_MID,
        api_key=None,
        max_tokens=4096,
        system=_SYSTEM_PROMPT.format(ticker=ticker),
    )
    return _finish_llm_overview(ticker, generated)


def generate_overview_from_upload_bytes(
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
            content = _call_llm_overview_pdf(ticker=normalized_ticker, pdf_bytes=upload_bytes)
        except (ValidationError, DataFetchError):
            raise
        except Exception as e:
            raise DataFetchError(source="llm", detail=f"Failed to generate overview: {e}") from e
    else:
        markdown = _decode_markdown_upload(upload_bytes)
        try:
            content = _call_llm_overview_markdown(ticker=normalized_ticker, markdown=markdown)
        except (ValidationError, DataFetchError):
            raise
        except Exception as e:
            raise DataFetchError(source="llm", detail=f"Failed to generate overview: {e}") from e

    _sync_overview_content_paths()
    stage_api_action(
        "save_overview_content",
        {"ticker": normalized_ticker, "content": content, "preserve_exact_content": True},
        source_id="overview.generate_overview",
        reason=f"Generate equity overview for {normalized_ticker} from upload",
        apply=True,
        approval_note="Approved generated overview upload.",
    )

    return {"status": "ok", "ticker": normalized_ticker, "content": content}


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
    return await enqueue_document_generation_upload(
        kind="overview",
        ticker=normalized_ticker,
        file=file,
        max_upload_size_bytes=MAX_UPLOAD_SIZE_BYTES,
    )


class SaveOverviewRequest(BaseModel):
    content: str = Field(..., max_length=MAX_UPLOAD_SIZE_BYTES)
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


@router.put("/overview/{ticker}")
def save_overview(ticker: str, body: SaveOverviewRequest, actor: ActorDep):
    """Stage an overview markdown save through the governed action path."""
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    content = body.content.strip()
    if not content:
        raise ValidationError("Overview content cannot be empty.")

    content = _normalize_overview_markdown(normalized_ticker, content)
    _sync_overview_content_paths()
    return stage_api_action(
        "save_overview_content",
        {"ticker": normalized_ticker, "content": content, "preserve_exact_content": True},
        source_id="overview.save_overview",
        actor=actor,
        reason=body.reason or f"Update equity overview for {normalized_ticker}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.get("/overview/{ticker}")
def get_overview(ticker: str):
    normalized_ticker = _normalize_ticker(ticker)
    _validate_ticker(normalized_ticker)

    if not _overview_exists(normalized_ticker):
        raise NotFoundError("Overview", normalized_ticker)
    try:
        content = _read_overview(normalized_ticker)
    except Exception as e:
        from api.exceptions import AppError

        raise AppError(f"Failed to read overview file: {e}") from e
    return {"status": "ok", "ticker": normalized_ticker, "content": content}
