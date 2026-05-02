"""
Industry earnings monitor:
- Read earnings call transcripts from local PDF files in macro/industry/files/
- Summarize with configured LLM provider (optional fallback if key/package is unavailable)
- Cache transcripts + summaries in SQLite
- Return structured data for frontend consumption
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional, TypedDict, cast

from dotenv import load_dotenv

from api import state_storage
from api.postgres import use_postgres_state
from api.postgres_compat import PostgresCompatConnection
from llm_utils import MODEL_MID, call_llm_text, has_llm_api_key, parse_json_text

LOGGER = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ---------- Config ----------
CompanyDef = tuple[str, str, str, str]


class SectorConfig(TypedDict):
    type: str
    companies: list[CompanyDef]


SECTORS: dict[str, SectorConfig] = {
    "Housing": {
        "type": "leading",
        "companies": [
            ("DHI", "D.R. Horton", "Homebuilder", "BMO"),
            ("LEN", "Lennar", "Homebuilder", "AMC"),
            ("NVR", "NVR", "Homebuilder", "BMO"),
            ("PHM", "PulteGroup", "Homebuilder", "BMO"),
            ("BLDR", "Builders FirstSource", "Building Materials", "BMO"),
            ("TOL", "Toll Brothers", "Homebuilder", "AMC"),
        ],
    },
    "Trucking": {
        "type": "leading",
        "companies": [
            ("ODFL", "Old Dominion Freight Line", "LTL", "AMC"),
            ("XPO", "XPO", "LTL", "BMO"),
            ("SAIA", "Saia", "LTL", "AMC"),
            ("ARCB", "ArcBest", "LTL", "BMO"),
            ("KNX", "Knight-Swift", "Truckload", "BMO"),
            ("SNDR", "Schneider", "Truckload", "BMO"),
            ("WERN", "Werner Enterprises", "Truckload", "BMO"),
            ("MRTN", "Marten Transport", "Truckload", "BMO"),
        ],
    },
    "Banks": {
        "type": "coincident",
        "companies": [
            ("JPM", "JPMorgan Chase", "Money Center", "BMO"),
            ("AXP", "American Express", "Card Issuer", "AMC"),
            ("C", "Citigroup", "Money Center", "BMO"),
            ("COF", "Capital One", "Card Issuer", "AMC"),
            ("BAC", "Bank of America", "Money Center", "BMO"),
        ],
    },
    "Retail": {
        "type": "coincident",
        "companies": [
            ("HD", "Home Depot", "Home Improvement", "BMO"),
            ("LOW", "Lowe's", "Home Improvement", "BMO"),
            ("DLTR", "Dollar Tree", "Discount", "BMO"),
            ("DG", "Dollar General", "Discount", "BMO"),
            ("WMT", "Walmart", "Big Box", "BMO"),
            ("TGT", "Target", "Big Box", "BMO"),
        ],
    },
    "Capital Goods": {
        "type": "lagging",
        "companies": [
            ("CAT", "Caterpillar", "Construction Machinery", "BMO"),
            ("DE", "Deere", "Agriculture Machinery", "BMO"),
            ("ETN", "Eaton", "Electrical Equipment", "BMO"),
            ("CMI", "Cummins", "Engines & Components", "BMO"),
        ],
    },
}

DB_PATH = "industry_transcripts.sqlite3"
SUMMARY_MODEL = MODEL_MID
SUMMARY_MAX_CHARS = int(os.environ.get("INDUSTRY_SUMMARY_MAX_CHARS", "32000"))
TRANSCRIPT_STALE_DAYS = int(os.environ.get("INDUSTRY_TRANSCRIPT_STALE_DAYS", "90"))


# ---------- Helpers ----------
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_db_path(db_path: str | None = None) -> str:
    if db_path:
        return db_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_PATH)


def _connect_db(db_path: str | None = None):
    if db_path is None and use_postgres_state():
        return PostgresCompatConnection(table_map={"transcripts": "industry_transcripts"})
    conn = sqlite3.connect(_resolve_db_path(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _make_id(ticker: str, year: int, quarter: int) -> str:
    return f"{ticker}_{year}_Q{quarter}"


def _extract_text_sample(text: str, max_words: int = 70) -> str:
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    words = [w for w in words if w]
    return " ".join(words[:max_words]).strip()


def _budget_text(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.45)
    middle = int(max_chars * 0.10)
    tail = max_chars - head - middle
    mid_start = max((len(text) - middle) // 2, 0)
    return text[:head] + "\n\n[...]\n\n" + text[mid_start : mid_start + middle] + "\n\n[...]\n\n" + text[-tail:]


def _sentiment_value(sentiment: str) -> int:
    v = str(sentiment or "").lower().strip()
    if v == "bullish":
        return 1
    if v == "bearish":
        return -1
    return 0


# ---------- PDF helpers ----------
_TICKER_FILENAME_MAP = {"ODFL": "ODL"}

INDUSTRY_TRANSCRIPTS_PREFIX = os.environ.get("INDUSTRY_TRANSCRIPTS_PREFIX", "industry-transcripts").strip("/")


def _sector_dir(sector: str) -> str:
    return sector.strip().lower().replace(" ", "_")


def _get_pdf_locator(sector: str, ticker: str) -> tuple[Path, str]:
    """Return (local_path, gcs_key) for a sector/ticker. Both forms apply _TICKER_FILENAME_MAP."""
    base = _TICKER_FILENAME_MAP.get(ticker, ticker)
    sector_dir = _sector_dir(sector)
    script_dir = Path(__file__).resolve().parent
    local_path = script_dir / "files" / sector_dir / f"{base}.pdf"
    gcs_key = f"{INDUSTRY_TRANSCRIPTS_PREFIX}/{sector_dir}/{base}.pdf"
    return local_path, gcs_key


def _extract_text_from_bytes(pdf_bytes: bytes) -> str:
    import logging

    from pdfminer.high_level import extract_text

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return extract_text(io.BytesIO(pdf_bytes)) or ""


def _sanitize_transcript_text(text: str) -> str:
    return (text or "").replace("\x00", "")


def _load_pdf_bytes(sector: str, ticker: str) -> tuple[bytes, datetime] | None:
    """Load PDF bytes + last-modified for sector/ticker. Returns None when missing."""
    local_path, gcs_key = _get_pdf_locator(sector, ticker)
    updated = state_storage.object_updated(local_path, gcs_key)
    if updated is None:
        return None
    pdf_bytes = state_storage.read_bytes(local_path, gcs_key)
    return pdf_bytes, updated


def _find_company(ticker: str) -> tuple[str, str, str, str, str] | None:
    normalized = (ticker or "").strip().upper()
    for sector, cfg in SECTORS.items():
        for company_ticker, company_name, sub_sector, report_time in cfg["companies"]:
            if company_ticker == normalized:
                return sector, company_ticker, company_name, sub_sector, report_time
    return None


def load_pdf_for_ticker(ticker: str) -> tuple[str, bytes] | None:
    """Return (download filename, PDF bytes) for a configured ticker."""
    found = _find_company(ticker)
    if found is None:
        return None
    sector, normalized, _company_name, _sub_sector, _report_time = found
    loaded = _load_pdf_bytes(sector, normalized)
    if loaded is None:
        return None
    local_path, _gcs_key = _get_pdf_locator(sector, normalized)
    return local_path.name, loaded[0]


_QUARTER_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
}

_MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = "|".join(sorted((re.escape(k) for k in _MONTH_ALIASES), key=len, reverse=True))
_QUARTER_CONTEXT_RE = re.compile(
    r"\b(earnings?|call|transcript|results?|reports?|announcement|presentation|release)\b",
    re.IGNORECASE,
)
_QUARTER_NOISE_RE = re.compile(
    r"\b(sequentially\s+from|compared\s+(?:to|with)|prior\s+quarter|previous\s+quarter|anticipated\s+to|expected\s+to)\b",
    re.IGNORECASE,
)


def _compact_spaced_token(match: re.Match[str]) -> str:
    return re.sub(r"\s+", "", match.group(0))


def _collapse_spaced_months(text: str) -> str:
    out = text
    for month in sorted(_MONTH_ALIASES, key=len, reverse=True):
        pattern = r"\b" + r"\s+".join(re.escape(ch) for ch in month) + r"\b"
        out = re.sub(pattern, month, out, flags=re.IGNORECASE)
    return out


def _canonicalize_header_text(text: str) -> str:
    out = (text or "").replace("\x00", " ")
    out = _collapse_spaced_months(out)
    out = re.sub(r"\b(?:\d\s+){1,3}\d\b", _compact_spaced_token, out)
    out = out.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    out = out.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", " ", out).strip()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_number(value: str) -> int | None:
    return _MONTH_ALIASES.get((value or "").strip().rstrip(".").lower())


def _normalize_year(value: str) -> int:
    year = int(value)
    if year < 100:
        year += 2000
    return year


def _date_candidates(header: str) -> list[tuple[int, date]]:
    candidates: list[tuple[int, date]] = []

    def add(start: int, year: int, month: int, day: int) -> None:
        parsed = _safe_date(year, month, day)
        if parsed is not None:
            candidates.append((start, parsed))

    for m in re.finditer(r"\b(20[2-3]\d)-(\d{1,2})-(\d{1,2})\b", header):
        add(m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3)))

    month_day_year = rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(20[2-3]\d)\b"
    for m in re.finditer(month_day_year, header, flags=re.IGNORECASE):
        month = _month_number(m.group(1))
        if month is not None:
            add(m.start(), int(m.group(3)), month, int(m.group(2)))

    day_month_year = rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*-\s*({_MONTH_PATTERN})\.?\s*-\s*(20[2-3]\d)\b"
    for m in re.finditer(day_month_year, header, flags=re.IGNORECASE):
        month = _month_number(m.group(2))
        if month is not None:
            add(m.start(), int(m.group(3)), month, int(m.group(1)))

    numeric_month_day_year = r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2}|20[2-3]\d)\b"
    for m in re.finditer(numeric_month_day_year, header):
        add(m.start(), _normalize_year(m.group(3)), int(m.group(1)), int(m.group(2)))

    return sorted(candidates, key=lambda item: item[0])


def _infer_reporting_period_from_date(call_date: date) -> tuple[int, int]:
    if call_date.month <= 3:
        return call_date.year - 1, 4
    if call_date.month <= 6:
        return call_date.year, 1
    if call_date.month <= 9:
        return call_date.year, 2
    return call_date.year, 3


def _infer_year_for_quarter(call_date: date, quarter: int) -> int:
    if quarter == 4 and call_date.month <= 3:
        return call_date.year - 1
    return call_date.year


def _quarter_score(header: str, start: int, end: int, has_year: bool) -> int | None:
    context = header[max(0, start - 140) : min(len(header), end + 160)]
    if not _QUARTER_CONTEXT_RE.search(context):
        return None
    score = 10
    if has_year:
        score += 2
    if start < 1200:
        score += 2
    if start < 300:
        score += 1
    if _QUARTER_NOISE_RE.search(context):
        return None
    if score < 8:
        return None
    return score


def _quarter_candidates(header: str, call_date: date) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []

    def add(start: int, end: int, quarter: int, year: int | None) -> None:
        score = _quarter_score(header, start, end, year is not None)
        if score is None:
            return
        candidates.append(
            (score, start, quarter, year if year is not None else _infer_year_for_quarter(call_date, quarter))
        )

    for m in re.finditer(r"\bQ([1-4])\s*(?:FY\s*)?(20[2-3]\d)\b", header, flags=re.IGNORECASE):
        add(m.start(), m.end(), int(m.group(1)), int(m.group(2)))

    for m in re.finditer(r"\b(20[2-3]\d)\s*Q([1-4])\b", header, flags=re.IGNORECASE):
        add(m.start(), m.end(), int(m.group(2)), int(m.group(1)))

    for m in re.finditer(r"\bQ([1-4])\s*[’']\s*(\d{2})\b", header, flags=re.IGNORECASE):
        add(m.start(), m.end(), int(m.group(1)), 2000 + int(m.group(2)))

    word_pattern = (
        r"\b(first|second|third|fourth)\s+quarter"
        r"(?:\s+(?:(?:and\s+)?(?:full[-\s]?year|fiscal\s+year)\s+|fiscal\s+|fy\s*|of\s+)?(20[2-3]\d))?\b"
    )
    for m in re.finditer(word_pattern, header, flags=re.IGNORECASE):
        quarter = _QUARTER_WORDS[m.group(1).lower()]
        year = int(m.group(2)) if m.group(2) else None
        add(m.start(), m.end(), quarter, year)

    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def _parse_period_from_text(text: str, fallback_dt: datetime) -> tuple[int, int, str]:
    header = _canonicalize_header_text(text[:5000])
    dates = _date_candidates(header)
    call_date = dates[0][1] if dates else fallback_dt.date()
    transcript_date = call_date.isoformat()

    quarter_matches = _quarter_candidates(header, call_date)
    if quarter_matches:
        _score, _start, quarter, year = quarter_matches[0]
        return year, quarter, transcript_date

    if dates:
        year, quarter = _infer_reporting_period_from_date(call_date)
        return year, quarter, transcript_date

    return fallback_dt.year, (fallback_dt.month - 1) // 3 + 1, transcript_date


# ---------- Storage ----------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            sector_type TEXT NOT NULL,
            sub_sector TEXT NOT NULL,
            quarter INTEGER NOT NULL,
            year INTEGER NOT NULL,
            transcript_text TEXT,
            content_sha256 TEXT,
            summary_json TEXT,
            fetched_at TEXT,
            summarized_at TEXT,
            transcript_date TEXT,
            is_stale INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_ticker ON transcripts(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_sector ON transcripts(sector)")
    conn.commit()
    # Migration: add price_reaction_2d column if it doesn't exist yet
    try:
        conn.execute("ALTER TABLE transcripts ADD COLUMN price_reaction_2d REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists


def _get_row_by_id(conn: sqlite3.Connection, row_id: str) -> sqlite3.Row | None:
    return cast(sqlite3.Row | None, conn.execute("SELECT * FROM transcripts WHERE id=?", (row_id,)).fetchone())


def _get_rows_by_content_sha(conn: sqlite3.Connection, ticker: str, content_sha256: str) -> list[sqlite3.Row]:
    return cast(
        list[sqlite3.Row],
        conn.execute(
            """
        SELECT * FROM transcripts
        WHERE ticker=? AND content_sha256=? AND summary_json IS NOT NULL
        """,
            (ticker, content_sha256),
        ).fetchall(),
    )


def _get_latest_row_for_ticker(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return cast(
        sqlite3.Row | None,
        conn.execute(
            """
        SELECT * FROM transcripts
        WHERE ticker=? AND COALESCE(transcript_text, '') != ''
        ORDER BY
            CASE WHEN is_stale THEN 1 ELSE 0 END ASC,
            COALESCE(transcript_date, '') DESC,
            year DESC,
            quarter DESC
        LIMIT 1
        """,
            (ticker,),
        ).fetchone(),
    )


def _set_fresh_row(conn: sqlite3.Connection, ticker: str, fresh_row_id: str | None) -> None:
    conn.execute("UPDATE transcripts SET is_stale=? WHERE ticker=?", (True, ticker))
    if fresh_row_id:
        conn.execute("UPDATE transcripts SET is_stale=? WHERE id=?", (False, fresh_row_id))
    conn.commit()


def _upsert_transcript(
    conn: sqlite3.Connection,
    *,
    row_id: str,
    ticker: str,
    company_name: str,
    sector: str,
    sector_type: str,
    sub_sector: str,
    year: int,
    quarter: int,
    transcript_text: str,
    transcript_date: str,
    content_sha256: str,
    fetched_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO transcripts (
            id, ticker, company_name, sector, sector_type, sub_sector, quarter, year,
            transcript_text, content_sha256, fetched_at, transcript_date, is_stale
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ticker=excluded.ticker,
            company_name=excluded.company_name,
            sector=excluded.sector,
            sector_type=excluded.sector_type,
            sub_sector=excluded.sub_sector,
            quarter=excluded.quarter,
            year=excluded.year,
            transcript_text=excluded.transcript_text,
            content_sha256=excluded.content_sha256,
            fetched_at=excluded.fetched_at,
            transcript_date=excluded.transcript_date,
            price_reaction_2d=CASE
                WHEN COALESCE(transcripts.transcript_date, '') != COALESCE(excluded.transcript_date, '')
                THEN NULL
                ELSE transcripts.price_reaction_2d
            END,
            is_stale=excluded.is_stale
        """,
        (
            row_id,
            ticker,
            company_name,
            sector,
            sector_type,
            sub_sector,
            quarter,
            year,
            transcript_text,
            content_sha256,
            fetched_at,
            transcript_date,
            False,
        ),
    )
    conn.commit()


def _set_summary(conn: sqlite3.Connection, row_id: str, summary: dict) -> None:
    conn.execute(
        """
        UPDATE transcripts
        SET summary_json=?, summarized_at=?, is_stale=?
        WHERE id=?
        """,
        (json.dumps(summary, ensure_ascii=False), _now_iso(), False, row_id),
    )
    conn.commit()


def _summary_looks_like_fallback(summary_json: str | None, company_name: str) -> bool:
    if not summary_json:
        return False
    try:
        summary = json.loads(summary_json)
    except Exception:
        return False
    headline = str(summary.get("summary_headline") or "").strip()
    return headline == f"{company_name} commentary is mixed; monitor demand and guidance closely."


def _choose_reusable_summary_row(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    company_name: str,
    content_sha256: str,
    target_row_id: str,
) -> sqlite3.Row | None:
    rows = _get_rows_by_content_sha(conn, ticker, content_sha256)
    if not rows:
        return None

    def sort_key(row: sqlite3.Row) -> tuple[int, str]:
        same_row = row["id"] == target_row_id
        fallback = _summary_looks_like_fallback(row["summary_json"], company_name)
        priority = 2 if same_row else 1 if fallback else 0
        return priority, str(row["summarized_at"] or "")

    return sorted(rows, key=sort_key)[0]


def _copy_summary(conn: sqlite3.Connection, *, source_row: sqlite3.Row, target_row_id: str) -> None:
    conn.execute(
        """
        UPDATE transcripts
        SET summary_json=?, summarized_at=?
        WHERE id=?
        """,
        (source_row["summary_json"], source_row["summarized_at"], target_row_id),
    )
    conn.commit()


def _today_utc() -> date:
    return datetime.now(UTC).date()


def _transcript_age_days(transcript_date: str) -> int | None:
    raw = (transcript_date or "").strip()
    if not raw:
        return None
    try:
        call_date = date.fromisoformat(raw[:10])
    except ValueError:
        return None
    return max((_today_utc() - call_date).days, 0)


# ---------- Price reaction ----------
def _fetch_price_reaction(ticker: str, transcript_date: str, report_time: str = "BMO") -> float | None:
    """Return 2-trading-day post-earnings price change (%).

    BMO (before market open): reaction starts on transcript_date itself.
        entry = close of last trading day *before* transcript_date (D-1)
        exit  = close 2 trading days later (D+1)
    AMC (after market close): reaction starts the morning *after* transcript_date.
        entry = close of transcript_date (D)
        exit  = close 2 trading days later (D+2)
    Returns None if data is unavailable or exit date is in the future.
    """
    from datetime import timedelta

    from utils.retry import yf_download

    try:
        dt = datetime.strptime(transcript_date, "%Y-%m-%d")
        start = (dt - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=10)).strftime("%Y-%m-%d")

        data = yf_download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if data.empty:
            return None

        close = data["Close"].dropna()
        if hasattr(close.columns, "__len__"):
            # Multi-ticker download — squeeze to Series
            close = close.squeeze()
        dates = [d.date() for d in close.index]

        if report_time == "BMO":
            # Entry: last trading day strictly before transcript_date
            entry_idx = max(
                (i for i, d in enumerate(dates) if d < dt.date()),
                default=None,
            )
        else:
            # AMC — Entry: last trading day on or before transcript_date
            entry_idx = max(
                (i for i, d in enumerate(dates) if d <= dt.date()),
                default=None,
            )

        if entry_idx is None or entry_idx + 2 >= len(dates):
            return None

        entry_price = float(close.iloc[entry_idx])
        exit_price = float(close.iloc[entry_idx + 2])
        if entry_price == 0:
            return None
        return (exit_price - entry_price) / entry_price * 100
    except Exception as ex:
        LOGGER.warning("Price reaction fetch failed for %s: %s", ticker, ex)
        return None


def _set_price_reaction(conn: sqlite3.Connection, row_id: str, value: float | None) -> None:
    conn.execute(
        "UPDATE transcripts SET price_reaction_2d=? WHERE id=?",
        (value, row_id),
    )
    conn.commit()


# Build a ticker → report_time lookup from SECTORS config
_TICKER_REPORT_TIME: dict[str, str] = {
    ticker: report_time for cfg in SECTORS.values() for ticker, _, _, report_time in cfg["companies"]
}


def _fetch_missing_price_reactions(conn: sqlite3.Connection) -> None:
    """Fetch and store price reactions for rows that have a summary but no reaction yet."""
    rows = conn.execute(
        "SELECT id, ticker, transcript_date FROM transcripts "
        "WHERE summary_json IS NOT NULL AND price_reaction_2d IS NULL"
    ).fetchall()

    if not rows:
        return

    LOGGER.info("Fetching price reactions for %d transcript(s)...", len(rows))
    for row in rows:
        row_id = row["id"]
        ticker = row["ticker"]
        transcript_date = row["transcript_date"] or ""
        if not transcript_date:
            continue
        report_time = _TICKER_REPORT_TIME.get(ticker, "BMO")
        value = _fetch_price_reaction(ticker, transcript_date, report_time)
        _set_price_reaction(conn, row_id, value)


# ---------- Summarization ----------
def _fallback_summary(text: str, meta: dict) -> dict:
    sample = _extract_text_sample(text, max_words=90)
    headline = f"{meta['company_name']} commentary is mixed; monitor demand and guidance closely."
    demand = "Demand commentary was mixed in the latest discussion."
    pricing = "Pricing commentary was mixed or not explicitly quantified."
    guidance = "Guidance tone was cautious to neutral."
    bullets = []
    if sample:
        bullets.append(sample + "...")

    return {
        "summary_headline": headline,
        "sentiment": "neutral",
        "business_conditions": bullets or ["No detailed transcript text available."],
        "demand_trends": demand,
        "pricing_commentary": pricing,
        "guidance_outlook": guidance,
        "macro_quotes": [],
    }


def _normalize_summary(summary: dict, text: str, meta: dict) -> dict:
    fallback = _fallback_summary(text, meta)
    out = dict(fallback)

    if isinstance(summary, dict):
        out["summary_headline"] = str(summary.get("summary_headline") or fallback["summary_headline"]).strip()

        sentiment = str(summary.get("sentiment") or "").strip().lower()
        if sentiment not in {"bullish", "neutral", "bearish"}:
            sentiment = "neutral"
        out["sentiment"] = sentiment

        raw_conditions = summary.get("business_conditions")
        if isinstance(raw_conditions, list):
            conds = [str(x).strip() for x in raw_conditions if str(x).strip()]
            out["business_conditions"] = conds[:6] if conds else fallback["business_conditions"]

        for key in ("demand_trends", "pricing_commentary", "guidance_outlook"):
            val = str(summary.get(key) or "").strip()
            if val:
                out[key] = val

        raw_quotes = summary.get("macro_quotes")
        if isinstance(raw_quotes, list):
            quotes = [str(x).strip() for x in raw_quotes if str(x).strip()]
            out["macro_quotes"] = quotes[:4]

    return out


def summarize_with_claude(text: str, meta: dict) -> dict:
    text_in = _budget_text(text)
    prompt = f"""
You are an analyst extracting macro signals from one earnings call transcript.

Return STRICT JSON:
{{
  "summary_headline": "...",
  "sentiment": "bullish|neutral|bearish",
  "business_conditions": ["...", "...", "..."],
  "demand_trends": "...",
  "pricing_commentary": "...",
  "guidance_outlook": "...",
  "macro_quotes": ["...", "..."]
}}

Rules:
- Keep each string concise and specific.
- Use only evidence from the transcript.
- sentiment must be one of bullish, neutral, bearish.
- macro_quotes should be short, high-signal excerpts (paraphrase if needed).

Company: {meta["company_name"]} ({meta["ticker"]})
Sector: {meta["sector"]} ({meta["sector_type"]})
Sub-sector: {meta["sub_sector"]}
Call Date: {meta.get("transcript_date") or "Unknown"}

Transcript:
{text_in}
""".strip()

    output_text, _citations, _resp = call_llm_text(
        prompt=prompt,
        model=SUMMARY_MODEL,
        api_key=None,
        max_tokens=2048,
    )
    if not output_text:
        raise ValueError("LLM returned empty response")
    parsed = parse_json_text(output_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned invalid JSON")
    return _normalize_summary(parsed, text, meta)


def summarize_with_llm(text: str, meta: dict) -> dict:
    if has_llm_api_key():
        try:
            return summarize_with_claude(text, meta)
        except Exception as ex:
            LOGGER.warning("LLM summarization failed for %s: %s", meta["ticker"], ex)
    return _fallback_summary(text, meta)


# ---------- Aggregation ----------
def _aggregate_sector(sector: str, sector_type: str, companies: list[dict]) -> dict:
    available = [c for c in companies if not c.get("missing_data")]
    if not available:
        return {
            "sector_headline": f"No transcript data cached yet for {sector}.",
            "key_themes": [],
            "economic_signal": "stable",
            "fresh_companies": 0,
            "total_companies": len(companies),
        }

    fresh = [c for c in available if not c.get("is_stale")]
    used = fresh if fresh else available
    avg = sum(_sentiment_value(c.get("sentiment", "neutral")) for c in used) / max(len(used), 1)

    if avg >= 0.5:
        signal = "expanding"
    elif avg >= -0.15:
        signal = "stable"
    elif avg >= -0.6:
        signal = "slowing"
    else:
        signal = "contracting"

    themes: list[str] = []
    for c in used:
        for bullet in c.get("business_conditions", []):
            b = str(bullet).strip()
            if b and b not in themes:
                themes.append(b)
            if len(themes) >= 6:
                break
        if len(themes) >= 6:
            break

    headline = (
        f"{sector} ({sector_type}) currently reads as {signal} based on {len(used)} company transcript summaries."
    )

    return {
        "sector_headline": headline,
        "key_themes": themes,
        "economic_signal": signal,
        "fresh_companies": len(fresh),
        "total_companies": len(companies),
    }


def _company_from_row(
    row: sqlite3.Row | None,
    *,
    ticker: str,
    company_name: str,
    sector: str,
    sector_type: str,
    sub_sector: str,
) -> dict:
    if row is None:
        return {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "sector_type": sector_type,
            "sub_sector": sub_sector,
            "quarter": None,
            "year": None,
            "call_date": "",
            "transcript_date": "",
            "summary_headline": "No transcript cached yet.",
            "sentiment": "neutral",
            "business_conditions": [],
            "demand_trends": "",
            "pricing_commentary": "",
            "guidance_outlook": "",
            "macro_quotes": [],
            "price_reaction_2d": None,
            "age_days": None,
            "is_stale": True,
            "missing_data": True,
        }

    text = row["transcript_text"] or ""
    meta = {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "sector_type": sector_type,
        "sub_sector": sub_sector,
        "quarter": row["quarter"],
        "year": row["year"],
        "transcript_date": row["transcript_date"] or "",
    }
    raw_summary = {}
    if row["summary_json"]:
        try:
            raw_summary = json.loads(row["summary_json"])
        except Exception:
            raw_summary = {}
    summary = _normalize_summary(raw_summary, text, meta)
    age_days = _transcript_age_days(row["transcript_date"] or "")
    is_age_stale = age_days is None or age_days > TRANSCRIPT_STALE_DAYS

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "sector_type": sector_type,
        "sub_sector": sub_sector,
        "quarter": row["quarter"],
        "year": row["year"],
        "call_date": row["transcript_date"] or "",
        "transcript_date": row["transcript_date"] or "",
        "summary_headline": summary["summary_headline"],
        "sentiment": summary["sentiment"],
        "business_conditions": summary["business_conditions"],
        "demand_trends": summary["demand_trends"],
        "pricing_commentary": summary["pricing_commentary"],
        "guidance_outlook": summary["guidance_outlook"],
        "macro_quotes": summary["macro_quotes"],
        "price_reaction_2d": (float(row["price_reaction_2d"]) if row["price_reaction_2d"] is not None else None),
        "age_days": age_days,
        "is_stale": bool(row["is_stale"]) or is_age_stale,
        "missing_data": False,
    }


# ---------- Main ----------
def _fetch_and_store(conn: sqlite3.Connection) -> None:
    # Phase 1: Extract PDFs and upsert transcripts (fast, sequential)
    to_summarize: list[tuple[str, str, dict]] = []  # (row_id, text, meta)

    for sector, cfg in SECTORS.items():
        sector_type = cfg["type"]
        for ticker, company_name, sub_sector, report_time in cfg["companies"]:  # noqa: B007
            try:
                loaded = _load_pdf_bytes(sector, ticker)
                if loaded is None:
                    LOGGER.warning("PDF not found for %s in sector %s", ticker, sector)
                    _set_fresh_row(conn, ticker, None)
                    continue

                pdf_bytes, fallback_dt = loaded

                try:
                    transcript_text = _extract_text_from_bytes(pdf_bytes)
                except Exception as ex:
                    LOGGER.warning("Failed to extract text from PDF for %s: %s", ticker, ex)
                    _set_fresh_row(conn, ticker, None)
                    continue
                transcript_text = _sanitize_transcript_text(transcript_text)

                if not transcript_text.strip():
                    LOGGER.warning("No text extracted from PDF for %s", ticker)
                    _set_fresh_row(conn, ticker, None)
                    continue

                try:
                    year, quarter, transcript_date = _parse_period_from_text(transcript_text, fallback_dt)
                except Exception as ex:
                    LOGGER.warning("Failed to parse period from PDF for %s: %s", ticker, ex)
                    _set_fresh_row(conn, ticker, None)
                    continue

                sha = hashlib.sha256(transcript_text.encode("utf-8", errors="ignore")).hexdigest()

                row_id = _make_id(ticker, year, quarter)
                existing = _get_row_by_id(conn, row_id)
                reusable_summary = _choose_reusable_summary_row(
                    conn,
                    ticker=ticker,
                    company_name=company_name,
                    content_sha256=sha,
                    target_row_id=row_id,
                )

                now_iso = _now_iso()
                _upsert_transcript(
                    conn,
                    row_id=row_id,
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    sector_type=sector_type,
                    sub_sector=sub_sector,
                    year=year,
                    quarter=quarter,
                    transcript_text=transcript_text,
                    transcript_date=transcript_date,
                    content_sha256=sha,
                    fetched_at=now_iso,
                )
                if reusable_summary is not None and reusable_summary["id"] != row_id:
                    _copy_summary(conn, source_row=reusable_summary, target_row_id=row_id)
                _set_fresh_row(conn, ticker, row_id)

                if existing and existing["content_sha256"] == sha and existing["summary_json"]:
                    continue
                if reusable_summary is not None:
                    continue

                meta = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector,
                    "sector_type": sector_type,
                    "sub_sector": sub_sector,
                    "quarter": quarter,
                    "year": year,
                    "transcript_date": transcript_date,
                }
                to_summarize.append((row_id, transcript_text, meta))
            except Exception as ex:
                LOGGER.exception("Unexpected error processing %s in sector %s: %s", ticker, sector, ex)
                _set_fresh_row(conn, ticker, None)

    # Phase 2: Summarize via LLM in parallel
    if not to_summarize:
        LOGGER.info("[INFO] Industry data fetch complete — all transcripts up to date, no new summaries needed.")
        return

    def _do_summarize(item: tuple[str, str, dict]) -> tuple[str, dict]:
        row_id, text, meta = item
        return row_id, summarize_with_llm(text, meta)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_do_summarize, item): item for item in to_summarize}
        for future in as_completed(futures):
            row_id, summary = future.result()
            _set_summary(conn, row_id, summary)

    LOGGER.info("Industry data fetch and summarization complete — %d transcript(s) summarized.", len(to_summarize))


def _query_data(conn: sqlite3.Connection) -> tuple[dict, list, dict]:
    by_sector: dict[str, dict] = {}
    sectors: list[dict] = []
    total_companies = 0
    total_fresh = 0
    total_stale = 0

    for sector, cfg in SECTORS.items():
        sector_type = cfg["type"]
        companies_out = []

        for ticker, company_name, sub_sector, _report_time in cfg["companies"]:
            row = _get_latest_row_for_ticker(conn, ticker)
            item = _company_from_row(
                row,
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                sector_type=sector_type,
                sub_sector=sub_sector,
            )
            companies_out.append(item)
            total_companies += 1
            if item["missing_data"] or item["is_stale"]:
                total_stale += 1
            else:
                total_fresh += 1

        sector_summary = _aggregate_sector(sector, sector_type, companies_out)
        by_sector[sector] = {
            "type": sector_type,
            "sector_summary": sector_summary,
            "companies": companies_out,
        }
        sectors.append(
            {
                "name": sector,
                "type": sector_type,
                "count": len(companies_out),
                "fresh": sector_summary["fresh_companies"],
            }
        )

    counts = {
        "total_companies": total_companies,
        "fresh_companies": total_fresh,
        "stale_or_missing_companies": total_stale,
    }
    return by_sector, sectors, counts


def get_data(db_path: str | None = None, refresh: bool = False) -> dict:
    conn = None
    try:
        conn = _connect_db(db_path)
        init_db(conn)
        if refresh:
            _fetch_and_store(conn)
        _fetch_missing_price_reactions(conn)
        by_sector, sectors, counts = _query_data(conn)
        return {
            "by_sector": by_sector,
            "sectors": sectors,
            "counts": counts,
            "last_updated": _now_iso(),
        }
    except Exception as ex:
        return {"error": str(ex)}
    finally:
        if conn is not None:
            conn.close()


def run() -> None:
    data = get_data()
    if "error" in data:
        LOGGER.error("Error: %s", data["error"])
        return

    for sector in SECTORS.keys():
        sec = data["by_sector"].get(sector, {})
        summary = sec.get("sector_summary", {})
        signal = summary.get("economic_signal", "stable")
        headline = summary.get("sector_headline", "")
        print(f"\n{sector}: {signal.upper()}")
        if headline:
            print(f"  {headline}")

        for company in sec.get("companies", []):
            td = company.get("transcript_date") or ""
            try:
                date_label = datetime.strptime(td, "%Y-%m-%d").strftime("%b %-d, %Y")
            except ValueError:
                date_label = "N/A"
            stale = " (stale/missing)" if company.get("is_stale") or company.get("missing_data") else ""
            print(
                f"  - {company['ticker']} [{date_label}] {company['sentiment']}: {company['summary_headline']}{stale}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Starting script execution: %s", __file__)
    run()
