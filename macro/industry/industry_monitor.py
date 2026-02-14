"""
Industry earnings monitor:
- Fetch latest earnings call transcripts from Financial Modeling Prep (FMP)
- Summarize with OpenAI (optional fallback if key/package is unavailable)
- Cache transcripts + summaries in SQLite
- Return structured data for Streamlit GUI consumption
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

# ---------- Config ----------
SECTORS = {
    "Housing": {
        "type": "leading",
        "companies": [
            ("DHI", "D.R. Horton", "Homebuilder"),
            ("LEN", "Lennar", "Homebuilder"),
            ("NVR", "NVR", "Homebuilder"),
            ("PHM", "PulteGroup", "Homebuilder"),
            ("BLDR", "Builders FirstSource", "Building Materials"),
            ("TOL", "Toll Brothers", "Homebuilder"),
        ],
    },
    "Trucking": {
        "type": "leading",
        "companies": [
            ("ODFL", "Old Dominion Freight Line", "LTL"),
            ("XPO", "XPO", "LTL"),
            ("SAIA", "Saia", "LTL"),
            ("ARCB", "ArcBest", "LTL"),
            ("KNX", "Knight-Swift", "Truckload"),
            ("SNDR", "Schneider", "Truckload"),
            ("WERN", "Werner Enterprises", "Truckload"),
            ("MRTN", "Marten Transport", "Truckload"),
        ],
    },
    "Banks": {
        "type": "coincident",
        "companies": [
            ("JPM", "JPMorgan Chase", "Money Center"),
            ("AXP", "American Express", "Card Issuer"),
            ("C", "Citigroup", "Money Center"),
            ("COF", "Capital One", "Card Issuer"),
            ("BAC", "Bank of America", "Money Center"),
        ],
    },
    "Retail": {
        "type": "coincident",
        "companies": [
            ("HD", "Home Depot", "Home Improvement"),
            ("LOW", "Lowe's", "Home Improvement"),
            ("DLTR", "Dollar Tree", "Discount"),
            ("DG", "Dollar General", "Discount"),
            ("WMT", "Walmart", "Big Box"),
            ("TGT", "Target", "Big Box"),
        ],
    },
}

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
DB_PATH = "industry_transcripts.sqlite3"
SUMMARY_MODEL = os.environ.get("INDUSTRY_SUMMARY_MODEL", "gpt-4.1-mini")
SUMMARY_MAX_CHARS = int(os.environ.get("INDUSTRY_SUMMARY_MAX_CHARS", "32000"))


# ---------- Data model ----------
@dataclass(frozen=True)
class TranscriptPeriod:
    ticker: str
    year: int
    quarter: int
    transcript_date: str


# ---------- Helpers ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    if db_path:
        return db_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_PATH)


def _make_id(ticker: str, year: int, quarter: int) -> str:
    return f"{ticker}_{year}_Q{quarter}"


def _coerce_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _parse_quarter(value) -> Optional[int]:
    q = _coerce_int(value)
    if q in (1, 2, 3, 4):
        return q
    if value is None:
        return None
    m = re.search(r"([1-4])", str(value).upper())
    if not m:
        return None
    return int(m.group(1))


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
    return (
        text[:head]
        + "\n\n[...]\n\n"
        + text[mid_start : mid_start + middle]
        + "\n\n[...]\n\n"
        + text[-tail:]
    )


def _sentiment_value(sentiment: str) -> int:
    v = str(sentiment or "").lower().strip()
    if v == "bullish":
        return 1
    if v == "bearish":
        return -1
    return 0


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


def _get_row_by_id(conn: sqlite3.Connection, row_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM transcripts WHERE id=?", (row_id,)).fetchone()


def _get_latest_row_for_ticker(conn: sqlite3.Connection, ticker: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM transcripts
        WHERE ticker=?
        ORDER BY year DESC, quarter DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()


def _set_fresh_row(conn: sqlite3.Connection, ticker: str, fresh_row_id: Optional[str]) -> None:
    conn.execute("UPDATE transcripts SET is_stale=1 WHERE ticker=?", (ticker,))
    if fresh_row_id:
        conn.execute("UPDATE transcripts SET is_stale=0 WHERE id=?", (fresh_row_id,))
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
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
            is_stale=0
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
        ),
    )
    conn.commit()


def _set_summary(conn: sqlite3.Connection, row_id: str, summary: dict) -> None:
    conn.execute(
        """
        UPDATE transcripts
        SET summary_json=?, summarized_at=?, is_stale=0
        WHERE id=?
        """,
        (json.dumps(summary, ensure_ascii=False), _now_iso(), row_id),
    )
    conn.commit()


# ---------- FMP ----------
def _fmp_get(client: httpx.Client, path: str, params: dict) -> list | dict:
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set")

    payload = dict(params)
    payload["apikey"] = api_key
    url = f"{FMP_BASE_URL}/{path.lstrip('/')}"
    resp = client.get(url, params=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        if data.get("Error Message"):
            raise RuntimeError(str(data.get("Error Message")))
        if data.get("error"):
            raise RuntimeError(str(data.get("error")))
    return data


def _parse_period_row(ticker: str, row: dict) -> Optional[TranscriptPeriod]:
    year = _coerce_int(row.get("year") or row.get("fiscalYear") or row.get("calendarYear"))
    quarter = _parse_quarter(row.get("quarter") or row.get("fiscalQuarter"))
    if year is None or quarter is None:
        return None

    transcript_date = str(
        row.get("date")
        or row.get("fillingDate")
        or row.get("acceptedDate")
        or row.get("publishedDate")
        or ""
    ).strip()

    return TranscriptPeriod(
        ticker=ticker,
        year=year,
        quarter=quarter,
        transcript_date=transcript_date,
    )


def fetch_latest_period(client: httpx.Client, ticker: str) -> Optional[TranscriptPeriod]:
    raw = _fmp_get(client, "earning-call-transcript-dates", {"symbol": ticker})
    if isinstance(raw, dict):
        rows = [raw]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    periods: list[TranscriptPeriod] = []
    for row in rows:
        if isinstance(row, dict):
            p = _parse_period_row(ticker, row)
            if p:
                periods.append(p)
    if not periods:
        return None

    periods.sort(key=lambda p: (p.year, p.quarter, p.transcript_date))
    return periods[-1]


def fetch_transcript_text(
    client: httpx.Client, ticker: str, year: int, quarter: int
) -> tuple[str, str]:
    raw = _fmp_get(
        client,
        "earning-call-transcript",
        {"symbol": ticker, "year": year, "quarter": quarter},
    )
    if isinstance(raw, dict):
        rows = [raw]
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    transcript_date = ""
    text_keys = (
        "content",
        "transcript",
        "text",
        "preparedRemarks",
        "prepared_remarks",
    )
    date_keys = ("date", "fillingDate", "acceptedDate", "publishedDate")

    for row in rows:
        if not isinstance(row, dict):
            continue
        if not transcript_date:
            for key in date_keys:
                val = str(row.get(key) or "").strip()
                if val:
                    transcript_date = val
                    break
        for key in text_keys:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip(), transcript_date

    return "", transcript_date


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


def summarize_with_openai(text: str, meta: dict) -> dict:
    from openai import OpenAI

    client = OpenAI()
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
Quarter: Q{meta["quarter"]} {meta["year"]}

Transcript:
{text_in}
""".strip()

    resp = client.responses.create(
        model=SUMMARY_MODEL,
        input=prompt,
    )

    out = (resp.output_text or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    if not out:
        raise ValueError("OpenAI returned empty response")
    parsed = json.loads(out)
    return _normalize_summary(parsed, text, meta)


def summarize_with_llm(text: str, meta: dict) -> dict:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return summarize_with_openai(text, meta)
        except Exception as ex:
            print(f"[WARN] OpenAI summarization failed for {meta['ticker']}: {ex}")
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
        f"{sector} ({sector_type}) currently reads as {signal} "
        f"based on {len(used)} company transcript summaries."
    )

    return {
        "sector_headline": headline,
        "key_themes": themes,
        "economic_signal": signal,
        "fresh_companies": len(fresh),
        "total_companies": len(companies),
    }


def _company_from_row(
    row: Optional[sqlite3.Row],
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
            "transcript_date": "",
            "summary_headline": "No transcript cached yet.",
            "sentiment": "neutral",
            "business_conditions": [],
            "demand_trends": "",
            "pricing_commentary": "",
            "guidance_outlook": "",
            "macro_quotes": [],
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
    }
    raw_summary = {}
    if row["summary_json"]:
        try:
            raw_summary = json.loads(row["summary_json"])
        except Exception:
            raw_summary = {}
    summary = _normalize_summary(raw_summary, text, meta)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "sector_type": sector_type,
        "sub_sector": sub_sector,
        "quarter": row["quarter"],
        "year": row["year"],
        "transcript_date": row["transcript_date"] or "",
        "summary_headline": summary["summary_headline"],
        "sentiment": summary["sentiment"],
        "business_conditions": summary["business_conditions"],
        "demand_trends": summary["demand_trends"],
        "pricing_commentary": summary["pricing_commentary"],
        "guidance_outlook": summary["guidance_outlook"],
        "macro_quotes": summary["macro_quotes"],
        "is_stale": bool(row["is_stale"]),
        "missing_data": False,
    }


# ---------- Main ----------
def _fetch_and_store(conn: sqlite3.Connection) -> None:
    headers = {"User-Agent": "industry-monitor/0.1 (+local)"}
    with httpx.Client(headers=headers) as client:
        for sector, cfg in SECTORS.items():
            sector_type = cfg["type"]
            for ticker, company_name, sub_sector in cfg["companies"]:
                latest = None
                try:
                    latest = fetch_latest_period(client, ticker)
                except Exception as ex:
                    print(f"[WARN] Failed to fetch transcript dates for {ticker}: {ex}")

                if latest is None:
                    _set_fresh_row(conn, ticker, None)
                    continue

                row_id = _make_id(ticker, latest.year, latest.quarter)
                existing = _get_row_by_id(conn, row_id)

                if existing and existing["transcript_text"] and existing["summary_json"]:
                    _set_fresh_row(conn, ticker, row_id)
                    continue

                transcript_text = ""
                transcript_date = latest.transcript_date
                try:
                    transcript_text, fetched_date = fetch_transcript_text(
                        client,
                        ticker,
                        latest.year,
                        latest.quarter,
                    )
                    if fetched_date:
                        transcript_date = fetched_date
                except Exception as ex:
                    print(f"[WARN] Failed to fetch transcript for {ticker}: {ex}")

                if not transcript_text.strip():
                    _set_fresh_row(conn, ticker, None)
                    continue

                sha = hashlib.sha256(
                    transcript_text.encode("utf-8", errors="ignore")
                ).hexdigest()
                now_iso = _now_iso()

                _upsert_transcript(
                    conn,
                    row_id=row_id,
                    ticker=ticker,
                    company_name=company_name,
                    sector=sector,
                    sector_type=sector_type,
                    sub_sector=sub_sector,
                    year=latest.year,
                    quarter=latest.quarter,
                    transcript_text=transcript_text,
                    transcript_date=transcript_date,
                    content_sha256=sha,
                    fetched_at=now_iso,
                )
                _set_fresh_row(conn, ticker, row_id)

                if existing and existing["content_sha256"] == sha and existing["summary_json"]:
                    continue

                summary = summarize_with_llm(
                    transcript_text,
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "sector": sector,
                        "sector_type": sector_type,
                        "sub_sector": sub_sector,
                        "quarter": latest.quarter,
                        "year": latest.year,
                    },
                )
                _set_summary(conn, row_id, summary)


def _query_data(conn: sqlite3.Connection) -> tuple[dict, list, dict]:
    by_sector: dict[str, dict] = {}
    sectors: list[dict] = []
    total_companies = 0
    total_fresh = 0
    total_stale = 0

    for sector, cfg in SECTORS.items():
        sector_type = cfg["type"]
        companies_out = []

        for ticker, company_name, sub_sector in cfg["companies"]:
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


def get_data(db_path: str = None) -> dict:
    db_path = _resolve_db_path(db_path)
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        init_db(conn)
        _fetch_and_store(conn)
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
        print(f"ERROR: {data['error']}")
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
            q = company.get("quarter")
            y = company.get("year")
            qy = f"Q{q} {y}" if q and y else "N/A"
            stale = " (stale/missing)" if company.get("is_stale") or company.get("missing_data") else ""
            print(
                f"  - {company['ticker']} [{qy}] {company['sentiment']}: "
                f"{company['summary_headline']}{stale}"
            )


if __name__ == "__main__":
    run()
