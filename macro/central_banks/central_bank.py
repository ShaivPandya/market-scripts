"""
pip install feedparser httpx beautifulsoup4 lxml readability-lxml pdfminer.six python-dotenv
Optional (OpenAI): pip install openai
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import feedparser
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from readability import Document

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

# ---------- Config ----------
FEEDS = {
    "ECB": [
        "https://www.ecb.europa.eu/rss/press.html",
    ],
    "FED": [
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "https://www.federalreserve.gov/feeds/press_other.xml",
    ],
    "BOJ": [
        "https://www.boj.or.jp/en/rss/whatsnew.xml",
    ],
    "BOE": [
        "https://www.bankofengland.co.uk/rss/news",
        "https://www.bankofengland.co.uk/rss/publications",
    ],
    "BOC": [
        "https://www.bankofcanada.ca/content_type/press-releases/feed/",
    ],
    "SNB": [
        "https://www.snb.ch/public/en/rss/pressrel",
        "https://www.snb.ch/public/en/rss/mopo",
    ],
    "NORGES": [
        "https://www.norges-bank.no/en/rss-feeds/Press-releases---Norges-Bank/",
    ],
    "RBA": [
        "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
    ],
    "RBNZ": [
        "https://www.rbnz.govt.nz/feeds/news",
    ],
    "RIKSBANK": [
        "https://www.riksbank.se/sv/rss/pressmeddelanden/",
    ],
}

DEFAULT_SOURCES = ["FED", "ECB", "BOJ", "BOE"]

# Heuristic classifiers (tune over time)
CLASSIFIERS = [
    ("FED", "FOMC statement", re.compile(r"\bFOMC statement\b", re.I)),
    ("FED", "FOMC minutes (press release)", re.compile(r"\bMinutes of the Federal Open Market Committee\b", re.I)),
    ("FED", "FOMC Economic Projections (SEP)", re.compile(r"\beconomic projections\b", re.I)),
    ("FED", "Beige Book", re.compile(r"\bBeige Book\b", re.I)),
    ("ECB", "Monetary policy decisions", re.compile(r"\bMonetary policy decisions?\b", re.I)),
    ("ECB", "Monetary policy statement", re.compile(r"\bMonetary policy statement\b", re.I)),
    ("ECB", "Economic Bulletin", re.compile(r"\bEconomic Bulletin\b", re.I)),
    ("ECB", "Macroeconomic projections", re.compile(r"\bmacroeconomic projections\b", re.I)),
    ("BOE", "Monetary Policy Summary and Minutes", re.compile(r"\bMonetary Policy Summary and Minutes\b", re.I)),
    ("BOE", "Monetary Policy Report", re.compile(r"\bMonetary Policy Report\b", re.I)),
    ("BOJ", "Summary of Opinions", re.compile(r"\bSummary of Opinions\b", re.I)),
    ("BOJ", "Regional Economic Report (Sakura Report)", re.compile(r"\bRegional Economic Report\b", re.I)),
    ("BOJ", "Statement on Monetary Policy", re.compile(r"\bStatement on Monetary Policy\b", re.I)),
    # Bank of Canada
    ("BOC", "Interest rate announcement", re.compile(r"\b(policy rate|interest rate|overnight rate)\b", re.I)),
    ("BOC", "Monetary Policy Report", re.compile(r"\bMonetary Policy Report\b", re.I)),
    ("BOC", "Summary of Deliberations", re.compile(r"\bSummary of (Governing Council )?Deliberations\b", re.I)),
    # Swiss National Bank
    ("SNB", "Monetary policy assessment", re.compile(r"\bMonetary policy assessment\b", re.I)),
    ("SNB", "Monetary policy decision", re.compile(r"\b(Monetary policy decision|policy rate)\b", re.I)),
    ("SNB", "Quarterly Bulletin", re.compile(r"\bQuarterly Bulletin\b", re.I)),
    # Norges Bank
    ("NORGES", "Policy rate decision", re.compile(r"\b[Pp]olicy rate\b", re.I)),
    ("NORGES", "Monetary Policy Report", re.compile(r"\bMonetary Policy Report\b", re.I)),
    # Reserve Bank of Australia
    ("RBA", "Monetary Policy Decision", re.compile(r"\bMonetary Policy Decision\b", re.I)),
    ("RBA", "Statement on Monetary Policy", re.compile(r"\bStatement on Monetary Policy\b", re.I)),
    ("RBA", "Board Minutes", re.compile(r"\bMinutes of the (Monetary Policy )?Board\b", re.I)),
    # Reserve Bank of New Zealand
    ("RBNZ", "Official Cash Rate", re.compile(r"\b(OCR|Official Cash Rate)\b", re.I)),
    ("RBNZ", "Monetary Policy Statement", re.compile(r"\bMonetary Policy Statement\b", re.I)),
    ("RBNZ", "Financial Stability Report", re.compile(r"\bFinancial Stability Report\b", re.I)),
    # Sveriges Riksbank
    ("RIKSBANK", "Policy rate decision", re.compile(r"\b(policy rate|repo rate)\b", re.I)),
    ("RIKSBANK", "Monetary Policy Report", re.compile(r"\bMonetary Policy Report\b", re.I)),
    ("RIKSBANK", "Minutes of monetary policy meeting", re.compile(r"\bMinutes of the (Executive Board.s )?monetary policy meeting\b", re.I)),
]

DB_PATH = "centralbank_summaries.sqlite3"

# ---------- Data model ----------
@dataclass
class Item:
    source: str
    title: str
    url: str
    published_at: datetime
    guid: str
    kind: str  # classified label

def _to_dt(entry) -> datetime:
    # feedparser provides published_parsed / updated_parsed as time.struct_time
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if not t:
        return datetime.now(timezone.utc)
    return datetime(*t[:6], tzinfo=timezone.utc)

def classify(source: str, title: str, url: str) -> Optional[str]:
    text = f"{title} {url}"
    for s, kind, rx in CLASSIFIERS:
        if s == source and rx.search(text):
            return kind
    return None

# ---------- Storage ----------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            guid TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            published_at TEXT NOT NULL,
            content_sha256 TEXT,
            content_text TEXT,
            summary_json TEXT
        )
        """
    )
    conn.commit()

def upsert_item(conn: sqlite3.Connection, item: Item) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO items (guid, source, kind, title, url, published_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (item.guid, item.source, item.kind, item.title, item.url, item.published_at.isoformat()),
    )
    conn.commit()

def set_content(conn: sqlite3.Connection, guid: str, text: str) -> None:
    sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    conn.execute(
        "UPDATE items SET content_sha256=?, content_text=? WHERE guid=?",
        (sha, text, guid),
    )
    conn.commit()

def set_summary(conn: sqlite3.Connection, guid: str, summary: dict) -> None:
    conn.execute(
        "UPDATE items SET summary_json=? WHERE guid=?",
        (json.dumps(summary, ensure_ascii=False), guid),
    )
    conn.commit()

# ---------- Extraction ----------
def fetch_url(client: httpx.Client, url: str) -> tuple[str, str]:
    """
    Returns (content_type, body_text_or_binary_decoded).
    For PDFs you'd normally keep bytes; here we'll keep it simple and detect later.
    """
    r = client.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    return ctype, r.text if "text" in ctype or "html" in ctype else r.content.decode("latin-1", errors="ignore")

def extract_text_from_html(html: str) -> str:
    # Readability to isolate main article, then soup get_text
    doc = Document(html)
    main_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(main_html, "lxml")
    text = soup.get_text("\n", strip=True)
    # basic cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def extract_text_from_pdf_url(client: httpx.Client, url: str) -> str:
    # Minimal PDF extraction
    from io import BytesIO
    from pdfminer.high_level import extract_text

    r = client.get(url, follow_redirects=True, timeout=60)
    r.raise_for_status()
    return extract_text(BytesIO(r.content)) or ""

def extract_full_text(client: httpx.Client, url: str) -> str:
    if url.lower().endswith(".pdf"):
        return extract_text_from_pdf_url(client, url)
    ctype, body = fetch_url(client, url)
    if "pdf" in ctype:
        return extract_text_from_pdf_url(client, url)
    return extract_text_from_html(body)

# ---------- Summarization ----------
def summarize_with_llm(text: str, meta: dict) -> dict:
    """
    Summarize using OpenAI if OPENAI_API_KEY is set, otherwise fall back to naive truncation.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return summarize_with_openai(text, meta)
        except Exception as ex:
            print(f"[WARN] OpenAI summarization failed: {ex}")
    # naive fallback summary if no LLM
    first = " ".join(text.split()[:60])
    return {
        "bullets": [
            f"{meta['kind']}: {first}…",
        ]
    }

# Example OpenAI Responses API call (optional).
# Docs: https://platform.openai.com/docs/api-reference/responses  (use current models per your account)
def summarize_with_openai(text: str, meta: dict) -> dict:
    """
    Requires: pip install openai ; export OPENAI_API_KEY=...
    """
    from openai import OpenAI
    client = OpenAI()

    # Chunk if needed (very simple chunking)
    max_chars = 40_000
    text_in = text[:max_chars]

    prompt = f"""
You are summarizing a central bank release.

Return STRICT JSON:
{{
  "bullets": ["...", "...", "..."],
  "signals": {{
    "policy_rate": "...",
    "inflation": "...",
    "growth": "...",
    "balance_sheet": "...",
    "forward_guidance": "..."
  }}
}}

Source: {meta['source']}
Type: {meta['kind']}
Title: {meta['title']}
Date: {meta['published_at']}
Text:
{text_in}
""".strip()

    resp = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    out = (resp.output_text or "").strip()
    # Strip markdown code fences if the model wrapped the JSON
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    if not out:
        raise ValueError("OpenAI returned empty response")
    return json.loads(out)

# ---------- Main ----------
def _resolve_db_path(db_path: Optional[str] = None) -> str:
    if db_path:
        return db_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_PATH)

def iter_feed_items(sources: Optional[list[str]] = None) -> Iterable[Item]:
    for source, feed_urls in FEEDS.items():
        if sources and source not in sources:
            continue
        for feed_url in feed_urls:
            feed = feedparser.parse(feed_url)
            for e in feed.entries:
                title = e.get("title", "").strip()
                url = (e.get("link") or "").strip()
                if not title or not url:
                    continue
                kind = classify(source, title, url)
                if not kind:
                    continue
                published_at = _to_dt(e)
                guid = (e.get("id") or url).strip()
                yield Item(source=source, title=title, url=url, published_at=published_at, guid=guid, kind=kind)

def _fetch_and_store(conn: sqlite3.Connection, sources: Optional[list[str]] = None) -> None:
    """Fetch RSS feeds, extract content, store in DB."""
    headers = {"User-Agent": "cb-summarizer/0.1 (+contact: you@example.com)"}
    fetched = 0
    to_summarize: list[tuple[str, str, dict]] = []  # (guid, text, meta)

    # Phase 1: Fetch feeds and extract content (sequential, involves HTTP + DB)
    with httpx.Client(headers=headers) as client:
        for item in iter_feed_items(sources=sources):
            upsert_item(conn, item)
            fetched += 1

            row = conn.execute("SELECT content_text, summary_json FROM items WHERE guid=?", (item.guid,)).fetchone()
            has_content = row and row[0]
            has_proper_summary = False
            if has_content and row[1]:
                try:
                    existing = json.loads(row[1])
                    has_proper_summary = bool(existing.get("signals"))
                except (json.JSONDecodeError, TypeError):
                    pass

            if has_content and has_proper_summary:
                continue

            meta = {
                "source": item.source,
                "kind": item.kind,
                "title": item.title,
                "published_at": item.published_at.isoformat(),
                "url": item.url,
            }

            if has_content:
                to_summarize.append((item.guid, row[0], meta))
            else:
                try:
                    text = extract_full_text(client, item.url)
                    if text.strip():
                        set_content(conn, item.guid, text)
                        to_summarize.append((item.guid, text, meta))
                except Exception as ex:
                    print(f"[WARN] {item.source} fetch failed: {item.url} -> {ex}")

    # Phase 2: Summarize via LLM in parallel
    if not to_summarize:
        print(f"[INFO] Central bank data fetch complete — {fetched} item(s) checked, no new summaries needed.")
        return

    def _do_summarize(item: tuple[str, str, dict]) -> tuple[str, dict]:
        guid, text, meta = item
        return guid, summarize_with_llm(text, meta)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_do_summarize, item): item for item in to_summarize}
        for future in as_completed(futures):
            try:
                guid, summary = future.result()
                set_summary(conn, guid, summary)
            except Exception as ex:
                failed = futures[future]
                print(f"[WARN] Summarization failed for {failed[2]['source']} {failed[2]['kind']}: {ex}")

    print(f"[INFO] Central bank data fetch and summarization complete — {fetched} item(s) fetched, {len(to_summarize)} new summary(ies) generated.")

def _query_items(conn: sqlite3.Connection, sources: Optional[list[str]] = None) -> list[dict]:
    """Query stored items and return as list of dicts."""
    if sources:
        placeholders = ",".join("?" for _ in sources)
        query = f"""
            SELECT source, kind, title, url, published_at,
                   summary_json, content_text
            FROM items
            WHERE source IN ({placeholders})
            ORDER BY published_at DESC
            LIMIT 100
        """
        rows = conn.execute(query, sources).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT source, kind, title, url, published_at,
                   summary_json, content_text
            FROM items
            ORDER BY published_at DESC
            LIMIT 100
            """
        ).fetchall()

    items = []
    for source, kind, title, url, published_at, summary_json, content_text in rows:
        summary = json.loads(summary_json) if summary_json else {}
        items.append({
            "source": source,
            "kind": kind,
            "title": title,
            "url": url,
            "published_at": published_at,
            "summary_bullets": summary.get("bullets", []),
            "signals": summary.get("signals", {}),
            "has_full_text": bool(content_text),
            "content_preview": (content_text or "")[:500],
        })
    return items

def get_data(db_path: str = None, refresh: bool = False, sources: Optional[list[str]] = None) -> dict:
    """
    Return structured data for GUI consumption.

    Only fetches RSS feeds and extracts content when refresh=True.
    *sources* filters which central banks to fetch/query (default: all in FEEDS).
    Returns dict with keys: items, by_source, counts, last_updated, error (on failure).
    """
    db_path = _resolve_db_path(db_path)
    try:
        conn = sqlite3.connect(db_path)
        init_db(conn)
        if refresh:
            _fetch_and_store(conn, sources=sources)
        items = _query_items(conn, sources=sources)
        conn.close()

        active_sources = sources or list(FEEDS.keys())
        by_source: dict[str, list] = {k: [] for k in active_sources}
        for item in items:
            if item["source"] in by_source:
                by_source[item["source"]].append(item)

        counts = {"total": len(items)}
        for k in active_sources:
            counts[k] = len(by_source[k])

        return {
            "items": items,
            "by_source": by_source,
            "counts": counts,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}

def run():
    data = get_data()
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return

    for item in data["items"]:
        if not item["summary_bullets"]:
            continue
        print(f"\n{item['source']}: {item['kind']} ({item['published_at'][:10]})")
        for b in item["summary_bullets"][:3]:
            print(f" - {b}")
        print(f"   {item['url']}")

if __name__ == "__main__":
    run()
