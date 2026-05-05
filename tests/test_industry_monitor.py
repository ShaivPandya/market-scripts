"""Tests for the cloud-aware Industry Monitor PDF pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.exceptions import DataFetchError
from api.routers import industry as industry_router
from macro.industry import industry_monitor as im


def _make_text_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        b"5 0 obj\n<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def test_get_pdf_locator_applies_filename_map():
    local_path, gcs_key = im._get_pdf_locator("Trucking", "ODFL")
    assert local_path.name == "ODL.pdf"
    assert gcs_key.endswith("/trucking/ODL.pdf")
    assert gcs_key.startswith(im.INDUSTRY_TRANSCRIPTS_PREFIX + "/")


def test_get_pdf_locator_normalizes_sector_with_space():
    _, gcs_key = im._get_pdf_locator("Capital Goods", "CAT")
    assert "/capital_goods/CAT.pdf" in gcs_key


def test_extract_text_from_bytes_roundtrip():
    pdf_bytes = _make_text_pdf("D.R. Horton Q3 2024 earnings")
    text = im._extract_text_from_bytes(pdf_bytes)
    assert text and text.strip()
    assert any(token in text for token in ("D.R. Horton", "Horton", "earnings"))


def test_sanitize_transcript_text_removes_nul_bytes():
    assert im._sanitize_transcript_text("abc\x00def\x00") == "abcdef"


def test_parse_period_from_text_uses_explicit_quarter_and_year():
    text = "D.R. Horton, Inc. Q3 2024 Earnings Conference Call July 18, 2024"
    fallback = datetime(2030, 1, 1, tzinfo=UTC)  # should not be consulted
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2024
    assert quarter == 3
    assert transcript_date == "2024-07-18"


def test_parse_period_handles_abbreviated_month_and_infers_reporting_quarter():
    text = "DATE Thursday, Apr. 23, 2026 at 8:30 a.m. ET CALL PARTICIPANTS"
    fallback = datetime(2030, 1, 1, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2026
    assert quarter == 1
    assert transcript_date == "2026-04-23"


def test_parse_period_handles_factset_day_month_date():
    text = "Corrected Transcript 18-Feb-2026 Toll Brothers, Inc. (TOL) Q1 2026 Earnings Call"
    fallback = datetime(2030, 1, 1, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2026
    assert quarter == 1
    assert transcript_date == "2026-02-18"


def test_parse_period_handles_spaced_slide_date():
    text = "Q1 2026 Earnings Presentation P E T E R J A C K S O N A p r i l   3 0 ,   2 0 2 6"
    fallback = datetime(2030, 1, 1, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2026
    assert quarter == 1
    assert transcript_date == "2026-04-30"


def test_parse_period_handles_fourth_quarter_results_after_year_end():
    text = (
        "Financial Release Saia Reports Fourth Quarter Results "
        "JOHNS CREEK, Ga., Feb. 10, 2026 -- Saia reported fourth quarter 2025 financial results."
    )
    fallback = datetime(2030, 1, 1, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2025
    assert quarter == 4
    assert transcript_date == "2026-02-10"


def test_parse_period_ignores_metric_quarter_references():
    text = (
        "DATE Tuesday, April 21, 2026 at 5 p.m. ET CALL PARTICIPANTS "
        "TAKEAWAYS Revenue -- Decreased 2% sequentially from Q4 2025; "
        "Brex Acquisition -- anticipated to decrease CET1 by just over 40 basis points in Q2 2026."
    )
    fallback = datetime(2030, 1, 1, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2026
    assert quarter == 1
    assert transcript_date == "2026-04-21"


def test_parse_period_falls_back_to_injected_dt(monkeypatch):
    # Header is intentionally devoid of date / quarter markers.
    text = "Welcome to the call." + (" filler" * 200)
    fallback = datetime(2025, 5, 7, tzinfo=UTC)
    year, quarter, transcript_date = im._parse_period_from_text(text, fallback)
    assert year == 2025
    assert quarter == 2  # May -> Q2
    assert transcript_date == "2025-05-07"


def test_load_pdf_bytes_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(im.state_storage, "object_updated", lambda local, key: None)

    # read_bytes should not be called when object_updated is None
    def _boom(*_args, **_kwargs):
        raise AssertionError("read_bytes must not run when object is missing")

    monkeypatch.setattr(im.state_storage, "read_bytes", _boom)
    assert im._load_pdf_bytes("Housing", "DHI") is None


def test_load_pdf_bytes_routes_through_state_storage(monkeypatch):
    canned_bytes = b"%PDF-FAKE"
    canned_dt = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

    seen = {}

    def _fake_object_updated(local: Path, key: str):
        seen["updated_key"] = key
        return canned_dt

    def _fake_read_bytes(local: Path, key: str) -> bytes:
        seen["read_key"] = key
        return canned_bytes

    monkeypatch.setattr(im.state_storage, "object_updated", _fake_object_updated)
    monkeypatch.setattr(im.state_storage, "read_bytes", _fake_read_bytes)

    result = im._load_pdf_bytes("Banks", "JPM")
    assert result == (canned_bytes, canned_dt)
    assert seen["updated_key"].endswith("/banks/JPM.pdf")
    assert seen["read_key"] == seen["updated_key"]


def test_industry_pdf_route_returns_pdf(monkeypatch):
    monkeypatch.setattr(
        im,
        "_load_pdf_bytes",
        lambda sector, ticker: (b"%PDF-FAKE", datetime(2026, 4, 23, tzinfo=UTC)),
    )

    response = industry_router.get_industry_transcript_pdf("jpm")

    assert response.media_type == "application/pdf"
    assert response.body == b"%PDF-FAKE"
    assert 'filename="JPM.pdf"' in response.headers["content-disposition"]


def test_industry_pdf_route_returns_404_for_unknown_ticker():
    with pytest.raises(HTTPException) as exc:
        industry_router.get_industry_transcript_pdf("NOPE")

    assert exc.value.status_code == 404


def _make_mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    im.init_db(conn)
    return conn


class _CaptureConn:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.commits = 0

    def execute(self, sql: str, params=()):
        self.calls.append((sql, tuple(params or ())))
        return None

    def commit(self):
        self.commits += 1


def test_set_fresh_row_binds_boolean_stale_flags():
    conn = _CaptureConn()

    im._set_fresh_row(conn, "DHI", "DHI_2025_Q1")  # type: ignore[arg-type]

    assert conn.calls[0][1] == (True, "DHI")
    assert type(conn.calls[0][1][0]) is bool
    assert conn.calls[1][1] == (False, "DHI_2025_Q1")
    assert type(conn.calls[1][1][0]) is bool
    assert conn.commits == 1


def test_upsert_transcript_binds_boolean_is_stale():
    conn = _CaptureConn()

    im._upsert_transcript(
        conn,  # type: ignore[arg-type]
        row_id="DHI_2025_Q1",
        ticker="DHI",
        company_name="D.R. Horton",
        sector="Housing",
        sector_type="leading",
        sub_sector="Homebuilder",
        year=2025,
        quarter=1,
        transcript_text="Q1 2025 Earnings Call placeholder body text",
        transcript_date="2025-04-30",
        content_sha256="abc123",
        fetched_at="2026-04-30T00:00:00+00:00",
    )

    sql, params = conn.calls[0]
    assert params[-1] is False
    assert type(params[-1]) is bool
    assert "is_stale=excluded.is_stale" in sql
    assert "is_stale=0" not in sql
    assert conn.commits == 1


def test_set_summary_binds_boolean_is_stale():
    conn = _CaptureConn()

    im._set_summary(conn, "DHI_2025_Q1", {"sentiment": "neutral"})  # type: ignore[arg-type]

    _sql, params = conn.calls[0]
    assert params[-2] is False
    assert type(params[-2]) is bool
    assert params[-1] == "DHI_2025_Q1"
    assert conn.commits == 1


def test_industry_route_raises_on_error_payload(monkeypatch):
    monkeypatch.setattr(im, "get_data", lambda refresh=False: {"error": "boom"})

    with pytest.raises(DataFetchError) as exc:
        industry_router.get_industry_monitor(refresh=True)

    assert exc.value.source == "industry"
    assert exc.value.detail == "boom"


def test_fetch_and_store_skips_missing_pdfs_without_crashing(monkeypatch):
    monkeypatch.setattr(im, "_load_pdf_bytes", lambda sector, ticker: None)

    conn = _make_mem_db()
    im._fetch_and_store(conn)
    # Every ticker in SECTORS should have been marked stale without crashing.
    rows = conn.execute("SELECT COUNT(*) AS n FROM transcripts").fetchone()
    assert rows["n"] == 0  # no upserts because every ticker missing


def test_fetch_and_store_persists_when_pdf_present(monkeypatch):
    fake_bytes = b"%PDF-FAKE"
    fake_dt = datetime(2025, 4, 30, tzinfo=UTC)

    monkeypatch.setattr(im, "_load_pdf_bytes", lambda sector, ticker: (fake_bytes, fake_dt))
    monkeypatch.setattr(im, "_extract_text_from_bytes", lambda b: "Q1 2025 Earnings Call placeholder body text")
    # Avoid LLM calls during the summarization phase.
    monkeypatch.setattr(
        im,
        "summarize_with_llm",
        lambda text, meta: {
            "ticker": meta["ticker"],
            "sentiment": "neutral",
            "headline": "stub",
            "demand_signal": "",
            "pricing_signal": "",
            "guidance_outlook": "",
            "key_themes": [],
            "macro_quotes": [],
        },
    )

    conn = _make_mem_db()
    im._fetch_and_store(conn)
    n = conn.execute("SELECT COUNT(*) AS n FROM transcripts WHERE is_stale=0").fetchone()["n"]
    # Sum across SECTORS — each ticker should have produced one fresh row.
    expected = sum(len(cfg["companies"]) for cfg in im.SECTORS.values())
    assert n == expected


def test_fetch_and_store_removes_nul_bytes_before_upsert(monkeypatch):
    monkeypatch.setattr(
        im,
        "SECTORS",
        {"Housing": {"type": "leading", "companies": [("DHI", "D.R. Horton", "Homebuilder", "BMO")]}},
    )
    monkeypatch.setattr(im, "_load_pdf_bytes", lambda sector, ticker: (b"%PDF-FAKE", datetime(2025, 4, 30, tzinfo=UTC)))
    monkeypatch.setattr(im, "_extract_text_from_bytes", lambda b: "Q1 2025 Earnings\x00 Call placeholder body text")
    monkeypatch.setattr(
        im,
        "summarize_with_llm",
        lambda text, meta: {
            "summary_headline": "stub",
            "sentiment": "neutral",
            "business_conditions": [],
            "demand_trends": "",
            "pricing_commentary": "",
            "guidance_outlook": "",
            "macro_quotes": [],
        },
    )

    conn = _make_mem_db()
    im._fetch_and_store(conn)

    row = conn.execute("SELECT transcript_text FROM transcripts WHERE ticker='DHI'").fetchone()
    assert row is not None
    assert "\x00" not in row["transcript_text"]
    assert row["transcript_text"] == "Q1 2025 Earnings Call placeholder body text"


def test_fetch_and_store_reuses_same_content_summary_after_reparse(monkeypatch):
    text = "DATE Thursday, Apr. 23, 2026 at 8:30 a.m. ET CALL PARTICIPANTS"
    sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    monkeypatch.setattr(
        im,
        "SECTORS",
        {"Banks": {"type": "coincident", "companies": [("AXP", "American Express", "Card Issuer", "AMC")]}},
    )
    monkeypatch.setattr(im, "_load_pdf_bytes", lambda sector, ticker: (b"%PDF-FAKE", datetime(2026, 4, 30, tzinfo=UTC)))
    monkeypatch.setattr(im, "_extract_text_from_bytes", lambda b: text)

    def _must_not_summarize(*_args, **_kwargs):
        raise AssertionError("summary should be reused from same-content row")

    monkeypatch.setattr(im, "summarize_with_llm", _must_not_summarize)
    conn = _make_mem_db()
    im._upsert_transcript(
        conn,
        row_id="AXP_2026_Q2",
        ticker="AXP",
        company_name="American Express",
        sector="Banks",
        sector_type="coincident",
        sub_sector="Card Issuer",
        year=2026,
        quarter=2,
        transcript_text=text,
        transcript_date="2026-04-23",
        content_sha256=sha,
        fetched_at="2026-05-01T00:00:00+00:00",
    )
    im._set_summary(conn, "AXP_2026_Q2", {"summary_headline": "prior summary", "sentiment": "bullish"})

    im._fetch_and_store(conn)

    row = conn.execute("SELECT summary_json FROM transcripts WHERE id='AXP_2026_Q1'").fetchone()
    assert row is not None
    assert json.loads(row["summary_json"])["summary_headline"] == "prior summary"


def test_latest_row_prefers_non_superseded_row_over_wrong_higher_quarter():
    conn = _make_mem_db()
    for row_id, year, quarter, internal_stale in (
        ("AXP_2026_Q2", 2026, 2, True),
        ("AXP_2026_Q1", 2026, 1, False),
    ):
        im._upsert_transcript(
            conn,
            row_id=row_id,
            ticker="AXP",
            company_name="American Express",
            sector="Banks",
            sector_type="coincident",
            sub_sector="Card Issuer",
            year=year,
            quarter=quarter,
            transcript_text="Q1 2026 Earnings Call placeholder body text",
            transcript_date="2026-04-23",
            content_sha256=row_id,
            fetched_at="2026-05-01T00:00:00+00:00",
        )
        conn.execute("UPDATE transcripts SET is_stale=? WHERE id=?", (internal_stale, row_id))
    conn.commit()

    row = im._get_latest_row_for_ticker(conn, "AXP")

    assert row is not None
    assert row["id"] == "AXP_2026_Q1"


def test_company_stale_uses_ninety_day_age_rule(monkeypatch):
    monkeypatch.setattr(im, "_today_utc", lambda: date(2026, 5, 1))
    conn = _make_mem_db()
    im._upsert_transcript(
        conn,
        row_id="AXP_2026_Q1",
        ticker="AXP",
        company_name="American Express",
        sector="Banks",
        sector_type="coincident",
        sub_sector="Card Issuer",
        year=2026,
        quarter=1,
        transcript_text="Q1 2026 Earnings Call placeholder body text",
        transcript_date="2026-04-23",
        content_sha256="fresh",
        fetched_at="2026-05-01T00:00:00+00:00",
    )

    row = conn.execute("SELECT * FROM transcripts WHERE id='AXP_2026_Q1'").fetchone()
    item = im._company_from_row(
        row,
        ticker="AXP",
        company_name="American Express",
        sector="Banks",
        sector_type="coincident",
        sub_sector="Card Issuer",
    )
    assert item["age_days"] == 8
    assert item["is_stale"] is False

    conn.execute("UPDATE transcripts SET transcript_date='2026-01-01' WHERE id='AXP_2026_Q1'")
    old_row = conn.execute("SELECT * FROM transcripts WHERE id='AXP_2026_Q1'").fetchone()
    old_item = im._company_from_row(
        old_row,
        ticker="AXP",
        company_name="American Express",
        sector="Banks",
        sector_type="coincident",
        sub_sector="Card Issuer",
    )
    assert old_item["age_days"] == 120
    assert old_item["is_stale"] is True
