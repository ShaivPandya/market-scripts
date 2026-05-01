"""Tests for the cloud-aware Industry Monitor PDF pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from api.exceptions import DataFetchError
from api.routers import industry as industry_router
from macro.industry import industry_monitor as im

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = REPO_ROOT / "macro" / "industry" / "files" / "housing" / "DHI.pdf"


def test_get_pdf_locator_applies_filename_map():
    local_path, gcs_key = im._get_pdf_locator("Trucking", "ODFL")
    assert local_path.name == "ODL.pdf"
    assert gcs_key.endswith("/trucking/ODL.pdf")
    assert gcs_key.startswith(im.INDUSTRY_TRANSCRIPTS_PREFIX + "/")


def test_get_pdf_locator_normalizes_sector_with_space():
    _, gcs_key = im._get_pdf_locator("Capital Goods", "CAT")
    assert "/capital_goods/CAT.pdf" in gcs_key


@pytest.mark.skipif(not SAMPLE_PDF.is_file(), reason="sample PDF not available locally")
def test_extract_text_from_bytes_roundtrip():
    text = im._extract_text_from_bytes(SAMPLE_PDF.read_bytes())
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
