"""Tests for thesis monitoring functions in auto_weekly_report."""

import json
from datetime import UTC, datetime, timedelta

import auto_report.auto_weekly_report as weekly


def test_load_theses_reads_files(tmp_path, monkeypatch):
    """Thesis files are loaded; missing tickers get None."""
    csv_path = tmp_path / "portfolio" / "portfolio.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "ticker,asset,direction,distressed,conviction\nAAA,equity,long,false,3\nBBB,equity,short,false,2\n"
    )

    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    (thesis_dir / "AAA.md").write_text("# AAA Thesis\nBuy because reasons.")

    monkeypatch.setattr(weekly, "THESES_DIR", thesis_dir)
    monkeypatch.setattr(weekly, "PROJECT_ROOT", tmp_path)

    result = weekly.load_theses()
    assert result["AAA"] == "# AAA Thesis\nBuy because reasons."
    assert result["BBB"] is None


def test_load_theses_empty_file(tmp_path, monkeypatch):
    """An empty thesis file returns None."""
    csv_path = tmp_path / "portfolio" / "portfolio.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("ticker,asset,direction,distressed,conviction\nXYZ,equity,long,false,3\n")

    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    (thesis_dir / "XYZ.md").write_text("")

    monkeypatch.setattr(weekly, "THESES_DIR", thesis_dir)
    monkeypatch.setattr(weekly, "PROJECT_ROOT", tmp_path)

    result = weekly.load_theses()
    assert result["XYZ"] is None


def test_filter_news_7day():
    """Only articles within 7 days are kept."""
    now_utc = datetime.now(UTC)
    articles = [
        {"ticker": "X", "title": "recent", "seendate": (now_utc - timedelta(days=1)).isoformat()},
        {"ticker": "X", "title": "old", "seendate": (now_utc - timedelta(days=15)).isoformat()},
        {"ticker": "X", "title": "edge", "seendate": (now_utc - timedelta(days=6, hours=23)).isoformat()},
    ]
    news_data = {"by_ticker": {"X": articles}}
    result = weekly.filter_news_7day(news_data)
    assert len(result["X"]) == 2
    titles = {a["title"] for a in result["X"]}
    assert "recent" in titles
    assert "edge" in titles
    assert "old" not in titles


def test_filter_news_7day_empty():
    """Empty input returns empty output."""
    result = weekly.filter_news_7day({})
    assert result == {}
    result2 = weekly.filter_news_7day({"by_ticker": {}})
    assert result2 == {}


def test_parse_thesis_response_valid():
    """Valid response with separator and JSON is parsed correctly."""
    md = "## Portfolio Thesis Monitoring\n\n### CRWD\nThesis intact."
    thesis_json = json.dumps(
        {
            "thesis_evaluations": [
                {
                    "ticker": "CRWD",
                    "thesis_status": "strengthen",
                    "technical_read": "improving",
                    "fundamental_read": "supportive",
                    "action": "hold",
                    "confidence": "high",
                    "key_developments": ["Strong earnings beat"],
                    "earnings_note": "Q4 beat estimates by 15%",
                    "risk_flag": None,
                }
            ],
            "positions_reviewed": ["CRWD"],
            "thesis_strengthened": ["CRWD"],
            "thesis_weakened": [],
            "positions_needing_reassessment": [],
            "missing_theses": [],
            "material_developments": [
                {
                    "ticker": "CRWD",
                    "type": "supports_thesis",
                    "summary": "Strong Q4 earnings beat",
                }
            ],
        }
    )
    text = f"{md}\n{weekly.THESIS_SEPARATOR}\n```json\n{thesis_json}\n```"

    result_md, result_summary = weekly.parse_thesis_response(text)
    assert result_md == md
    assert result_summary["thesis_evaluations"][0]["ticker"] == "CRWD"
    assert result_summary["positions_reviewed"] == ["CRWD"]
    assert result_summary["thesis_strengthened"] == ["CRWD"]
    assert "parse_error" not in result_summary


def test_parse_thesis_response_no_separator():
    """Response without separator uses fallback."""
    text = "Just some markdown without a separator."
    result_md, result_summary = weekly.parse_thesis_response(text)
    assert result_md == text
    assert result_summary["parse_error"] is True
    assert result_summary["thesis_evaluations"] == []


def test_parse_thesis_response_bad_json():
    """Response with separator but invalid JSON uses fallback."""
    text = f"Some markdown\n{weekly.THESIS_SEPARATOR}\nnot valid json at all"
    result_md, result_summary = weekly.parse_thesis_response(text)
    assert result_md == "Some markdown"
    assert result_summary["parse_error"] is True


def test_merge_thesis_into_summary():
    """Thesis monitoring is added without clobbering existing keys."""
    base = {"stance": "bullish", "confidence": "high", "drivers": ["x"], "watchlist_triggers": ["y"]}
    thesis = {
        "thesis_evaluations": [{"ticker": "CRWD", "thesis_status": "strengthen"}],
        "positions_reviewed": ["CRWD"],
        "thesis_strengthened": ["CRWD"],
        "thesis_weakened": [],
        "positions_needing_reassessment": [],
        "missing_theses": [],
        "material_developments": [],
    }
    merged = weekly._merge_thesis_into_summary(base, thesis)
    assert merged["stance"] == "bullish"
    assert merged["confidence"] == "high"
    assert merged["drivers"] == ["x"]
    assert merged["thesis_monitoring"] == thesis
