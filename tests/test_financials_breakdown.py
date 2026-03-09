import portfolio.momentum.fundamental_momentum.financials_single as fs


def _patch_common(monkeypatch):
    monkeypatch.setattr(
        fs,
        "_candidate_revenue_filings",
        lambda us_gaap: [{"accn": "A1", "form": "10-K", "filed": "2025-02-01"}],
    )
    monkeypatch.setattr(
        fs,
        "build_filing_url",
        lambda cik_str, accn, submissions=None: f"https://sec.test/{accn}",
    )


def test_xbrl_both_axes_skips_ai(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [{"label": "Cloud", "value": 100.0, "pct_of_total": 0.5}],
            "by_region": [{"label": "US", "value": 120.0, "pct_of_total": 0.6}],
        },
    )

    def _should_not_run_ai(**kwargs):
        raise AssertionError("AI fallback should not be called when XBRL has both axes")

    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", _should_not_run_ai)

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"]
    assert out["by_region"]
    assert out["extraction_meta"]["segment"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is False


def test_missing_segment_filled_by_ai(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [{"label": "US", "value": 120.0, "pct_of_total": 0.6}],
        },
    )

    calls = []

    def _ai_fill_segment(**kwargs):
        calls.append(set(kwargs["wanted_axes"]))
        return {
            "accn": kwargs["accn"],
            "period_end": "2024-12-31",
            "filed": kwargs["filed"],
            "form": kwargs["form"],
            "by_segment": [{"label": "Devices", "value": 80.0, "pct_of_total": 0.4}],
            "by_region": [],
            "segment_disclosed": True,
            "region_disclosed": None,
        }

    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", _ai_fill_segment)

    out = fs._build_breakdown({}, "0000123456", None)
    assert calls == [{"segment"}]
    assert out["by_segment"]
    assert out["by_region"]
    assert out["extraction_meta"]["segment"] == {"status": "found", "source": "ai"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is True


def test_missing_segment_not_disclosed(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [{"label": "US", "value": 120.0, "pct_of_total": 0.6}],
        },
    )
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_via_nlp",
        lambda **kwargs: {
            "accn": kwargs["accn"],
            "period_end": "2024-12-31",
            "filed": kwargs["filed"],
            "form": kwargs["form"],
            "by_segment": [],
            "by_region": [],
            "segment_disclosed": False,
            "region_disclosed": None,
        },
    )

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"] == []
    assert out["by_region"]
    assert out["extraction_meta"]["segment"] == {"status": "not_disclosed", "source": "none"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is True


def test_missing_segment_ai_invalid_or_failed(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [{"label": "US", "value": 120.0, "pct_of_total": 0.6}],
        },
    )
    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", lambda **kwargs: None)

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"] == []
    assert out["extraction_meta"]["segment"] == {"status": "unavailable", "source": "none"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is True


def test_missing_axis_no_anthropic_key(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [{"label": "US", "value": 120.0, "pct_of_total": 0.6}],
        },
    )

    def _should_not_run_ai(**kwargs):
        raise AssertionError("AI fallback should not run without ANTHROPIC_API_KEY")

    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", _should_not_run_ai)

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"] == []
    assert out["by_region"]
    assert out["extraction_meta"]["segment"] == {"status": "unavailable", "source": "none"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "xbrl"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is False


def test_missing_axes_filled_by_html(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # 1. XBRL returns nothing
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [],
        },
    )

    # 2. HTML returns both axes
    def _mock_html(**kwargs):
        return {
            "accn": kwargs["accn"],
            "period_end": "2024-12-31",
            "filed": kwargs["filed"],
            "form": "10-K",
            "by_segment": [{"label": "Hardware", "value": 100.0, "pct_of_total": 0.5}],
            "by_region": [{"label": "Americas", "value": 120.0, "pct_of_total": 0.6}],
        }

    monkeypatch.setattr(fs, "_extract_breakdown_from_html", _mock_html)

    # 3. NLP should not run since HTML found both
    def _should_not_run_ai(**kwargs):
        raise AssertionError("AI fallback should not be called when HTML has both axes")

    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", _should_not_run_ai)

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"]
    assert out["by_region"]
    assert out["extraction_meta"]["segment"] == {"status": "found", "source": "html"}
    assert out["extraction_meta"]["region"] == {"status": "found", "source": "html"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is False


def test_html_fails_falls_back_to_nlp(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # 1. XBRL returns nothing
    monkeypatch.setattr(
        fs,
        "_extract_breakdown_for_filing",
        lambda us_gaap, accn: {
            "accn": accn,
            "period_end": "2024-12-31",
            "filed": "2025-02-01",
            "form": "10-K",
            "by_segment": [],
            "by_region": [],
        },
    )

    # 2. HTML returns nothing (fails)
    monkeypatch.setattr(fs, "_extract_breakdown_from_html", lambda **kwargs: None)

    # 3. NLP returns data
    def _mock_nlp(**kwargs):
        return {
            "accn": kwargs["accn"],
            "period_end": "2024-12-31",
            "filed": kwargs["filed"],
            "form": "10-K",
            "by_segment": [{"label": "Software", "value": 80.0, "pct_of_total": 1.0}],
            "by_region": [],
            "segment_disclosed": True,
            "region_disclosed": False,
        }

    monkeypatch.setattr(fs, "_extract_breakdown_via_nlp", _mock_nlp)

    out = fs._build_breakdown({}, "0000123456", None)
    assert out["by_segment"]
    assert out["by_region"] == []
    assert out["extraction_meta"]["segment"] == {"status": "found", "source": "ai"}
    assert out["extraction_meta"]["region"] == {"status": "not_disclosed", "source": "none"}
    assert out["extraction_meta"]["ai_fallback_attempted"] is True
