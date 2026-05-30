from __future__ import annotations

import json
import sys
import types
from typing import Any

from auto_report import auto_daily_report, auto_weekly_report


def _module(monkeypatch, name: str, **attrs: Any):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _install_collect_data_fakes(monkeypatch):
    _module(
        monkeypatch,
        "equities.index_dashboard.index_dashboard",
        INDEX_ORDER=["SPX"],
        get_data=lambda _period: {"indices": {"SPX": [100, 101]}},
    )
    _module(
        monkeypatch,
        "fx.fx_dashboard.fx_dashboard",
        PAIR_ORDER=["EURUSD"],
        get_data=lambda _period: {"pairs": {"EURUSD": [1.0, 1.1]}},
    )
    _module(
        monkeypatch,
        "commodities.commodities_dashboard",
        COMMODITY_ORDER=["Gold"],
        get_data=lambda _period: {"commodities": {"Gold": [2000, 2010]}},
    )
    _module(monkeypatch, "equities.market_technicals.market_breadth", get_data=lambda period="1y": {"breadth": 1})
    _module(monkeypatch, "equities.market_technicals.top50_breadth", get_data=lambda: {"top50": 1})
    _module(monkeypatch, "equities.market_technicals.vix_term_structure", get_data=lambda: {"ratio": 1.1})
    _module(monkeypatch, "equities.sector_metrics.sector_metrics", get_data=lambda: {"sectors": []})
    _module(
        monkeypatch,
        "macro.positioning.positioning",
        DATASETS={"tff_futures_only": "dataset"},
        DEFAULT_DOMAIN="example.com",
        fetch_multiple_instruments=lambda **_kwargs: [{"instrument": "SP500"}],
    )
    _module(
        monkeypatch,
        "portfolio.technical_analysis.technical_analysis",
        get_ratio_data=lambda *_args, **_kwargs: {"stats": {"start_ratio": 1, "end_ratio": 2}},
    )
    _module(monkeypatch, "macro.economic_growth.economic_growth", get_data=lambda: {"growth": "ok"})
    _module(monkeypatch, "macro.labor_market.labor_market", get_data=lambda: {"labor": "ok"})
    _module(monkeypatch, "macro.housing.housing", get_data=lambda: {"housing": "ok"})
    _module(monkeypatch, "macro.liquidity.liquidity", get_snapshot=lambda: {"liquidity": "ok"})
    _module(monkeypatch, "macro.central_banks.central_bank", get_data=lambda refresh=False: {"central": "ok"})
    _module(monkeypatch, "government_bonds.yield_curve", get_data=lambda lookback_days=90: {"curve": "ok"})
    _module(monkeypatch, "government_bonds.bond_dashboard", get_data=lambda: {"bonds": "ok"})
    _module(
        monkeypatch,
        "macro.country_dashboard.country_dashboard",
        METRICS=["Inflation", "Unemployment", "GDP"],
        get_data=lambda metric="Inflation": {
            "metric": metric,
            "countries": {"US": [{"date": "2024-01-01", "value": 1.0}]},
        },
    )
    _module(monkeypatch, "macro.industry.industry_monitor", get_data=lambda refresh=False: {"industry": "ok"})
    _module(
        monkeypatch,
        "macro.sentiment.sentiment",
        get_surveys=lambda: {"aaii": [], "naaim": [], "errors": {}},
        get_put_call=lambda: {},
        get_volatility=lambda: [],
    )


def test_collect_data_includes_expanded_macro_sources(monkeypatch):
    _install_collect_data_fakes(monkeypatch)

    data = auto_weekly_report.collect_data()

    expected = {
        "economic_growth",
        "labor_market",
        "housing",
        "liquidity",
        "central_banks",
        "yield_curve",
        "bond_dashboard",
        "country_dashboard",
        "industry",
        "sentiment",
        "news_digests",
    }
    assert expected.issubset(data.keys())
    assert data["country_dashboard"]["Inflation"]["metric"] == "Inflation"
    assert data["country_dashboard"]["Unemployment"]["metric"] == "Unemployment"
    assert data["country_dashboard"]["GDP"]["metric"] == "GDP"


def test_collect_data_keeps_source_failures_isolated(monkeypatch):
    _install_collect_data_fakes(monkeypatch)

    def fail_labor():
        raise RuntimeError("labor unavailable")

    sys.modules["macro.labor_market.labor_market"].get_data = fail_labor

    data = auto_weekly_report.collect_data()

    assert data["labor_market"] == {"error": "labor unavailable"}
    assert data["housing"] == {"housing": "ok"}
    assert data["central_banks"] == {"central": "ok"}


def test_prepare_prompt_bundle_slims_expanded_macro_sources():
    raw = {
        "labor_market": {
            "timestamp": "2024-01-02T00:00:00",
            "latest": {"initial_claims": {"value": 210, "date": "2024-01-01", "change": -5}},
            "series": {
                "initial_claims": {
                    "dates": ["2023-12-25", "2024-01-01"],
                    "values": [215, 210],
                    "label": "Initial Jobless Claims",
                    "unit": "thousands",
                }
            },
        },
        "housing": {
            "latest": {"housing_starts": {"value": 1400, "date": "2024-01-01", "change": 20}},
            "series": {
                "housing_starts": {
                    "dates": ["2023-12-01", "2024-01-01"],
                    "values": [1380, 1400],
                    "label": "Housing Starts",
                    "unit": "thousands",
                }
            },
        },
        "central_banks": {
            "counts": {"total": 1, "FED": 1},
            "by_source": {"FED": [{"title": "duplicate"}]},
            "items": [
                {
                    "source": "FED",
                    "kind": "FOMC statement",
                    "title": "Fed statement",
                    "published_at": "2024-01-01",
                    "url": "https://example.com",
                    "summary_bullets": ["held rates"],
                    "signals": {"policy_rate": "hold"},
                    "content_preview": "long text",
                }
            ],
        },
        "bond_dashboard": {
            "countries": {
                "US": {
                    "tenors": {
                        "10Y": {
                            "series": [{"date": "2023-01-01", "value": 3.5}],
                            "latest": 4.1,
                            "year_ago": 3.5,
                            "change_bps": 60,
                        }
                    }
                }
            }
        },
        "country_dashboard": {
            "Inflation": {
                "metric": "Inflation",
                "countries": {
                    "US": [{"date": "2023-12-01", "value": 3.4}, {"date": "2024-01-01", "value": 3.1}],
                    "Japan": [],
                },
                "series_used": {"US": {"source": "fred"}},
                "latest_observation_dates": {"US": "2024-01-01"},
                "max_age_days": {"inflation": 90},
            }
        },
    }

    slim = auto_weekly_report._prepare_prompt_bundle(raw)

    assert "dates" not in slim["labor_market"]["series"]["initial_claims"]
    assert "values" not in slim["housing"]["series"]["housing_starts"]
    assert slim["labor_market"]["series"]["initial_claims"]["latest"]["value"] == 210
    assert "by_source" not in slim["central_banks"]
    assert "content_preview" not in slim["central_banks"]["items"][0]
    assert "series" not in slim["bond_dashboard"]["countries"]["US"]["tenors"]["10Y"]
    assert slim["country_dashboard"]["Inflation"]["countries"]["US"] == {"date": "2024-01-01", "value": 3.1}
    assert slim["country_dashboard"]["Inflation"]["countries"]["Japan"] is None


def test_weekly_hedge_position_detection_covers_role_type_and_metadata():
    assert auto_weekly_report._is_hedge_position({"ticker": "SH", "role": "hedge"})
    assert auto_weekly_report._is_hedge_position({"ticker": "QQQ", "type": "hedge"})
    assert auto_weekly_report._is_hedge_position({"ticker": "IWM", "position_type": "hedge"})
    assert auto_weekly_report._is_hedge_position({"ticker": "TLT", "object_type": "HedgePosition"})
    assert auto_weekly_report._is_hedge_position({"ticker": "GLD", "_meta": {"object_type": "HedgePosition"}})
    assert not auto_weekly_report._is_hedge_position({"ticker": "AAA", "role": "position"})


def test_weekly_thesis_data_excludes_hedges_from_cached_portfolio_state(monkeypatch, tmp_path):
    state_path = tmp_path / "portfolio_state.json"
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    for ticker in ("AAA", "SH", "QQQ", "IWM", "TLT", "GLD"):
        (thesis_dir / f"{ticker}.md").write_text(f"{ticker} thesis", encoding="utf-8")

    state_path.write_text(
        json.dumps(
            {
                "positions": [
                    {
                        "ticker": "AAA",
                        "role": "position",
                        "direction": "long",
                        "conviction": 3,
                        "asset": "equity",
                        "group_name": "Core",
                        "group_conviction": 3,
                    },
                    {"ticker": "SH", "role": "hedge", "direction": "short", "conviction": 1, "asset": "equity"},
                    {"ticker": "QQQ", "type": "hedge", "direction": "short", "conviction": 1, "asset": "equity"},
                    {
                        "ticker": "IWM",
                        "position_type": "hedge",
                        "direction": "short",
                        "conviction": 1,
                        "asset": "equity",
                    },
                    {
                        "ticker": "TLT",
                        "object_type": "HedgePosition",
                        "direction": "long",
                        "conviction": 1,
                        "asset": "bond",
                    },
                    {
                        "ticker": "GLD",
                        "_meta": {"object_type": "HedgePosition"},
                        "direction": "long",
                        "conviction": 1,
                        "asset": "commodity",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    ta_tickers: list[str] = []

    def fake_ta(ticker, lookback="2Y"):
        ta_tickers.append(ticker)
        return {"summary": [{"Indicator": "Trend", "Value": "ok", "Signal": "neutral", "Bias": "neutral"}]}

    monkeypatch.setenv("AUTO_REPORT_PORTFOLIO_STATE_PATH", str(state_path))
    monkeypatch.setattr(auto_weekly_report, "THESES_DIR", thesis_dir)
    monkeypatch.setattr(
        auto_weekly_report,
        "collect_news_digest_context",
        lambda days: {"window_days": days, "digests": [], "counts": {"digests": 0, "stories": 0}},
    )
    _module(monkeypatch, "portfolio.technical_analysis.technical_analysis", get_data=fake_ta)
    _module(
        monkeypatch,
        "portfolio.momentum.price_momentum.momentum",
        get_data=lambda: {
            "results": [
                {"ticker": "AAA", "avg20_roc63": 1.0, "rel_roc42": 2.0, "avg10_rel_roc": 3.0},
                {"ticker": "QQQ", "avg20_roc63": -1.0, "rel_roc42": -2.0, "avg10_rel_roc": -3.0},
            ]
        },
    )

    theses = auto_weekly_report.load_theses()
    data = auto_weekly_report.collect_thesis_data()
    _system_msg, user_msg = auto_weekly_report._build_thesis_prompt(data, web_search=False)

    assert theses == {"AAA": "AAA thesis"}
    assert [row["ticker"] for row in data["portfolio"]] == ["AAA"]
    assert data["portfolio_groups"][0]["members"] == ["AAA"]
    assert ta_tickers == ["AAA"]
    assert "### AAA" in user_msg
    for hedge_ticker in ("SH", "QQQ", "IWM", "TLT", "GLD"):
        assert f"### {hedge_ticker}" not in user_msg
        assert f"{hedge_ticker} thesis" not in user_msg


def test_weekly_commentary_finalizer_does_not_append_source_footer():
    commentary = "## Weekly Commentary\n\nMarket text with inline citations preserved by the model."

    result = auto_weekly_report._finalize_commentary_markdown(
        commentary,
        [
            ("Fed release", "https://example.com/fed"),
            ("Duplicate Fed release", "https://example.com/fed"),
        ],
    )

    assert result == commentary
    assert "## Sources" not in result
    assert "https://example.com/fed" not in result


def test_weekly_thesis_summary_filter_removes_unexpected_hedge_tickers():
    summary = {
        "thesis_evaluations": [{"ticker": "AAA"}, {"ticker": "SH"}],
        "positions_reviewed": ["AAA", "SH"],
        "thesis_strengthened": ["AAA"],
        "thesis_weakened": ["SH"],
        "positions_needing_reassessment": ["SH"],
        "missing_theses": ["SH"],
        "material_developments": [{"ticker": "AAA"}, {"ticker": "SH"}],
    }

    result = auto_weekly_report._filter_thesis_summary_to_portfolio(summary, [{"ticker": "AAA"}])

    assert result["thesis_evaluations"] == [{"ticker": "AAA"}]
    assert result["positions_reviewed"] == ["AAA"]
    assert result["thesis_strengthened"] == ["AAA"]
    assert result["thesis_weakened"] == []
    assert result["positions_needing_reassessment"] == []
    assert result["missing_theses"] == []
    assert result["material_developments"] == [{"ticker": "AAA"}]


def test_weekly_and_daily_prompts_call_out_expanded_macro_sources():
    bundle = {
        "labor_market": {},
        "housing": {},
        "central_banks": {},
        "yield_curve": {},
        "bond_dashboard": {},
        "country_dashboard": {},
        "news_digests": {
            "window_days": 8,
            "digests": [
                {
                    "title": "User Digest",
                    "generated_date": "2026-05-01",
                    "sections": [{"name": "macro", "stories": [{"headline": "Fed story", "notes": []}]}],
                }
            ],
            "counts": {"digests": 1, "stories": 1},
        },
    }

    weekly_msg = auto_weekly_report._build_user_message(bundle, "## Weekly Performance", web_search=False)
    daily_msg = auto_daily_report._build_pass1_user_message(bundle, "## Weekly Performance")

    for key in (
        "labor_market",
        "housing",
        "central_banks",
        "yield_curve",
        "bond_dashboard",
        "country_dashboard",
        "news_digests",
    ):
        assert key in weekly_msg
        assert key in daily_msg
    assert "policy, rates, growth" in weekly_msg
    assert "Liquidity/Rates" in daily_msg
    assert "curated news context" in weekly_msg
    assert "user-curated high-signal leads" in daily_msg
