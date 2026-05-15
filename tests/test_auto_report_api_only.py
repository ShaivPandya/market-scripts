from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_top50_breadth_api_only_falls_back_when_postgres_unavailable(monkeypatch):
    from equities.market_technicals import top50_breadth

    monkeypatch.setenv("AUTO_REPORT_API_ONLY", "1")
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(top50_breadth, "compute_top50", lambda: pd.DataFrame({"ticker": ["AAA", "BBB"]}))

    captured: dict[str, list[str]] = {}

    def fake_compute_metrics(tickers, period="2y", prices_df=None):
        captured["tickers"] = list(tickers)
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "rows": 30,
                    "below_50dma": False,
                    "dist_days_last20": 0,
                    "has_3plus_dist_days": False,
                    "broke_prior20_low_last_week": False,
                }
                for ticker in tickers
            ]
        )

    monkeypatch.setattr(top50_breadth, "compute_metrics", fake_compute_metrics)

    result = top50_breadth.get_data()

    assert captured["tickers"] == ["AAA", "BBB"]
    assert result["universe_size"] == 2


def test_optional_report_caches_use_local_sqlite_in_api_only_mode(monkeypatch, tmp_path):
    from macro.central_banks import central_bank
    from macro.industry import industry_monitor

    monkeypatch.setenv("AUTO_REPORT_API_ONLY", "1")
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(central_bank, "_resolve_db_path", lambda db_path=None: str(tmp_path / "central.sqlite3"))
    monkeypatch.setattr(industry_monitor, "_resolve_db_path", lambda db_path=None: str(tmp_path / "industry.sqlite3"))
    monkeypatch.setattr(industry_monitor, "_fetch_missing_price_reactions", lambda _conn: None)

    central = central_bank.get_data(refresh=False)
    industry = industry_monitor.get_data(refresh=False)

    assert "error" not in central
    assert "error" not in industry


def test_runtime_position_reads_use_cached_state_in_api_only_mode(monkeypatch, tmp_path):
    state_path = tmp_path / "portfolio_state.json"
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
                    },
                    {
                        "ticker": "IWM",
                        "role": "hedge",
                        "direction": "short",
                        "conviction": 1,
                        "asset": "equity",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTO_REPORT_API_ONLY", "1")
    monkeypatch.setenv("AUTO_REPORT_PORTFOLIO_STATE_PATH", str(state_path))
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from ontology import runtime_read_service

    df = runtime_read_service.get_positions_df()
    rows_with_hedges = runtime_read_service.get_positions(include_hedges=True)
    hedges = runtime_read_service.get_hedge_positions()

    assert df["ticker"].tolist() == ["AAA"]
    assert [row["ticker"] for row in rows_with_hedges] == ["AAA", "IWM"]
    assert [row["ticker"] for row in hedges] == ["IWM"]


def test_auto_daily_report_module_api_only_runs_without_database_url(tmp_path):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    _write_llm_utils_stub(stub_dir / "llm_utils.py")
    _write_sitecustomize(stub_dir / "sitecustomize.py")

    state_path = tmp_path / "portfolio_state.json"
    state_path.write_text(
        json.dumps(
            {
                "book_size": 80500,
                "positions": [
                    {
                        "ticker": "MU",
                        "role": "position",
                        "direction": "long",
                        "conviction": 3,
                        "group_name": "Semis",
                        "group_conviction": 3,
                        "shares": 10,
                        "cost_basis": 100,
                        "asset": "equity",
                        "contrarian": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PYTHONPATH": f"{stub_dir}{os.pathsep}{PROJECT_ROOT}",
        "AUTO_REPORT_API_ONLY": "1",
        "AUTO_REPORT_PORTFOLIO_STATE_PATH": str(state_path),
        "FORCE_RUN": "1",
        "STATE_DB_BACKEND": "postgres",
        "LLM_PROVIDER": "anthropic",
    }
    env.pop("DATABASE_URL", None)
    env.pop("GITHUB_TOKEN", None)

    result = subprocess.run(
        [sys.executable, "-m", "auto_report.auto_daily_report", "--force", "--no-search", "--book", "80500"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined_output
    assert "DATABASE_URL is required" not in combined_output
    assert "Daily risk report run complete" in combined_output


def _write_llm_utils_stub(path: Path) -> None:
    path.write_text(
        r"""
from __future__ import annotations

import json
from types import SimpleNamespace

MODEL_LOW = "low"
MODEL_MID = "mid"
MODEL_HIGH = "high"


def reasoning_effort_for_tier(_tier):
    return None


def extract_citations(_response):
    return []


def extract_text(_response):
    return ""


def call_llm_text(prompt, model="high", api_key=None, max_tokens=16384, system=None, enable_web_search=False, reasoning_effort=None):
    response = SimpleNamespace(usage=None)
    if "PASS1_STANCE_JSON" in prompt:
        payload = {
            "stance": "Neutral / Watchful",
            "target_leverage": 1.5,
            "leverage_rationale": "Stubbed neutral stance.",
            "confidence": "medium",
            "market_regime_assessment": {
                "headline": "Neutral test regime.",
                "dominant_character": "Markets are balanced in the stubbed test fixture.",
                "main_tension": "The test fixture is designed to avoid external dependencies.",
            },
            "regime_evidence": [
                {"dimension": "Market Behavior", "rating": "Neutral", "evidence": "Stub market behavior is neutral.", "stance_implication": "Keep leverage neutral."},
                {"dimension": "Macro Momentum", "rating": "Neutral", "evidence": "Stub macro momentum is neutral.", "stance_implication": "Keep leverage neutral."},
                {"dimension": "Liquidity", "rating": "Neutral", "evidence": "Stub liquidity is neutral.", "stance_implication": "Keep leverage neutral."},
                {"dimension": "Positioning", "rating": "Neutral", "evidence": "Stub positioning is neutral.", "stance_implication": "Keep leverage neutral."},
                {"dimension": "Risk Sentiment", "rating": "Neutral", "evidence": "Stub risk sentiment is neutral.", "stance_implication": "Keep leverage neutral."},
                {"dimension": "Cycle Position", "rating": "Neutral", "evidence": "Stub cycle position is neutral.", "stance_implication": "Keep leverage neutral."},
            ],
            "six_dimensions": {
                "market_behavior": "Neutral",
                "macro_momentum": "Neutral",
                "liquidity": "Neutral",
                "positioning": "Neutral",
                "risk_sentiment": "Neutral",
                "cycle_position": "Neutral",
            },
            "drivers": ["Stub fixture"],
            "watchlist": {
                "risks_to_upside": [{"trigger": "Stub upside trigger", "implication": "Increase only after confirmation."}],
                "risks_to_downside": [{"trigger": "Stub downside trigger", "implication": "Reduce if risk rises."}],
            },
            "watchlist_triggers": ["Stub upside trigger", "Stub downside trigger"],
        }
        return "# Stance Rationale\n\nStub rationale.\n\n<!-- PASS1_STANCE_JSON -->\n" + json.dumps(payload), [], response
    if "DAILY_SUMMARY_JSON" in prompt:
        payload = {
            "risk_level": "moderate",
            "top_risks": ["Stub risk"],
            "positions_flagged": [],
            "largest_adjustments": ["MU"],
        }
        return "## Risk Summary\n\nStub risk summary.\n\n<!-- DAILY_SUMMARY_JSON -->\n" + json.dumps(payload), [], response
    if "RECOMMENDATIONS_JSON" in prompt:
        payload = {
            "report_type": "daily",
            "as_of": "2026-05-15",
            "stance": "Neutral / Watchful",
            "recommendation_status": "clear",
            "critical_data_quality": "ok",
            "blocked_reasons": [],
            "do_nothing_rationale": "No stub action required.",
            "what_changed": ["Stub fixture"],
            "recommended_actions": [
                {
                    "action": "do_nothing",
                    "ticker": None,
                    "instrument": "portfolio",
                    "horizon": "1 trading day",
                    "target_change": "",
                    "rationale": "Stubbed no-action recommendation.",
                    "evidence": ["Stub evidence"],
                    "disconfirming_evidence": [],
                    "catalyst": "",
                    "invalidation": "",
                    "expected_onset_window": "",
                    "confidence": 0.5,
                    "source_quality": "ok",
                    "approval_required": False,
                    "decision_quality": {},
                }
            ],
            "alternatives": [],
            "opportunity_cost": [],
        }
        return "Stub recommendations.\n\n<!-- RECOMMENDATIONS_JSON -->\n" + json.dumps(payload), [], response
    return "Stub response.", [], response
""",
        encoding="utf-8",
    )


def _write_sitecustomize(path: Path) -> None:
    path.write_text(
        r"""
from __future__ import annotations

import sys
import types

import pandas as pd


def install(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


dates = pd.bdate_range("2026-05-01", periods=40)
prices = pd.DataFrame({"MU": [100 + i for i in range(40)], "SPY": [500 + i for i in range(40)], "SH": [20 for _ in range(40)]}, index=dates)

install("equities.index_dashboard.index_dashboard", INDEX_ORDER=["SPX"], get_data=lambda _period: {"indices": {"SPX": [100, 101]}})
install("fx.fx_dashboard.fx_dashboard", PAIR_ORDER=["EURUSD"], get_data=lambda _period: {"pairs": {"EURUSD": [1.0, 1.1]}})
install("commodities.commodities_dashboard", COMMODITY_ORDER=["Gold"], get_data=lambda _period: {"commodities": {"Gold": [2000, 2010]}})
install("equities.market_technicals.market_breadth", get_data=lambda period="1y": {"pct_above_200dma": 50.0, "pct_above_20dma": 50.0, "pct_at_20day_low": 5.0})
install("equities.market_technicals.top50_breadth", get_data=lambda: {"pct_below_50dma": 10.0, "pct_3plus_dist": 5.0, "pct_broke_20low": 2.0, "universe_size": 50})
install("equities.market_technicals.vix_term_structure", get_data=lambda: {"ratio": 1.1, "signal": "Neutral"})
install("equities.sector_metrics.sector_metrics", get_data=lambda: {"weights_df": pd.DataFrame({"Technology": [1.0]}), "summary": {}})
install("macro.positioning.positioning", DATASETS={"tff_futures_only": "dataset"}, DEFAULT_DOMAIN="example.com", fetch_multiple_instruments=lambda **_kwargs: [{"instrument": "SP500"}])
install("macro.economic_growth.economic_growth", get_data=lambda: {"growth": "ok"})
install("macro.labor_market.labor_market", get_data=lambda: {"labor": "ok"})
install("macro.housing.housing", get_data=lambda: {"housing": "ok"})
install("macro.liquidity.liquidity", get_snapshot=lambda: {"liquidity": "ok"})
install("macro.central_banks.central_bank", get_data=lambda refresh=False: {"items": [], "counts": {"total": 0}})
install("government_bonds.yield_curve", get_data=lambda lookback_days=90: {"curve": "ok"})
install("government_bonds.bond_dashboard", get_data=lambda: {"bonds": "ok"})
install("macro.country_dashboard.country_dashboard", METRICS=["Inflation"], get_data=lambda metric="Inflation": {"metric": metric, "countries": {"US": [{"date": "2026-05-01", "value": 3.0}]}})
install("macro.industry.industry_monitor", get_data=lambda refresh=False: {"industry": "ok"})
install("macro.sentiment.sentiment", get_surveys=lambda: {"aaii": [], "naaim": [], "errors": {}}, get_put_call=lambda: {}, get_volatility=lambda: [])
install("portfolio.news_digests", get_report_context=lambda days: {"window_days": days, "digests": [], "counts": {"digests": 0, "stories": 0}})


def get_ta_data(_ticker, lookback="2Y"):
    return {"summary": {"trend": "neutral"}}


def get_ratio_data(*_args, **_kwargs):
    return {"stats": {"start_ratio": 1.0, "end_ratio": 1.0, "change_pct": 0.0}}


install("portfolio.technical_analysis.technical_analysis", get_data=get_ta_data, get_ratio_data=get_ratio_data)
install("portfolio.momentum.price_momentum.momentum", get_data=lambda: {"momentum": "ok"})


def fetch_currencies(tickers):
    return {ticker: "USD" for ticker in tickers}


def get_required_fx_tickers(_currencies):
    return []


def download_prices(tickers, _fx_tickers):
    return pd.DataFrame({ticker: prices.get(ticker, prices["MU"]) for ticker in tickers}, index=prices.index)


def to_usd_price(local_px, _ccy, _prices_all):
    return local_px


def compute_defense_volatility(_usd_prices, tickers):
    return pd.Series({ticker: 0.2 for ticker in tickers})


def compute_severe_drawdown_flags(_usd_prices, tickers):
    return {ticker: False for ticker in tickers}


def compute_contrarian_long_metrics(_prices_all, tickers):
    return pd.DataFrame(index=tickers)


def compute_beta_frame(_rets, valid_tickers):
    return pd.DataFrame({"beta_spy": [1.0 for _ in valid_tickers], "beta_iwm": [1.0 for _ in valid_tickers]}, index=valid_tickers), None, None


install(
    "portfolio.portfolio_optimizer.portfolio_analyzer",
    MARKET_TICKER_LONG="SPY",
    MARKET_TICKER_SHORT="SH",
    compute_beta_frame=compute_beta_frame,
    compute_contrarian_long_metrics=compute_contrarian_long_metrics,
    compute_defense_volatility=compute_defense_volatility,
    compute_severe_drawdown_flags=compute_severe_drawdown_flags,
    download_prices=download_prices,
    fetch_currencies=fetch_currencies,
    get_required_fx_tickers=get_required_fx_tickers,
    to_usd_price=to_usd_price,
)


def size_portfolio(positions, book, target_leverage=1.5):
    weights_df = pd.DataFrame(
        [
            {
                "ticker": "MU",
                "group_name": "Semis",
                "group_conviction": 3,
                "direction": "long",
                "conviction": 3,
                "weight": 0.5,
                "beta_spy": 1.0,
                "beta_iwm": 1.0,
                "realized_vol": 0.2,
                "shares": 100,
                "dollar_weight": 40000,
            }
        ]
    )
    return {
        "exposures": {"equity_gross": 0.5, "equity_net": 0.5, "total_gross": 0.5, "total_net": 0.5},
        "constraints": {},
        "net_beta_spy": 0.5,
        "net_beta_iwm": 0.5,
        "weights_df": weights_df,
        "hedges_df": pd.DataFrame(),
    }


install("portfolio.portfolio_optimizer.portfolio_sizer", size_portfolio=size_portfolio)
""",
        encoding="utf-8",
    )
