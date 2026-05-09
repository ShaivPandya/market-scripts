import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import portfolio.momentum.fundamental_momentum.financials_single as fs


def _fact(
    *,
    start: str,
    end: str,
    val: float,
    fy: int,
    fp: str = "FY",
    form: str = "10-K",
    filed: str,
    accn: str,
    frame: str | None = None,
) -> dict:
    out = {
        "start": start,
        "end": end,
        "val": val,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
        "accn": accn,
    }
    if frame is not None:
        out["frame"] = frame
    return out


def _us_gaap(concept: str, unit: str, rows: list[dict]) -> dict:
    return {concept: {"units": {unit: rows}}}


def _metric_row(period_end: str, value: float) -> dict:
    return {"period_end": period_end, "value": value}


def test_annual_revenue_excludes_interim_10k_facts_and_prefers_own_filing():
    rows = [
        _fact(
            start="2024-09-29",
            end="2025-09-27",
            val=1443276000,
            fy=2025,
            filed="2025-11-14",
            accn="fy2025",
            frame="CY2025",
        ),
        _fact(
            start="2023-10-01",
            end="2024-09-28",
            val=1518056000,
            fy=2025,
            filed="2025-11-14",
            accn="fy2025-comparative",
            frame="CY2024",
        ),
        _fact(
            start="2023-10-01",
            end="2024-09-28",
            val=1518056000,
            fy=2024,
            filed="2024-11-15",
            accn="fy2024-own",
        ),
        _fact(
            start="2022-10-02",
            end="2023-09-30",
            val=1655255000,
            fy=2023,
            filed="2023-11-20",
            accn="fy2023-own",
        ),
        _fact(
            start="2023-07-02",
            end="2023-09-30",
            val=305147000,
            fy=2023,
            filed="2023-11-20",
            accn="fy2023-q4-inside-10k",
            frame="CY2023Q3",
        ),
        _fact(
            start="2023-04-02",
            end="2023-07-01",
            val=373356000,
            fy=2023,
            filed="2023-11-20",
            accn="fy2023-q3-inside-10k",
        ),
        _fact(
            start="2022-01-01",
            end="2022-12-31",
            val=10968000000,
            fy=2022,
            filed="2023-03-01",
            accn="fy2022-own",
        ),
        _fact(
            start="2022-07-01",
            end="2022-09-30",
            val=2979000000,
            fy=2023,
            filed="2024-02-27",
            accn="fy2022-q3-inside-10k",
            frame="CY2022Q3",
        ),
    ]
    us_gaap = _us_gaap("RevenueFromContractWithCustomerExcludingAssessedTax", "USD", rows)

    annual, _quarterly = fs._build_revenue_rows(us_gaap, "0000000000", None)

    assert [r["period_end"] for r in annual] == [
        "2025-09-27",
        "2024-09-28",
        "2023-09-30",
        "2022-12-31",
    ]
    assert len({r["period_label"] for r in annual}) == len(annual)
    assert annual[1]["accn"] == "fy2024-own"
    assert all(r["period_end"] not in {"2023-07-01", "2022-09-30"} for r in annual)


def test_direct_annual_eps_uses_full_year_filter():
    rows = [
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=2.5,
            fy=2025,
            filed="2026-02-25",
            accn="fy2025",
            frame="CY2025",
        ),
        _fact(
            start="2024-01-01",
            end="2024-12-31",
            val=1.5,
            fy=2025,
            filed="2026-02-25",
            accn="fy2025-comparative",
            frame="CY2024",
        ),
        _fact(
            start="2024-01-01",
            end="2024-12-31",
            val=1.5,
            fy=2024,
            filed="2025-02-24",
            accn="fy2024-own",
        ),
        _fact(
            start="2024-10-01",
            end="2024-12-31",
            val=0.4,
            fy=2024,
            filed="2025-02-24",
            accn="fy2024-q4-inside-10k",
            frame="CY2024Q4",
        ),
    ]
    us_gaap = _us_gaap("EarningsPerShareDiluted", "USD/shares", rows)

    annual, _quarterly = fs._build_eps_rows(us_gaap, "0000000000", None)

    assert [r["period_end"] for r in annual] == ["2025-12-31", "2024-12-31"]
    assert annual[1]["accn"] == "fy2024-own"
    assert all(r["period_end"] != "2024-12-31" or r["value"] == 1.5 for r in annual)


def test_derived_annual_eps_uses_full_year_filter():
    net_income = [
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=1000,
            fy=2025,
            filed="2026-02-25",
            accn="fy2025",
        ),
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="q1-2025",
        ),
    ]
    shares = [
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=100,
            fy=2025,
            filed="2026-02-25",
            accn="fy2025",
        ),
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="q1-2025",
        ),
    ]
    us_gaap = {
        "NetIncomeLoss": {"units": {"USD": net_income}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": shares}},
    }

    annual, _quarterly = fs._build_eps_rows(us_gaap, "0000000000", None)

    assert len(annual) == 1
    assert annual[0]["period_end"] == "2025-12-31"
    assert annual[0]["value"] == 10


def test_quarterly_eps_fills_missing_q4_from_derived_net_income_and_shares():
    direct_eps = [
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=1.0,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="eps-q1",
        ),
        _fact(
            start="2025-04-01",
            end="2025-06-30",
            val=2.0,
            fy=2025,
            fp="Q2",
            form="10-Q",
            filed="2025-08-01",
            accn="eps-q2",
        ),
        _fact(
            start="2025-07-01",
            end="2025-09-30",
            val=3.0,
            fy=2025,
            fp="Q3",
            form="10-Q",
            filed="2025-11-01",
            accn="eps-q3",
        ),
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=10.0,
            fy=2025,
            filed="2026-02-20",
            accn="eps-fy",
        ),
    ]
    net_income = [
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="ni-q1",
        ),
        _fact(
            start="2025-01-01",
            end="2025-06-30",
            val=300,
            fy=2025,
            fp="Q2",
            form="10-Q",
            filed="2025-08-01",
            accn="ni-q2-ytd",
        ),
        _fact(
            start="2025-01-01",
            end="2025-09-30",
            val=600,
            fy=2025,
            fp="Q3",
            form="10-Q",
            filed="2025-11-01",
            accn="ni-q3-ytd",
        ),
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=1000,
            fy=2025,
            filed="2026-02-20",
            accn="ni-fy",
        ),
    ]
    shares = [
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="shares-q1",
        ),
        _fact(
            start="2025-01-01",
            end="2025-06-30",
            val=100,
            fy=2025,
            fp="Q2",
            form="10-Q",
            filed="2025-08-01",
            accn="shares-q2-ytd",
        ),
        _fact(
            start="2025-01-01",
            end="2025-09-30",
            val=100,
            fy=2025,
            fp="Q3",
            form="10-Q",
            filed="2025-11-01",
            accn="shares-q3-ytd",
        ),
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=100,
            fy=2025,
            filed="2026-02-20",
            accn="shares-fy",
        ),
    ]
    us_gaap = {
        "EarningsPerShareDiluted": {"units": {"USD/shares": direct_eps}},
        "NetIncomeLoss": {"units": {"USD": net_income}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": shares}},
    }

    _annual, quarterly = fs._build_eps_rows(us_gaap, "0000000000", None)

    assert [r["period_label"] for r in quarterly[:4]] == ["Q4 2025", "Q3 2025", "Q2 2025", "Q1 2025"]
    assert quarterly[0]["period_end"] == "2025-12-31"
    assert quarterly[0]["value"] == 4.0
    assert quarterly[0]["accn"] == "ni-fy"
    assert quarterly[1]["accn"] == "eps-q3"


def test_quarterly_selection_prefers_period_own_filing_over_later_comparative():
    rows = [
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=1000,
            fy=2025,
            filed="2026-02-25",
            accn="fy2025",
            frame="CY2025",
        ),
        _fact(
            start="2024-01-01",
            end="2024-12-31",
            val=900,
            fy=2024,
            filed="2025-02-24",
            accn="fy2024",
            frame="CY2024",
        ),
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2025,
            fp="Q1",
            form="10-Q",
            filed="2025-05-01",
            accn="q1-2025-own",
            frame="CY2025Q1",
        ),
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fy=2026,
            fp="Q1",
            form="10-Q",
            filed="2026-05-01",
            accn="q1-2025-comparative",
            frame="CY2025Q1",
        ),
    ]

    quarterly = fs._quarterly_fact_entries(rows)
    q1 = next(r for r in quarterly if r["end"] == "2025-03-31")

    assert q1["accn"] == "q1-2025-own"


def test_interest_expense_rows_preserve_edgar_concept_priority():
    us_gaap = {
        "InterestExpense": {
            "units": {
                "USD": [
                    _fact(
                        start="2024-01-01",
                        end="2024-12-31",
                        val=10.0,
                        fy=2024,
                        filed="2025-02-01",
                        accn="interest-expense",
                    )
                ]
            }
        },
        "InterestAndDebtExpense": {
            "units": {
                "USD": [
                    _fact(
                        start="2025-01-01",
                        end="2025-12-31",
                        val=99.0,
                        fy=2025,
                        filed="2026-02-01",
                        accn="interest-and-debt-expense",
                    )
                ]
            }
        },
    }

    annual, _quarterly = fs._build_flow_rows(
        us_gaap,
        fs.INTEREST_EXPENSE_RATIO_CONCEPTS,
        "0000000000",
        None,
        preserve_concept_order=True,
    )

    assert annual[0]["value"] == 10.0
    assert annual[0]["accn"] == "interest-expense"


def test_get_data_falls_back_to_yfinance_when_edgar_has_no_revenue_or_eps(monkeypatch):
    income_stmt = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [160.0, 4.0, 32.0, 24.0, -8.0],
            pd.Timestamp("2024-12-31"): [140.0, 3.0, 28.0, 21.0, -7.0],
            pd.Timestamp("2023-12-31"): [120.0, 2.5, 18.0, 12.0, -6.0],
            pd.Timestamp("2022-12-31"): [100.0, 2.0, 10.0, 8.0, -5.0],
        },
        index=["Total Revenue", "Diluted EPS", "Operating Income", "Net Income", "Interest Expense"],
    )

    class FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker
            self.info = {
                "longName": "Taiwan Semiconductor Manufacturing Company Limited",
                "financialCurrency": "TWD",
            }
            self.income_stmt = income_stmt
            self.financials = income_stmt

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    monkeypatch.setattr(fs, "get_cik_for_ticker", lambda ticker: "0001046179")
    monkeypatch.setattr(
        fs,
        "fetch_companyfacts_by_cik",
        lambda cik: {"entityName": "Taiwan Semiconductor Manufacturing Company Limited", "facts": {"us-gaap": {}}},
    )
    monkeypatch.setattr(fs, "fetch_submissions_by_cik", lambda cik: {})

    out = fs.get_data("tsm")

    assert out["ticker"] == "TSM"
    assert out["data_source"] == "yfinance"
    assert out["cik"] is None
    assert out["financial_currency"] == "TWD"
    assert out["quarterly"] == {"revenue": [], "eps": []}
    assert out["annual"]["revenue"][0]["period_end"] == "2025-12-31"
    assert out["annual"]["revenue"][0]["form"] == "Yahoo Finance"
    assert out["annual"]["revenue"][0]["yoy_growth"] == pytest.approx((160.0 - 140.0) / 140.0)
    assert out["metrics"]["revenue_cagr_3y"] == pytest.approx((160.0 / 100.0) ** (1.0 / 3.0) - 1.0)
    assert out["metrics"]["eps_cagr_3y"] == pytest.approx((4.0 / 2.0) ** (1.0 / 3.0) - 1.0)
    assert out["metrics"]["avg_yoy_revenue_growth_3q"] is None
    assert out["metrics"]["avg_yoy_eps_growth_3q"] is None
    assert out["metrics"]["operating_margin"] == pytest.approx(32.0 / 160.0)
    assert out["metrics"]["operating_margin_basis"] == "annual"
    assert out["metrics"]["operating_margin_period_end"] == "2025-12-31"
    assert out["metrics"]["net_income_margin"] == pytest.approx(24.0 / 160.0)
    assert out["metrics"]["interest_coverage"] == pytest.approx(32.0 / 8.0)
    assert out["metrics"]["interest_coverage_flag"] is False
    assert out["metrics"]["interest_coverage_warning_threshold"] == fs.INTEREST_COVERAGE_WARNING_THRESHOLD


def test_profitability_metrics_use_aligned_ttm_periods():
    periods = ["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"]
    quarterly_revenue = [
        _metric_row(period, value) for period, value in zip(periods, [100.0, 90.0, 80.0, 70.0], strict=True)
    ]
    quarterly_operating_income = [
        _metric_row(period, value) for period, value in zip(periods, [20.0, 18.0, 16.0, 14.0], strict=True)
    ]
    quarterly_net_income = [
        _metric_row(period, value) for period, value in zip(periods, [10.0, 9.0, 8.0, 7.0], strict=True)
    ]
    quarterly_interest_expense = [_metric_row(period, 5.0) for period in periods]

    metrics = fs._build_profitability_metrics(
        [],
        quarterly_revenue,
        [],
        quarterly_operating_income,
        [],
        quarterly_net_income,
        [],
        quarterly_interest_expense,
    )

    assert metrics["operating_margin"] == pytest.approx(68.0 / 340.0)
    assert metrics["operating_margin_basis"] == "ttm"
    assert metrics["operating_margin_period_end"] == "2025-12-31"
    assert metrics["net_income_margin"] == pytest.approx(34.0 / 340.0)
    assert metrics["net_income_margin_basis"] == "ttm"
    assert metrics["interest_coverage"] == pytest.approx(68.0 / 20.0)
    assert metrics["interest_coverage_basis"] == "ttm"
    assert metrics["interest_coverage_flag"] is True


def test_profitability_metrics_fall_back_to_latest_aligned_annual_period():
    quarterly_revenue = [_metric_row("2025-12-31", 100.0), _metric_row("2025-09-30", 90.0)]
    quarterly_operating_income = [_metric_row("2025-12-31", 25.0), _metric_row("2025-09-30", 20.0)]
    annual_revenue = [_metric_row("2024-12-31", 200.0), _metric_row("2023-12-31", 180.0)]
    annual_operating_income = [_metric_row("2024-12-31", 50.0), _metric_row("2023-12-31", 36.0)]
    annual_net_income = [_metric_row("2024-12-31", 30.0), _metric_row("2023-12-31", 27.0)]
    annual_interest_expense = [_metric_row("2024-12-31", 10.0), _metric_row("2023-12-31", 9.0)]

    metrics = fs._build_profitability_metrics(
        annual_revenue,
        quarterly_revenue,
        annual_operating_income,
        quarterly_operating_income,
        annual_net_income,
        [],
        annual_interest_expense,
        [],
    )

    assert metrics["operating_margin"] == pytest.approx(50.0 / 200.0)
    assert metrics["operating_margin_basis"] == "annual"
    assert metrics["operating_margin_period_end"] == "2024-12-31"
    assert metrics["net_income_margin"] == pytest.approx(30.0 / 200.0)
    assert metrics["interest_coverage"] == pytest.approx(50.0 / 10.0)
    assert metrics["interest_coverage_flag"] is False


def test_profitability_metrics_return_null_for_mismatched_periods():
    metrics = fs._build_profitability_metrics(
        [_metric_row("2024-12-31", 200.0)],
        [],
        [_metric_row("2023-12-31", 50.0)],
        [],
        [_metric_row("2023-12-31", 30.0)],
        [],
        [_metric_row("2022-12-31", 10.0)],
        [],
    )

    assert metrics["operating_margin"] is None
    assert metrics["operating_margin_basis"] is None
    assert metrics["net_income_margin"] is None
    assert metrics["interest_coverage"] is None
    assert metrics["interest_coverage_flag"] is False


def test_negative_operating_income_interest_coverage_is_flagged():
    metrics = fs._build_profitability_metrics(
        [],
        [],
        [_metric_row("2024-12-31", -20.0)],
        [],
        [],
        [],
        [_metric_row("2024-12-31", -10.0)],
        [],
    )

    assert metrics["interest_coverage"] == pytest.approx(-2.0)
    assert metrics["interest_coverage_basis"] == "annual"
    assert metrics["interest_coverage_flag"] is True


def test_zero_or_missing_interest_expense_returns_null_without_flag():
    zero_metrics = fs._build_profitability_metrics(
        [],
        [],
        [_metric_row("2024-12-31", 20.0)],
        [],
        [],
        [],
        [_metric_row("2024-12-31", 0.0)],
        [],
    )
    missing_metrics = fs._build_profitability_metrics(
        [],
        [],
        [_metric_row("2024-12-31", 20.0)],
        [],
        [],
        [],
        [],
        [],
    )

    assert zero_metrics["interest_coverage"] is None
    assert zero_metrics["interest_coverage_flag"] is False
    assert missing_metrics["interest_coverage"] is None
    assert missing_metrics["interest_coverage_flag"] is False
