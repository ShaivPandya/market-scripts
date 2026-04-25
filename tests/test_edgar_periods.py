from datetime import date

import portfolio.momentum.fundamental_momentum._edgar_periods as ep
from portfolio.momentum.fundamental_momentum.edgar_fetcher import (
    extract_quarterly_eps,
    extract_quarterly_revenue,
)


def _fact(
    *,
    end: str,
    val: float,
    fp: str,
    fy: int = 2025,
    form: str = "10-Q",
    filed: str = "2025-05-01",
    accn: str = "accn",
    start: str | None = None,
    frame: str | None = None,
) -> dict:
    out = {
        "end": end,
        "val": val,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
        "accn": accn,
    }
    if start is not None:
        out["start"] = start
    if frame is not None:
        out["frame"] = frame
    return out


def _facts(concept: str, unit: str, rows: list[dict]) -> dict:
    return {"facts": {"us-gaap": {concept: {"units": {unit: rows}}}}}


def test_quarterly_flow_entries_convert_ytd_to_discrete_quarters():
    rows = [
        _fact(start="2025-01-01", end="2025-03-31", val=100, fp="Q1", filed="2025-05-01", accn="q1"),
        _fact(start="2025-01-01", end="2025-06-30", val=250, fp="Q2", filed="2025-08-01", accn="q2-ytd"),
        _fact(start="2025-01-01", end="2025-09-30", val=450, fp="Q3", filed="2025-11-01", accn="q3-ytd"),
        _fact(
            start="2025-01-01",
            end="2025-12-31",
            val=700,
            fp="FY",
            form="10-K",
            filed="2026-02-20",
            accn="fy",
        ),
    ]

    out = ep._quarterly_flow_entries(rows)
    values_by_fp = {row["fp"]: row["val"] for row in out}

    assert values_by_fp == {"Q4": 250, "Q3": 200, "Q2": 150, "Q1": 100}


def test_ytd_flow_value_is_not_returned_raw_without_prior_anchor():
    rows = [
        _fact(start="2025-01-01", end="2025-06-30", val=250, fp="Q2", filed="2025-08-01", accn="q2-ytd"),
        _fact(start="2025-01-01", end="2025-09-30", val=450, fp="Q3", filed="2025-11-01", accn="q3-ytd"),
    ]

    out = ep._quarterly_flow_entries(rows)

    assert [row["end"] for row in out] == ["2025-09-30"]
    assert out[0]["val"] == 200


def test_direct_quarter_fact_beats_ytd_fact_for_same_period():
    rows = [
        _fact(start="2025-01-01", end="2025-03-31", val=100, fp="Q1", filed="2025-05-01", accn="q1"),
        _fact(start="2025-04-01", end="2025-06-30", val=150, fp="Q2", filed="2025-08-01", accn="q2-direct"),
        _fact(start="2025-01-01", end="2025-06-30", val=260, fp="Q2", filed="2025-08-02", accn="q2-ytd"),
    ]

    out = ep._quarterly_flow_entries(rows)
    q2 = next(row for row in out if row["end"] == "2025-06-30")

    assert q2["val"] == 150
    assert q2["accn"] == "q2-direct"


def test_period_owned_filing_beats_later_comparative_fact():
    rows = [
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=100,
            fp="Q1",
            fy=2025,
            filed="2025-05-01",
            accn="q1-own",
            frame="CY2025Q1",
        ),
        _fact(
            start="2025-01-01",
            end="2025-03-31",
            val=999,
            fp="Q1",
            fy=2026,
            filed="2026-05-01",
            accn="q1-comparative",
            frame="CY2025Q1",
        ),
    ]

    out = ep._quarterly_flow_entries(rows)

    assert len(out) == 1
    assert out[0]["accn"] == "q1-own"
    assert out[0]["val"] == 100


def test_quarterly_average_entries_derive_weighted_quarter_averages_from_ytd():
    rows = [
        _fact(end="2025-03-31", val=10, fp="Q1", filed="2025-05-01", accn="q1"),
        _fact(end="2025-06-30", val=12, fp="Q2", filed="2025-08-01", accn="q2-ytd"),
        _fact(end="2025-09-30", val=15, fp="Q3", filed="2025-11-01", accn="q3-ytd"),
        _fact(end="2025-12-31", val=16, fp="FY", form="10-K", filed="2026-02-20", accn="fy"),
    ]

    out = ep._quarterly_average_entries(rows)
    values_by_fp = {row["fp"]: row["val"] for row in out}

    assert values_by_fp == {"Q4": 19, "Q3": 21, "Q2": 14, "Q1": 10}


def test_extract_quarterly_revenue_returns_normalized_values_newest_first_and_respects_n():
    facts = _facts(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "USD",
        [
            _fact(start="2025-01-01", end="2025-03-31", val=100, fp="Q1", filed="2025-05-01", accn="q1"),
            _fact(start="2025-01-01", end="2025-06-30", val=250, fp="Q2", filed="2025-08-01", accn="q2"),
            _fact(start="2025-01-01", end="2025-09-30", val=450, fp="Q3", filed="2025-11-01", accn="q3"),
            _fact(
                start="2025-01-01",
                end="2025-12-31",
                val=700,
                fp="FY",
                form="10-K",
                filed="2026-02-20",
                accn="fy",
            ),
        ],
    )

    out = extract_quarterly_revenue(facts, n=2)

    assert out == [(date(2025, 12, 31), 250.0), (date(2025, 9, 30), 200.0)]


def test_extract_quarterly_eps_derives_from_normalized_net_income_and_average_shares():
    facts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _fact(
                                start="2025-01-01",
                                end="2025-03-31",
                                val=100,
                                fp="Q1",
                                filed="2025-05-01",
                                accn="q1-ni",
                            ),
                            _fact(
                                start="2025-01-01",
                                end="2025-06-30",
                                val=250,
                                fp="Q2",
                                filed="2025-08-01",
                                accn="q2-ni-ytd",
                            ),
                        ]
                    }
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            _fact(end="2025-03-31", val=10, fp="Q1", filed="2025-05-01", accn="q1-sh"),
                            _fact(end="2025-06-30", val=12, fp="Q2", filed="2025-08-01", accn="q2-sh-ytd"),
                        ]
                    }
                },
            }
        }
    }

    out = extract_quarterly_eps(facts, n=2)

    assert out[0][0] == date(2025, 6, 30)
    assert abs(out[0][1] - (150 / 14)) < 1e-12
    assert out[1] == (date(2025, 3, 31), 10.0)


def test_extract_quarterly_eps_fills_direct_series_gaps_from_derived_eps():
    facts = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            _fact(
                                start="2025-01-01",
                                end="2025-03-31",
                                val=1,
                                fp="Q1",
                                filed="2025-05-01",
                                accn="eps-q1",
                            ),
                            _fact(
                                start="2025-04-01",
                                end="2025-06-30",
                                val=2,
                                fp="Q2",
                                filed="2025-08-01",
                                accn="eps-q2",
                            ),
                            _fact(
                                start="2025-07-01",
                                end="2025-09-30",
                                val=3,
                                fp="Q3",
                                filed="2025-11-01",
                                accn="eps-q3",
                            ),
                            _fact(
                                start="2025-01-01",
                                end="2025-12-31",
                                val=10,
                                fp="FY",
                                form="10-K",
                                filed="2026-02-20",
                                accn="eps-fy",
                            ),
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _fact(
                                start="2025-01-01",
                                end="2025-03-31",
                                val=100,
                                fp="Q1",
                                filed="2025-05-01",
                                accn="ni-q1",
                            ),
                            _fact(
                                start="2025-01-01",
                                end="2025-06-30",
                                val=300,
                                fp="Q2",
                                filed="2025-08-01",
                                accn="ni-q2-ytd",
                            ),
                            _fact(
                                start="2025-01-01",
                                end="2025-09-30",
                                val=600,
                                fp="Q3",
                                filed="2025-11-01",
                                accn="ni-q3-ytd",
                            ),
                            _fact(
                                start="2025-01-01",
                                end="2025-12-31",
                                val=1000,
                                fp="FY",
                                form="10-K",
                                filed="2026-02-20",
                                accn="ni-fy",
                            ),
                        ]
                    }
                },
                "WeightedAverageNumberOfDilutedSharesOutstanding": {
                    "units": {
                        "shares": [
                            _fact(end="2025-03-31", val=100, fp="Q1", filed="2025-05-01", accn="shares-q1"),
                            _fact(end="2025-06-30", val=100, fp="Q2", filed="2025-08-01", accn="shares-q2"),
                            _fact(end="2025-09-30", val=100, fp="Q3", filed="2025-11-01", accn="shares-q3"),
                            _fact(
                                end="2025-12-31",
                                val=100,
                                fp="FY",
                                form="10-K",
                                filed="2026-02-20",
                                accn="shares-fy",
                            ),
                        ]
                    }
                },
            }
        }
    }

    out = extract_quarterly_eps(facts, n=4)

    assert out == [
        (date(2025, 12, 31), 4.0),
        (date(2025, 9, 30), 3.0),
        (date(2025, 6, 30), 2.0),
        (date(2025, 3, 31), 1.0),
    ]
