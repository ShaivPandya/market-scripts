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
