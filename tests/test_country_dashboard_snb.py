from __future__ import annotations

from datetime import datetime


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_fetch_snb_series_parses_bom_preamble_csv(monkeypatch):
    from macro.country_dashboard import country_dashboard

    payload = (
        '\ufeff"CubeId";"plkopr"\n'
        '"PublishingDate";"2026-04-21 09:00"\n'
        "\n"
        '"Date";"D0";"Value"\n'
        '"2026-02";"VVP";"0.13072156"\n'
        '"2026-03";"VVP";"0.3146541"\n'
    )

    def fake_requests_get(url: str, timeout: int = 20):
        assert "data.snb.ch/api/cube/plkopr/data/csv/en" in url
        assert "dimSel=D0(VVP)" in url
        return _FakeResponse(payload)

    monkeypatch.setattr(country_dashboard, "requests_get", fake_requests_get)

    series = country_dashboard._fetch_snb_series(
        cube="plkopr",
        dim_sel="D0(VVP)",
        observation_start=datetime(2026, 1, 1),
    )

    assert list(series.index.strftime("%Y-%m")) == ["2026-02", "2026-03"]
    assert series.iloc[-1] == 0.3146541
