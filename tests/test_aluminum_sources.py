from __future__ import annotations

import pandas as pd


def test_world_bank_parser_with_synthetic_workbook(tmp_path):
    from commodities.aluminum.sources.world_bank import parse_world_bank_pink_sheet

    path = tmp_path / "pink_sheet.xlsx"
    raw = pd.DataFrame(
        [
            ["Commodity", "Unit", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")],
            ["Copper", "$/mt", 6000.0, 6100.0, 6200.0],
            ["Aluminum", "$/mt", 1800.0, 1850.0, 1900.0],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        raw.to_excel(writer, index=False, header=False)

    parsed = parse_world_bank_pink_sheet(path)

    assert list(parsed.columns) == ["date", "aluminum_price_usd_tonne", "source"]
    assert len(parsed) == 3
    assert parsed["date"].iloc[0] == pd.Timestamp("2020-01-31")
    assert parsed["aluminum_price_usd_tonne"].tolist() == [1800.0, 1850.0, 1900.0]


def test_shfe_parser_with_synthetic_html():
    from commodities.aluminum.sources.shfe import parse_shfe_aluminum_inventory_html

    html = """
    <html><body>
      <p>Report date: 2024-01-05</p>
      <table>
        <tr><th>Product</th><th>Inventory</th></tr>
        <tr><td>Aluminum</td><td>123,456</td></tr>
        <tr><td>Copper</td><td>99,999</td></tr>
      </table>
    </body></html>
    """

    parsed = parse_shfe_aluminum_inventory_html(html)

    assert len(parsed) == 1
    assert parsed.loc[0, "date"] == pd.Timestamp("2024-01-05")
    assert parsed.loc[0, "contract_or_product"] == "Aluminum"
    assert parsed.loc[0, "inventory_tonnes"] == 123456.0


def test_lme_parser_with_synthetic_xml():
    from commodities.aluminum.sources.lme import parse_lme_price_xml, parse_lme_stock_xml

    xml = """
    <Root>
      <PriceRecord>
        <Date>2024-01-31</Date>
        <Metal>Aluminum</Metal>
        <Cash>2200.50</Cash>
        <ThreeMonth>2240.25</ThreeMonth>
      </PriceRecord>
      <StockRecord>
        <Date>2024-01-31</Date>
        <Metal>Aluminum</Metal>
        <WarehouseLocation>Rotterdam</WarehouseLocation>
        <StockTonnes>50000</StockTonnes>
        <CancelledTonnes>7500</CancelledTonnes>
      </StockRecord>
    </Root>
    """

    prices = parse_lme_price_xml(xml)
    stocks = parse_lme_stock_xml(xml)

    assert len(prices) == 1
    assert prices.loc[0, "lme_aluminum_cash"] == 2200.50
    assert prices.loc[0, "lme_aluminum_3m"] == 2240.25
    assert len(stocks) == 1
    assert stocks.loc[0, "warehouse_location"] == "Rotterdam"
    assert stocks.loc[0, "stock_tonnes"] == 50000.0
    assert stocks.loc[0, "cancelled_tonnes"] == 7500.0


def test_eia_missing_key_returns_empty_frame(monkeypatch):
    import commodities.aluminum.sources.eia as eia

    monkeypatch.delenv("EIA_API_KEY", raising=False)
    monkeypatch.setattr(eia, "load_env", lambda: None)

    df = eia.fetch_eia_power_proxy()

    assert df.empty
    assert list(df.columns) == ["date", "eia_series_id_or_route", "value", "unit", "source"]


def test_eia_normalizer_with_synthetic_payload():
    from commodities.aluminum.sources.eia import normalize_eia_power_proxy_response

    payload = {
        "response": {
            "data": [
                {
                    "period": "2024-01",
                    "price": "7.25",
                    "price-units": "cents per kilowatthour",
                }
            ]
        }
    }

    df = normalize_eia_power_proxy_response(payload)

    assert len(df) == 1
    assert df.loc[0, "date"] == pd.Timestamp("2024-01-31")
    assert df.loc[0, "value"] == 7.25
