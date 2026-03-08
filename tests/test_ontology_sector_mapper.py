from __future__ import annotations

import json

from ontology.sector_mapper import SectorMapper, _fetch_sector_from_yfinance


def test_sector_mapper_precedence_manual_then_yfinance_then_unknown(tmp_path, monkeypatch):
    map_path = tmp_path / "sector_map.json"
    map_path.write_text(json.dumps({"MANUAL": "Information Technology"}), encoding="utf-8")

    _fetch_sector_from_yfinance.cache_clear()
    monkeypatch.setattr("ontology.sector_mapper._fetch_sector_from_yfinance", lambda ticker: "Energy")

    mapper = SectorMapper(map_path=map_path)

    manual = mapper.resolve_sector("MANUAL", "equity")
    assert manual.sector == "Information Technology"
    assert manual.source == "manual_map"

    fallback = mapper.resolve_sector("UNKNOWN", "equity")
    assert fallback.sector == "Energy"
    assert fallback.source == "yfinance"


def test_sector_mapper_synthetic_and_unknown(monkeypatch):
    _fetch_sector_from_yfinance.cache_clear()
    monkeypatch.setattr("ontology.sector_mapper._fetch_sector_from_yfinance", lambda ticker: None)

    mapper = SectorMapper()

    commodity = mapper.resolve_sector("DBB", "commodity")
    assert commodity.sector == "Commodities"
    assert commodity.source == "synthetic"

    unknown = mapper.resolve_sector("NOPE", "equity")
    assert unknown.sector == "Unknown Equity"
    assert unknown.source == "unknown"
