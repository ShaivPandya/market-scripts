from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger("api.ontology")

DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "config" / "sector_map.json"

SYNTHETIC_SECTORS = {
    "fx": "FX",
    "commodity": "Commodities",
    "bond": "Rates",
    "crypto": "Digital Assets",
}


@dataclass(slots=True)
class SectorResolution:
    sector: str
    source: str


class SectorMapper:
    """Resolves position sectors with precedence: manual map > yfinance > synthetic."""

    def __init__(self, map_path: Path | None = None):
        resolved_path = map_path or DEFAULT_MAP_PATH
        self._manual_map = _load_manual_map(resolved_path)

    def resolve_sector(self, ticker: str, asset: str) -> SectorResolution:
        ticker_norm = (ticker or "").strip().upper()
        asset_norm = (asset or "").strip().lower()

        if asset_norm != "equity":
            sector = SYNTHETIC_SECTORS.get(asset_norm, "Other Assets")
            return SectorResolution(sector=sector, source="synthetic")

        manual = self._manual_map.get(ticker_norm)
        if manual:
            return SectorResolution(sector=manual, source="manual_map")

        yf_sector = _fetch_sector_from_yfinance(ticker_norm)
        if yf_sector:
            return SectorResolution(sector=yf_sector, source="yfinance")

        return SectorResolution(sector="Unknown Equity", source="unknown")


@lru_cache(maxsize=512)
def _fetch_sector_from_yfinance(ticker: str) -> str | None:
    try:
        import yfinance as yf

        obj = yf.Ticker(ticker)
        info = obj.get_info() or obj.info or {}
        sector = info.get("sector")
        if isinstance(sector, str) and sector.strip():
            return sector.strip()
    except Exception as exc:
        LOGGER.debug("yfinance sector lookup failed for %s: %s", ticker, exc)
    return None


def _load_manual_map(path: Path) -> dict[str, str]:
    if not path.exists():
        LOGGER.warning("manual sector map missing at %s", path)
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("failed to parse manual sector map %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        LOGGER.warning("manual sector map has invalid format at %s", path)
        return {}

    out: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key.strip().upper()] = value.strip()
    return out
