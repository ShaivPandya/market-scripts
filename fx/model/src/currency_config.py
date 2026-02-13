"""Currency pair configurations for FX models."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CurrencyPairConfig:
    """Configuration for a currency pair.

    All spot rates are expressed as quote_per_base (e.g., USDCAD = CAD per USD).
    RER = log(spot) + log(CPI_base) - log(CPI_quote)
    Carry = (r_quote - r_base) / 100
    """

    pair_name: str  # e.g., "USDCAD"
    base_ccy: str  # e.g., "USD"
    quote_ccy: str  # e.g., "CAD"

    # FRED spot rate
    fred_spot_id: str  # FRED series ID for spot (quote per base)
    fred_spot_invert: bool = False  # True if FRED series needs inversion

    # CPI series (candidates, tried in order)
    cpi_base_ids: list = field(default_factory=list)  # CPI for base currency
    cpi_quote_ids: list = field(default_factory=list)  # CPI for quote currency

    # Interest rate series
    rate_base_id: str = ""  # 3m rate for base currency
    rate_quote_id: str = ""  # 3m rate for quote currency

    # IMF DataMapper country codes
    imf_iso3_base: str = ""  # e.g., "USA"
    imf_iso3_quote: str = ""  # e.g., "CAN"

    # BIS REER key (for quote currency effective exchange rate)
    bis_key: str = ""  # e.g., "M.R.B.CA"

    # Display labels
    spot_label: str = ""  # e.g., "CAD per USD"

    # Common series (oil, VIX)
    fred_oil_id: str = "DCOILWTICO"
    fred_vix_id: str = "VIXCLS"

    # Yahoo Finance tickers (for spot, oil, VIX)
    yf_spot_ticker: str = ""
    yf_oil_ticker: str = "CL=F"
    yf_vix_ticker: str = "^VIX"


# Pre-defined configurations
CURRENCY_CONFIGS = {
    "USDCAD": CurrencyPairConfig(
        pair_name="USDCAD",
        base_ccy="USD",
        quote_ccy="CAD",
        fred_spot_id="DEXCAUS",
        fred_spot_invert=False,
        cpi_base_ids=["CPIAUCSL"],
        cpi_quote_ids=["CANCPIALLMINMEI", "CPALTT01CAM657N"],
        rate_base_id="TB3MS",
        rate_quote_id="IR3TIB01CAM156N",
        imf_iso3_base="USA",
        imf_iso3_quote="CAN",
        bis_key="M.R.B.CA",
        spot_label="CAD per USD",
        yf_spot_ticker="CAD=X",
    ),
    "GBPUSD": CurrencyPairConfig(
        pair_name="GBPUSD",
        base_ccy="GBP",
        quote_ccy="USD",
        fred_spot_id="DEXUSUK",
        fred_spot_invert=False,
        cpi_base_ids=["GBRCPIALLMINMEI", "CPALTT01GBM657N"],
        cpi_quote_ids=["CPIAUCSL"],
        rate_base_id="IR3TIB01GBM156N",
        rate_quote_id="TB3MS",
        imf_iso3_base="GBR",
        imf_iso3_quote="USA",
        bis_key="M.R.B.GB",
        spot_label="USD per GBP",
        yf_spot_ticker="GBPUSD=X",
    ),
    "AUDUSD": CurrencyPairConfig(
        pair_name="AUDUSD",
        base_ccy="AUD",
        quote_ccy="USD",
        fred_spot_id="DEXUSAL",
        fred_spot_invert=False,
        # Australia CPI is quarterly in FRED; will forward-fill to monthly
        cpi_base_ids=["CPALTT01AUQ657N"],
        cpi_quote_ids=["CPIAUCSL"],
        rate_base_id="IR3TIB01AUM156N",
        rate_quote_id="TB3MS",
        imf_iso3_base="AUS",
        imf_iso3_quote="USA",
        bis_key="M.R.B.AU",
        spot_label="USD per AUD",
        yf_spot_ticker="AUDUSD=X",
    ),
    "USDJPY": CurrencyPairConfig(
        pair_name="USDJPY",
        base_ccy="USD",
        quote_ccy="JPY",
        fred_spot_id="DEXJPUS",
        fred_spot_invert=False,
        cpi_base_ids=["CPIAUCSL"],
        cpi_quote_ids=["JPNCPIALLMINMEI", "CPALTT01JPM657N"],
        rate_base_id="TB3MS",
        rate_quote_id="IR3TIB01JPM156N",
        imf_iso3_base="USA",
        imf_iso3_quote="JPN",
        bis_key="M.R.B.JP",
        spot_label="JPY per USD",
        yf_spot_ticker="JPY=X",
    ),
}


def get_config(pair_name: str) -> CurrencyPairConfig:
    """Get configuration for a currency pair."""
    pair_upper = pair_name.upper()
    if pair_upper not in CURRENCY_CONFIGS:
        available = ", ".join(CURRENCY_CONFIGS.keys())
        raise ValueError(f"Unknown currency pair: {pair_name}. Available: {available}")
    return CURRENCY_CONFIGS[pair_upper]


def list_pairs() -> list:
    """List available currency pairs."""
    return list(CURRENCY_CONFIGS.keys())
