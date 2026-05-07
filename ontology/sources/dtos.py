from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PortfolioMetadata:
    ticker: str
    asset: str
    direction: str
    instrument_type: str = "security"
    price_symbol: str | None = None
    quantity: float | None = None
    contract_multiplier: float = 1.0
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioPosition:
    ticker: str
    asset: str
    direction: str
    latest_price: float | None
    series_points: int
    as_of: str | None
    metadata: PortfolioMetadata
    instrument_type: str = "security"
    price_symbol: str | None = None
    quantity: float | None = None
    contract_multiplier: float = 1.0
    fx_base_currency: str | None = None
    fx_quote_currency: str | None = None


@dataclass(slots=True)
class PortfolioSnapshot:
    positions: dict[str, PortfolioPosition]
    timeframe: str
    timestamp: str | None
    position_order: list[str] = field(default_factory=list)
    analytics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VixTermStructureSnapshot:
    date: str | None
    vix: float | None
    vix3m: float | None
    ratio: float | None
    signal: str
    used_ticker: str | None = None


@dataclass(slots=True)
class MarketBreadthSnapshot:
    total_analyzed: int | None
    pct_above_200dma: float | None
    pct_above_20dma: float | None
    pct_at_20day_low: float | None
    pct_at_52wk_low: float | None
    as_of_date: str | None
    failed_ticker_count: int = 0


@dataclass(slots=True)
class Top50BreadthSnapshot:
    pct_below_50dma: float | None
    pct_3plus_dist: float | None
    pct_broke_20low: float | None
    universe_size: int | None


@dataclass(slots=True)
class SectorMetricRow:
    sector: str
    weight_now: float | None
    chg_1m_pp: float | None
    chg_3m_pp: float | None
    chg_6m_pp: float | None
    relperf_3m_pp: float | None
    relperf_12m_pp: float | None
    pct_above_200dma: float | None


@dataclass(slots=True)
class SectorMetricsSnapshot:
    rows: list[SectorMetricRow]
    timestamp: str | None
    d_1m: str | None = None
    d_3m: str | None = None
    d_6m: str | None = None


@dataclass(slots=True)
class LiquiditySnapshot:
    composite_score: float | None
    regime: str
    latest_date: str | None
    regional_scores: dict[str, Any] = field(default_factory=dict)
    components: list[dict[str, Any]] = field(default_factory=list)
    changes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SentimentSnapshot:
    put_call: dict[str, Any] = field(default_factory=dict)
    surveys: dict[str, Any] = field(default_factory=dict)
    volatility: list[dict[str, Any]] = field(default_factory=list)
    latest_vvix: float | None = None


@dataclass(slots=True)
class PositioningRow:
    instrument: str
    report_date: str | None
    lf_net: float | None
    lf_net_pct_oi: float | None
    lf_z: float | None
    lf_deleveraging_z: float | None
    lf_forced: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PositioningSnapshot:
    rows: list[PositioningRow]


@dataclass(slots=True)
class EconomicGrowthSnapshot:
    commodities: dict[str, Any] = field(default_factory=dict)
    equities: dict[str, Any] = field(default_factory=dict)
    equity_relative_returns: dict[str, Any] = field(default_factory=dict)
    currencies: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    crb_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LaborIndicator:
    key: str
    value: float | None
    date: str | None
    change: float | None
    label: str | None = None
    unit: str | None = None


@dataclass(slots=True)
class LaborMarketSnapshot:
    latest: dict[str, LaborIndicator]
    timestamp: str | None
    series_labels: dict[str, str] = field(default_factory=dict)
    series_units: dict[str, str] = field(default_factory=dict)
    initial_claims_change: float | None = None
