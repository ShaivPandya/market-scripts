"""Stable keys for daily computed API snapshots."""

SNAPSHOT_SCHEMA_VERSION = 1

SNAPSHOT_MARKET_BREADTH = "market_breadth:sp500:1y"
SNAPSHOT_TOP50_BREADTH = "top50_breadth:sp500:2y"
SNAPSHOT_SECTOR_METRICS = "sector_metrics:sp500:2y"
SNAPSHOT_LIQUIDITY = "liquidity:current:v1"
SNAPSHOT_HOUSING = "housing:current:v1"
SNAPSHOT_VIX_TERM_STRUCTURE = "vix_term_structure:current:v1"
SNAPSHOT_SENTIMENT = "sentiment:current:v1"
SNAPSHOT_POSITIONING_SUMMARY = "positioning_summary:current:v1"
SNAPSHOT_ECONOMIC_GROWTH = "economic_growth:current:v1"
SNAPSHOT_LABOR_MARKET = "labor_market:current:v1"
SNAPSHOT_MOMENTUM = "momentum:portfolio:5y"
SNAPSHOT_SIGNAL_AGGREGATOR = "signal_aggregator:current:v1"

# Daily after-close data should be considered stale if a weekday refresh was
# missed. Stale snapshots are still served; callers surface the metadata.
DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 36 * 60 * 60
