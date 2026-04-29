"""Configuration for the aluminum research/backtest pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALUMINUM_CACHE_DIR = PROJECT_ROOT / "data_cache" / "aluminum"
RAW_DIR = ALUMINUM_CACHE_DIR / "raw"
PROCESSED_DIR = ALUMINUM_CACHE_DIR / "processed"
LME_XML_DIR = ALUMINUM_CACHE_DIR / "lme_xml"
RESULTS_DIR = PROJECT_ROOT / "results" / "aluminum"

WORLD_BANK_MONTHLY_XLS_URL = (
    "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
WORLD_BANK_RAW_XLS = RAW_DIR / "world_bank_pink_sheet_monthly.xlsx"
WORLD_BANK_PROCESSED_CSV = PROCESSED_DIR / "world_bank_aluminum_prices.csv"

EIA_API_BASE_URL = "https://api.eia.gov/v2"
EIA_RETAIL_SALES_ROUTE = "electricity/retail-sales"
EIA_POWER_PROXY_KEY = "electricity_retail_sales_us_industrial_price"
EIA_PROCESSED_CSV = PROCESSED_DIR / "eia_power_proxy.csv"

SHFE_WEEKLY_DATA_URL = "https://www.shfe.com.cn/eng/reports/StatisticalData/WeeklyData/"
SHFE_ALUMINUM_FUTURES_URL = "https://www.shfe.com.cn/eng/Market/Futures/Metal/al_f/"
SHFE_PROCESSED_CSV = PROCESSED_DIR / "shfe_aluminum_inventory.csv"

LME_PRICES_PROCESSED_CSV = PROCESSED_DIR / "lme_aluminum_prices.csv"
LME_STOCKS_PROCESSED_CSV = PROCESSED_DIR / "lme_aluminum_stocks.csv"

FEATURES_CSV = PROCESSED_DIR / "aluminum_monthly_features.csv"
TRADES_CSV = RESULTS_DIR / "backtest_trades.csv"
METRICS_CSV = RESULTS_DIR / "backtest_metrics.csv"
EQUITY_CURVE_CSV = RESULTS_DIR / "equity_curve.csv"
EQUITY_CURVE_PNG = RESULTS_DIR / "equity_curve.png"
DRAWDOWN_PNG = RESULTS_DIR / "drawdown.png"
FACTOR_DIAGNOSTICS_CSV = RESULTS_DIR / "factor_diagnostics.csv"

DEFAULT_RANDOM_SEED = 42
DEFAULT_MODEL_TYPE = "ridge"
DEFAULT_FORECAST_THRESHOLD = 0.005
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_MIN_TRAIN_MONTHS = 120

VALIDATION_MIN_FORECASTS = 60
VALIDATION_MIN_TRAIN_MONTHS = 120
VALIDATION_MIN_NET_SHARPE = 0.75
VALIDATION_MIN_SHARPE_EDGE_VS_BUY_HOLD = 0.25
VALIDATION_MIN_PREDICTION_SPEARMAN_IC = 0.05
VALIDATION_MIN_FUNDAMENTAL_FEATURE_IC = 0.05
VALIDATION_MIN_OPTIONAL_SOURCE_MONTHS = 36
VALIDATION_MIN_POSITIVE_YEAR_RATIO = 0.60
VALIDATION_MAX_SINGLE_YEAR_PNL_SHARE = 0.50


@dataclass(frozen=True)
class AluminumBacktestConfig:
    start_date: str | None = None
    end_date: str | None = None
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS
    model_type: str = DEFAULT_MODEL_TYPE
    forecast_threshold: float = DEFAULT_FORECAST_THRESHOLD
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    refresh: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    lme_xml_dir: Path = LME_XML_DIR


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, LME_XML_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
