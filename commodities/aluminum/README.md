# Aluminum Research Backtest

This package builds a research-only aluminum fundamental data and backtest layer.
It is intentionally not wired into `commodities/commodity_research.py`.

## Data Sources

- World Bank Pink Sheet monthly commodity XLS: free required baseline for the aluminum target price series.
- EIA API v2 retail electricity sales: optional U.S. industrial electricity price proxy when `EIA_API_KEY` is set.
  Override the default route/facets with `EIA_ALUMINUM_POWER_ROUTE`, `EIA_ALUMINUM_POWER_STATEID`,
  `EIA_ALUMINUM_POWER_SECTORID`, and `EIA_ALUMINUM_POWER_DATA_FIELD`.
- SHFE public HTML pages: optional aluminum inventory/futures tables where `pandas.read_html` can parse them.
- LME XML: optional licensed/local XML only. The adapter reads `data_cache/aluminum/lme_xml/` or a configured licensed endpoint using `LME_XML_URL`, `LME_USERNAME`, and `LME_PASSWORD`. It does not scrape or bypass licensed LME sources.

## Running

```bash
python3 backtest/aluminum_backtest.py fetch-data --refresh
python3 backtest/aluminum_backtest.py build-features
python3 backtest/aluminum_backtest.py run-backtest --model ridge
python3 backtest/aluminum_backtest.py all --model random_forest --transaction-cost-bps 5
```

Outputs are written to:

- `data_cache/aluminum/processed/aluminum_monthly_features.csv`
- `results/aluminum/backtest_trades.csv`
- `results/aluminum/backtest_metrics.csv`
- `results/aluminum/equity_curve.csv`
- `results/aluminum/equity_curve.png`
- `results/aluminum/drawdown.png`
- `results/aluminum/factor_diagnostics.csv`

## Validation Bar

Production integration requires all checks to pass:

- At least 60 out-of-sample monthly forecasts with `min_train_months >= 120`.
- Net-of-cost model strategy Sharpe at least 0.75.
- Sharpe at least 0.25 above buy-and-hold aluminum.
- Prediction Spearman IC at least 0.05.
- RMSE at least 5% better than zero-return forecast or hit rate at least 53%.
- At least one non-price fundamental/cost feature has absolute Spearman IC at least 0.05.
- Positive net returns in at least 60% of out-of-sample calendar years.
- No single year contributes more than 50% of positive net P&L.
- At least 36 monthly observations from optional non-price sources for production-facing fundamental scoring.

If the gate fails, the module remains research-only. Price-only features may still be useful diagnostics, but they should not be presented as aluminum fundamentals.

## Diagnostics

`factor_diagnostics.csv` reports feature/target correlation, Spearman rank IC,
tercile forward-return spreads, observation counts, and a classification:
`useful`, `weak`, `unstable`, or `unavailable`.

## Limitations

- Source histories may be revised.
- The model operates at monthly frequency.
- Optional source coverage may be sparse.
- Public SHFE pages are fragile and may change structure.
- Backtest performance is not a guarantee of profitability.
- Future source additions must preserve lagging rules to avoid lookahead and survivorship errors.
