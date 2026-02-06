# Multi-Currency FX Macro Model

A quantitative "fundamentals + market/plumbing" FX framework for G10 currency pairs over a 1-2 year horizon.

## Supported Currency Pairs

| Pair | Base | Quote | Spot Convention |
|------|------|-------|-----------------|
| USDCAD | USD | CAD | CAD per USD |
| GBPUSD | GBP | USD | USD per GBP |
| AUDUSD | AUD | USD | USD per AUD |
| USDJPY | USD | JPY | JPY per USD |

## Data Sources

- **FRED** (spot, CPI, rates, oil, VIX) via the FRED API
- **IMF DataMapper (WEO)** (current account % GDP) via the IMF DataMapper API (no key required)
- **BIS** (optional) via SDMX using `pandaSDMX` (model runs even if BIS download fails)

## Setup

1. Create a FRED API key and set it as an environment variable:
   - FRED API docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html

2. Use `.env` (optional):
```bash
cp .env.example .env
# edit .env and paste your key
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Run a single currency pair
```bash
# USDCAD (default)
python fx_model.py

# GBPUSD
python fx_model.py --pair GBPUSD

# AUDUSD
python fx_model.py --pair AUDUSD

# USDJPY
python fx_model.py --pair USDJPY
```

### Options
```bash
# Skip BIS downloads
python fx_model.py --pair USDCAD --no-bis

# Force refresh cached data
python fx_model.py --pair GBPUSD --refresh

# Reduce bootstrap draws (faster)
python fx_model.py --pair AUDUSD --bootstrap 1000

# Change horizons (months)
python fx_model.py --pair USDJPY --horizons 12,18,24
```

## Output Structure

Outputs are saved to pair-specific subdirectories:

```
outputs/
├── usdcad/
│   ├── usdcad_monthly_features.csv
│   ├── usdcad_reference_points.csv
│   ├── forecast_latest.json
│   ├── spot_vs_reference_points.png
│   ├── valuation_zscore.png
│   ├── forecast_distribution_12m.png
│   └── forecast_distribution_24m.png
├── gbpusd/
│   └── ...
├── audusd/
│   └── ...
└── usdjpy/
    └── ...
```

## Method Summary

- Monthly frequency
- Targets: 12- and 24-month ahead log returns
- Core features:
  - **Valuation**: Bilateral real exchange rate (PPP-style) and rolling z-score
  - **Macro**: CA%GDP differential (quote - base), oil (real)
  - **Market/plumbing**: Carry (rate differential), momentum, realized vol, carry-to-vol, VIX
- Model: OLS with HAC standard errors; forecast distribution via residual bootstrap
- "Reference points": Implied spot levels if the real exchange rate reverted to rolling median/p25/p75

### Formulas

**Real Exchange Rate (RER)**:
```
RER = log(spot_quote_per_base) + log(CPI_base) - log(CPI_quote)
```

**Carry**:
```
carry = (r_quote - r_base) / 100
```

Positive carry means going long the base currency earns positive carry.

## Adding New Currency Pairs

Edit `src/currency_config.py` to add a new `CurrencyPairConfig` entry with:
- FRED series IDs for spot, CPI, and interest rates
- IMF country codes
- BIS REER key
- Display labels
