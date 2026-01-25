# USDCAD 1–2Y Macro-FX Model (FRED + IMF + BIS)

This project builds a quantitative “fundamentals + market/plumbing” FX framework for **USDCAD** over a **1–2 year horizon**.

It downloads data from:
- **FRED** (spot, CPI, rates, oil, VIX) via the FRED API.
- **IMF DataMapper (WEO)** (current account % GDP) via the IMF DataMapper API (no key required).
- **BIS** (optional) via SDMX using `pandaSDMX` (the model runs even if BIS download fails).

Outputs (created after you run the script):
- `outputs/usdcad_monthly_features.csv`
- `outputs/usdcad_reference_points.csv`
- `outputs/forecast_latest.json`
- `outputs/spot_vs_reference_points.png`
- `outputs/valuation_zscore.png`
- `outputs/forecast_distribution_12m.png`
- `outputs/forecast_distribution_24m.png` (if horizon 24 is used)

By default, `outputs/` and `data_cache/` are created next to `usdcad_model.py` (not relative to where you run the command).

## Setup

1) Create a FRED API key and set it as an environment variable:
- FRED `series/observations` API docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html

2) Use `.env` (optional):
```bash
cp .env.example .env
# edit .env and paste your key
```

3) Install requirements:
```bash
pip install -r requirements.txt
```

## Run

Baseline (tries BIS; continues if BIS fails):
```bash
python usdcad_model.py
```

Skip BIS:
```bash
python usdcad_model.py --no-bis
```

Force refresh cached data:
```bash
python usdcad_model.py --refresh
```

Reduce bootstrap draws (faster):
```bash
python usdcad_model.py --bootstrap 1000
```

Change horizons (months):
```bash
python usdcad_model.py --horizons 12,18,24
```

## Method summary

- Monthly frequency.
- Targets: 12- and 24-month ahead log returns in USDCAD.
- Core features:
  - valuation: bilateral real exchange rate (PPP-style) and rolling z-score
  - macro: CA%GDP differential (Canada – US), oil (real)
  - market/plumbing: rate differential (carry), momentum, realized vol, carry-to-vol, VIX
- Model: OLS with HAC standard errors; forecast distribution via residual bootstrap.
- “Reference points”: implied spot levels if the real exchange rate reverted to rolling median / p25 / p75.
