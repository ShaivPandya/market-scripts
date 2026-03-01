# Multi-Currency FX Macro Model (`fx/model/`)

A quantitative “fundamentals + market/plumbing” FX framework for a small G10 universe over a ~1–2 year horizon.

For the overall repo architecture (API + UI), see the repo root `README.md`.

## Where this is used

- Standalone CLI: `fx/model/fx_model.py`
- Web dashboard:
  - API: `POST /api/fx-model`, `GET /api/fx-model/pairs` (see `api/routers/fx_model.py`)
  - UI: “FX Model” page in `frontend/`

## Supported currency pairs

| Pair | Base | Quote | Spot convention |
|------|------|-------|-----------------|
| USDCAD | USD | CAD | CAD per USD |
| EURUSD | EUR | USD | USD per EUR |
| GBPUSD | GBP | USD | USD per GBP |
| AUDUSD | AUD | USD | USD per AUD |
| USDJPY | USD | JPY | JPY per USD |

## Data sources

- **FRED**: spot, CPI candidates, interest rates, oil, VIX (`FRED_API_KEY` required)
- **IMF DataMapper (WEO)**: current account % GDP (no key)
- **BIS** (optional): monthly REER/effective exchange rate series via SDMX (`pandasdmx` / `pandaSDMX`)
- **Statistics Canada WDS** (for Canada CPI candidate): no key
- **Japan e-Stat** (optional CPI candidate for JPY): `ESTAT_APP_ID` if that candidate is used

## Setup

### Python dependencies

If you’re running the full repo (API/UI), install from the repo root:

```bash
pip install -r requirements.txt
```

If you only want the FX model in isolation, install the model-only deps:

```bash
pip install -r fx/model/requirements.txt
```

Notes:
- The model uses `statsmodels` for OLS + HAC standard errors.
- BIS downloads require `pandasdmx` (installed via the `pandaSDMX` package). You can always skip BIS via `--no-bis` / `skip_bis`.

### Environment variables

The pipeline calls `dotenv.load_dotenv()` (see `fx/model/src/pipeline.py`), so putting a `.env` at the repo root works even when you run from `fx/model/`.

- Required:
  - `FRED_API_KEY`
- Optional:
  - `ESTAT_APP_ID` (only needed if the USDJPY CPI fetch uses the e-Stat candidate)

## Usage (CLI)

Run from `fx/model/`:

```bash
# USDCAD (default)
python fx_model.py

python fx_model.py --pair EURUSD
python fx_model.py --pair GBPUSD
python fx_model.py --pair AUDUSD
python fx_model.py --pair USDJPY
```

Options:

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

## Usage (web API)

The backend exposes the model under:
- `GET /api/fx-model/pairs` (available pairs)
- `POST /api/fx-model` (run the pipeline)

The request body matches `FXModelRequest` in `api/routers/fx_model.py`:

```json
{
  "pair": "USDCAD",
  "bootstrap": 1000,
  "skip_bis": false,
  "horizons": "12,24"
}
```

The API writes outputs under `fx/model/outputs/<pair>/` and uses `fx/model/data_cache/` for cached downloads.

## Output structure

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
└── ...
```

## Method summary

- Monthly frequency
- Targets: 12- and 24-month ahead log returns (configurable)
- Core features:
  - **Valuation**: bilateral real exchange rate (PPP-style) and rolling z-score
  - **Macro**: current-account % GDP differential (quote − base), oil (real)
  - **Market / plumbing**: carry (rate differential), momentum, realized vol, carry-to-vol, VIX
- Model: OLS with HAC standard errors; forecast distribution via residual bootstrap
- “Reference points”: implied spot levels if the real exchange rate reverted to rolling median/p25/p75

### Key formulas

**Real Exchange Rate (RER)**:

```
RER = log(spot_quote_per_base) + log(CPI_base) - log(CPI_quote)
```

**Carry**:

```
carry = (r_quote - r_base) / 100
```

Positive carry means going long the base currency earns positive carry.

## Adding new currency pairs

Edit `fx/model/src/currency_config.py` and add a new `CurrencyPairConfig` entry with:
- FRED series IDs for spot, CPI, and interest rates
- IMF country codes
- BIS REER key (optional)
- Display labels / conventions
