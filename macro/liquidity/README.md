# Liquidity Dashboard

A macro liquidity monitoring tool that fetches FRED and optional ECB SDMX data,
computes regional liquidity scores, rolls them into a composite score, classifies
market regimes, and displays results in a rich terminal dashboard.

## Where this is used

- Standalone CLI: `python3 macro/liquidity/liquidity.py`
- FastAPI:
  - `GET /api/liquidity`
  - `POST /api/liquidity/analyze` for the LLM-assisted dashboard summary
- React UI: “Liquidity” page in `frontend/`

## Overview

This script tracks global liquidity conditions by combining regional components
into one composite score. US liquidity carries the largest weight and emphasizes
net-liquidity/reserve trends, credit spreads, NFCI, and M2/GDP. Europe uses ECB
excess liquidity and net liquidity effect when `sdmx1` is available. Japan uses
BoJ assets, M3, and private-credit growth.

## Features

- Fetches real-time data from FRED API
- Calculates regional scores for the US, Europe, and Japan, then combines them
  into one weighted composite
- Classifies market regimes: ample, normal, tight, or stress
- Rich terminal dashboard with color-coded output
- Historical change tracking (1w, 1m, 3m)
- Optional matplotlib charts with `--plot` flag
- Optional `--no-ecb` mode to skip ECB SDMX fetches

## Installation

### Requirements

```bash
# From the repo root:
pip install -r requirements.txt

# Or minimal:
# pip install pandas fredapi rich sdmx1
```

### Optional (for charts)

```bash
pip install matplotlib
```

## Setup

You need a FRED API key to fetch data. Get one for free at https://fred.stlouisfed.org/docs/api/api_key.html

Set your API key as an environment variable:

```bash
export FRED_API_KEY=your_api_key_here
```

This repo also supports `.env` files:

```bash
cp .env.example .env
# edit .env and set FRED_API_KEY=...
```

## Usage

### Basic dashboard

```bash
python3 macro/liquidity/liquidity.py
```

### With charts

```bash
python3 macro/liquidity/liquidity.py --plot
```

### Without ECB SDMX

```bash
python3 macro/liquidity/liquidity.py --no-ecb
```

## How It Works

### Data Sources

The script fetches the following series from FRED:

**Fed Balance Sheet (H.4.1)**
- WALCL: Total assets
- WRESBAL: Reserve balances
- WTREGEN: Treasury General Account
- RRPONTSYD: Overnight reverse repo

**Global Central Banks**
- JPNASSETS: BoJ total assets
- JPNMABMM301GYSAM: Japan M3 YoY growth
- CRDQJPAPABIS: Japan total private-sector credit, adjusted for breaks

**Credit & Conditions**
- BAMLC0A0CM: IG corporate OAS
- BAMLH0A0HYM2: HY corporate OAS
- NFCI: National Financial Conditions Index

**Money & Activity**
- M2SL: M2 money stock
- GDP: Nominal GDP

ECB data is fetched separately through `sdmx1`:

- Excess liquidity
- Net liquidity effect

### Composite Score Calculation

Each component is converted to a z-score using a 104-week rolling window, then
weighted within its region. The regional scores are combined as:

| Region | Composite Weight |
|--------|------------------|
| US | 60% |
| Europe | 30% |
| Japan | 10% |

Current component weights:

| Component | Weight | Polarity |
|-----------|--------|----------|
| US Net Liquidity (4w change) | 25% of US | Positive |
| US Net Liquidity (level) | 20% of US | Positive |
| US Reserve Balances (4w change) | 20% of US | Positive |
| IG OAS | 15% of US | Negative |
| HY OAS | 10% of US | Negative |
| NFCI | 5% of US | Negative |
| M2 / GDP | 5% of US | Positive |
| ECB Excess Liquidity | 60% of Europe | Positive |
| ECB Net Liquidity Effect | 40% of Europe | Positive |
| BoJ Assets YoY | 35% of Japan | Positive |
| Japan M3 YoY | 35% of Japan | Positive |
| Japan Private Credit YoY | 30% of Japan | Positive |

**Net Liquidity** = Fed Total Assets - TGA - ON RRP

### Regime Classification

The composite score maps to four regimes:

- **Ample** (>1.0): Very supportive liquidity conditions
- **Normal** (-0.5 to 1.0): Neutral liquidity environment
- **Tight** (-1.5 to -0.5): Restrictive liquidity conditions
- **Stress** (<-1.5): Severe liquidity tightening

## Output

The dashboard displays:

1. **Current regime** with color-coded composite score
2. **Regional scores** for US, Europe, and Japan
3. **Component breakdown** showing each input's value, z-score, weight, contribution, and signal
4. **Historical changes** for composite score and key series across 1w, 1m, and 3m periods

## Example Output

```
╭─────────────────────────────────╮
│ Liquidity Dashboard             │
│ Last update: 2025-01-15         │
╰─────────────────────────────────╯

╭─ Liquidity Regime ──────────────╮
│ Composite Score: +1.23          │
│ Regime: AMPLE                   │
╰─────────────────────────────────╯
```

## Notes

- Data is resampled to weekly frequency (Wednesday-ending)
- Z-scores use a 2-year (104-week) rolling window
- Trend components use 4-week changes to capture momentum
- Missing data is forward-filled when resampling
