# Liquidity Dashboard

A macro liquidity monitoring tool that fetches data from FRED, computes a composite liquidity score, classifies market regimes, and displays results in a rich terminal dashboard.

## Overview

This script tracks global liquidity conditions by combining multiple macro indicators into a single composite score. The score emphasizes trend-based measures (4-week changes in net liquidity, reserves, and global central bank assets) while incorporating credit spreads and financial conditions.

## Features

- Fetches real-time data from FRED API
- Calculates composite liquidity score from 8 weighted components
- Classifies market regimes: ample, normal, tight, or stress
- Rich terminal dashboard with color-coded output
- Historical change tracking (1w, 1m, 3m)
- Optional matplotlib charts with `--plot` flag

## Installation

### Requirements

```bash
pip install pandas fredapi rich
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

## Usage

### Basic dashboard

```bash
python macro/liquidity/liquidity.py
```

### With charts

```bash
python macro/liquidity/liquidity.py --plot
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
- ECBASSETSW: ECB total assets
- JPNASSETS: BoJ total assets

**Credit & Conditions**
- BAMLC0A0CM: IG corporate OAS
- BAMLH0A0HYM2: HY corporate OAS
- NFCI: National Financial Conditions Index

**Money & Activity**
- M2SL: M2 money stock
- GDP: Nominal GDP

### Composite Score Calculation

Each component is converted to a z-score using a 104-week rolling window, then weighted and summed:

| Component | Weight | Polarity |
|-----------|--------|----------|
| Net Liquidity (4w change) | 35% | Positive |
| Reserve Balances (4w change) | 15% | Positive |
| ECB Assets (4w change) | 10% | Positive |
| BoJ Assets (4w change) | 10% | Positive |
| IG OAS | 10% | Negative |
| HY OAS | 10% | Negative |
| NFCI | 5% | Negative |
| M2 / GDP | 5% | Positive |

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
2. **Component breakdown** showing each input's value, z-score, weight, contribution, and signal
3. **Historical changes** for composite score and key series across 1w, 1m, and 3m periods

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
