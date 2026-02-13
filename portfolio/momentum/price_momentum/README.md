# Momentum ROC Analysis

A Python script for analyzing stock momentum using Rate of Change (ROC) indicators with absolute and relative price movements.

## Overview

This script computes three key momentum metrics:

1. **20-day average of 63-day ROC (%)** - Measures absolute price momentum over a ~3-month period
2. **42-day ROC (%) of relative price to benchmark** - Compares stock performance vs. benchmark over ~2 months
3. **10-day average of relative ROC (%)** - Short-term average of relative performance

The script uses color-coded output to highlight strong (green) vs. weak (red) momentum signals.

## Installation

```bash
pip install yfinance pandas
```

## Usage

### Single Ticker Analysis

```bash
python3 momentum.py AAPL --benchmark SPY
```

### Multiple Tickers from File

```bash
python3 momentum.py --tickers-file tickers.txt --benchmark SPY
```

### Using Built-in Universe

```bash
python3 momentum.py --tickers-file us_mega_cap --benchmark QQQ --years 10
```

### List Available Universes

```bash
python3 momentum.py --list-universes
```

## Command Line Arguments

- `ticker` - Single ticker symbol (e.g., AAPL)
- `--tickers-file` - Universe name or path to file containing tickers (one per line)
- `--benchmark` - Benchmark ticker for relative comparisons (required, e.g., SPY, QQQ)
- `--years` - Years of historical data to download (default: 5)
- `--list-universes` - List available universe files and exit

## Output

The script displays color-coded results:

```
Benchmark: SPY
As of: 2026-01-17

Ticker: AAPL    Close: 150.2500
  20-day avg of 63-day ROC (%):        2.345678
  42-day ROC of relative price (%):   1.234567
  10-day avg of relative ROC (%):     0.987654
```

### Color Coding

- **Green**: Strong momentum (above threshold)
- **Red**: Weak momentum (below threshold)

Thresholds:
- 20-day avg of 63-day ROC: 1.5%
- Relative metrics: 0%

## How It Works

1. Downloads historical adjusted close prices from Yahoo Finance
2. Aligns ticker and benchmark data on common trading dates
3. Calculates absolute price momentum using 63-day ROC
4. Computes relative price (ticker/benchmark) momentum
5. Applies moving averages to smooth signals
6. Displays results with visual indicators

## Data Requirements

The script requires at least 83 trading days of overlapping data between the ticker and benchmark (63 days for ROC calculation + 20 days for averaging).

## Dependencies

- Python 3.7+
- yfinance
- pandas

## Notes

- Uses adjusted close prices to account for splits and dividends
- All calculations are based on overlapping trading days between ticker and benchmark
- Relative price calculations show how the ticker performs versus the benchmark
