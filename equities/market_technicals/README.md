# Market Technicals

Analyzes market breadth and price/volume signals for U.S. equities.

## Quick Start

```bash
cd equities/market_technicals
python3 market_technicals.py
```

This runs all three analyses in sequence.

## Dependencies

```bash
pip install pandas yfinance requests lxml numpy
```

## What It Does

### 1. Market Breadth (`market_breadth.py`)

Calculates S&P 500 breadth metrics:
- % of stocks above 200-day moving average
- % of stocks above 20-day moving average
- % of stocks at 20-day highs
- % of stocks at 20-day lows

Color-coded output highlights extreme readings (potential turning points).

### 2. Top 50 Breadth (`top50_breadth.py`)

Analyzes the top 50 S&P 500 performers (by 6-month return):
- % below 50-day moving average
- % with 3+ distribution days in last 20 sessions
- % that closed below prior 20-day low in last 5 days

Distribution days = down days with above-average volume (50-day avg).

### 3. Price/Volume Signals (`price_volume_signals.py`)

Detects signals on major indices (S&P 500, Nasdaq, Russell 2000):
- **Downside record volume**: Down day with volume at 252-day high
- **New high on low volume**: New 252-day high with volume below 80% of average
- **High-volume churn**: 3+ high-volume days in 5 sessions with minimal price progress

Falls back to ETF proxies (SPY, QQQ, IWM) if index volume data is unavailable.

## Supporting Files

| File | Description |
|------|-------------|
| `get_top50.py` | Fetches S&P 500 constituents and ranks by 6-month return |
| `sp500_top50_6mo.csv` | Output from get_top50 (auto-generated) |
| `tickers.txt` | Optional custom ticker list for market_breadth.py |

## Running Individual Scripts

```bash
python3 market_breadth.py              # S&P 500 breadth
python3 market_breadth.py --universe /path/to/tickers.txt  # Custom universe

python3 top50_breadth.py               # Top 50 analysis

python3 price_volume_signals.py        # Index signals
```
