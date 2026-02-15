# Breakout Detector (Daily)

This module scans the FX/commodity universe for a strict congestion regime and prior-bar breakout signals.

## Universe

### FX (spot via Yahoo Finance `=X` tickers)
- EURUSD (`EURUSD=X`)
- GBPUSD (`GBPUSD=X`)
- AUDUSD (`AUDUSD=X`)
- NZDUSD (`NZDUSD=X`)
- USDCAD (`CAD=X`)
- USDCHF (`CHF=X`)
- USDJPY (`JPY=X`)

### Commodities (futures via Yahoo Finance `=F` tickers)
- Gold (`GC=F`)
- Silver (`SI=F`)
- Copper (`HG=F`)
- Platinum (`PL=F`)
- Aluminum (`ALI=F`)
- Palladium (`PA=F`)

## Formula

Inputs:
- `ATR period = 14` (Wilder ATR / RMA)
- `Congestion lookback = 30`
- `ATR compression lookback = 100`
- `SMA period = 20`

Definitions:
- `atr = ATR(14)`
- `range_high = highest(high, 30)`
- `range_low = lowest(low, 30)`

Congestion conditions (all required):
- `atr <= lowest(atr, 100) * 1.25`
- `(range_high - range_low) / atr <= 6`
- `abs(close - SMA(20)) <= atr`

Regime flag:
- `congestion = all conditions true`

Breakouts (no stops, no exits):
- `long_breakout = congestion[1] AND close > range_high[1]`
- `short_breakout = congestion[1] AND close < range_low[1]`

## Data API (`get_data()`)

`get_data()` returns:

- `latest`: one row per asset for the latest bar with flags and core indicator values.
- `events`: all historical breakout events (`LONG`/`SHORT`).
- `history`: per-ticker full daily rows (OHLC + formula columns + flags).
- `params`: active parameter values.
- `error`: included only if analysis fails.

### `latest` row fields

- `market`, `name`, `ticker`, `date`
- `close`, `atr`, `atr_min100`, `range_high30`, `range_low30`, `sma20`, `range_atr_ratio`
- `cond_atr_compression`, `cond_range_atr`, `cond_sma_distance`
- `congestion`, `long_breakout`, `short_breakout`, `direction`

### `events` row fields

- `market`, `name`, `ticker`, `date`, `direction`
- `close`, `atr`, `range_high30_prev`, `range_low30_prev`, `congestion_prev`

### `history[ticker].rows` row fields

- `date`, `open`, `high`, `low`, `close`
- `atr`, `atr_min100`, `range_high30`, `range_low30`, `sma20`, `range_atr_ratio`
- `cond_atr_compression`, `cond_range_atr`, `cond_sma_distance`
- `congestion`, `long_breakout`, `short_breakout`

## CLI

Run:

```bash
python3 macro/breakout/breakout.py
```

CLI output shows:
- latest long breakouts
- latest short breakouts
- assets currently in congestion
- recent historical breakout events

## Requirements

Install:

```bash
pip install yfinance pandas numpy
```
