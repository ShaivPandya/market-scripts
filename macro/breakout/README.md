# Breakout Detector (Daily)

This project contains a Python script that scans a small universe of **major currency pairs** and **major commodity futures** for:

1) **Tight congestion** (a compressed, low-volatility range), followed by  
2) A **close-based breakout** from that congestion, confirmed by **volume expansion** (or a proxy when volume is unavailable).

---

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

---

## What the Script Does

For each symbol:
1. Downloads **daily OHLCV** data from Yahoo Finance using `yfinance`.
2. Computes features/indicators needed for congestion and breakout detection.
3. Detects an **active congestion box** and checks whether the **latest daily bar** is a confirmed breakout.
4. Prints a table of any confirmed breakouts found on the most recent bar.

If there is no confirmed breakout on the latest bar for any instrument, it prints a message indicating none were found.

---

## Congestion Definition (Step 3)

The script defines congestion as a **Donchian range box** + **compression filters** + **minimum duration**.

### Box boundaries (Donchian)
Using a rolling window `Wc` (default 20):
- `upper = highest High over Wc`
- `lower = lowest Low over Wc`
- `box_range = upper - lower`

### Tightness filters
A day is considered “tight” when:
- `box_range <= range_atr_mult * ATR(atr_n)`  
- Bollinger Band width is low relative to history:  
  `BBWidth <= rolling_quantile(BBWidth, lookback=252, q=0.20)`

### Duration requirement
Congestion is “confirmed” when at least `k_min` of the last `Wc` days are “tight”.

Default: `k_min = 8` of the last `Wc = 20`.

---

## Box Handling: Commodities vs FX

### Commodities: Frozen Box
Once congestion is confirmed, the script freezes `upper/lower` until the box resolves (breakout or invalidation).

### FX: Hybrid Box
Once congestion is confirmed, the script freezes `upper/lower` but allows small boundary updates **only if they are within a tolerance**:
- `tol = fx_tol_atr * ATR(atr_n)` (default `0.15 * ATR(20)`)

This helps capture slow “drifting” compressions in FX without letting the boundary creep too much.

---

## Breakout Definition (Step 4)

### Close-based trigger
A breakout occurs when the **Close** exits the congestion box by a volatility-scaled buffer:

- `buffer = buffer_k * ATR(atr_n)` (default `0.20 * ATR(20)`)

Then:
- Up breakout: `Close > upper + buffer`
- Down breakout: `Close < lower - buffer`

### Confirmation: Volume expansion
Breakouts are confirmed only if:
- `Volume > vol_mult * VolMA(vol_ma_n)`  
Default: `Volume > 1.20 * SMA(Volume, 20)`

---

## Important Note About FX Volume (Yahoo Finance)

Yahoo Finance spot FX data frequently has **no reliable volume** (often 0 or missing).

In that case, the script uses a **proxy for participation/expansion**:
- `TR > vol_mult * SMA(TR, vol_ma_n)`  
It will label the output with `VolMethod = TR_PROXY`.

For commodities, Yahoo futures volume is typically present, so the script uses real `Volume`.

---

## Requirements

- Python 3.9+
- Packages:
  - `yfinance`
  - `pandas`
  - `numpy`

Install:
```bash
pip install yfinance pandas numpy
