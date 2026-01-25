# CFTC COT Positioning (`positioning.py`)

Fetch Commitments of Traders (COT) futures positioning from the CFTC Public Reporting Environment (PRE) API (Socrata/SODA),
compute net positioning for participant groups, and derive simple “forced flow” proxies (deleveraging / short covering).

This script is at `macro/positioning/positioning.py`.

## Requirements

- Python 3.8+
- Dependencies from the repo root: `pip install -r requirements.txt`
- Network access (the script pulls data from the CFTC PRE API)

Optional:
- A Socrata App Token to reduce throttling: set `SODA_APP_TOKEN` or pass `--app-token`.

## Quick start

### 1) Fetch a single market (exact name)

```bash
python3 macro/positioning/positioning.py \
  --market "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE" \
  --start 2015-01-01 \
  --out sp500_cot.csv
```

The script prints a compact “latest row” and (optionally) writes a CSV with the full time series.

### 2) Fetch multiple instruments (summary table)

```bash
python3 macro/positioning/positioning.py --all
```

Or a subset:

```bash
python3 macro/positioning/positioning.py --instruments SP500,EUR,US10Y
```

## Finding a market name

`--market` must match the PRE field `market_and_exchange_names` exactly.

Use one of these approaches:

- **Use built-in aliases**: run `python3 macro/positioning/positioning.py --list-instruments`
- **Look it up in the CFTC Data Hub / PRE UI** and copy the exact `market_and_exchange_names` value

## Participant groups

Leveraged funds are always computed and exposed as `lf_*` columns for backward compatibility.

Add more participant groups with `--groups`:

```bash
python3 macro/positioning/positioning.py \
  --market "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE" \
  --start 2015-01-01 \
  --groups dealer,asset_mgr,other_rept,nonrept \
  --z-window 156
```

Valid canonical group keys:
- `dealer`
- `asset_mgr`
- `other_rept`
- `nonrept`
- `all` (includes all of the above + leveraged funds)

## Output columns (high level)

The output includes:

- Market identifiers: `report_date`, `market_and_exchange_names`
- Raw positions (when available): `open_interest`, `leveraged_funds_long`, `leveraged_funds_short`, plus optional `dealer_*`, `asset_mgr_*`, `other_rept_*`, `nonrept_*`
- Derived metrics per group prefix (e.g. `lf_*`, `dealer_*`):
  - `*_net`: long - short
  - `*_net_pct_oi`: `*_net / open_interest * 100` (if open interest is present in the dataset)
  - `*_z`: z-score of net positioning (on `*_net_pct_oi` if available, otherwise `*_net`)
  - `*_d_*`: week-over-week changes
  - `*_deleveraging`: a “toward flat” proxy (positive when reducing exposure)
  - `*_deleveraging_z`: z-score of `*_deleveraging`
  - `*_forced`: classification when `*_deleveraging_z >= --force-threshold` (`long_liquidation` or `short_covering`)

## Useful flags

- `--start YYYY-MM-DD` / `--end YYYY-MM-DD`: bound the date range
- `--z-window N`: rolling z-score window in weeks (`0` uses the full sample)
- `--force-threshold X`: threshold for `*_forced` (default `2.0`)
- `--dataset KEY_OR_ID`: defaults to `tff_futures_only` (currently `gpe5-46if`)
- `--domain DOMAIN`: defaults to `publicreportinghub.cftc.gov`

## Notes

- If you see rate limits (HTTP 429), pass an app token via `--app-token` or `SODA_APP_TOKEN`, or reduce the date range.
- `macro/positioning/tff.csv` is a local CSV artifact in this repo; the script itself does not require it.
