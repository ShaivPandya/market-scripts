#!/usr/bin/env python3
"""
Fetch CFTC Commitments of Traders (COT) futures positioning via the CFTC Public Reporting Environment (PRE) API
(Socrata/SODA backend), and compute Leveraged Funds net positioning for a selected market (e.g., S&P futures).

Sources:
- CFTC notes the COT PRE provides an API with filtering/search and downloads.  :contentReference[oaicite:0]{index=0}
- Example TFF Futures Only dataset identifier (gpe5-46if) is on the CFTC Data Hub / Socrata backend. :contentReference[oaicite:1]{index=1}

Usage examples:
  # 1) Find the exact market_and_exchange_names string you want:
  python3 positioning.py --search "S&P"

  # 2) Pull Leveraged Funds positioning for an exact market name (copy from search results):
  python3 positioning.py --market "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" --start 2015-01-01 --out es_cot.csv

  # 3) Same, but set an app token (optional) to reduce throttling risk:
  export SODA_APP_TOKEN="YOUR_TOKEN"
  python3 positioning.py --market "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" --start 2015-01-01

Notes:
- Without an App Token, requests can be rate-limited. You can obtain an App Token from the Socrata developer portal.
- This script uses metadata to auto-detect column field names, so it is resilient to minor schema differences.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd


DEFAULT_DOMAIN = "publicreportinghub.cftc.gov"

# COT datasets (PRE/Socrata). TFF Futures Only is typically used for "Leveraged Funds" in financial futures.
DATASETS = {
    "tff_futures_only": "gpe5-46if",
    # Add others if you want:
    # "disagg_futures_only": "72hh-3qpy",
    # "disagg_combined": "kh3c-gbw2",
    # "legacy_futures_only": "6dca-aqww",
}

# Predefined instruments mapping friendly names to exact CFTC market names
INSTRUMENTS = {
    # Equity Indices
    "SP500": "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
    "NASDAQ": "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE",
    "RUSSELL": "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE",
    "NIKKEI": "NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE",
    # Currency Futures (vs USD)
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "NZD": "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    # Treasuries
    "US10Y": "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
}


@dataclass(frozen=True)
class Fields:
    report_date: str
    market_name: str
    open_interest: Optional[str]
    lf_long: str
    lf_short: str


def _http_get_json(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 6,
    backoff_s: float = 1.25,
) -> Any:
    """GET JSON with basic retry/backoff for 429/5xx."""
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            sleep_s = backoff_s * (2 ** attempt)
            time.sleep(min(sleep_s, 30))
            continue
        # Non-retryable
        raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:500]}")
    raise RuntimeError(f"Exceeded retries for {url}")


def get_dataset_metadata(domain: str, dataset_id: str, app_token: Optional[str]) -> Dict[str, Any]:
    # Socrata metadata endpoint
    url = f"https://{domain}/api/views/{dataset_id}.json"
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token
    return _http_get_json(url, headers=headers)


def _normalize(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in s).strip()


def detect_fields(meta: Dict[str, Any]) -> Fields:
    cols = meta.get("columns", [])
    if not cols:
        raise RuntimeError("No columns found in metadata; cannot detect fields.")

    col_pairs: List[Tuple[str, str]] = []
    for c in cols:
        human = str(c.get("name", "") or "")
        field = str(c.get("fieldName", "") or "")
        if field:
            col_pairs.append((human, field))

    def find_one(required_terms: List[str], prefer_field_terms: Optional[List[str]] = None) -> str:
        required_terms_n = [_normalize(t) for t in required_terms]
        prefer_terms_n = [_normalize(t) for t in (prefer_field_terms or [])]

        candidates: List[Tuple[int, int, str]] = []
        for human, field in col_pairs:
            h = _normalize(human)
            f = _normalize(field)
            blob = f"{h} {f}"
            if all(t in blob for t in required_terms_n):
                prefer_hits = sum(1 for t in prefer_terms_n if t and t in blob)
                field_hits = sum(1 for t in required_terms_n if t and t in f)
                candidates.append((prefer_hits, field_hits, field))

        if not candidates:
            raise RuntimeError(f"Could not detect required field with terms={required_terms}")

        candidates.sort(reverse=True)
        return candidates[0][2]

    def find_first(alternatives: List[List[str]], prefer: Optional[List[str]] = None) -> str:
        last_err: Optional[Exception] = None
        for terms in alternatives:
            try:
                return find_one(terms, prefer_field_terms=prefer)
            except RuntimeError as e:
                last_err = e
        # Helpful debug: show a few likely candidates
        likely = sorted(
            {f for _, f in col_pairs if "lev" in _normalize(f) or "lever" in _normalize(f) or "money" in _normalize(f)}
        )[:50]
        raise RuntimeError(f"{last_err}\nLikely related fields (sample): {likely}")

    report_date = find_first(
        [["report", "date"], ["as", "of", "date"], ["report_date"]],
        prefer=["report_date"],
    )
    market_name = find_first(
        [["market", "exchange"], ["market_and_exchange"], ["contract", "market"]],
        prefer=["market_and_exchange_names"],
    )

    # Find open interest, but exclude percentage fields
    open_interest = None
    for human, field in col_pairs:
        h = _normalize(human)
        f = _normalize(field)
        blob = f"{h} {f}"
        # Must contain "open" and "interest", but NOT "pct" or "percent"
        if ("open" in blob and "interest" in blob and
            "pct" not in blob and "percent" not in blob and "change" not in blob):
            open_interest = field
            break

    # IMPORTANT: TFF uses "Leveraged Money" in many schemas (e.g., lev_money_positions_long_all)
    lf_long = find_first(
        [
            ["leveraged", "fund", "long"],
            ["leveraged", "money", "long"],
            ["lev", "money", "positions", "long"],
            ["lev_money", "positions", "long"],
        ],
        prefer=["lev_money", "leveraged", "long", "positions", "all"],
    )
    lf_short = find_first(
        [
            ["leveraged", "fund", "short"],
            ["leveraged", "money", "short"],
            ["lev", "money", "positions", "short"],
            ["lev_money", "positions", "short"],
        ],
        prefer=["lev_money", "leveraged", "short", "positions", "all"],
    )

    return Fields(
        report_date=report_date,
        market_name=market_name,
        open_interest=open_interest,
        lf_long=lf_long,
        lf_short=lf_short,
    )



def soda_iter_rows(
    domain: str,
    dataset_id: str,
    app_token: Optional[str],
    soql_params: Dict[str, Any],
    page_size: int = 50000,
) -> Iterable[Dict[str, Any]]:
    """
    Iterate rows from the Socrata SODA endpoint using $limit/$offset pagination.
    """
    base_url = f"https://{domain}/resource/{dataset_id}.json"
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    offset = 0
    while True:
        params = dict(soql_params)
        params["$limit"] = page_size
        params["$offset"] = offset
        rows = _http_get_json(base_url, headers=headers, params=params)
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected response type: {type(rows)}")
        if not rows:
            break
        for r in rows:
            yield r
        offset += len(rows)


def search_markets(domain: str, dataset_id: str, app_token: Optional[str], query: str, limit: int = 200) -> List[str]:
    """
    Return distinct market_and_exchange_names values matching query (case-insensitive LIKE).
    """
    # We still need the exact field name; fetch metadata and use the detected market field.
    meta = get_dataset_metadata(domain, dataset_id, app_token)
    fields = detect_fields(meta)

    q = query.replace("'", "''")
    soql = {
        "$select": f"distinct({fields.market_name})",
        "$where": f"upper({fields.market_name}) like upper('%{q}%')",
        "$order": fields.market_name,
    }

    markets: List[str] = []
    for row in soda_iter_rows(domain, dataset_id, app_token, soql_params=soql, page_size=50000):
        # Socrata returns distinct field under its field name, but with a suffix when using distinct()
        # Try both the original field name and the suffixed version
        val = row.get(fields.market_name) or row.get(f"{fields.market_name}_1")
        if isinstance(val, str) and val.strip():
            markets.append(val.strip())
        if len(markets) >= limit:
            break
    return markets


def fetch_market_timeseries(
    domain: str,
    dataset_id: str,
    app_token: Optional[str],
    market_exact: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """
    Fetch rows for a single exact market_and_exchange_names, optionally bounded by start/end (YYYY-MM-DD).
    """
    meta = get_dataset_metadata(domain, dataset_id, app_token)
    fields = detect_fields(meta)

    # Build SoQL
    market_escaped = market_exact.replace("'", "''")
    where_parts = [f"{fields.market_name} = '{market_escaped}'"]

    # Socrata fixed timestamp comparisons want full timestamp strings; T00:00:00.000 is fine.
    if start:
        where_parts.append(f"{fields.report_date} >= '{start}T00:00:00.000'")
    if end:
        where_parts.append(f"{fields.report_date} <= '{end}T23:59:59.999'")

    select_fields = [fields.report_date, fields.market_name, fields.lf_long, fields.lf_short]
    if fields.open_interest:
        select_fields.append(fields.open_interest)

    soql = {
        "$select": ", ".join(select_fields),
        "$where": " AND ".join(where_parts),
        "$order": fields.report_date,
    }

    rows = list(soda_iter_rows(domain, dataset_id, app_token, soql_params=soql, page_size=50000))
    if not rows:
        raise RuntimeError("No rows returned. Check the market name (use --search) and date range.")

    df = pd.DataFrame(rows)

    # Parse and coerce numeric fields
    df[fields.report_date] = pd.to_datetime(df[fields.report_date], errors="coerce").dt.date
    for c in [fields.lf_long, fields.lf_short] + ([fields.open_interest] if fields.open_interest else []):
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics
    df["lf_net"] = df[fields.lf_long] - df[fields.lf_short]
    if fields.open_interest and fields.open_interest in df.columns:
        df["lf_net_pct_oi"] = (df["lf_net"] / df[fields.open_interest]) * 100.0
    else:
        df["lf_net_pct_oi"] = pd.NA

    # z-score on lf_net_pct_oi (or lf_net if OI missing)
    series = df["lf_net_pct_oi"]
    if series.isna().all():
        series = df["lf_net"]
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    df["lf_z"] = (series - mu) / sd if sd and sd > 0 else pd.NA

    # Standardize column names for output readability
    rename_map = {
        fields.report_date: "report_date",
        fields.market_name: "market_and_exchange_names",
        fields.lf_long: "leveraged_funds_long",
        fields.lf_short: "leveraged_funds_short",
    }
    if fields.open_interest:
        rename_map[fields.open_interest] = "open_interest"

    df = df.rename(columns=rename_map)
    return df


def fetch_multiple_instruments(
    domain: str,
    dataset_id: str,
    app_token: Optional[str],
    instruments: List[str],
    start: Optional[str],
    end: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Fetch latest positioning for multiple instruments and return summary data.
    """
    results = []
    for alias in instruments:
        market_name = INSTRUMENTS.get(alias)
        if not market_name:
            print(f"Warning: Unknown instrument '{alias}', skipping.", file=sys.stderr)
            continue
        try:
            df = fetch_market_timeseries(
                domain=domain,
                dataset_id=dataset_id,
                app_token=app_token,
                market_exact=market_name,
                start=start,
                end=end,
            )
            latest = df.dropna(subset=["report_date"]).iloc[-1]
            results.append({
                "instrument": alias,
                "report_date": latest["report_date"],
                "lf_net": latest["lf_net"],
                "lf_net_pct_oi": latest.get("lf_net_pct_oi"),
                "lf_z": latest.get("lf_z"),
            })
        except Exception as e:
            print(f"Warning: Failed to fetch {alias}: {e}", file=sys.stderr)
    return results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary table of positioning results, sorted by Z-score.
    """
    if not results:
        print("No results to display.")
        return

    # Sort by absolute Z-score (most extreme first)
    sorted_results = sorted(
        results,
        key=lambda x: abs(x["lf_z"]) if x["lf_z"] is not None and not pd.isna(x["lf_z"]) else 0,
        reverse=True,
    )

    # Print header
    print("\n" + "=" * 75)
    print(f"{'Instrument':<12} {'Date':<12} {'Net Position':>14} {'Net % OI':>10} {'Z-Score':>10}")
    print("=" * 75)

    # Print rows
    for r in sorted_results:
        net = f"{r['lf_net']:,.0f}" if r['lf_net'] is not None and not pd.isna(r['lf_net']) else "N/A"
        pct = f"{r['lf_net_pct_oi']:.1f}%" if r['lf_net_pct_oi'] is not None and not pd.isna(r['lf_net_pct_oi']) else "N/A"
        z = f"{r['lf_z']:.2f}" if r['lf_z'] is not None and not pd.isna(r['lf_z']) else "N/A"
        print(f"{r['instrument']:<12} {str(r['report_date']):<12} {net:>14} {pct:>10} {z:>10}")

    print("=" * 75)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch CFTC COT positioning via PRE/Socrata API.")
    p.add_argument("--domain", default=DEFAULT_DOMAIN, help="Socrata domain (default: publicreportinghub.cftc.gov).")
    p.add_argument(
        "--dataset",
        default="tff_futures_only",
        help=f"Dataset key in {list(DATASETS.keys())} or a raw dataset id (e.g. gpe5-46if).",
    )
    p.add_argument("--app-token", default=os.getenv("SODA_APP_TOKEN"), help="Optional Socrata App Token.")
    p.add_argument("--search", default=None, help="Search markets by substring (prints matches and exits).")
    p.add_argument("--market", default=None, help="Exact market_and_exchange_names value to fetch.")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional).")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional).")
    p.add_argument("--out", default=None, help="Write CSV to this path (optional).")
    p.add_argument("--all", action="store_true", help="Fetch all predefined instruments.")
    p.add_argument("--instruments", default=None, help="Comma-separated list of instrument aliases (e.g., SP500,EUR,US10Y).")
    p.add_argument("--list-instruments", action="store_true", help="List available instrument aliases and exit.")

    args = p.parse_args()

    dataset_id = DATASETS.get(args.dataset, args.dataset)

    # Handle --list-instruments
    if args.list_instruments:
        print("Available instrument aliases:")
        print("-" * 60)
        for alias, market in INSTRUMENTS.items():
            print(f"  {alias:<10} -> {market}")
        return 0

    # Handle --search
    if args.search:
        markets = search_markets(args.domain, dataset_id, args.app_token, args.search)
        if not markets:
            print("No matches.")
            return 0
        print("Matches (use one with --market):")
        for m in markets:
            print(f"  {m}")
        return 0

    # Handle --all or --instruments (multi-instrument mode)
    if args.all or args.instruments:
        if args.all:
            instrument_list = list(INSTRUMENTS.keys())
        else:
            instrument_list = [s.strip().upper() for s in args.instruments.split(",")]
            # Validate
            invalid = [i for i in instrument_list if i not in INSTRUMENTS]
            if invalid:
                print(f"Error: Unknown instruments: {invalid}", file=sys.stderr)
                print(f"Use --list-instruments to see available aliases.", file=sys.stderr)
                return 2

        print(f"Fetching positioning for {len(instrument_list)} instruments...")
        results = fetch_multiple_instruments(
            domain=args.domain,
            dataset_id=dataset_id,
            app_token=args.app_token,
            instruments=instrument_list,
            start=args.start,
            end=args.end,
        )
        print_summary_table(results)
        return 0

    # Handle --market (single market mode)
    if not args.market:
        print("Error: provide --market, --all, --instruments, or --search", file=sys.stderr)
        return 2

    df = fetch_market_timeseries(
        domain=args.domain,
        dataset_id=dataset_id,
        app_token=args.app_token,
        market_exact=args.market,
        start=args.start,
        end=args.end,
    )

    # Print a compact "latest reading"
    latest = df.dropna(subset=["report_date"]).iloc[-1]
    print("\nLatest row:")
    print(latest[["report_date", "lf_net", "lf_net_pct_oi", "lf_z"]].to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(df):,} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
