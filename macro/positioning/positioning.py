#!/usr/bin/env python3
"""
Fetch CFTC Commitments of Traders (COT) futures positioning via the CFTC Public Reporting Environment (PRE) API
(Socrata/SODA backend), and compute participant net positioning + simple "forced flow" proxies (deleveraging / short-covering)
for a selected market (e.g., S&P futures).

Sources:
- CFTC notes the COT PRE provides an API with filtering/search and downloads.  :contentReference[oaicite:0]{index=0}
- Example TFF Futures Only dataset identifier (gpe5-46if) is on the CFTC Data Hub / Socrata backend. :contentReference[oaicite:1]{index=1}

Usage examples:
  # 1) Pull Leveraged Funds positioning for an exact market name (copy from the CFTC Data Hub / PRE UI):
  python3 positioning.py --market "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" --start 2015-01-01 --out es_cot.csv

  # 2) Same, but set an app token (optional) to reduce throttling risk:
  export SODA_APP_TOKEN="YOUR_TOKEN"
  python3 positioning.py --market "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" --start 2015-01-01

  # 3) Include other participant groups and use a rolling z-score window (~3y):
  python3 positioning.py --market "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE" --start 2015-01-01 --groups all --z-window 156

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
import numpy as np
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

CANONICAL_GROUPS = ["lev_money", "dealer", "asset_mgr", "other_rept", "nonrept"]

GROUP_ALIASES = {
    "lf": "lev_money",
    "lev": "lev_money",
    "leveraged": "lev_money",
    "leveraged_funds": "lev_money",
    "leveraged_money": "lev_money",
    "lev_money": "lev_money",
    "dealer": "dealer",
    "dealers": "dealer",
    "asset": "asset_mgr",
    "asset_mgr": "asset_mgr",
    "asset_manager": "asset_mgr",
    "asset_managers": "asset_mgr",
    "other": "other_rept",
    "other_rept": "other_rept",
    "other_reportables": "other_rept",
    "nonrept": "nonrept",
    "non_rept": "nonrept",
    "nonreportable": "nonrept",
    "non_reportable": "nonrept",
    "non_reportables": "nonrept",
}

# Keep historical column naming for leveraged funds to avoid breaking downstream users (GUI, etc).
GROUP_PREFIX = {
    "lev_money": "lf",
    "dealer": "dealer",
    "asset_mgr": "asset_mgr",
    "other_rept": "other_rept",
    "nonrept": "nonrept",
}

GROUP_LABEL = {
    "lev_money": "Leveraged Funds",
    "dealer": "Dealer",
    "asset_mgr": "Asset Manager",
    "other_rept": "Other Reportables",
    "nonrept": "Non-Reportables",
}


@dataclass(frozen=True)
class Fields:
    report_date: str
    market_name: str
    open_interest: Optional[str]
    lf_long: str
    lf_short: str
    dealer_long: Optional[str] = None
    dealer_short: Optional[str] = None
    asset_mgr_long: Optional[str] = None
    asset_mgr_short: Optional[str] = None
    other_rept_long: Optional[str] = None
    other_rept_short: Optional[str] = None
    nonrept_long: Optional[str] = None
    nonrept_short: Optional[str] = None


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

    def try_find_first(alternatives: List[List[str]], prefer: Optional[List[str]] = None) -> Optional[str]:
        try:
            return find_first(alternatives, prefer=prefer)
        except RuntimeError:
            return None

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

    # Optional: additional participant groups (TFF Futures Only schema)
    dealer_long = try_find_first(
        [["dealer", "positions", "long"]],
        prefer=["dealer_positions_long_all"],
    )
    dealer_short = try_find_first(
        [["dealer", "positions", "short"]],
        prefer=["dealer_positions_short_all"],
    )
    asset_mgr_long = try_find_first(
        [["asset", "mgr", "positions", "long"], ["asset", "manager", "positions", "long"]],
        prefer=["asset_mgr_positions_long_all"],
    )
    asset_mgr_short = try_find_first(
        [["asset", "mgr", "positions", "short"], ["asset", "manager", "positions", "short"]],
        prefer=["asset_mgr_positions_short_all"],
    )
    other_rept_long = try_find_first(
        [["other", "rept", "positions", "long"], ["other", "report", "positions", "long"]],
        prefer=["other_rept_positions_long_all"],
    )
    other_rept_short = try_find_first(
        [["other", "rept", "positions", "short"], ["other", "report", "positions", "short"]],
        prefer=["other_rept_positions_short_all"],
    )
    nonrept_long = try_find_first(
        [["nonrept", "positions", "long"], ["non", "rept", "positions", "long"]],
        prefer=["nonrept_positions_long_all"],
    )
    nonrept_short = try_find_first(
        [["nonrept", "positions", "short"], ["non", "rept", "positions", "short"]],
        prefer=["nonrept_positions_short_all"],
    )

    return Fields(
        report_date=report_date,
        market_name=market_name,
        open_interest=open_interest,
        lf_long=lf_long,
        lf_short=lf_short,
        dealer_long=dealer_long,
        dealer_short=dealer_short,
        asset_mgr_long=asset_mgr_long,
        asset_mgr_short=asset_mgr_short,
        other_rept_long=other_rept_long,
        other_rept_short=other_rept_short,
        nonrept_long=nonrept_long,
        nonrept_short=nonrept_short,
    )


def parse_groups(groups: Optional[str]) -> List[str]:
    """
    Parse comma-separated participant groups.

    Note: leveraged funds ("lev_money") are always computed as lf_* for backward compatibility.
    """
    if not groups:
        return ["lev_money"]

    requested = [g.strip().lower() for g in groups.split(",") if g.strip()]
    if not requested:
        return ["lev_money"]

    if any(g in ("all", "*") for g in requested):
        return list(CANONICAL_GROUPS)

    resolved: List[str] = []
    for g in requested:
        key = GROUP_ALIASES.get(g)
        if not key:
            raise ValueError(f"Unknown group '{g}'. Valid: {CANONICAL_GROUPS} or 'all'.")
        if key not in resolved:
            resolved.append(key)

    # Always keep leveraged funds present for downstream consumers.
    if "lev_money" not in resolved:
        resolved.insert(0, "lev_money")

    return resolved


def _zscore(series: pd.Series, window: Optional[int]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if window and window > 0:
        mu = s.rolling(window, min_periods=window).mean()
        sd = s.rolling(window, min_periods=window).std(ddof=1)
        return (s - mu) / sd.replace(0, np.nan)

    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def _add_group_metrics(
    df: pd.DataFrame,
    long_col: str,
    short_col: str,
    open_interest_col: Optional[str],
    prefix: str,
    z_window: int,
    force_threshold: float,
) -> None:
    """
    Add net positioning metrics for a participant group.

    "Forced" here is a simple proxy: unusually large movement toward flat (deleveraging / short covering).
    """
    net_col = f"{prefix}_net"
    net_pct_col = f"{prefix}_net_pct_oi"

    df[net_col] = df[long_col] - df[short_col]

    if open_interest_col and open_interest_col in df.columns:
        df[net_pct_col] = (df[net_col] / df[open_interest_col]) * 100.0
    else:
        df[net_pct_col] = pd.NA

    base_series = df[net_pct_col]
    if base_series.isna().all():
        base_series = df[net_col]
    df[f"{prefix}_z"] = _zscore(base_series, window=z_window if z_window > 0 else None)

    # "Flow" proxy: week-over-week changes
    df[f"{prefix}_d_net"] = df[net_col].diff()
    if df[net_pct_col].isna().all():
        d_series = df[f"{prefix}_d_net"]
        df[f"{prefix}_d_net_pct_oi"] = pd.NA
    else:
        df[f"{prefix}_d_net_pct_oi"] = df[net_pct_col].diff()
        d_series = df[f"{prefix}_d_net_pct_oi"]
    df[f"{prefix}_d_z"] = _zscore(d_series, window=z_window if z_window > 0 else None)

    # Toward-flat positioning change: positive when reducing exposure (long liquidation or short covering)
    df[f"{prefix}_deleveraging"] = (-np.sign(df[net_col])) * df[f"{prefix}_d_net"]
    df[f"{prefix}_deleveraging_z"] = _zscore(df[f"{prefix}_deleveraging"], window=z_window if z_window > 0 else None)

    forced = pd.Series(pd.NA, index=df.index, dtype="object")
    forced_long = ((df[net_col] > 0) & (df[f"{prefix}_deleveraging_z"] >= force_threshold)).fillna(False)
    forced_short = ((df[net_col] < 0) & (df[f"{prefix}_deleveraging_z"] >= force_threshold)).fillna(False)
    forced.loc[forced_long] = "long_liquidation"
    forced.loc[forced_short] = "short_covering"
    df[f"{prefix}_forced"] = forced



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


def fetch_market_timeseries(
    domain: str,
    dataset_id: str,
    app_token: Optional[str],
    market_exact: str,
    start: Optional[str],
    end: Optional[str],
    groups: Optional[str] = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch rows for a single exact market_and_exchange_names, optionally bounded by start/end (YYYY-MM-DD).
    """
    meta = get_dataset_metadata(domain, dataset_id, app_token)
    fields = detect_fields(meta)
    group_keys = parse_groups(groups)

    # Build SoQL
    market_escaped = market_exact.replace("'", "''")
    where_parts = [f"{fields.market_name} = '{market_escaped}'"]

    # Socrata fixed timestamp comparisons want full timestamp strings; T00:00:00.000 is fine.
    if start:
        where_parts.append(f"{fields.report_date} >= '{start}T00:00:00.000'")
    if end:
        where_parts.append(f"{fields.report_date} <= '{end}T23:59:59.999'")

    # Always include leveraged funds fields for backward-compatible lf_* outputs.
    select_fields = [fields.report_date, fields.market_name, fields.lf_long, fields.lf_short]
    if fields.open_interest:
        select_fields.append(fields.open_interest)

    group_cols: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        "dealer": (fields.dealer_long, fields.dealer_short),
        "asset_mgr": (fields.asset_mgr_long, fields.asset_mgr_short),
        "other_rept": (fields.other_rept_long, fields.other_rept_short),
        "nonrept": (fields.nonrept_long, fields.nonrept_short),
    }
    for g in group_keys:
        if g == "lev_money":
            continue
        long_col, short_col = group_cols.get(g, (None, None))
        if not long_col or not short_col:
            print(f"Warning: Could not detect fields for group '{g}' in this dataset; skipping.", file=sys.stderr)
            continue
        select_fields.extend([long_col, short_col])

    soql = {
        "$select": ", ".join(select_fields),
        "$where": " AND ".join(where_parts),
        "$order": fields.report_date,
    }

    rows = list(soda_iter_rows(domain, dataset_id, app_token, soql_params=soql, page_size=50000))
    if not rows:
        raise RuntimeError(
            "No rows returned. Check the exact market name (market_and_exchange_names) and date range."
        )

    df = pd.DataFrame(rows)

    # Parse and coerce numeric fields
    df[fields.report_date] = pd.to_datetime(df[fields.report_date], errors="coerce").dt.date
    numeric_cols: List[str] = [fields.lf_long, fields.lf_short]
    if fields.open_interest:
        numeric_cols.append(fields.open_interest)
    for g in group_keys:
        if g == "lev_money":
            continue
        long_col, short_col = group_cols.get(g, (None, None))
        if long_col and short_col:
            numeric_cols.extend([long_col, short_col])
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics (always compute leveraged funds under lf_* names)
    _add_group_metrics(
        df=df,
        long_col=fields.lf_long,
        short_col=fields.lf_short,
        open_interest_col=fields.open_interest,
        prefix="lf",
        z_window=z_window,
        force_threshold=force_threshold,
    )

    # Optional additional participant groups
    for g in group_keys:
        if g == "lev_money":
            continue
        long_col, short_col = group_cols.get(g, (None, None))
        if not long_col or not short_col:
            continue
        prefix = GROUP_PREFIX[g]
        _add_group_metrics(
            df=df,
            long_col=long_col,
            short_col=short_col,
            open_interest_col=fields.open_interest,
            prefix=prefix,
            z_window=z_window,
            force_threshold=force_threshold,
        )

    # Standardize column names for output readability
    rename_map = {
        fields.report_date: "report_date",
        fields.market_name: "market_and_exchange_names",
        fields.lf_long: "leveraged_funds_long",
        fields.lf_short: "leveraged_funds_short",
    }
    if fields.open_interest:
        rename_map[fields.open_interest] = "open_interest"
    if fields.dealer_long:
        rename_map[fields.dealer_long] = "dealer_long"
    if fields.dealer_short:
        rename_map[fields.dealer_short] = "dealer_short"
    if fields.asset_mgr_long:
        rename_map[fields.asset_mgr_long] = "asset_mgr_long"
    if fields.asset_mgr_short:
        rename_map[fields.asset_mgr_short] = "asset_mgr_short"
    if fields.other_rept_long:
        rename_map[fields.other_rept_long] = "other_rept_long"
    if fields.other_rept_short:
        rename_map[fields.other_rept_short] = "other_rept_short"
    if fields.nonrept_long:
        rename_map[fields.nonrept_long] = "nonrept_long"
    if fields.nonrept_short:
        rename_map[fields.nonrept_short] = "nonrept_short"

    df = df.rename(columns=rename_map)
    return df


def fetch_multiple_instruments(
    domain: str,
    dataset_id: str,
    app_token: Optional[str],
    instruments: List[str],
    start: Optional[str],
    end: Optional[str],
    groups: Optional[str] = None,
    z_window: int = 0,
    force_threshold: float = 2.0,
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
                groups=groups,
                z_window=z_window,
                force_threshold=force_threshold,
            )
            latest = df.dropna(subset=["report_date"]).iloc[-1]
            row = {
                "instrument": alias,
                "report_date": latest["report_date"],
                "lf_net": latest["lf_net"],
                "lf_net_pct_oi": latest.get("lf_net_pct_oi"),
                "lf_z": latest.get("lf_z"),
                "lf_deleveraging_z": latest.get("lf_deleveraging_z"),
                "lf_forced": latest.get("lf_forced"),
            }

            # Optionally include additional groups in the summary output.
            group_keys = parse_groups(groups)
            for g in group_keys:
                if g == "lev_money":
                    continue
                prefix = GROUP_PREFIX[g]
                row[f"{prefix}_net_pct_oi"] = latest.get(f"{prefix}_net_pct_oi")
                row[f"{prefix}_z"] = latest.get(f"{prefix}_z")
                row[f"{prefix}_deleveraging_z"] = latest.get(f"{prefix}_deleveraging_z")
                row[f"{prefix}_forced"] = latest.get(f"{prefix}_forced")

            results.append(row)
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
    print("\n" + "=" * 110)
    print(
        f"{'Instrument':<12} {'Date':<12} {'Net Position':>14} {'Net % Open Int':>15} {'Position Z':>11} {'Delev Z':>11} {'Forced Flow':>16}"
    )
    print("=" * 110)

    # Print rows
    for r in sorted_results:
        net = f"{r['lf_net']:,.0f}" if r['lf_net'] is not None and not pd.isna(r['lf_net']) else "N/A"
        pct = f"{r['lf_net_pct_oi']:.1f}%" if r['lf_net_pct_oi'] is not None and not pd.isna(r['lf_net_pct_oi']) else "N/A"
        z = f"{r['lf_z']:.2f}" if r['lf_z'] is not None and not pd.isna(r['lf_z']) else "N/A"
        dz = (
            f"{r['lf_deleveraging_z']:.2f}"
            if r.get("lf_deleveraging_z") is not None and not pd.isna(r.get("lf_deleveraging_z"))
            else "N/A"
        )
        forced = str(r.get("lf_forced") or "").replace("_", " ").title()
        print(f"{r['instrument']:<12} {str(r['report_date']):<12} {net:>14} {pct:>15} {z:>11} {dz:>11} {forced:>16}")

    print("=" * 110)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch CFTC COT positioning via PRE/Socrata API.")
    p.add_argument("--domain", default=DEFAULT_DOMAIN, help="Socrata domain (default: publicreportinghub.cftc.gov).")
    p.add_argument(
        "--dataset",
        default="tff_futures_only",
        help=f"Dataset key in {list(DATASETS.keys())} or a raw dataset id (e.g. gpe5-46if).",
    )
    p.add_argument("--app-token", default=os.getenv("SODA_APP_TOKEN"), help="Optional Socrata App Token.")
    p.add_argument("--market", default=None, help="Exact market_and_exchange_names value to fetch.")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional).")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional).")
    p.add_argument("--out", default=None, help="Write CSV to this path (optional).")
    p.add_argument(
        "--groups",
        default=None,
        help="Comma-separated participant groups to include (leveraged funds always included): dealer,asset_mgr,other_rept,nonrept,all",
    )
    p.add_argument(
        "--z-window",
        type=int,
        default=0,
        help="Rolling z-score window in weeks (0 = use full sample).",
    )
    p.add_argument(
        "--force-threshold",
        type=float,
        default=2.0,
        help="Threshold for *_forced classification using *_deleveraging_z.",
    )
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
            groups=args.groups,
            z_window=args.z_window,
            force_threshold=args.force_threshold,
        )
        print_summary_table(results)
        return 0

    # Handle --market (single market mode)
    if not args.market:
        print("Error: provide --market, --all, or --instruments", file=sys.stderr)
        return 2

    df = fetch_market_timeseries(
        domain=args.domain,
        dataset_id=dataset_id,
        app_token=args.app_token,
        market_exact=args.market,
        start=args.start,
        end=args.end,
        groups=args.groups,
        z_window=args.z_window,
        force_threshold=args.force_threshold,
    )

    # Print a compact "latest reading"
    latest = df.dropna(subset=["report_date"]).iloc[-1]
    print("\nLatest row:")
    latest_cols = ["report_date", "lf_net", "lf_net_pct_oi", "lf_z", "lf_deleveraging_z", "lf_forced"]
    latest_cols = [c for c in latest_cols if c in df.columns]
    print(latest[latest_cols].to_string())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(df):,} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
