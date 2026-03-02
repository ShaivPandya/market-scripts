#!/usr/bin/env python3
"""
Single-company financials from SEC EDGAR companyfacts.

Outputs annual and quarterly Revenue/EPS history with filing links, plus key growth
metrics and latest-filing segment/region revenue breakdown.
"""

from __future__ import annotations
import logging

import re
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from edgar_fetcher import (
    build_filing_url,
    fetch_companyfacts_by_cik,
    fetch_submissions_by_cik,
    get_cik_for_ticker,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_ANNUAL_FORMS = {"10-K", "10-K/A"}
ALLOWED_QUARTERLY_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
QUARTER_FPS = {"Q1", "Q2", "Q3", "Q4"}

REVENUE_CONCEPTS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

EPS_CONCEPTS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
EPS_UNITS = ("USD/shares", "USD-per-shares")


def _as_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def _safe_growth(numer: Optional[float], denom: Optional[float], denom_abs: bool = False) -> Optional[float]:
    if numer is None or denom is None:
        return None
    d = abs(denom) if denom_abs else denom
    if d == 0:
        return None
    return numer / d


def _parse_iso_date(s: object) -> Optional[date]:
    if not isinstance(s, str) or not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _entries_for(us_gaap: dict, concept: str, unit: str) -> List[dict]:
    try:
        rows = us_gaap[concept]["units"][unit]
    except (KeyError, TypeError):
        return []
    return rows if isinstance(rows, list) else []


def _is_valid_fact_row(e: dict) -> bool:
    return bool(e.get("end")) and _as_float(e.get("val")) is not None


def _keep_latest_by(entries: Iterable[dict], key_fn) -> List[dict]:
    best: Dict[str, dict] = {}
    for e in entries:
        key = key_fn(e)
        if not key:
            continue
        filed = str(e.get("filed") or "")
        curr = best.get(key)
        if curr is None or filed > str(curr.get("filed") or ""):
            best[key] = e
    return list(best.values())


def _sort_newest(entries: List[dict]) -> List[dict]:
    def _k(e: dict):
        d = _parse_iso_date(e.get("end"))
        return (
            d or date.min,
            str(e.get("filed") or ""),
        )

    return sorted(entries, key=_k, reverse=True)


def _latest_end_date(entries: List[dict]) -> date:
    latest = date.min
    for e in entries:
        d = _parse_iso_date(e.get("end"))
        if d is not None and d > latest:
            latest = d
    return latest


def _latest_filed_date(entries: List[dict]) -> str:
    if not entries:
        return ""
    return max(str(e.get("filed") or "") for e in entries)


def _pick_best_concept_entries(
    us_gaap: dict,
    concepts: Iterable[str],
    unit: str,
    extractor: Callable[[List[dict]], List[dict]],
) -> List[dict]:
    """
    Choose the strongest concept series for a metric.
    Priority:
      1) most recent period end date
      2) largest history length
      3) latest filing date
    """
    best_entries: List[dict] = []
    best_score: Optional[Tuple[date, int, str]] = None

    for concept in concepts:
        raw = _entries_for(us_gaap, concept, unit)
        if not raw:
            continue
        candidate = extractor(raw)
        if not candidate:
            continue

        score = (
            _latest_end_date(candidate),
            len(candidate),
            _latest_filed_date(candidate),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_entries = candidate

    return best_entries


def _annual_fact_entries(entries: List[dict]) -> List[dict]:
    filtered = [
        e
        for e in entries
        if _is_valid_fact_row(e)
        and e.get("fp") == "FY"
        and str(e.get("form") or "") in ALLOWED_ANNUAL_FORMS
    ]

    def _key(e: dict) -> str:
        fy = e.get("fy")
        if fy is not None:
            return f"FY:{fy}"
        return f"END:{e.get('end', '')}"

    return _sort_newest(_keep_latest_by(filtered, _key))


def _quarterly_fact_entries(entries: List[dict]) -> List[dict]:
    filtered = [
        e
        for e in entries
        if _is_valid_fact_row(e)
        and str(e.get("fp") or "") in QUARTER_FPS
        and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
    ]
    return _sort_newest(_keep_latest_by(filtered, lambda e: str(e.get("end") or "")))


def _period_label(e: dict, frequency: str) -> str:
    end = str(e.get("end") or "")
    fy = e.get("fy")
    fp = str(e.get("fp") or "")

    if frequency == "annual":
        if fy is not None:
            return f"FY{fy}"
        if len(end) >= 4:
            return f"FY{end[:4]}"
        return "FY"

    if fp in QUARTER_FPS and fy is not None:
        return f"{fp} {fy}"

    d = _parse_iso_date(end)
    if d is not None:
        q = ((d.month - 1) // 3) + 1
        return f"Q{q} {d.year}"
    return fp or "Quarter"


def _rows_from_entries(
    entries: List[dict],
    *,
    frequency: str,
    limit: int,
    cik_str: str,
    submissions: Optional[dict],
    yoy_step: int,
    yoy_abs_denom: bool,
) -> List[dict]:
    rows: List[dict] = []
    for e in entries[:limit]:
        val = _as_float(e.get("val"))
        if val is None:
            continue
        accn = str(e.get("accn") or "")
        rows.append(
            {
                "period_label": _period_label(e, frequency),
                "period_end": str(e.get("end") or ""),
                "value": val,
                "yoy_growth": None,
                "form": str(e.get("form") or ""),
                "filed": str(e.get("filed") or ""),
                "accn": accn,
                "filing_url": build_filing_url(cik_str, accn, submissions=submissions),
            }
        )

    for i, row in enumerate(rows):
        j = i + yoy_step
        if j >= len(rows):
            continue
        curr = _as_float(row.get("value"))
        prev = _as_float(rows[j].get("value"))
        if curr is None or prev is None:
            continue
        row["yoy_growth"] = _safe_growth(curr - prev, prev, denom_abs=yoy_abs_denom)

    return rows


def _build_revenue_rows(us_gaap: dict, cik_str: str, submissions: Optional[dict]) -> Tuple[List[dict], List[dict]]:
    annual_entries = _pick_best_concept_entries(
        us_gaap,
        REVENUE_CONCEPTS,
        "USD",
        _annual_fact_entries,
    )
    quarterly_entries = _pick_best_concept_entries(
        us_gaap,
        REVENUE_CONCEPTS,
        "USD",
        _quarterly_fact_entries,
    )

    annual_rows = _rows_from_entries(
        annual_entries,
        frequency="annual",
        limit=5,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=1,
        yoy_abs_denom=False,
    )
    quarterly_rows = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        limit=20,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=4,
        yoy_abs_denom=False,
    )
    return annual_rows, quarterly_rows


def _derived_eps_entries(us_gaap: dict, frequency: str) -> List[dict]:
    if frequency == "annual":
        fp_filter = {"FY"}
        forms = ALLOWED_ANNUAL_FORMS
        period_key = lambda e: f"FY:{e.get('fy') or ''}|END:{e.get('end') or ''}"
    else:
        fp_filter = QUARTER_FPS
        forms = ALLOWED_QUARTERLY_FORMS
        period_key = lambda e: f"END:{e.get('end') or ''}"

    ni_raw = _entries_for(us_gaap, "NetIncomeLoss", "USD")
    ni = [
        e
        for e in ni_raw
        if _is_valid_fact_row(e)
        and str(e.get("fp") or "") in fp_filter
        and str(e.get("form") or "") in forms
    ]

    shares = []
    for concept in (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ):
        sh_raw = _entries_for(us_gaap, concept, "shares")
        sh = [
            e
            for e in sh_raw
            if _is_valid_fact_row(e)
            and str(e.get("fp") or "") in fp_filter
            and str(e.get("form") or "") in forms
        ]
        if sh:
            shares = sh
            break

    if not ni or not shares:
        return []

    shares_by_period = {period_key(e): e for e in _keep_latest_by(shares, period_key)}

    derived: List[dict] = []
    for ni_row in _keep_latest_by(ni, period_key):
        key = period_key(ni_row)
        sh_row = shares_by_period.get(key)
        if not sh_row:
            continue
        ni_val = _as_float(ni_row.get("val"))
        sh_val = _as_float(sh_row.get("val"))
        eps = _safe_growth(ni_val, sh_val, denom_abs=False)
        if eps is None:
            continue
        clone = dict(ni_row)
        clone["val"] = eps
        derived.append(clone)

    return _sort_newest(derived)


def _build_eps_rows(us_gaap: dict, cik_str: str, submissions: Optional[dict]) -> Tuple[List[dict], List[dict]]:
    annual_entries: List[dict] = []
    quarterly_entries: List[dict] = []

    for concept in EPS_CONCEPTS:
        for unit in EPS_UNITS:
            raw = _entries_for(us_gaap, concept, unit)
            if not raw:
                continue
            a = _annual_fact_entries(raw)
            q = _quarterly_fact_entries(raw)
            if a and not annual_entries:
                annual_entries = a
            if q and not quarterly_entries:
                quarterly_entries = q
            if annual_entries and quarterly_entries:
                break
        if annual_entries and quarterly_entries:
            break

    if not annual_entries:
        annual_entries = _derived_eps_entries(us_gaap, "annual")
    if not quarterly_entries:
        quarterly_entries = _derived_eps_entries(us_gaap, "quarterly")

    annual_rows = _rows_from_entries(
        annual_entries,
        frequency="annual",
        limit=5,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=1,
        yoy_abs_denom=True,
    )
    quarterly_rows = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        limit=20,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=4,
        yoy_abs_denom=True,
    )
    return annual_rows, quarterly_rows


def _calc_cagr(rows: List[dict], years: int = 3) -> Optional[float]:
    values = [_as_float(r.get("value")) for r in rows]
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    n = min(years, len(clean) - 1)
    if n < 1:
        return None
    latest = clean[0]
    prior = clean[n]
    if latest <= 0 or prior <= 0:
        return None
    return (latest / prior) ** (1.0 / n) - 1.0


def _calc_avg_3q_yoy(rows: List[dict], denom_abs: bool) -> Optional[float]:
    values = [_as_float(r.get("value")) for r in rows]
    if len(values) < 7:
        return None
    changes: List[float] = []
    for i in (0, 1, 2):
        a = values[i]
        b = values[i + 4]
        if a is None or b is None:
            continue
        ch = _safe_growth(a - b, b, denom_abs=denom_abs)
        if ch is not None:
            changes.append(ch)
    if not changes:
        return None
    return sum(changes) / len(changes)


def _iter_segment_pairs(segment_obj: object) -> List[Tuple[str, str]]:
    if isinstance(segment_obj, dict):
        if "dimension" in segment_obj and "value" in segment_obj:
            dim = str(segment_obj.get("dimension") or "").strip()
            val = str(segment_obj.get("value") or "").strip()
            if dim and val:
                return [(dim, val)]
        out: List[Tuple[str, str]] = []
        for k, v in segment_obj.items():
            if isinstance(v, str):
                out.append((str(k), v))
            elif isinstance(v, dict):
                out.extend(_iter_segment_pairs(v))
            elif isinstance(v, list):
                for item in v:
                    out.extend(_iter_segment_pairs(item))
        return out
    if isinstance(segment_obj, list):
        out: List[Tuple[str, str]] = []
        for item in segment_obj:
            out.extend(_iter_segment_pairs(item))
        return out
    return []


def _classify_dimension(dimension: str) -> Optional[str]:
    d = dimension.lower()
    if any(k in d for k in ("geo", "geograph", "region", "country", "market", "area")):
        return "region"
    if any(k in d for k in ("product", "service", "segment", "lineofbusiness", "business")):
        return "segment"
    return None


def _normalize_member_label(raw_member: str) -> str:
    m = str(raw_member or "")
    if ":" in m:
        m = m.split(":", 1)[1]
    m = re.sub(r"Member$", "", m)
    m = re.sub(r"Segment$", "", m)
    m = re.sub(r"Region$", "", m)
    m = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", m)
    m = m.replace("_", " ").strip()
    return m


def _is_total_like_member(member: str) -> bool:
    s = member.lower()
    return "consolidated" in s or "total" == s


def _pct_rows(values_by_label: Dict[str, float], total: Optional[float]) -> List[dict]:
    rows = []
    for label, value in sorted(values_by_label.items(), key=lambda kv: kv[1], reverse=True):
        pct = None
        if total not in (None, 0):
            pct = value / float(total)
        rows.append({"label": label, "value": value, "pct_of_total": pct})
    return rows


def _extract_breakdown_for_filing(us_gaap: dict, accn: str) -> dict:
    for concept in REVENUE_CONCEPTS:
        all_rows = _entries_for(us_gaap, concept, "USD")
        if not all_rows:
            continue

        in_filing = [
            e
            for e in all_rows
            if str(e.get("accn") or "") == accn
            and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
            and _is_valid_fact_row(e)
        ]
        if not in_filing:
            continue

        target_end = max(str(e.get("end") or "") for e in in_filing)
        period_rows = [e for e in in_filing if str(e.get("end") or "") == target_end]
        if not period_rows:
            continue

        total_candidates = [e for e in period_rows if not e.get("segment")]
        total_val = _as_float(total_candidates[0].get("val")) if total_candidates else None

        by_segment: Dict[str, float] = {}
        by_region: Dict[str, float] = {}

        for row in period_rows:
            seg = row.get("segment")
            if not seg:
                continue

            seg_pairs = _iter_segment_pairs(seg)
            if len(seg_pairs) != 1:
                continue

            dim, member = seg_pairs[0]
            kind = _classify_dimension(dim)
            if kind is None:
                continue

            label = _normalize_member_label(member)
            if not label or _is_total_like_member(label):
                continue

            val = _as_float(row.get("val"))
            if val is None:
                continue

            bucket = by_region if kind == "region" else by_segment
            prev = bucket.get(label)
            if prev is None or abs(val) > abs(prev):
                bucket[label] = val

        if by_segment or by_region:
            best = max(period_rows, key=lambda e: str(e.get("filed") or ""))
            return {
                "accn": accn,
                "period_end": target_end,
                "filed": str(best.get("filed") or ""),
                "form": str(best.get("form") or ""),
                "by_segment": _pct_rows(by_segment, total_val),
                "by_region": _pct_rows(by_region, total_val),
            }

    return {"accn": accn, "by_segment": [], "by_region": []}


def _candidate_revenue_filings(us_gaap: dict) -> List[dict]:
    by_accn: Dict[str, dict] = {}
    for concept in REVENUE_CONCEPTS:
        for e in _entries_for(us_gaap, concept, "USD"):
            form = str(e.get("form") or "")
            accn = str(e.get("accn") or "")
            filed = str(e.get("filed") or "")
            if form not in ALLOWED_QUARTERLY_FORMS or not accn or not filed:
                continue
            cur = by_accn.get(accn)
            if cur is None or filed > cur["filed"]:
                by_accn[accn] = {"accn": accn, "form": form, "filed": filed}
    return sorted(by_accn.values(), key=lambda x: x["filed"], reverse=True)


def _build_breakdown(us_gaap: dict, cik_str: str, submissions: Optional[dict]) -> dict:
    filings = _candidate_revenue_filings(us_gaap)
    if not filings:
        return {
            "source_filing": None,
            "by_segment": [],
            "by_region": [],
        }

    latest_filing = filings[0]
    chosen: Optional[dict] = None
    for f in filings:
        candidate = _extract_breakdown_for_filing(us_gaap, f["accn"])
        if candidate.get("by_segment") or candidate.get("by_region"):
            chosen = {**f, **candidate}
            break

    if chosen is None:
        chosen = latest_filing
        by_segment: List[dict] = []
        by_region: List[dict] = []
    else:
        by_segment = chosen.get("by_segment") or []
        by_region = chosen.get("by_region") or []

    accn = str(chosen.get("accn") or "")
    source_filing = {
        "form": str(chosen.get("form") or ""),
        "filed": str(chosen.get("filed") or ""),
        "accn": accn,
        "period_end": str(chosen.get("period_end") or ""),
        "filing_url": build_filing_url(cik_str, accn, submissions=submissions),
    }

    return {
        "source_filing": source_filing,
        "by_segment": by_segment,
        "by_region": by_region,
    }


def get_data(ticker: str) -> dict:
    tk = str(ticker or "").strip().upper()
    if not tk:
        raise ValueError("Ticker is required")

    cik_str = get_cik_for_ticker(tk)
    if not cik_str:
        raise ValueError(f"CIK not found for ticker: {tk}")

    facts = fetch_companyfacts_by_cik(cik_str)
    if facts is None:
        raise ValueError(f"No SEC EDGAR companyfacts available for ticker: {tk}")

    submissions = fetch_submissions_by_cik(cik_str)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    annual_revenue, quarterly_revenue = _build_revenue_rows(us_gaap, cik_str, submissions)
    annual_eps, quarterly_eps = _build_eps_rows(us_gaap, cik_str, submissions)

    if not annual_revenue and not quarterly_revenue and not annual_eps and not quarterly_eps:
        raise ValueError(f"No Revenue or EPS history found in EDGAR companyfacts for ticker: {tk}")

    metrics = {
        "revenue_cagr_3y": _calc_cagr(annual_revenue, years=3),
        "eps_cagr_3y": _calc_cagr(annual_eps, years=3),
        "avg_yoy_eps_growth_3q": _calc_avg_3q_yoy(quarterly_eps, denom_abs=True),
        "avg_yoy_revenue_growth_3q": _calc_avg_3q_yoy(quarterly_revenue, denom_abs=False),
    }

    breakdown = _build_breakdown(us_gaap, cik_str, submissions)

    return {
        "ticker": tk,
        "company_name": str(facts.get("entityName") or tk),
        "cik": cik_str,
        "metrics": metrics,
        "annual": {
            "revenue": annual_revenue,
            "eps": annual_eps,
        },
        "quarterly": {
            "revenue": quarterly_revenue,
            "eps": quarterly_eps,
        },
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    LOGGER.info('Starting script execution: %s', __file__)
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR financials for one ticker")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    print(json.dumps(get_data(args.ticker), indent=2))
