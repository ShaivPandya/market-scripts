#!/usr/bin/env python3
"""
Single-company financials from SEC EDGAR companyfacts.

Outputs annual and quarterly Revenue/EPS history with filing links, plus key growth
metrics and latest-filing segment/region revenue breakdown.
"""

from __future__ import annotations
import logging
import os
import warnings

import re
from collections import defaultdict
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import requests

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

ANNUAL_DISPLAY_LIMIT = 5
QUARTERLY_DISPLAY_LIMIT = 20
ANNUAL_YOY_STEP = 1
QUARTERLY_YOY_STEP = 4


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
        return f"END:{e.get('end', '')}"

    return _sort_newest(_keep_latest_by(filtered, _key))


def _duration_days(e: dict) -> Optional[int]:
    start = _parse_iso_date(e.get("start"))
    end = _parse_iso_date(e.get("end"))
    if start is None or end is None or end < start:
        return None
    return (end - start).days + 1


def _infer_ytd_quarters(e: dict) -> Optional[int]:
    dur = _duration_days(e)
    if dur is not None:
        if dur <= 130:
            return 1
        if dur <= 220:
            return 2
        if dur <= 320:
            return 3
        return 4

    fp = str(e.get("fp") or "")
    if fp == "Q1":
        return 1
    if fp == "Q2":
        return 2
    if fp == "Q3":
        return 3
    if fp in {"Q4", "FY"}:
        return 4
    return None


def _as_int(v: object) -> Optional[int]:
    try:
        return int(str(v))
    except Exception:
        return None


def _annual_calendar(entries: List[dict]) -> List[Tuple[date, int]]:
    """
    Build a clean fiscal-year anchor calendar: one trusted fiscal-year-end date per FY.

    SEC companyfacts can include comparative prior-year columns in later 10-K filings,
    where deduping by period-end and "latest filed" can attach misleading FY metadata.
    Grouping by FY first avoids mapping old period-ends into a newer fiscal year.
    """
    grouped: Dict[int, List[dict]] = defaultdict(list)
    fallback: List[Tuple[date, int]] = []

    for e in entries:
        if not _is_valid_fact_row(e):
            continue
        if str(e.get("fp") or "") != "FY":
            continue
        if str(e.get("form") or "") not in ALLOWED_ANNUAL_FORMS:
            continue

        end = _parse_iso_date(e.get("end"))
        if end is None:
            continue

        fy = _as_int(e.get("fy"))
        if fy is None:
            fallback.append((end, end.year))
            continue
        grouped[fy].append(e)

    out: List[Tuple[date, int]] = []
    for fy, rows in grouped.items():
        # Prefer the row with the latest period-end; tie-break by filed date.
        best = max(rows, key=lambda r: (_parse_iso_date(r.get("end")) or date.min, str(r.get("filed") or "")))
        end = _parse_iso_date(best.get("end"))
        if end is not None:
            out.append((end, fy))

    if out:
        return sorted(out, key=lambda x: x[0])
    return sorted(fallback, key=lambda x: x[0])


def _assign_fiscal_year(end: date, annual_calendar: List[Tuple[date, int]]) -> int:
    if not annual_calendar:
        return end.year

    for annual_end, fy in annual_calendar:
        gap = (annual_end - end).days
        if end <= annual_end and gap <= 370:
            return fy

    first_end, first_fy = annual_calendar[0]
    last_end, last_fy = annual_calendar[-1]

    if end > last_end:
        step = max(1, ((end - last_end).days // 320) + 1)
        return last_fy + step

    if end < first_end:
        step = max(1, ((first_end - end).days // 320) + 1)
        return first_fy - step

    # end is between calendar entries but no annual_end is within 370 days.
    # Find the nearest annual end and interpolate by year distance.
    nearest_end, nearest_fy = min(annual_calendar, key=lambda ac: abs((ac[0] - end).days))
    offset = round((end - nearest_end).days / 365.25)
    return nearest_fy + offset


def _pick_best_for_target(entries: List[dict], target_quarters: int, prefer_form: str) -> Optional[dict]:
    if not entries:
        return None

    target_days = 91 * max(1, target_quarters)

    def _score(e: dict) -> Tuple[int, str, int]:
        form = str(e.get("form") or "")
        filed = str(e.get("filed") or "")
        dur = _duration_days(e)
        diff = abs((dur if dur is not None else target_days) - target_days)

        form_pref = 0
        if prefer_form == "quarterly" and form.startswith("10-Q"):
            form_pref = 1
        elif prefer_form == "annual" and form.startswith("10-K"):
            form_pref = 1
        return (form_pref, filed, -diff)

    return max(entries, key=_score)


def _quarterly_fact_entries(entries: List[dict]) -> List[dict]:
    filtered = [
        e
        for e in entries
        if _is_valid_fact_row(e)
        and str(e.get("fp") or "") in (QUARTER_FPS | {"FY"})
        and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
    ]
    if not filtered:
        return []

    by_end: Dict[str, List[dict]] = defaultdict(list)
    for e in filtered:
        end = str(e.get("end") or "")
        if end:
            by_end[end].append(e)

    ordered_ends: List[Tuple[date, str]] = []
    for end_str in by_end.keys():
        d = _parse_iso_date(end_str)
        if d is not None:
            ordered_ends.append((d, end_str))
    if not ordered_ends:
        return []
    ordered_ends.sort(key=lambda x: x[0])

    annual_calendar = _annual_calendar(entries)
    ends_by_fy: Dict[int, List[Tuple[date, str]]] = defaultdict(list)
    for end_date, end_str in ordered_ends:
        fy = _assign_fiscal_year(end_date, annual_calendar)
        ends_by_fy[fy].append((end_date, end_str))

    normalized: List[dict] = []
    for fy, ends in ends_by_fy.items():
        fy_ends = sorted(ends, key=lambda x: x[0])
        if len(fy_ends) > 4:
            fy_ends = fy_ends[-4:]

        ytd_prev: Optional[float] = None
        for q_idx, (_, end_str) in enumerate(fy_ends, start=1):
            candidates = by_end[end_str]
            prefer_form = "annual" if q_idx == 4 else "quarterly"

            direct_candidates = [e for e in candidates if _infer_ytd_quarters(e) == 1]
            direct_entry = _pick_best_for_target(direct_candidates, 1, prefer_form="quarterly")

            ytd_candidates = [
                e
                for e in candidates
                if _infer_ytd_quarters(e) == q_idx or (q_idx == 4 and str(e.get("fp") or "") == "FY")
            ]
            ytd_entry = _pick_best_for_target(ytd_candidates, q_idx, prefer_form=prefer_form)

            value: Optional[float] = None
            source: Optional[dict] = None

            if direct_entry is not None:
                value = _as_float(direct_entry.get("val"))
                source = direct_entry

            if value is None and ytd_entry is not None:
                ytd_val = _as_float(ytd_entry.get("val"))
                if ytd_val is not None:
                    value = ytd_val if q_idx == 1 or ytd_prev is None else (ytd_val - ytd_prev)
                    source = ytd_entry

            if value is None or source is None:
                continue

            ytd_val_from_entry = _as_float(ytd_entry.get("val")) if ytd_entry is not None else None
            if ytd_val_from_entry is not None:
                ytd_prev = ytd_val_from_entry
            else:
                ytd_prev = value if ytd_prev is None else (ytd_prev + value)

            row = dict(source)
            row["val"] = value
            row["fy"] = fy
            row["fp"] = f"Q{q_idx}"
            row["end"] = end_str
            normalized.append(row)

    return _sort_newest(_keep_latest_by(normalized, lambda e: str(e.get("end") or "")))


def _period_label(e: dict, frequency: str) -> str:
    end = str(e.get("end") or "")
    fy = e.get("fy")
    fp = str(e.get("fp") or "")

    if frequency == "annual":
        if len(end) >= 4:
            return f"FY{end[:4]}"
        if fy is not None:
            return f"FY{fy}"
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

    annual_rows_full = _rows_from_entries(
        annual_entries,
        frequency="annual",
        # Pull one extra year so oldest displayed row still has YoY.
        limit=ANNUAL_DISPLAY_LIMIT + ANNUAL_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=ANNUAL_YOY_STEP,
        yoy_abs_denom=False,
    )
    quarterly_rows_full = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        # Pull four extra quarters so oldest displayed row still has YoY.
        limit=QUARTERLY_DISPLAY_LIMIT + QUARTERLY_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=QUARTERLY_YOY_STEP,
        yoy_abs_denom=False,
    )
    return annual_rows_full[:ANNUAL_DISPLAY_LIMIT], quarterly_rows_full[:QUARTERLY_DISPLAY_LIMIT]


def _derived_eps_entries(us_gaap: dict, frequency: str) -> List[dict]:
    period_key = lambda e: f"END:{e.get('end') or ''}"

    ni_raw = _entries_for(us_gaap, "NetIncomeLoss", "USD")
    if frequency == "annual":
        ni = [
            e
            for e in ni_raw
            if _is_valid_fact_row(e)
            and str(e.get("fp") or "") == "FY"
            and str(e.get("form") or "") in ALLOWED_ANNUAL_FORMS
        ]
    else:
        ni = _quarterly_fact_entries(ni_raw)

    shares = []
    for concept in (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ):
        sh_raw = _entries_for(us_gaap, concept, "shares")
        if frequency == "annual":
            sh = [
                e
                for e in sh_raw
                if _is_valid_fact_row(e)
                and str(e.get("fp") or "") == "FY"
                and str(e.get("form") or "") in ALLOWED_ANNUAL_FORMS
            ]
        else:
            sh = _quarterly_fact_entries(sh_raw)
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

    annual_rows_full = _rows_from_entries(
        annual_entries,
        frequency="annual",
        limit=ANNUAL_DISPLAY_LIMIT + ANNUAL_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=ANNUAL_YOY_STEP,
        yoy_abs_denom=True,
    )
    quarterly_rows_full = _rows_from_entries(
        quarterly_entries,
        frequency="quarterly",
        limit=QUARTERLY_DISPLAY_LIMIT + QUARTERLY_YOY_STEP,
        cik_str=cik_str,
        submissions=submissions,
        yoy_step=QUARTERLY_YOY_STEP,
        yoy_abs_denom=True,
    )
    return annual_rows_full[:ANNUAL_DISPLAY_LIMIT], quarterly_rows_full[:QUARTERLY_DISPLAY_LIMIT]


def _calc_cagr(rows: List[dict], years: int = 3, *, abs_fallback: bool = False) -> Optional[float]:
    values = [_as_float(r.get("value")) for r in rows]
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    n = min(years, len(clean) - 1)
    if n < 1:
        return None
    latest = clean[0]
    prior = clean[n]
    if latest > 0 and prior > 0:
        return (latest / prior) ** (1.0 / n) - 1.0

    # EPS can cross zero (loss to profit), where strict CAGR is undefined.
    # Fallback to CAGR on absolute magnitude so the card remains informative.
    if not abs_fallback or latest == 0 or prior == 0:
        return None
    return (abs(latest) / abs(prior)) ** (1.0 / n) - 1.0


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
    if any(k in d for k in ("geo", "geograph", "region", "country", "market", "area", "location", "territor")):
        return "region"
    if any(k in d for k in ("product", "service", "segment", "lineofbusiness", "business", "operatingsegment", "reportablesegment")):
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
    return (
        s in {"total", "all"}
        or s.startswith("total ")
        or s in {"worldwide", "global"}
        or s.startswith("worldwide ")
        or s.startswith("global ")
        or "all geograph" in s
        or "all region" in s
        or "all countr" in s
        or "all market" in s
        or "entire company" in s
        or "whole company" in s
        or "consolidated" in s
        or "elimination" in s
        or "all other" in s
        or s.startswith("all ")
        or " total " in s
        or s.endswith(" total")
        or s.endswith(" totals")
    )


def _classified_row_members(segment_obj: object) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for dim, member in _iter_segment_pairs(segment_obj):
        kind = _classify_dimension(dim)
        if kind is None:
            continue
        label = _normalize_member_label(member)
        if not label or _is_total_like_member(label):
            continue
        key = (kind, label)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _row_has_only_total_context(row: dict) -> bool:
    seg = row.get("segment")
    if not seg:
        return True

    for dim, member in _iter_segment_pairs(seg):
        kind = _classify_dimension(dim)
        if kind is None:
            continue
        label = _normalize_member_label(member)
        if label and not _is_total_like_member(label):
            return False
    return True


def _pick_best_period_rows(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("fp") or ""), str(row.get("start") or ""))
        grouped[key].append(row)

    def _score(group_rows: List[dict]) -> Tuple[int, int, int, str]:
        informative = 0
        total_like = 0
        for row in group_rows:
            if _classified_row_members(row.get("segment")):
                informative += 1
            if _row_has_only_total_context(row):
                total_like += 1
        latest_filed = max(str(r.get("filed") or "") for r in group_rows)
        return informative, total_like, len(group_rows), latest_filed

    return max(grouped.values(), key=_score)


def _pct_rows(values_by_label: Dict[str, float], total: Optional[float]) -> List[dict]:
    rows = []
    for label, value in sorted(values_by_label.items(), key=lambda kv: kv[1], reverse=True):
        pct = None
        if total not in (None, 0):
            pct = value / float(total)
        rows.append({"label": label, "value": value, "pct_of_total": pct})
    return rows


def _parse_money_like(v: object) -> Optional[float]:
    if isinstance(v, (int, float)):
        return _as_float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        neg = True
        s = s[1:]
    try:
        out = float(s)
    except Exception:
        return None
    return -out if neg else out


def _rows_to_value_map(rows: object) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if not label or _is_total_like_member(label):
            continue
        val = _parse_money_like(row.get("value"))
        if val is None:
            continue
        cur = out.get(label)
        if cur is None or abs(val) > abs(cur):
            out[label] = val
    return out


def _filing_context_for_nlp(html: str) -> str:
    try:
        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    except Exception:
        return ""

    head = html.lstrip()[:5000].lower()
    parser = "lxml"
    if head.startswith("<?xml") or "<xbrl" in head or "<ix:" in head:
        parser = "xml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, parser)
    blocks: List[str] = []
    keywords = ("revenue", "segment", "geograph", "region", "united states", "foreign")

    for table in soup.find_all("table"):
        txt = " ".join(table.get_text(" ", strip=True).split())
        if not txt:
            continue
        lower = txt.lower()
        if "revenue" not in lower:
            continue
        if not any(k in lower for k in keywords[1:]):
            continue
        blocks.append(txt[:2500])
        if len(blocks) >= 8:
            break

    if not blocks:
        body = " ".join(soup.get_text(" ", strip=True).split())
        if body:
            lower = body.lower()
            idx = lower.find("revenue")
            if idx >= 0:
                start = max(0, idx - 5000)
                end = min(len(body), idx + 15000)
                blocks.append(body[start:end])
            else:
                blocks.append(body[:12000])

    joined = "\n\n".join(blocks).strip()
    return joined[:18000]


def _extract_breakdown_via_nlp(
    *,
    cik_str: str,
    accn: str,
    form: str,
    filed: str,
    submissions: Optional[dict],
) -> Optional[dict]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    filing_url = build_filing_url(cik_str, accn, submissions=submissions)
    if not filing_url:
        return None

    try:
        resp = requests.get(
            filing_url,
            headers={"User-Agent": "market-scripts research@example.com"},
            timeout=25,
        )
        if resp.status_code != 200 or not resp.text:
            return None
    except Exception:
        return None

    context = _filing_context_for_nlp(resp.text)
    if not context:
        return None

    prompt = (
        "Extract ONLY the latest-period revenue breakdown from this SEC filing excerpt.\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "period_end": "YYYY-MM-DD or empty",\n'
        '  "total_revenue": number or null,\n'
        '  "unit_scale": "ones" | "thousands" | "millions" | "billions",\n'
        '  "by_segment": [{"label": string, "value": number}],\n'
        '  "by_region": [{"label": string, "value": number}]\n'
        "}\n"
        "Rules: include only revenue rows, exclude totals/eliminations, keep latest quarter in this filing.\n"
        "No markdown, no explanation, JSON only.\n\n"
        f"FORM: {form}\nFILED: {filed}\nACCN: {accn}\nURL: {filing_url}\n\n"
        f"EXCERPT:\n{context}"
    )

    try:
        from openai import OpenAI

        client = OpenAI()
        out = client.responses.create(model="gpt-5-mini", input=prompt)
        txt = (out.output_text or "").strip()
        if not txt:
            return None
    except Exception:
        return None

    # Be resilient to minor wrapper text around JSON.
    try:
        import json

        start = txt.find("{")
        end = txt.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        payload = json.loads(txt[start : end + 1])
    except Exception:
        return None

    by_segment = _rows_to_value_map(payload.get("by_segment"))
    by_region = _rows_to_value_map(payload.get("by_region"))
    if not by_segment and not by_region:
        return None

    scale_map = {
        "ones": 1.0,
        "thousands": 1_000.0,
        "millions": 1_000_000.0,
        "billions": 1_000_000_000.0,
    }
    scale_key = str(payload.get("unit_scale") or "ones").strip().lower()
    scale = scale_map.get(scale_key, 1.0)
    if scale != 1.0:
        by_segment = {k: v * scale for k, v in by_segment.items()}
        by_region = {k: v * scale for k, v in by_region.items()}

    total_val = _parse_money_like(payload.get("total_revenue"))
    if total_val is not None:
        total_val *= scale
    else:
        seg_sum = sum(abs(v) for v in by_segment.values())
        reg_sum = sum(abs(v) for v in by_region.values())
        total_val = seg_sum if seg_sum >= reg_sum and seg_sum > 0 else (reg_sum if reg_sum > 0 else None)

    return {
        "accn": accn,
        "period_end": str(payload.get("period_end") or ""),
        "filed": filed,
        "form": form,
        "by_segment": _pct_rows(by_segment, total_val),
        "by_region": _pct_rows(by_region, total_val),
    }


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

        period_rows = _pick_best_period_rows(period_rows)

        total_candidates = [e for e in period_rows if _row_has_only_total_context(e)]
        total_val: Optional[float] = None
        if total_candidates:
            total_row = max(
                total_candidates,
                key=lambda e: (abs(_as_float(e.get("val")) or 0.0), str(e.get("filed") or "")),
            )
            total_val = _as_float(total_row.get("val"))

        by_segment: Dict[str, float] = {}
        by_region: Dict[str, float] = {}

        for row in period_rows:
            classified = _classified_row_members(row.get("segment"))
            if not classified:
                continue
            if len(classified) != 1:
                # Cross-dimensional intersections are ambiguous for a simple
                # one-axis breakdown; skip to avoid double counting.
                continue
            kind, label = classified[0]
            if not kind or not label:
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

    annual_filings = [f for f in filings if str(f.get("form") or "") in ALLOWED_ANNUAL_FORMS]
    search_filings = annual_filings if annual_filings else filings
    latest_filing = search_filings[0]
    chosen: Optional[dict] = None
    for f in search_filings:
        candidate = _extract_breakdown_for_filing(us_gaap, f["accn"])
        if candidate.get("by_segment") or candidate.get("by_region"):
            chosen = {**f, **candidate}
            break

    if chosen is None:
        nlp_candidate = _extract_breakdown_via_nlp(
            cik_str=cik_str,
            accn=str(latest_filing.get("accn") or ""),
            form=str(latest_filing.get("form") or ""),
            filed=str(latest_filing.get("filed") or ""),
            submissions=submissions,
        )
        if nlp_candidate and (nlp_candidate.get("by_segment") or nlp_candidate.get("by_region")):
            chosen = {**latest_filing, **nlp_candidate}
            by_segment = chosen.get("by_segment") or []
            by_region = chosen.get("by_region") or []
        else:
            chosen = latest_filing
            by_segment = []
            by_region = []
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
        "revenue_cagr_3y": _calc_cagr(annual_revenue, years=3, abs_fallback=False),
        "eps_cagr_3y": _calc_cagr(annual_eps, years=3, abs_fallback=True),
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
