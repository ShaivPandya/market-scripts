"""Shared SEC companyfacts period selection and normalization helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import date

ALLOWED_ANNUAL_FORMS = {"10-K", "10-K/A"}
ALLOWED_QUARTERLY_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A"}
QUARTER_FPS = {"Q1", "Q2", "Q3", "Q4"}

ANNUAL_DURATION_MIN_DAYS = 330
ANNUAL_DURATION_MAX_DAYS = 380
QUARTER_DURATION_MIN_DAYS = 60
QUARTER_DURATION_MAX_DAYS = 130
PERIOD_OWN_FILING_MAX_LAG_DAYS = 240


def _as_float(v: object) -> float | None:
    if not isinstance(v, (int, float, str)):
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


def _safe_growth(numer: float | None, denom: float | None, denom_abs: bool = False) -> float | None:
    if numer is None or denom is None:
        return None
    d = abs(denom) if denom_abs else denom
    if d == 0:
        return None
    return numer / d


def _parse_iso_date(s: object) -> date | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _entries_for(us_gaap: dict, concept: str, unit: str) -> list[dict]:
    try:
        rows = us_gaap[concept]["units"][unit]
    except (KeyError, TypeError):
        return []
    return rows if isinstance(rows, list) else []


def _is_valid_fact_row(e: dict) -> bool:
    return bool(e.get("end")) and _as_float(e.get("val")) is not None


def _frame_is_quarterly(e: dict) -> bool:
    frame = str(e.get("frame") or "").upper()
    return bool(re.search(r"Q[1-4](?:$|[^0-9])", frame))


def _duration_days(e: dict) -> int | None:
    start = _parse_iso_date(e.get("start"))
    end = _parse_iso_date(e.get("end"))
    if start is None or end is None or end < start:
        return None
    return (end - start).days + 1


def _is_full_year_duration(e: dict) -> bool:
    dur = _duration_days(e)
    return dur is not None and ANNUAL_DURATION_MIN_DAYS <= dur <= ANNUAL_DURATION_MAX_DAYS


def _is_direct_quarter_duration(e: dict) -> bool:
    dur = _duration_days(e)
    return dur is not None and QUARTER_DURATION_MIN_DAYS <= dur <= QUARTER_DURATION_MAX_DAYS


def _is_annual_fact_row(e: dict) -> bool:
    return (
        _is_valid_fact_row(e)
        and str(e.get("fp") or "") == "FY"
        and str(e.get("form") or "") in ALLOWED_ANNUAL_FORMS
        and _is_full_year_duration(e)
        and not _frame_is_quarterly(e)
    )


def _filed_lag_days(e: dict) -> int | None:
    filed = _parse_iso_date(e.get("filed"))
    end = _parse_iso_date(e.get("end"))
    if filed is None or end is None:
        return None
    return (filed - end).days


def _is_amended_form(e: dict) -> bool:
    return str(e.get("form") or "").endswith("/A")


def _as_int(v: object) -> int | None:
    try:
        return int(str(v))
    except Exception:
        return None


def _canonical_fiscal_year(e: dict) -> int | None:
    end = _parse_iso_date(e.get("end"))
    if end is not None:
        return end.year
    return _as_int(e.get("fy"))


def _period_ownership_rank(e: dict, target_fiscal_year: int | None) -> int:
    fy = _as_int(e.get("fy"))
    if target_fiscal_year is not None and fy == target_fiscal_year:
        return 2

    lag = _filed_lag_days(e)
    if lag is not None and 0 <= lag <= PERIOD_OWN_FILING_MAX_LAG_DAYS:
        return 1
    return 0


def _keep_latest_by(entries: Iterable[dict], key_fn) -> list[dict]:
    best: dict[str, dict] = {}
    for e in entries:
        key = key_fn(e)
        if not key:
            continue
        filed = str(e.get("filed") or "")
        curr = best.get(key)
        if curr is None or filed > str(curr.get("filed") or ""):
            best[key] = e
    return list(best.values())


def _sort_newest(entries: list[dict]) -> list[dict]:
    def _k(e: dict):
        d = _parse_iso_date(e.get("end"))
        return (
            d or date.min,
            str(e.get("filed") or ""),
        )

    return sorted(entries, key=_k, reverse=True)


def _latest_end_date(entries: list[dict]) -> date:
    latest = date.min
    for e in entries:
        d = _parse_iso_date(e.get("end"))
        if d is not None and d > latest:
            latest = d
    return latest


def _latest_filed_date(entries: list[dict]) -> str:
    if not entries:
        return ""
    return max(str(e.get("filed") or "") for e in entries)


def _pick_best_concept_entries(
    us_gaap: dict,
    concepts: Iterable[str],
    unit: str,
    extractor: Callable[[list[dict]], list[dict]],
) -> list[dict]:
    """
    Choose the strongest concept series for a metric.
    Priority:
      1) most recent period end date
      2) largest history length
      3) latest filing date
    """
    best_entries: list[dict] = []
    best_score: tuple[date, int, str] | None = None

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


def _annual_fact_entries(entries: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        if not _is_annual_fact_row(e):
            continue
        fiscal_year = _canonical_fiscal_year(e)
        if fiscal_year is None:
            continue
        grouped[fiscal_year].append(e)

    out: list[dict] = []
    for fiscal_year, rows in grouped.items():
        best_row: dict | None = None
        best_score: tuple[int, int, str, int] | None = None
        for row in rows:
            dur = _duration_days(row)
            diff = abs((dur if dur is not None else 365) - 365)
            row_score = (
                _period_ownership_rank(row, fiscal_year),
                1 if _is_amended_form(row) else 0,
                str(row.get("filed") or ""),
                -diff,
            )
            if best_score is None or row_score > best_score:
                best_score = row_score
                best_row = row
        if best_row is None:
            continue
        best = dict(best_row)
        best["fy"] = fiscal_year
        best["fp"] = "FY"
        out.append(best)

    return _sort_newest(out)


def _infer_ytd_quarters(e: dict) -> int | None:
    dur = _duration_days(e)
    if dur is not None:
        if QUARTER_DURATION_MIN_DAYS <= dur <= QUARTER_DURATION_MAX_DAYS:
            return 1
        if dur <= 220:
            return 2
        if dur <= 320:
            return 3
        if ANNUAL_DURATION_MIN_DAYS <= dur <= ANNUAL_DURATION_MAX_DAYS:
            return 4
        return None

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


def _quarter_index_from_fp(e: dict) -> int | None:
    fp = str(e.get("fp") or "")
    if fp in QUARTER_FPS:
        return int(fp[1])
    return None


def _quarter_index_for_end(candidates: list[dict], fallback: int) -> int:
    fp_indexes: list[int] = [i for e in candidates if (i := _quarter_index_from_fp(e)) is not None]
    if fp_indexes:
        return max(set(fp_indexes), key=fp_indexes.count)

    inferred: list[int] = [i for e in candidates if (i := _infer_ytd_quarters(e)) is not None]
    if inferred:
        return max(set(inferred), key=inferred.count)

    return fallback


def _annual_calendar(entries: list[dict]) -> list[tuple[date, int]]:
    """
    Build a clean fiscal-year anchor calendar: one trusted fiscal-year-end date per FY.

    SEC companyfacts can include comparative prior-year columns in later 10-K filings,
    where deduping by period-end and "latest filed" can attach misleading FY metadata.
    Grouping by FY first avoids mapping old period-ends into a newer fiscal year.
    """
    annual_entries = _annual_fact_entries(entries)
    out: list[tuple[date, int]] = []
    for e in annual_entries:
        end = _parse_iso_date(e.get("end"))
        fy = _as_int(e.get("fy"))
        if end is not None and fy is not None:
            out.append((end, fy))
    return sorted(out, key=lambda x: x[0])


def _assign_fiscal_year(end: date, annual_calendar: list[tuple[date, int]]) -> int:
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

    nearest_end, nearest_fy = min(annual_calendar, key=lambda ac: abs((ac[0] - end).days))
    offset = round((end - nearest_end).days / 365.25)
    return nearest_fy + offset


def _pick_best_for_target(
    entries: list[dict],
    target_quarters: int,
    prefer_form: str,
    target_fiscal_year: int | None = None,
) -> dict | None:
    if not entries:
        return None

    target_days = 91 * max(1, target_quarters)

    def _score(e: dict) -> tuple[int, int, int, str, int]:
        form = str(e.get("form") or "")
        filed = str(e.get("filed") or "")
        dur = _duration_days(e)
        diff = abs((dur if dur is not None else target_days) - target_days)

        form_pref = 0
        if prefer_form == "quarterly" and form.startswith("10-Q"):
            form_pref = 1
        elif prefer_form == "annual" and form.startswith("10-K"):
            form_pref = 1
        return (
            _period_ownership_rank(e, target_fiscal_year),
            form_pref,
            1 if _is_amended_form(e) else 0,
            filed,
            -diff,
        )

    return max(entries, key=_score)


def _quarterly_candidate_groups(entries: list[dict]) -> dict[int, list[tuple[date, str, int]]]:
    filtered = [
        e
        for e in entries
        if _is_valid_fact_row(e)
        and str(e.get("fp") or "") in (QUARTER_FPS | {"FY"})
        and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
    ]
    if not filtered:
        return {}

    by_end: dict[str, list[dict]] = defaultdict(list)
    for e in filtered:
        end = str(e.get("end") or "")
        if end:
            by_end[end].append(e)

    annual_calendar = _annual_calendar(entries)
    ends_by_fy: dict[int, list[tuple[date, str, int]]] = defaultdict(list)
    fallback_by_fy: dict[int, int] = defaultdict(int)

    ordered_ends: list[tuple[date, str, list[dict]]] = []
    for end_str, candidates in by_end.items():
        end_date = _parse_iso_date(end_str)
        if end_date is None:
            continue
        ordered_ends.append((end_date, end_str, candidates))

    for end_date, end_str, candidates in sorted(ordered_ends, key=lambda x: x[0]):
        fy = _assign_fiscal_year(end_date, annual_calendar)
        fallback_by_fy[fy] += 1
        q_idx = _quarter_index_for_end(candidates, fallback_by_fy[fy])
        if q_idx in {1, 2, 3, 4}:
            ends_by_fy[fy].append((end_date, end_str, q_idx))

    return ends_by_fy


def _filtered_quarterly_rows_by_end(entries: list[dict]) -> dict[str, list[dict]]:
    by_end: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if not (
            _is_valid_fact_row(e)
            and str(e.get("fp") or "") in (QUARTER_FPS | {"FY"})
            and str(e.get("form") or "") in ALLOWED_QUARTERLY_FORMS
        ):
            continue
        end = str(e.get("end") or "")
        if end:
            by_end[end].append(e)
    return by_end


def _clone_quarter_row(source: dict, value: float, fy: int, q_idx: int, end_str: str) -> dict:
    row = dict(source)
    row["val"] = value
    row["fy"] = fy
    row["fp"] = f"Q{q_idx}"
    row["end"] = end_str
    return row


def _ytd_weight(e: dict | None, q_idx: int) -> float:
    if e is not None:
        dur = _duration_days(e)
        if dur is not None:
            return float(dur)
    return float(q_idx)


def _quarter_weight(e: dict | None) -> float:
    if e is not None:
        dur = _duration_days(e)
        if dur is not None:
            return float(dur)
    return 1.0


def _quarterly_direct_entries(entries: list[dict]) -> list[dict]:
    """Return only discrete quarter-duration facts, normalized and newest-first."""
    by_end = _filtered_quarterly_rows_by_end(entries)
    if not by_end:
        return []

    normalized: list[dict] = []
    groups = _quarterly_candidate_groups(entries)
    for fy, ends in groups.items():
        fy_ends = sorted(ends, key=lambda x: x[0])
        if len(fy_ends) > 4:
            fy_ends = fy_ends[-4:]

        for _, end_str, q_idx in fy_ends:
            candidates = by_end[end_str]
            direct_entry = _pick_best_for_target(
                [e for e in candidates if _is_direct_quarter_duration(e)],
                1,
                prefer_form="quarterly",
                target_fiscal_year=fy,
            )
            if direct_entry is None:
                continue
            value = _as_float(direct_entry.get("val"))
            if value is None:
                continue
            normalized.append(_clone_quarter_row(direct_entry, value, fy, q_idx, end_str))

    return _sort_newest(_keep_latest_by(normalized, lambda e: str(e.get("end") or "")))


def _quarterly_flow_entries(entries: list[dict]) -> list[dict]:
    """Normalize flow facts so returned values are discrete quarters, not YTD totals."""
    by_end = _filtered_quarterly_rows_by_end(entries)
    if not by_end:
        return []

    normalized: list[dict] = []
    groups = _quarterly_candidate_groups(entries)
    for fy, ends in groups.items():
        fy_ends = sorted(ends, key=lambda x: x[0])
        if len(fy_ends) > 4:
            fy_ends = fy_ends[-4:]

        ytd_prev: float | None = None
        for _, end_str, q_idx in fy_ends:
            candidates = by_end[end_str]
            prefer_form = "annual" if q_idx == 4 else "quarterly"

            direct_entry = _pick_best_for_target(
                [e for e in candidates if _is_direct_quarter_duration(e)],
                1,
                prefer_form="quarterly",
                target_fiscal_year=fy,
            )
            ytd_entry = _pick_best_for_target(
                [
                    e
                    for e in candidates
                    if _infer_ytd_quarters(e) == q_idx
                    or (
                        q_idx == 4
                        and str(e.get("fp") or "") == "FY"
                        and _is_full_year_duration(e)
                        and not _frame_is_quarterly(e)
                    )
                ],
                q_idx,
                prefer_form=prefer_form,
                target_fiscal_year=fy,
            )

            value: float | None = None
            source: dict | None = None

            if direct_entry is not None:
                value = _as_float(direct_entry.get("val"))
                source = direct_entry

            ytd_val = _as_float(ytd_entry.get("val")) if ytd_entry is not None else None
            if value is None and ytd_val is not None:
                if q_idx == 1:
                    value = ytd_val
                    source = ytd_entry
                elif ytd_prev is not None:
                    value = ytd_val - ytd_prev
                    source = ytd_entry

            if value is not None and source is not None:
                normalized.append(_clone_quarter_row(source, value, fy, q_idx, end_str))

            if ytd_val is not None:
                ytd_prev = ytd_val
            elif value is not None:
                if q_idx == 1 or ytd_prev is not None:
                    ytd_prev = value if ytd_prev is None else ytd_prev + value

    return _sort_newest(_keep_latest_by(normalized, lambda e: str(e.get("end") or "")))


def _quarterly_average_entries(entries: list[dict]) -> list[dict]:
    """
    Normalize weighted-average facts to discrete quarter averages.

    YTD average facts are converted with duration-weighted math:
    Qn_avg = (YTD_avg * YTD_weight - prior_YTD_avg * prior_weight) / quarter_weight.
    """
    by_end = _filtered_quarterly_rows_by_end(entries)
    if not by_end:
        return []

    normalized: list[dict] = []
    groups = _quarterly_candidate_groups(entries)
    for fy, ends in groups.items():
        fy_ends = sorted(ends, key=lambda x: x[0])
        if len(fy_ends) > 4:
            fy_ends = fy_ends[-4:]

        prev_ytd_avg: float | None = None
        prev_ytd_weight: float | None = None

        for _, end_str, q_idx in fy_ends:
            candidates = by_end[end_str]
            prefer_form = "annual" if q_idx == 4 else "quarterly"

            direct_entry = _pick_best_for_target(
                [e for e in candidates if _is_direct_quarter_duration(e)],
                1,
                prefer_form="quarterly",
                target_fiscal_year=fy,
            )
            ytd_entry = _pick_best_for_target(
                [
                    e
                    for e in candidates
                    if _infer_ytd_quarters(e) == q_idx
                    or (
                        q_idx == 4
                        and str(e.get("fp") or "") == "FY"
                        and _is_full_year_duration(e)
                        and not _frame_is_quarterly(e)
                    )
                ],
                q_idx,
                prefer_form=prefer_form,
                target_fiscal_year=fy,
            )

            value: float | None = None
            source: dict | None = None

            if direct_entry is not None:
                value = _as_float(direct_entry.get("val"))
                source = direct_entry

            ytd_avg = _as_float(ytd_entry.get("val")) if ytd_entry is not None else None
            ytd_weight = _ytd_weight(ytd_entry, q_idx) if ytd_entry is not None else None
            if value is None and ytd_avg is not None and ytd_weight is not None:
                if q_idx == 1:
                    value = ytd_avg
                    source = ytd_entry
                elif prev_ytd_avg is not None and prev_ytd_weight is not None:
                    quarter_weight = ytd_weight - prev_ytd_weight
                    if quarter_weight > 0:
                        value = (ytd_avg * ytd_weight - prev_ytd_avg * prev_ytd_weight) / quarter_weight
                        source = ytd_entry

            if value is not None and source is not None:
                normalized.append(_clone_quarter_row(source, value, fy, q_idx, end_str))

            if ytd_avg is not None and ytd_weight is not None:
                prev_ytd_avg = ytd_avg
                prev_ytd_weight = ytd_weight
            elif value is not None:
                q_weight = _quarter_weight(direct_entry)
                if q_idx == 1 or (prev_ytd_avg is not None and prev_ytd_weight is not None):
                    if prev_ytd_avg is None or prev_ytd_weight is None:
                        prev_ytd_avg = value
                        prev_ytd_weight = q_weight
                    else:
                        new_weight = prev_ytd_weight + q_weight
                        prev_ytd_avg = (prev_ytd_avg * prev_ytd_weight + value * q_weight) / new_weight
                        prev_ytd_weight = new_weight

    return _sort_newest(_keep_latest_by(normalized, lambda e: str(e.get("end") or "")))


# Backward-compatible alias for existing financials tests and callers that
# treat generic quarterly facts as flow concepts.
_quarterly_fact_entries = _quarterly_flow_entries
