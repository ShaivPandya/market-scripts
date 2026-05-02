"""Helpers for serving signal-aggregator snapshots with request-time slicing."""

from __future__ import annotations

import copy
from typing import Any

from api.signal_aggregator import _build_episodes
from api.snapshot_keys import SNAPSHOT_SIGNAL_AGGREGATOR
from api.snapshot_store import get_snapshot_response


def _history_score(row: dict[str, Any]) -> float | None:
    try:
        value = row.get("score")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _slice_history(data: dict[str, Any], lookback_weeks: int) -> None:
    history = data.get("history")
    if not isinstance(history, dict):
        return
    series = history.get("series")
    if not isinstance(series, list):
        return
    lookback = max(26, min(int(lookback_weeks), 520))
    rows = [row for row in series if isinstance(row, dict)][-lookback:]
    history["lookback_weeks"] = lookback
    history["series"] = rows
    history["episodes"] = _build_episodes(rows)

    regime = data.get("regime")
    if not isinstance(regime, dict):
        return
    try:
        regime_score = regime.get("score")
        if regime_score is None:
            return
        composite = float(regime_score)
    except (TypeError, ValueError):
        return
    scores = [score for score in (_history_score(row) for row in rows) if score is not None]
    if scores:
        regime["history_percentile"] = round(
            (sum(1 for score in scores if score <= composite) / len(scores)) * 100.0, 2
        )


def get_signal_aggregator_snapshot_response(
    *,
    lookback_weeks: int,
    include_raw_modules: bool,
) -> dict[str, Any] | None:
    snapshot = get_snapshot_response(SNAPSHOT_SIGNAL_AGGREGATOR)
    if snapshot is None:
        return None
    data = copy.deepcopy(snapshot)
    if not include_raw_modules:
        data.pop("raw_modules", None)
    _slice_history(data, lookback_weeks)
    return data
