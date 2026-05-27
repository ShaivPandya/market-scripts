from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from api.async_job_runner import enqueue_registered_job, enqueue_response, poll_registered_job
from api.decision_state import analysis_metadata
from api.exceptions import DataFetchError
from api.portfolio_settings import get_portfolio_book_size
from api.serializers import serialize_dataframe, serialize_value
from ontology.runtime_read_service import OntologyRuntimeReadService
from portfolio.position_groups import (
    canonicalize_position_group_rows,
    group_key,
    normalize_group_conviction,
    normalize_group_name,
)

router = APIRouter()

HEDGE_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_-]{0,31}$")
BETA_HEDGE_MODE_TICKERS: dict[str, tuple[str, ...]] = {
    "spy": ("SPY",),
    "iwm": ("IWM",),
    "qqq": ("QQQ",),
    "spy_iwm": ("SPY", "IWM"),
    "spy_qqq": ("SPY", "QQQ"),
    "iwm_qqq": ("IWM", "QQQ"),
    "spy_iwm_qqq": ("SPY", "IWM", "QQQ"),
}


def _normalize_hedge_tickers_input(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if not ticker:
            raise ValueError("hedge_tickers cannot contain empty tickers.")
        if not HEDGE_TICKER_PATTERN.fullmatch(ticker):
            raise ValueError(f"Invalid hedge ticker '{value}'. Use a yfinance-compatible ticker symbol.")
        if ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)
    if not normalized:
        raise ValueError("hedge_tickers must contain at least one ticker.")
    return normalized


def _effective_hedge_tickers(req: SizerRequest) -> list[str]:
    if req.hedge_tickers is not None:
        return list(req.hedge_tickers)
    return list(BETA_HEDGE_MODE_TICKERS[req.beta_hedge_mode])


class SizerPosition(BaseModel):
    ticker: str = ""
    conviction: int = 3
    group_name: str | None = None
    group_conviction: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def _normalize_group(self) -> SizerPosition:
        self.group_name = normalize_group_name(self.group_name)
        if self.group_name:
            self.group_conviction = normalize_group_conviction(self.group_conviction)
            if self.group_conviction is None:
                raise ValueError(f"Group '{self.group_name}' requires a group conviction.")
        else:
            self.group_conviction = None
        return self


class SizerRequest(BaseModel):
    book: float | None = None
    target_leverage: float = 2.0
    beta_hedge_mode: Literal["spy", "iwm", "qqq", "spy_iwm", "spy_qqq", "iwm_qqq", "spy_iwm_qqq"] = "spy_iwm"
    hedge_tickers: list[str] | None = None
    positions: list[SizerPosition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_request(self) -> SizerRequest:
        if self.book is None:
            self.book = float(get_portfolio_book_size())
        self.hedge_tickers = _normalize_hedge_tickers_input(self.hedge_tickers)
        return self


def _effective_book(req: SizerRequest) -> float:
    return float(req.book) if req.book is not None else float(get_portfolio_book_size())


def _canonical_positions(req: SizerRequest) -> list[tuple[str, int, str, int]]:
    aggregated: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(req.positions):  # noqa: B007
        ticker = str(row.ticker).strip().upper()
        conviction = int(row.conviction)
        if not ticker:
            continue
        if conviction < 1 or conviction > 5:
            raise ValueError(f"Position '{ticker}' conviction must be 1-5, got {conviction}.")
        name = normalize_group_name(row.group_name)
        gkey = group_key(name)
        group_conviction = normalize_group_conviction(row.group_conviction) if gkey else None
        if gkey:
            if group_conviction is None:
                raise ValueError(f"Group '{name}' requires a group conviction.")
            group = groups.setdefault(gkey, {"name": name, "conviction": group_conviction})
            if group["conviction"] != group_conviction:
                raise ValueError(
                    f"Group '{group['name']}' has inconsistent group convictions "
                    f"({group['conviction']} and {group_conviction})."
                )
            name = group["name"]
        existing = aggregated.get(ticker)
        if existing is None or conviction > existing["conviction"]:
            aggregated[ticker] = {
                "ticker": ticker,
                "conviction": conviction,
                "group_name": name,
                "group_conviction": group_conviction,
            }
    if not aggregated:
        raise ValueError("No valid positions provided.")
    return [
        (
            row["ticker"],
            int(row["conviction"]),
            str(row["group_name"] or ""),
            int(row["group_conviction"] or 0),
        )
        for row in sorted(aggregated.values(), key=lambda x: str(x["ticker"]))
    ]


def _cache_key(req: SizerRequest) -> str:
    strategy_version = "v3_conviction_sizing_custom_equity_beta"
    canonical = _canonical_positions(req)
    token = (
        "|".join(
            f"{ticker}:{conviction}:group={group_name}:{group_conviction}"
            for ticker, conviction, group_name, group_conviction in canonical
        )
        or "none"
    )
    hedge_token = ",".join(_effective_hedge_tickers(req))
    return (
        f"portfolio_sizer:{strategy_version}:book={_effective_book(req):.4f}:"
        f"lev={float(req.target_leverage):.4f}:beta_hedge_mode={req.beta_hedge_mode}:"
        f"hedge_tickers={hedge_token}:positions={token}"
    )


def _compute_sizer_result(req: SizerRequest) -> dict[str, Any]:
    try:
        from portfolio.portfolio_optimizer.portfolio_sizer import get_data

        payload = canonicalize_position_group_rows([row.model_dump() for row in req.positions])
        data = get_data(
            positions=payload,
            book=_effective_book(req),
            target_leverage=float(req.target_leverage),
            beta_hedge_mode=req.beta_hedge_mode,
            hedge_tickers=req.hedge_tickers,
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(str(e)) from e

    if "error" in data and data["error"]:
        raise RuntimeError(str(data["error"]))

    import pandas as pd

    result: dict[str, Any] = {}
    for k, v in data.items():
        if k == "max_scaled" and isinstance(v, dict):
            inner: dict[str, Any] = {}
            for ik, iv in v.items():
                if isinstance(iv, pd.DataFrame):
                    inner[ik] = serialize_dataframe(iv.reset_index())
                else:
                    inner[ik] = serialize_value(iv)
            result[k] = inner
        elif isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    result.update(analysis_metadata(quality_state="ok"))
    return result


@router.post("/portfolio-sizer")
def run_portfolio_sizer(req: SizerRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("sizer", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/portfolio-sizer/async/{job_id}")


@router.post("/portfolio-sizer/async")
def start_portfolio_sizer(req: SizerRequest):
    try:
        key = _cache_key(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # noqa: B904

    row, _disposition = enqueue_registered_job("sizer", req.model_dump(), cache_key=key)
    return enqueue_response(row, "/api/portfolio-sizer/async/{job_id}")


@router.get("/portfolio-sizer/async/{job_id}")
def get_portfolio_sizer_job(job_id: str):
    try:
        return poll_registered_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")  # noqa: B904


@router.get("/portfolio-sizer/prefill")
def get_sizer_prefill():
    try:
        df = OntologyRuntimeReadService().positions_df()
        if "ticker" not in df.columns:
            raise ValueError("Ontology positions are missing required 'ticker' column.")

        tickers = df["ticker"].astype(str).str.strip().str.upper()
        directions = (
            df["direction"].astype(str).str.strip().str.lower()
            if "direction" in df.columns
            else pd.Series([""] * len(df))
        )
        convictions = (
            pd.to_numeric(df["conviction"], errors="coerce").fillna(3).astype(int).clip(1, 5)
            if "conviction" in df.columns
            else pd.Series([3] * len(df))
        )
        instrument_types = (
            df["instrument_type"].astype(str).str.strip().str.lower()
            if "instrument_type" in df.columns
            else pd.Series(["security"] * len(df))
        )
        assets = (
            df["asset"].astype(str).str.strip().str.lower()
            if "asset" in df.columns
            else pd.Series(["equity"] * len(df))
        )
        group_names = (
            df["group_name"].apply(normalize_group_name) if "group_name" in df.columns else pd.Series([None] * len(df))
        )
        group_convictions = (
            pd.to_numeric(df["group_conviction"], errors="coerce").astype("Int64")
            if "group_conviction" in df.columns
            else pd.Series([pd.NA] * len(df), dtype="Int64")
        )

        deduped_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ticker, direction, conviction, instrument_type, asset, group_name, group_conviction in zip(  # noqa: B905
            tickers.tolist(),
            directions.tolist(),
            convictions.tolist(),
            instrument_types.tolist(),
            assets.tolist(),
            group_names.tolist(),
            group_convictions.tolist(),
        ):
            if asset != "equity":
                continue
            if ticker and ticker not in seen:
                seen.add(ticker)
                group_conviction_out = None if pd.isna(group_conviction) else int(group_conviction)
                deduped_rows.append(
                    {
                        "ticker": ticker,
                        "conviction": conviction,
                        "direction": direction,
                        "instrument_type": instrument_type,
                        "group_name": group_name,
                        "group_conviction": group_conviction_out if group_name else None,
                    }
                )

        return {
            "positions": deduped_rows,
            "book_size": get_portfolio_book_size(),
            "source": "ontology",
            "count": len(deduped_rows),
        }
    except Exception as e:
        raise DataFetchError(source="portfolio_sizer", detail=str(e)) from e
