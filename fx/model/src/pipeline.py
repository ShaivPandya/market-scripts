"""FX model pipeline - config-driven for multiple currency pairs."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .currency_config import CurrencyPairConfig
from .data_bis import BisError, fetch_bis_ws_eer_m
from .data_estat import EStatError, fetch_estat_cpi
from .data_eurostat import fetch_euro_area_current_account_pct_gdp
from .data_fred import fetch_fred_series
from .data_imf import fetch_imf_datamapper_indicator
from .data_statcan import fetch_statcan_cpi
from .features import build_monthly_panel, compute_features, implied_spot_reference_points
from .models import bootstrap_forecast_distribution, fit_horizon_ols, predict_from_row
from .report import (
    build_driver_explanation,
    plot_forecast_distribution,
    plot_spot_vs_reference,
    plot_valuation_zscore,
    save_csv,
    save_json,
    summarize_distribution,
)

log = logging.getLogger(__name__)


def _fetch_with_candidates(
    candidates: list,
    name: str,
    start: str,
    cache_dir: Path,
    refresh: bool,
) -> pd.Series:
    """Try fetching from a list of candidates (FRED series IDs or source dicts), return first success."""
    last_err = None
    for candidate in candidates:
        if isinstance(candidate, dict):
            source = candidate.get("source", "fred")
            if source == "statcan":
                vector_id = candidate["vector_id"]
                try:
                    log.info(f"Downloading StatCan v{vector_id} -> {name}")
                    return fetch_statcan_cpi(vector_id, start=start, cache_dir=cache_dir, refresh=refresh)
                except Exception as e:
                    log.warning(f"StatCan v{vector_id} failed: {e}")
                    last_err = e
                    continue
            if source == "estat":
                stats_data_id = candidate.get("stats_data_id", "0003427113")
                try:
                    log.info(f"Downloading e-Stat {stats_data_id} -> {name}")
                    return fetch_estat_cpi(stats_data_id, start=start, cache_dir=cache_dir, refresh=refresh)
                except Exception as e:
                    log.warning(f"e-Stat {stats_data_id} failed: {e}")
                    last_err = e
                    continue
            sid = candidate.get("id", "")
        else:
            sid = candidate
        try:
            log.info(f"Downloading FRED {sid} -> {name}")
            return fetch_fred_series(sid, start=start, cache_dir=cache_dir, refresh=refresh)
        except Exception as e:
            log.warning(f"FRED {sid} failed: {e}")
            last_err = e
    raise RuntimeError(f"All candidates failed for {name}: {candidates}") from last_err


def run_pipeline(
    config: CurrencyPairConfig,
    start: str,
    outdir: Path,
    cache_dir: Path,
    refresh: bool,
    use_bis: bool,
    bootstrap_draws: int,
    horizons: list,
) -> dict:
    """Run the FX model pipeline for a given currency pair config."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    pair = config.pair_name
    log.info(f"Running pipeline for {pair}")

    # -------- Download FRED --------
    fred = {}

    # Spot rate
    log.info(f"Downloading FRED {config.fred_spot_id} -> spot")
    spot_raw = fetch_fred_series(config.fred_spot_id, start=start, cache_dir=cache_dir, refresh=refresh)
    if config.fred_spot_invert:
        spot_raw = 1.0 / spot_raw
    fred["spot"] = spot_raw

    # CPI for base currency
    fred["cpi_base"] = _fetch_with_candidates(config.cpi_base_ids, "cpi_base", start, cache_dir, refresh)

    # CPI for quote currency
    fred["cpi_quote"] = _fetch_with_candidates(config.cpi_quote_ids, "cpi_quote", start, cache_dir, refresh)

    # Interest rates
    log.info(f"Downloading FRED {config.rate_base_id} -> r_base")
    fred["r_base"] = fetch_fred_series(config.rate_base_id, start=start, cache_dir=cache_dir, refresh=refresh)

    log.info(f"Downloading FRED {config.rate_quote_id} -> r_quote")
    fred["r_quote"] = fetch_fred_series(config.rate_quote_id, start=start, cache_dir=cache_dir, refresh=refresh)

    # Oil and VIX (common to all pairs)
    log.info(f"Downloading FRED {config.fred_oil_id} -> wti")
    fred["wti"] = fetch_fred_series(config.fred_oil_id, start=start, cache_dir=cache_dir, refresh=refresh)

    log.info(f"Downloading FRED {config.fred_vix_id} -> vix")
    fred["vix"] = fetch_fred_series(config.fred_vix_id, start=start, cache_dir=cache_dir, refresh=refresh)

    # -------- Download CA%GDP differential (quote - base) --------
    # Primary source: IMF DataMapper (BCA_NGDPD).
    # EUR base has an additional fallback: Eurostat (quarterly CA balance % GDP).
    imf_ca_available = False
    ca_diff_available = False
    ca_quote = None
    ca_base = None

    if config.imf_iso3_quote:
        try:
            log.info(f"Downloading IMF BCA_NGDPD for {config.imf_iso3_quote}")
            ca_quote = fetch_imf_datamapper_indicator(
                "BCA_NGDPD", config.imf_iso3_quote, cache_dir=cache_dir, refresh=refresh
            )
        except Exception as e:
            log.warning(f"IMF CA%GDP download failed for {config.imf_iso3_quote}: {e}")

    if config.imf_iso3_base:
        try:
            log.info(f"Downloading IMF BCA_NGDPD for {config.imf_iso3_base}")
            ca_base = fetch_imf_datamapper_indicator(
                "BCA_NGDPD", config.imf_iso3_base, cache_dir=cache_dir, refresh=refresh
            )
        except Exception as e:
            log.warning(f"IMF CA%GDP download failed for {config.imf_iso3_base}: {e}")
            if config.imf_iso3_base.upper() == "EMU" or config.base_ccy.upper() == "EUR":
                try:
                    log.info("Attempting Eurostat fallback for Euro area CA%GDP")
                    ca_base = fetch_euro_area_current_account_pct_gdp(start=start, cache_dir=cache_dir, refresh=refresh)
                except Exception as e2:
                    log.warning(f"Eurostat CA%GDP fallback failed: {e2}")

    ca_diff_m = None
    if ca_quote is not None and ca_base is not None:
        ca_quote_m = ca_quote.resample("ME").ffill()
        ca_base_m = ca_base.resample("ME").ffill()
        ca_diff_m = (ca_quote_m - ca_base_m).rename("ca_diff")
        ca_diff_available = True
        imf_ca_available = (
            isinstance(getattr(ca_quote, "name", None), str)
            and isinstance(getattr(ca_base, "name", None), str)
            and str(ca_quote.name).startswith("BCA_NGDPD_")
            and str(ca_base.name).startswith("BCA_NGDPD_")
        )

    # -------- Optional BIS --------
    bis_reer = None
    if use_bis and config.bis_key:
        try:
            log.info(f"Downloading BIS WS_EER_M for {config.bis_key}")
            bis_reer = fetch_bis_ws_eer_m(config.bis_key, cache_dir=cache_dir, refresh=refresh).rename("bis_reer")
        except BisError as e:
            log.warning(f"BIS download failed (continuing without BIS): {e}")
        except Exception as e:
            log.warning(f"BIS download failed (continuing without BIS): {e}")

    # -------- Build monthly panel --------
    series = dict(fred)
    if ca_diff_available and ca_diff_m is not None:
        series["ca_diff"] = ca_diff_m
    if bis_reer is not None:
        series["bis_reer"] = bis_reer

    panel = build_monthly_panel(series)

    # -------- Features + reference points --------
    df = compute_features(panel)
    ref = implied_spot_reference_points(df)

    # -------- Fit models + forecast --------
    feature_cols = ["rer_z", "carry", "oil_mom12", "mom12", "ca_diff", "carry_to_vol", "vix"]
    if (not ca_diff_available) or ("ca_diff" not in df.columns):
        feature_cols = [c for c in feature_cols if c != "ca_diff"]
    results_by_h = {}
    latest_forecast = {}

    FEATURE_LAG_MONTHS = 1

    df_clean = df.replace([np.inf, -np.inf], np.nan)

    # Align inference with training: fit_horizon_ols uses X = features.shift(1),
    # so for a forecast "as of" spot date t we must use feature snapshot t-1.
    X_pred = df_clean[feature_cols].shift(FEATURE_LAG_MONTHS)
    pred_panel = pd.concat([df_clean["spot"], X_pred], axis=1)

    asof_df = pred_panel.dropna(subset=["spot"] + feature_cols)
    if asof_df.empty:
        raise ValueError(
            f"No rows have a complete lagged feature set for forecasting {pair}. "
            f"Check inputs for missing/invalid values in (lag={FEATURE_LAG_MONTHS}m): {feature_cols}"
        )
    asof_date = asof_df.index.max()
    spot_now = float(asof_df.loc[asof_date, "spot"])
    x_row = asof_df.loc[[asof_date], feature_cols]
    rer_z_now = float(x_row["rer_z"].iloc[0])

    idx = df_clean.index
    asof_loc = int(idx.get_indexer([asof_date])[0])
    feature_asof_date = idx[asof_loc - FEATURE_LAG_MONTHS] if asof_loc >= FEATURE_LAG_MONTHS else pd.NaT

    for h in horizons:
        log.info(f"Fitting OLS for {pair} horizon {h} months")
        res, _ = fit_horizon_ols(df_clean, horizon=h, feature_cols=feature_cols, target_col="logS")

        point = predict_from_row(res, x_row)
        draws = bootstrap_forecast_distribution(res, x_row, draws=bootstrap_draws, seed=42)
        dist = summarize_distribution(draws)

        level_point = spot_now * float(np.exp(point))
        level_dist = {k: spot_now * float(np.exp(v)) for k, v in dist.items() if k.startswith("q") or k == "mean"}

        results_by_h[h] = {
            "params": {k: float(v) for k, v in res.params.to_dict().items()},
            "nobs": int(res.nobs),
            "r2": float(res.rsquared),
            "point_log_return": float(point),
            "point_level": float(level_point),
            "dist_log_return": dist,
            "dist_level": {k: float(v) for k, v in level_dist.items()},
        }

        # Driver explanation
        feature_values = {col: float(x_row[col].iloc[0]) for col in feature_cols}
        driver_explanation = build_driver_explanation(
            params={k: float(v) for k, v in res.params.to_dict().items()},
            feature_values=feature_values,
            pair_name=pair,
            base_ccy=config.base_ccy,
            quote_ccy=config.quote_ccy,
            horizon=h,
            spot_now=spot_now,
            point_level=float(level_point),
            r2=float(res.rsquared),
            nobs=int(res.nobs),
        )
        results_by_h[h]["driver_explanation"] = driver_explanation

        latest_forecast[str(h)] = {
            "spot_now": spot_now,
            "point_level": float(level_point),
            "level_q05_q50_q95": {
                "q05": float(level_dist.get("q05", float("nan"))),
                "q50": float(level_dist.get("q50", float("nan"))),
                "q95": float(level_dist.get("q95", float("nan"))),
            },
            "valuation_rer_z": rer_z_now,
            "driver_explanation": driver_explanation,
        }

        plot_forecast_distribution(
            draws,
            point,
            spot_now,
            outdir / f"forecast_distribution_{h}m.png",
            horizon=h,
            pair_name=pair,
            spot_label=config.spot_label,
        )

    # -------- Save artifacts --------
    pair_lower = pair.lower()
    save_csv(df, outdir / f"{pair_lower}_monthly_features.csv")
    save_csv(ref, outdir / f"{pair_lower}_reference_points.csv")
    plot_spot_vs_reference(
        ref.dropna(),
        outdir / "spot_vs_reference_points.png",
        pair_name=pair,
        spot_label=config.spot_label,
    )
    plot_valuation_zscore(
        df.dropna(subset=["rer_z"]),
        outdir / "valuation_zscore.png",
        pair_name=pair,
    )

    out_json = {
        "pair": pair,
        "latest_date": str(asof_date.date()),
        "feature_asof_date": (None if pd.isna(feature_asof_date) else str(pd.Timestamp(feature_asof_date).date())),
        "feature_lag_months": int(FEATURE_LAG_MONTHS),
        "horizons_months": horizons,
        "feature_cols": feature_cols,
        "imf_ca_available": imf_ca_available,
        "ca_diff_available": ca_diff_available,
        "latest_forecast": latest_forecast,
        "models": results_by_h,
        "notes": {
            "rer_definition": f"rer = log({pair}) + log(CPI_{config.base_ccy}) - log(CPI_{config.quote_ccy})",
            "reference_points": "implied spot if rer reverted to rolling median/p25/p75",
            "bootstrap_draws": bootstrap_draws,
            "feature_lag_months": int(FEATURE_LAG_MONTHS),
        },
    }
    save_json(out_json, outdir / "forecast_latest.json")

    return {
        "panel": df,
        "reference_points": ref,
        "latest_forecast": latest_forecast,
        "models": results_by_h,
        "latest_date": str(asof_date.date()),
        "feature_asof_date": (None if pd.isna(feature_asof_date) else str(pd.Timestamp(feature_asof_date).date())),
        "feature_lag_months": int(FEATURE_LAG_MONTHS),
        "imf_ca_available": imf_ca_available,
        "ca_diff_available": ca_diff_available,
    }
