"""FX model pipeline - config-driven for multiple currency pairs."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .currency_config import CurrencyPairConfig
from .data_fred import fetch_fred_series
from .data_yfinance import fetch_yfinance_series
from .data_imf import fetch_imf_datamapper_indicator
from .data_bis import fetch_bis_ws_eer_m, BisError
from .data_worldbank import fetch_worldbank_cpi
from .features import build_monthly_panel, compute_features, implied_spot_reference_points
from .models import fit_horizon_ols, predict_latest, bootstrap_forecast_distribution
from .report import (
    save_csv,
    save_json,
    plot_spot_vs_reference,
    plot_valuation_zscore,
    plot_forecast_distribution,
    summarize_distribution,
    build_driver_explanation,
)

log = logging.getLogger(__name__)


def _fetch_with_candidates(
    candidates: list,
    name: str,
    start: str,
    cache_dir: Path,
    refresh: bool,
) -> pd.Series:
    """Try fetching from a list of FRED series IDs, return first success."""
    last_err = None
    for sid in candidates:
        try:
            log.info(f"Downloading FRED {sid} -> {name}")
            return fetch_fred_series(sid, start=start, cache_dir=cache_dir, refresh=refresh)
        except Exception as e:
            log.warning(f"FRED {sid} failed: {e}")
            last_err = e
    raise RuntimeError(f"All candidates failed for {name}: {candidates}") from last_err


def _extend_cpi_if_stale(
    fred_cpi: pd.Series,
    iso3: str,
    name: str,
    cache_dir: Path,
    refresh: bool,
    stale_months: int = 12,
) -> pd.Series:
    """If FRED CPI ends more than stale_months ago, extend with World Bank data."""
    last_date = fred_cpi.dropna().index.max()
    months_old = (pd.Timestamp.now() - last_date).days / 30
    if months_old <= stale_months:
        return fred_cpi

    log.warning(
        f"FRED {name} is stale (last: {last_date.date()}, {months_old:.0f}m old). "
        f"Extending with World Bank CPI for {iso3}."
    )
    try:
        wb = fetch_worldbank_cpi(iso3, cache_dir=cache_dir, refresh=refresh)
    except Exception as e:
        log.warning(f"World Bank CPI fallback failed for {iso3}: {e}")
        return fred_cpi

    # Resample both to month-end so dates align
    fred_me = fred_cpi.dropna().resample("ME").last().dropna()
    wb_me = wb.dropna().resample("ME").last().dropna()

    overlap = fred_me.index.intersection(wb_me.index)
    if overlap.empty:
        log.warning("No overlap between FRED and World Bank CPI; cannot extend.")
        return fred_cpi

    anchor_date = overlap.max()
    scale = float(fred_me.loc[anchor_date]) / float(wb_me.loc[anchor_date])
    wb_scaled = wb_me * scale

    # Append World Bank data after FRED ends
    fred_last_me = fred_me.index.max()
    extension = wb_scaled[wb_scaled.index > fred_last_me]
    if extension.empty:
        log.warning("World Bank CPI has no data beyond FRED end date.")
        return fred_cpi

    combined = pd.concat([fred_me, extension])
    log.info(f"Extended {name} from {fred_last_me.date()} to {combined.index.max().date()} via World Bank")
    return combined


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

    # Spot rate (from Yahoo Finance)
    log.info(f"Downloading yFinance {config.yf_spot_ticker} -> spot")
    spot_raw = fetch_yfinance_series(config.yf_spot_ticker, start=start, cache_dir=cache_dir, refresh=refresh)
    if config.fred_spot_invert:
        spot_raw = 1.0 / spot_raw
    fred["spot"] = spot_raw

    # CPI for base currency
    fred["cpi_base"] = _fetch_with_candidates(
        config.cpi_base_ids, "cpi_base", start, cache_dir, refresh
    )
    fred["cpi_base"] = _extend_cpi_if_stale(
        fred["cpi_base"], config.imf_iso3_base, "cpi_base", cache_dir, refresh
    )

    # CPI for quote currency
    fred["cpi_quote"] = _fetch_with_candidates(
        config.cpi_quote_ids, "cpi_quote", start, cache_dir, refresh
    )
    fred["cpi_quote"] = _extend_cpi_if_stale(
        fred["cpi_quote"], config.imf_iso3_quote, "cpi_quote", cache_dir, refresh
    )

    # Interest rates
    log.info(f"Downloading FRED {config.rate_base_id} -> r_base")
    fred["r_base"] = fetch_fred_series(config.rate_base_id, start=start, cache_dir=cache_dir, refresh=refresh)

    log.info(f"Downloading FRED {config.rate_quote_id} -> r_quote")
    fred["r_quote"] = fetch_fred_series(config.rate_quote_id, start=start, cache_dir=cache_dir, refresh=refresh)

    # Oil and VIX (from Yahoo Finance)
    log.info(f"Downloading yFinance {config.yf_oil_ticker} -> wti")
    fred["wti"] = fetch_yfinance_series(config.yf_oil_ticker, start=start, cache_dir=cache_dir, refresh=refresh)

    log.info(f"Downloading yFinance {config.yf_vix_ticker} -> vix")
    fred["vix"] = fetch_yfinance_series(config.yf_vix_ticker, start=start, cache_dir=cache_dir, refresh=refresh)

    # -------- Download IMF (current account % GDP) --------
    log.info(f"Downloading IMF BCA_NGDPD for {config.imf_iso3_quote} and {config.imf_iso3_base}")
    ca_quote = fetch_imf_datamapper_indicator("BCA_NGDPD", config.imf_iso3_quote, cache_dir=cache_dir, refresh=refresh)
    ca_base = fetch_imf_datamapper_indicator("BCA_NGDPD", config.imf_iso3_base, cache_dir=cache_dir, refresh=refresh)

    # CA differential: quote - base (positive = quote has CA surplus vs base)
    ca_diff = (ca_quote - ca_base).rename("ca_diff")
    ca_diff_m = ca_diff.resample("ME").ffill()

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
    series = {**fred, "ca_diff": ca_diff_m}
    if bis_reer is not None:
        series["bis_reer"] = bis_reer

    panel = build_monthly_panel(series)

    # -------- Features + reference points --------
    df = compute_features(panel)
    ref = implied_spot_reference_points(df)

    # -------- Fit models + forecast --------
    feature_cols = ["rer_z", "carry", "oil_mom12", "mom12", "ca_diff", "carry_to_vol", "vix"]
    results_by_h = {}
    latest_forecast = {}

    df_clean = df.replace([np.inf, -np.inf], np.nan)
    asof_df = df_clean.dropna(subset=["spot"] + feature_cols)
    if asof_df.empty:
        raise ValueError(
            f"No rows have a complete feature set for forecasting {pair}. "
            f"Check inputs for missing/invalid values in: {feature_cols}"
        )
    asof_date = asof_df.index.max()
    spot_now = float(asof_df.loc[asof_date, "spot"])
    x_row = asof_df.loc[[asof_date], feature_cols]
    rer_z_now = float(asof_df.loc[asof_date, "rer_z"])

    for h in horizons:
        log.info(f"Fitting OLS for {pair} horizon {h} months")
        res, _ = fit_horizon_ols(df_clean, horizon=h, feature_cols=feature_cols, target_col="logS")

        point = predict_latest(df_clean, res, feature_cols)
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
            draws, point, spot_now,
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
        "horizons_months": horizons,
        "feature_cols": feature_cols,
        "latest_forecast": latest_forecast,
        "models": results_by_h,
        "notes": {
            "rer_definition": f"rer = log({pair}) + log(CPI_{config.base_ccy}) - log(CPI_{config.quote_ccy})",
            "reference_points": "implied spot if rer reverted to rolling median/p25/p75",
            "bootstrap_draws": bootstrap_draws,
        },
    }
    save_json(out_json, outdir / "forecast_latest.json")

    return {
        "panel": df,
        "reference_points": ref,
        "latest_forecast": latest_forecast,
        "models": results_by_h,
    }
