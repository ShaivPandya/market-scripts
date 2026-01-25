import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .data_fred import fetch_fred_series
from .data_imf import fetch_imf_datamapper_indicator
from .data_bis import fetch_bis_ws_eer_m, BisError
from .features import build_monthly_panel, compute_features, implied_spot_reference_points
from .models import fit_horizon_ols, predict_latest, bootstrap_forecast_distribution
from .report import (
    save_csv,
    save_json,
    plot_spot_vs_reference,
    plot_valuation_zscore,
    plot_forecast_distribution,
    summarize_distribution,
)

log = logging.getLogger(__name__)

FRED_SERIES = {
    "usdcad": "DEXCAUS",            # CAD per USD (daily)
    "cpi_us": "CPIAUCSL",           # US CPI (monthly)
    "cpi_ca": "CPALTT01CAM657N",    # Canada CPI (monthly)
    "wti": "DCOILWTICO",            # WTI (daily)
    "r_ca": "IR3TIB01CAM156N",      # Canada 3m (monthly)
    "r_us": "TB3MS",                # US 3m (monthly)
    "vix": "VIXCLS",                # VIX (daily)
}

CPI_CA_SERIES_CANDIDATES = [
    "CANCPIALLMINMEI",   # price level index (preferred, if available)
    "CPALTT01CAM657N",   # fallback (often percent change; handled in feature engineering)
]

def run_pipeline(
    start: str,
    outdir: Path,
    cache_dir: Path,
    refresh: bool,
    use_bis: bool,
    bootstrap_draws: int,
    horizons: list,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    # -------- Download FRED --------
    fred = {}
    for name, sid in FRED_SERIES.items():
        if name == "cpi_ca":
            last_err = None
            for candidate in CPI_CA_SERIES_CANDIDATES:
                try:
                    log.info(f"Downloading FRED {candidate} -> {name}")
                    fred[name] = fetch_fred_series(candidate, start=start, cache_dir=cache_dir, refresh=refresh)
                    break
                except Exception as e:
                    last_err = e
            if name not in fred:
                raise last_err
            continue

        log.info(f"Downloading FRED {sid} -> {name}")
        fred[name] = fetch_fred_series(sid, start=start, cache_dir=cache_dir, refresh=refresh)

    # -------- Download IMF (WEO/DataMapper) --------
    # Indicator list endpoint: https://www.imf.org/external/datamapper/api/v1/indicators
    log.info("Downloading IMF DataMapper BCA_NGDPD (current account % GDP) for CAN and USA")
    ca_can = fetch_imf_datamapper_indicator("BCA_NGDPD", "CAN", cache_dir=cache_dir, refresh=refresh)
    ca_usa = fetch_imf_datamapper_indicator("BCA_NGDPD", "USA", cache_dir=cache_dir, refresh=refresh)

    ca_diff = (ca_can - ca_usa).rename("ca_diff")
    ca_diff_m = ca_diff.resample("M").ffill()

    # -------- Optional BIS --------
    bis_reer = None
    if use_bis:
        try:
            # Common key pattern from BIS WS_EER_M examples: M.<type>.<basket>.<country>
            log.info("Downloading BIS WS_EER_M (optional) for Canada: key=M.R.B.CA")
            bis_reer = fetch_bis_ws_eer_m("M.R.B.CA", cache_dir=cache_dir, refresh=refresh).rename("bis_reer_can")
        except BisError as e:
            log.warning(f"BIS download failed (continuing without BIS): {e}")
        except Exception as e:
            log.warning(f"BIS download failed (continuing without BIS): {e}")

    # -------- Monthly panel --------
    series = {**fred, "ca_diff": ca_diff_m}
    if bis_reer is not None:
        series["bis_reer_can"] = bis_reer

    panel = build_monthly_panel(series)

    # -------- Features + reference points --------
    df = compute_features(panel)
    ref = implied_spot_reference_points(df)

    # -------- Fit models + forecast --------
    feature_cols = ["rer_z", "carry", "oil_mom12", "mom12", "ca_diff", "carry_to_vol", "vix"]
    results_by_h = {}
    latest_forecast = {}

    df_clean = df.replace([np.inf, -np.inf], np.nan)
    asof_df = df_clean.dropna(subset=["usdcad"] + feature_cols)
    if asof_df.empty:
        raise ValueError(
            "No rows have a complete feature set for forecasting. "
            f"Check inputs for missing/invalid values in: {feature_cols}"
        )
    asof_date = asof_df.index.max()
    spot_now = float(asof_df.loc[asof_date, "usdcad"])
    x_row = asof_df.loc[[asof_date], feature_cols]
    rer_z_now = float(asof_df.loc[asof_date, "rer_z"])

    for h in horizons:
        log.info(f"Fitting OLS for horizon {h} months")
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

        latest_forecast[str(h)] = {
            "spot_now": spot_now,
            "point_level": float(level_point),
            "level_q05_q50_q95": {
                "q05": float(level_dist.get("q05", float("nan"))),
                "q50": float(level_dist.get("q50", float("nan"))),
                "q95": float(level_dist.get("q95", float("nan"))),
            },
            "valuation_rer_z": rer_z_now,
        }

        plot_forecast_distribution(draws, point, spot_now, outdir / f"forecast_distribution_{h}m.png", horizon=h)

    # -------- Save artifacts --------
    save_csv(df, outdir / "usdcad_monthly_features.csv")
    save_csv(ref, outdir / "usdcad_reference_points.csv")
    plot_spot_vs_reference(ref.dropna(), outdir / "spot_vs_reference_points.png")
    plot_valuation_zscore(df.dropna(subset=["rer_z"]), outdir / "valuation_zscore.png")

    out_json = {
        "latest_date": str(asof_date.date()),
        "horizons_months": horizons,
        "feature_cols": feature_cols,
        "latest_forecast": latest_forecast,
        "models": results_by_h,
        "notes": {
            "rer_definition": "rer = log(USDCAD) + log(CPI_US) - log(CPI_CA)",
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
