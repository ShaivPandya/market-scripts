from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.serializers import serialize_value

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent


class FXModelRequest(BaseModel):
    pair: str
    bootstrap: int = 1000
    skip_bis: bool = False
    horizons: str = "12,24"


def _to_compact_response(data: dict, pair: str, bootstrap: int, skip_bis: bool) -> dict:
    latest_forecast = data.get("latest_forecast", {}) or {}
    models = data.get("models", {}) or {}

    forecast_rows = []
    ci_series = []
    driver_breakdown = []

    for horizon_key, forecast in latest_forecast.items():
        if not isinstance(forecast, dict):
            continue

        try:
            horizon = int(horizon_key)
        except (TypeError, ValueError):
            continue

        spot_now = forecast.get("spot_now")
        point_level = forecast.get("point_level")
        dist = forecast.get("level_q05_q50_q95", {}) or {}
        model = models.get(horizon) or models.get(str(horizon)) or {}
        driver_explanation = forecast.get("driver_explanation") or model.get("driver_explanation") or {}

        expected_move_pct = None
        if isinstance(spot_now, (int, float)) and isinstance(point_level, (int, float)) and spot_now != 0:
            expected_move_pct = ((point_level / spot_now) - 1.0) * 100.0

        row = {
            "horizon_months": horizon,
            "spot_now": spot_now,
            "point_level": point_level,
            "expected_move_pct": expected_move_pct,
            "q05": dist.get("q05"),
            "q50": dist.get("q50"),
            "q95": dist.get("q95"),
            "valuation_rer_z": forecast.get("valuation_rer_z"),
            "r2": model.get("r2"),
            "nobs": model.get("nobs"),
        }
        forecast_rows.append(row)
        ci_series.append(
            {
                "horizon": horizon,
                "p05": dist.get("q05"),
                "p50": dist.get("q50"),
                "p95": dist.get("q95"),
                "value": point_level,
            }
        )

        raw_drivers = driver_explanation.get("drivers", [])
        drivers = []
        if isinstance(raw_drivers, list):
            for d in raw_drivers:
                if not isinstance(d, dict):
                    continue
                drivers.append(
                    {
                        "name": d.get("name"),
                        "label": d.get("label"),
                        "coefficient": d.get("coefficient"),
                        "value": d.get("value"),
                        "contribution": d.get("contribution"),
                        "description": d.get("description"),
                    }
                )

        drivers.sort(
            key=lambda d: abs(d.get("contribution", 0))
            if isinstance(d.get("contribution"), (int, float))
            else 0,
            reverse=True,
        )
        driver_breakdown.append(
            {
                "horizon_months": horizon,
                "conclusion": driver_explanation.get("conclusion"),
                "drivers": drivers,
            }
        )

    forecast_rows.sort(key=lambda r: r["horizon_months"])
    ci_series.sort(key=lambda r: r["horizon"])
    driver_breakdown.sort(key=lambda r: r["horizon_months"])

    return {
        "pair": pair,
        "latest_date": data.get("latest_date"),
        "feature_asof_date": data.get("feature_asof_date"),
        "feature_lag_months": data.get("feature_lag_months"),
        "bootstrap_draws": bootstrap,
        "skip_bis": skip_bis,
        "imf_ca_available": data.get("imf_ca_available"),
        "ca_diff_available": data.get("ca_diff_available"),
        "forecast": forecast_rows,
        "ci_series": ci_series,
        "driver_breakdown": driver_breakdown,
    }


def _missing_dependency_error(e: ModuleNotFoundError) -> HTTPException:
    missing = e.name or "unknown"
    return HTTPException(
        status_code=500,
        detail=(
            f"Missing backend dependency '{missing}'. "
            "Install dependencies with `pip install -r requirements.txt` and restart the API server."
        ),
    )


@router.post("/fx-model")
def run_fx_model(req: FXModelRequest):
    try:
        horizons = [int(x.strip()) for x in req.horizons.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid horizons. Use comma-separated integers, e.g. `12,24`.")

    if not horizons:
        raise HTTPException(status_code=422, detail="At least one horizon is required, e.g. `12,24`.")

    try:
        cache_dir = PROJECT_ROOT / "fx" / "model" / "data_cache"
        outdir = PROJECT_ROOT / "fx" / "model" / "outputs" / req.pair.lower()

        from src.currency_config import get_config
        from src.pipeline import run_pipeline

        config = get_config(req.pair)
        outdir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        data = run_pipeline(
            config=config,
            start="1990-01-01",
            cache_dir=cache_dir,
            outdir=outdir,
            refresh=False,
            use_bis=not req.skip_bis,
            bootstrap_draws=req.bootstrap,
            horizons=horizons,
        )
    except ModuleNotFoundError as e:
        raise _missing_dependency_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    compact = _to_compact_response(
        data=data,
        pair=req.pair,
        bootstrap=req.bootstrap,
        skip_bis=req.skip_bis,
    )
    return serialize_value(compact)


@router.get("/fx-model/pairs")
def list_pairs():
    try:
        from src.currency_config import list_pairs as _list_pairs
        return {"pairs": _list_pairs()}
    except ModuleNotFoundError as e:
        raise _missing_dependency_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
