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


@router.post("/fx-model")
def run_fx_model(req: FXModelRequest):
    try:
        horizons = [int(x.strip()) for x in req.horizons.split(",") if x.strip()]
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return serialize_value(data)


@router.get("/fx-model/pairs")
def list_pairs():
    try:
        from src.currency_config import list_pairs as _list_pairs
        return {"pairs": _list_pairs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
