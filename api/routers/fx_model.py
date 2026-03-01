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
        data = run_pipeline(
            config=config,
            cache_dir=str(cache_dir),
            outdir=str(outdir),
            bootstrap_draws=req.bootstrap,
            skip_bis=req.skip_bis,
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
