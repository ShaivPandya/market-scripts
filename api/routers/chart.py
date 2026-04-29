from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from api.cache import stamp_fresh
from api.exceptions import DataFetchError
from api.serializers import serialize_dataframe, serialize_value

router = APIRouter()


def _csv_download_filename(symbol: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in symbol.strip().upper()).strip("_")
    return f"{clean or 'ticker'}_price_history.csv"


class ChartRequest(BaseModel):
    ticker: str
    lookback: str = "2Y"


class RatioChartRequest(BaseModel):
    symbol_a: str
    symbol_b: str
    start_date: str | None = None
    end_date: str | None = None
    method: str = "price_ratio"


@router.post("/chart")
def run_chart(req: ChartRequest):
    ticker = req.ticker.strip().upper()
    try:
        from portfolio.technical_analysis.technical_analysis import get_data

        data = get_data(ticker, lookback=req.lookback)
    except Exception as e:
        raise DataFetchError(source="chart", detail=str(e)) from e

    if "error" in data and data["error"]:
        raise HTTPException(status_code=400, detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return stamp_fresh(result)


@router.get("/chart/price-history/{ticker}")
def download_price_history(ticker: str):
    normalized = ticker.strip().upper()
    if not normalized:
        raise HTTPException(status_code=400, detail="Ticker is required.")

    try:
        from portfolio.technical_analysis.technical_analysis import fetch_full_price_history

        df = fetch_full_price_history(normalized)
    except Exception as e:
        raise DataFetchError(source="chart", detail=str(e)) from e

    csv_text = df.to_csv(index=False)
    filename = _csv_download_filename(normalized)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/chart/ratio")
def run_chart_ratio(req: RatioChartRequest):
    symbol_a = req.symbol_a.strip().upper()
    symbol_b = req.symbol_b.strip().upper()
    if not symbol_a or not symbol_b:
        raise HTTPException(status_code=400, detail="Both symbol_a and symbol_b are required.")

    try:
        from portfolio.technical_analysis.technical_analysis import get_ratio_data

        data = get_ratio_data(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            start_date=req.start_date,
            end_date=req.end_date,
            method=req.method,
        )
    except Exception as e:
        raise DataFetchError(source="chart", detail=str(e)) from e

    if "error" in data and data["error"]:
        raise HTTPException(status_code=400, detail=data["error"])

    import pandas as pd

    result = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            result[k] = serialize_dataframe(v.reset_index())
        else:
            result[k] = serialize_value(v)
    return stamp_fresh(result)
