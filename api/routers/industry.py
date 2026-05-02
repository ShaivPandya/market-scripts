from fastapi import APIRouter, HTTPException, Response

from api.cache import get_cached, long_cache, set_cached
from api.exceptions import DataFetchError
from api.serializers import serialize_response

router = APIRouter()


@router.get("/industry-monitor")
def get_industry_monitor(refresh: bool = False):
    key = f"industry_monitor:{refresh}"
    if not refresh:
        cached = get_cached(long_cache, key)
        if cached is not None:
            return cached
    try:
        from macro.industry.industry_monitor import get_data

        data = get_data(refresh=refresh)
        if isinstance(data, dict) and data.get("error"):
            raise DataFetchError(source="industry", detail=str(data.get("error")))
    except Exception as e:
        if isinstance(e, DataFetchError):
            raise
        raise DataFetchError(source="industry", detail=str(e)) from e

    result = serialize_response(data)
    set_cached(long_cache, key, result)
    return result


@router.get("/industry-monitor/transcripts/{ticker}/pdf")
def get_industry_transcript_pdf(ticker: str):
    try:
        from macro.industry.industry_monitor import load_pdf_for_ticker

        loaded = load_pdf_for_ticker(ticker)
    except Exception as e:
        raise DataFetchError(source="industry", detail=str(e)) from e

    if loaded is None:
        raise HTTPException(status_code=404, detail="Transcript PDF not found")

    filename, pdf_bytes = loaded
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "private, max-age=300",
        },
    )
