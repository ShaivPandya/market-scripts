import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from api.cache import delete_cached, get_or_set_cached, short_cache
from api.exceptions import ConfigurationError, DataFetchError, ValidationError
from api.request_limits import read_upload_file_bytes
from api.serializers import serialize_response
from api.state_storage import exists_text, read_bytes, read_text, write_bytes, write_text
from llm_utils import MODEL_LOW, api_key_env, call_llm_text, has_llm_api_key
from paths import PROJECT_ROOT

router = APIRouter()
REQUIRED_CURRENCY_PERIODS = ["1-mo", "3-mo", "6-mo", "1-yr"]
ECONOMIC_GROWTH_CACHE_KEY = "economic_growth"

CRB_LOCAL_PATH = PROJECT_ROOT / "data_cache" / "economic_growth" / "crb.xlsx"
CRB_METADATA_LOCAL_PATH = PROJECT_ROOT / "data_cache" / "economic_growth" / "crb.json"
CRB_GCS_KEY = "live/economic_growth/crb.xlsx"
CRB_METADATA_GCS_KEY = "live/economic_growth/crb.json"
MAX_CRB_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def _normalize_currency_payload(payload: dict) -> dict:
    periods = payload.get("currency_periods")
    normalized_periods = [p for p in periods if isinstance(p, str)] if isinstance(periods, list) else []
    for period in REQUIRED_CURRENCY_PERIODS:
        if period not in normalized_periods:
            normalized_periods.append(period)
    payload["currency_periods"] = normalized_periods

    currencies = payload.get("currencies")
    if isinstance(currencies, dict):
        for returns in currencies.values():
            if isinstance(returns, dict):
                for period in normalized_periods:
                    returns.setdefault(period, None)
    return payload


def _is_excel_upload(file: UploadFile) -> bool:
    filename = Path(file.filename or "").name
    return Path(filename).suffix.lower() in _EXCEL_EXTENSIONS


def _crb_content_type(filename: str) -> str:
    return (
        "application/vnd.ms-excel"
        if filename.lower().endswith(".xls")
        else ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )


def _read_crb_metadata() -> dict:
    if not exists_text(CRB_METADATA_LOCAL_PATH, CRB_METADATA_GCS_KEY):
        return {}
    try:
        raw = read_text(CRB_METADATA_LOCAL_PATH, CRB_METADATA_GCS_KEY, encoding="utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _load_managed_crb() -> tuple[bytes | None, dict]:
    if not exists_text(CRB_LOCAL_PATH, CRB_GCS_KEY):
        return None, {}
    return read_bytes(CRB_LOCAL_PATH, CRB_GCS_KEY), _read_crb_metadata()


def _crb_metadata_from_upload(payload: bytes, filename: str) -> dict:
    from macro.economic_growth.economic_growth import read_crb_from_xls

    df = read_crb_from_xls(payload, filename=filename)
    if df is None or df.empty:
        raise ValidationError("Uploaded CRB workbook does not contain valid date/value rows.")

    latest = df.iloc[-1]
    return {
        "filename": filename,
        "uploaded_at": datetime.now(UTC).isoformat(),
        "rows": int(len(df)),
        "latest_date": latest["date"].date().isoformat(),
        "latest_value": float(latest["value"]),
        "size_bytes": len(payload),
    }


def _write_managed_crb(payload: bytes, metadata: dict) -> None:
    filename = str(metadata.get("filename") or "crb.xlsx")
    string_metadata = {key: str(value) for key, value in metadata.items() if value is not None}
    write_bytes(
        CRB_LOCAL_PATH,
        CRB_GCS_KEY,
        payload,
        content_type=_crb_content_type(filename),
        metadata=string_metadata,
    )
    write_text(
        CRB_METADATA_LOCAL_PATH,
        CRB_METADATA_GCS_KEY,
        json.dumps(metadata, sort_keys=True),
        content_type="application/json; charset=utf-8",
    )


@router.get("/economic-growth")
def get_economic_growth():
    key = ECONOMIC_GROWTH_CACHE_KEY

    def loader():
        try:
            from macro.economic_growth.economic_growth import get_data

            crb_bytes, crb_metadata = _load_managed_crb()
            if crb_bytes is not None:
                data = get_data(
                    crb_bytes=crb_bytes,
                    crb_filename=crb_metadata.get("filename")
                    if isinstance(crb_metadata.get("filename"), str)
                    else None,
                    crb_uploaded_at=(
                        crb_metadata.get("uploaded_at") if isinstance(crb_metadata.get("uploaded_at"), str) else None
                    ),
                )
            else:
                data = get_data()
        except Exception as e:
            raise DataFetchError(source="economic_growth", detail=str(e)) from e
        return _normalize_currency_payload(serialize_response(data))

    return _normalize_currency_payload(get_or_set_cached(short_cache, key, loader))


@router.post("/economic-growth/crb-file")
async def upload_economic_growth_crb_file(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI parameter declaration
):
    if not _is_excel_upload(file):
        raise ValidationError("File must be an Excel workbook (.xlsx or .xls).")

    payload = await read_upload_file_bytes(file, limit_bytes=MAX_CRB_UPLOAD_SIZE_BYTES, limit_label="10 MiB")
    if not payload:
        raise ValidationError("Uploaded file is empty.")

    filename = Path(file.filename or "crb.xlsx").name
    metadata = _crb_metadata_from_upload(payload, filename)
    _write_managed_crb(payload, metadata)
    delete_cached(short_cache, ECONOMIC_GROWTH_CACHE_KEY)
    return serialize_response({"status": "ok", "crb": metadata})


class EconomicGrowthAnalyzeRequest(BaseModel):
    commodities: dict
    equities: dict
    currencies: dict
    equity_periods: list[str]
    currency_periods: list[str]


def _format_table(data: dict, periods: list[str]) -> str:
    lines = []
    header = "  ".join(f"{p:>8}" for p in periods)
    lines.append(f"{'Asset':<28}  {header}")
    lines.append("-" * (30 + 10 * len(periods)))
    for name, returns in data.items():
        vals = "  ".join(f"{returns.get(p):>+8.1f}" if returns.get(p) is not None else f"{'N/A':>8}" for p in periods)
        lines.append(f"{name:<28}  {vals}")
    return "\n".join(lines)


@router.post("/economic-growth/analyze")
def analyze_economic_growth(req: EconomicGrowthAnalyzeRequest):
    if not has_llm_api_key():
        raise ConfigurationError(api_key_env())

    commodities_table = _format_table(req.commodities, req.equity_periods)
    equities_table = _format_table(req.equities, req.equity_periods)
    currencies_table = _format_table(req.currencies, req.currency_periods)

    prompt = f"""You are an experienced macro strategist. Analyze the following market performance data and provide a concise but insightful overview of what it indicates about the current global economic growth environment.

The data shows percentage returns over 1-month (30 days), 3-month (91 days), 6-month (182 days), and 1-year (365 days) periods. US equities are benchmarked against S&P 500, Europe Banks against STOXX 600. Outperformance = bullish growth signal, underperformance = bearish.

COMMODITIES (returns %):
{commodities_table}

Key context:
- Copper ("Dr. Copper"): Highly sensitive to global economic activity due to widespread use in construction, electrical equipment, and manufacturing. Rising prices signal expansion, falling prices suggest contraction.
- CRB Industrial Spot Index: A broad index of non-traded industrial commodities (metals, textiles, agricultural inputs). Less influenced by investor speculation than futures-based indices, making it a purer measure of real industrial demand.
- GS Commodity Index (GSG): Broad commodity exposure including energy, metals, and agriculture. Identifies inflationary pressures and global demand trends.

EQUITIES vs BENCHMARK (returns %):
{equities_table}

Key context:
- Russell 2000 (IWM) & S&P 600 (IJR): Small-cap stocks with less diversified revenue, more domestic economic dependence, and less pricing power. Outperformance vs S&P 500 signals risk-on sentiment and economic optimism.
- DJ Transport (IYT): Direct beneficiary of goods movement. Dow Theory holds that transports should confirm trends in industrials — divergence signals economic weakness ahead.
- KBW Banks (KBWB) & Europe Banks (EXV1.DE): Highly cyclical financials whose profitability depends on loan demand, net interest margins, and credit quality. Strong performance signals confidence in growth and credit conditions.
- US Retail (XRT): Consumer discretionary spending indicator sensitive to household confidence and income growth.
- US Staples (XLP): Defensive sector that typically underperforms during expansions and outperforms during slowdowns. Relative strength signals defensive positioning and economic concern.
- US Utilities (XLU): Another defensive sector; outperformance suggests investors are seeking safety and yield over growth.
- MSCI Korea (EWY): Export-dependent and cyclical, sensitive to global manufacturing (semiconductors, electronics, autos), China's economic health, and global trade volumes. Used by macro investors as a proxy for global economic optimism.
- STOXX 600 (^STOXX): European equity benchmark for gauging European economic health and investor sentiment toward the region.

CURRENCY PAIRS (returns %):
{currencies_table}

Key context:
- AUD/JPY & CAD/JPY: Classic risk-on/risk-off indicators. JPY is a safe-haven currency; AUD and CAD are commodity currencies. Rising pairs suggest risk appetite and commodity demand (economic growth). Falling pairs suggest risk aversion and economic uncertainty. Both correlate with global risk sentiment and commodity cycles.

BULLISH GROWTH SIGNALS: Small-caps outperforming S&P 500, banks outperforming benchmarks, copper and CRB rising, transports strong, Korea outperforming, staples/utilities underperforming, AUD/JPY and CAD/JPY rising.
BEARISH/DEFENSIVE SIGNALS: Small-caps underperforming, banks underperforming, commodities falling, staples/utilities outperforming S&P 500, Korea underperforming, currency pairs falling (yen strength).

Write 2-3 flowing paragraphs of plain text (no bullet points, no markdown, no headers). Be concise. Cover:
1. What commodity moves and equity breadth signals indicate about industrial demand, growth depth, and risk appetite
2. What European/EM signals and currency pairs imply about global growth synchronization and risk sentiment
3. An overall conclusion about where we are in the growth cycle and what to watch

Be specific about the numbers. Write for a professional investor audience."""

    try:
        analysis, _citations, _resp = call_llm_text(
            prompt=prompt,
            model=MODEL_LOW,
            api_key=None,
            max_tokens=2048,
        )
        if not analysis:
            raise ValueError("LLM returned empty response")
    except Exception as exc:
        raise DataFetchError(source="ai_analysis", detail=str(exc)) from exc

    return {"analysis": analysis}
