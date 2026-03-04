import os
from fastapi import APIRouter, HTTPException
from api.cache import long_cache, get_cached, set_cached

router = APIRouter()

@router.get("/weekly-report")
def get_weekly_report():
    key = "weekly_report_generated"
    cached = get_cached(long_cache, key)
    if cached is not None:
        return cached

    # 1. Fetch all required data
    try:
        from index_dashboard import get_index_data
        indices = get_index_data("This Week")
    except Exception as e:
        indices = {"error": str(e)}

    try:
        from fx_dashboard import get_fx_data
        fx = get_fx_data("This Week")
    except Exception as e:
        fx = {"error": str(e)}

    try:
        import sys
        # Commodities isn't easily exposed without sys.path hacks that main.py does,
        # but the router should have access if it's imported properly. 
        from commodities_dashboard import get_data as get_commodity_data
        commodities = get_commodity_data("This Week")
    except Exception as e:
        commodities = {"error": str(e)}

    try:
        from market_breadth import get_data as get_breadth_data
        breadth = get_breadth_data(period="1y")
    except Exception as e:
        breadth = {"error": str(e)}

    try:
        from top50_breadth import get_data as get_top50_data
        top50 = get_top50_data()
    except Exception as e:
        top50 = {"error": str(e)}

    try:
        from vix_term_structure import get_data as get_vix_data
        vix = get_vix_data()
    except Exception as e:
        vix = {"error": str(e)}

    try:
        from sector_metrics import get_data as get_sector_data
        sector = get_sector_data()
        
        # We need to process sector_metrics as it returns a DataFrame for weights_df
        weights_df = sector.get("weights_df")
        if weights_df is not None:
            # We just want top-level summary for the prompt
            import pandas as pd
            if isinstance(weights_df, pd.DataFrame):
                sector["weights_summary"] = weights_df.to_dict(orient="records")
                del sector["weights_df"]
                
    except Exception as e:
        sector = {"error": str(e)}

    try:
        from positioning import fetch_multiple_instruments, DEFAULT_DOMAIN, DATASETS
        # Fetching basic summary for positioning
        pos = fetch_multiple_instruments(
            domain=DEFAULT_DOMAIN,
            dataset_id=DATASETS.get("tff_futures_only", "tff_futures_only"),
            app_token=os.environ.get("SODA_APP_TOKEN"),
            instruments=["SP500", "NASDAQ", "US10Y", "EUR", "GOLD", "OIL"],
        )
    except Exception as e:
        pos = {"error": str(e)}

    try:
        from technical_analysis import get_ratio_data
        silver_gold = get_ratio_data("SI=F", "GC=F", "This Week")
        sp_eq = get_ratio_data("^GSPC", "RSP", "This Week")
    except Exception as e:
        silver_gold = {"error": str(e)}
        sp_eq = {"error": str(e)}

    # 2. Extract specific rules for Breadth and VIX to include in prompt
    rules_text = """
STRICT FORMATTING RULES (Apply these to the data provided below):

MARKET BREADTH THRESHOLDS:
- 200-day MA: Flag if > 80% or < 15%
- 20-day MA: Flag if > 80% or < 20%
- 20-day Highs: Flag if > 50%
- 20-day Lows: Flag if > 50% (Capitulation signal)
- 52-week Highs: Flag if > 15%
- 52-week Lows: Flag if > 15%
- 24-week Highs: Flag if > 20%
- 24-week Lows: Flag if > 20%

TOP 50 S&P 500 BREADTH:
- Simply state the % below 50-DMA, % with >=3 distribution days (last 20), and % that broke prior 20-day low in last 5 days.

VIX TERM STRUCTURE:
- Signal is 'Complacency' if 3M/1M Ratio >= 1.25
- Signal is 'Fear' if Ratio < 1.0
- Otherwise 'Neutral'
"""

    data_context = f"""
==== RAW WEEKLY DATA ====

INDICES (This Week):
{indices}

FX (This Week):
{fx}

COMMODITIES (This Week):
{commodities}

MARKET BREADTH:
{breadth}

TOP 50 BREADTH:
{top50}

VIX TERM STRUCTURE:
{vix}

SECTOR METRICS:
{sector}

POSITIONING:
{pos}

RATIOS (Silver/Gold, S&P500/RSP):
Silver/Gold: {silver_gold}
SP500/RSP: {sp_eq}
=======================
"""

    prompt = f"""You are a quantitative market analyst compiling a weekly catch-up report.
Your goal is to summarize the moves of the past week into a clean report to catch up the user on what happened in the markets. 
FLAG anything that stands out, but strictly AVOID commentary. Do not explain *why* something happened, just note *that* it happened.

Use the explicit rules provided below to flag technicals.
For other dashboards (Indices, FX, Commodities, Sectors, Positioning, Ratios), use your best judgment as an LLM to identify and highlight significant outliers, major percentage moves, or extremes.

{rules_text}

{data_context}

Output the report in clean Markdown format. Group it into logical sections (e.g., Dashboards, Technicals & Breadth, Sectors & Positioning, Key Ratios).
Remember: No commentary, no editorializing. Just the facts and explicitly flagged threshold breaches.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        report_md = (resp.output_text or "").strip()
        if not report_md:
            raise ValueError("OpenAI returned empty response")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {exc}")

    result = {"report": report_md}
    # Cache for 1 hour (long_cache) to prevent spamming the LLM
    set_cached(long_cache, key, result)
    
    return result
