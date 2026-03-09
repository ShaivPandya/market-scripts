import os
from datetime import datetime, timedelta

from load_env import load_env

load_env()

from utils.retry import fred_get_series

SERIES_CONFIG = {
    "initial_claims": ("ICSA", "Initial Jobless Claims", "thousands", None),
    "continuing_claims": ("CCSA", "Continuing Claims", "thousands", None),
    "median_weeks_unemployed": ("UEMPMED", "Median Weeks Unemployed", "weeks", None),
    "weekly_hours": ("AWHAETP", "Avg Weekly Hours Worked", "hours", None),
    "wage_growth": ("AHETPI", "Wage Growth (YoY)", "%", "yoy12"),
    "job_openings": ("JTSJOL", "Job Openings (JOLTS)", "thousands", None),
}

# FRED claims series are returned as counts, but we display them in "thousands".
_CLAIMS_SERIES_KEYS = {"initial_claims", "continuing_claims"}


def get_data() -> dict:
    from fredapi import Fred

    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    start = datetime.now() - timedelta(days=365 * 15)

    result: dict = {
        "series": {},
        "latest": {},
        "timestamp": datetime.now().isoformat(),
    }

    for key, (fred_id, label, unit, transform) in SERIES_CONFIG.items():
        try:
            s = fred_get_series(fred, fred_id, observation_start=start).dropna()
        except Exception as exc:
            result["series"][key] = {"dates": [], "values": [], "label": label, "unit": unit, "error": str(exc)}
            continue

        if transform == "yoy12":
            s = s.pct_change(12) * 100
            s = s.dropna()
        elif key in _CLAIMS_SERIES_KEYS:
            s = s / 1000.0

        dates = [d.strftime("%Y-%m-%d") for d in s.index]
        values = [round(float(v), 3) for v in s.values]

        result["series"][key] = {
            "dates": dates,
            "values": values,
            "label": label,
            "unit": unit,
        }

        if dates:
            prev = values[-2] if len(values) > 1 else None
            result["latest"][key] = {
                "value": values[-1],
                "date": dates[-1],
                "change": round(values[-1] - prev, 3) if prev is not None else None,
            }

    return result


if __name__ == "__main__":
    import json

    data = get_data()
    print("Latest values:")
    for key, latest in data["latest"].items():
        label = data["series"][key]["label"]
        unit = data["series"][key]["unit"]
        print(f"  {label}: {latest['value']} {unit} (as of {latest['date']}, chg: {latest['change']})")
