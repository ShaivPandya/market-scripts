import os
from datetime import datetime, timedelta

from load_env import load_env

load_env()

from utils.retry import fred_get_series

SERIES_CONFIG = {
    "housing_starts": ("HOUST", "Housing Starts", "thousands", None),
    "housing_permits": ("PERMIT", "Building Permits", "thousands", None),
    "nahb_index": ("NAHBHMI", "NAHB Housing Market Index", "index", None),
    "existing_home_sales": ("EXHOSLUSM495S", "Existing Home Sales", "millions", None),
}


def get_data() -> dict:
    from fredapi import Fred

    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    start = datetime.now() - timedelta(days=365 * 15)

    result: dict = {
        "series": {},
        "latest": {},
        "timestamp": datetime.now().isoformat(),
    }

    for key, (fred_id, label, unit, _transform) in SERIES_CONFIG.items():
        try:
            s = fred_get_series(fred, fred_id, observation_start=start).dropna()
        except Exception as exc:
            result["series"][key] = {"dates": [], "values": [], "label": label, "unit": unit, "error": str(exc)}
            continue

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
