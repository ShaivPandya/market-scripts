import os

from dotenv import load_dotenv

load_dotenv()

from api.routers.weekly_report import get_weekly_report

print("Running weekly report generation...")
try:
    res = get_weekly_report()
    print("Success:", res.keys())
except Exception:
    import traceback

    traceback.print_exc()
