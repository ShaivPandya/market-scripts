import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline

def main():
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="USDCAD 1–2Y FX model using FRED + IMF (+ optional BIS).")
    ap.add_argument("--start", default="1970-01-01", help="Start date for downloads (YYYY-MM-DD).")
    ap.add_argument("--outdir", default="outputs", help="Output directory (relative paths resolve from this file).")
    ap.add_argument("--cache", default="data_cache", help="Cache directory (relative paths resolve from this file).")
    ap.add_argument("--refresh", action="store_true", help="Force refresh (ignore cached files).")
    ap.add_argument("--no-bis", action="store_true", help="Skip BIS downloads.")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws for forecast distribution.")
    ap.add_argument("--horizons", default="12,24", help="Comma-separated horizons in months, e.g. 12,24.")
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    outdir = Path(args.outdir).expanduser()
    cache = Path(args.cache).expanduser()
    if not outdir.is_absolute():
        outdir = base_dir / outdir
    if not cache.is_absolute():
        cache = base_dir / cache
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    results = run_pipeline(
        start=args.start,
        outdir=outdir,
        cache_dir=cache,
        refresh=args.refresh,
        use_bis=not args.no_bis,
        bootstrap_draws=args.bootstrap,
        horizons=horizons,
    )

    print("\nLatest forecast summary:")
    print(json.dumps(results["latest_forecast"], indent=2))

if __name__ == "__main__":
    main()
