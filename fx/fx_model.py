#!/usr/bin/env python3
"""FX macro model CLI - supports multiple currency pairs.

Usage:
    python fx_model.py --pair USDCAD
    python fx_model.py --pair GBPUSD --no-bis
    python fx_model.py --pair USDJPY --bootstrap 1000
"""
import argparse
import json
from pathlib import Path

from src.currency_config import get_config, list_pairs
from src.pipeline import run_pipeline


def main():
    available_pairs = list_pairs()

    ap = argparse.ArgumentParser(
        description="FX macro model using FRED + IMF (+ optional BIS)."
    )
    ap.add_argument(
        "--pair",
        default="USDCAD",
        choices=available_pairs,
        help=f"Currency pair to model. Available: {', '.join(available_pairs)}",
    )
    ap.add_argument(
        "--start",
        default="1970-01-01",
        help="Start date for downloads (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--outdir",
        default="outputs",
        help="Base output directory (will create pair-specific subdirectory).",
    )
    ap.add_argument(
        "--cache",
        default="data_cache",
        help="Cache directory.",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh (ignore cached files).",
    )
    ap.add_argument(
        "--no-bis",
        action="store_true",
        help="Skip BIS downloads.",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap draws for forecast distribution.",
    )
    ap.add_argument(
        "--horizons",
        default="12,24",
        help="Comma-separated horizons in months, e.g. 12,24.",
    )
    args = ap.parse_args()

    # Parse horizons
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    # Get currency config
    config = get_config(args.pair)

    # Set up directories (pair-specific output subdirectory)
    pair_lower = args.pair.lower()
    outdir = Path(args.outdir) / pair_lower
    cache = Path(args.cache)
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    results = run_pipeline(
        config=config,
        start=args.start,
        outdir=outdir,
        cache_dir=cache,
        refresh=args.refresh,
        use_bis=not args.no_bis,
        bootstrap_draws=args.bootstrap,
        horizons=horizons,
    )

    print(f"\nLatest forecast summary for {args.pair}:")
    print(json.dumps(results["latest_forecast"], indent=2))
    print(f"\nOutputs saved to: {outdir}/")


if __name__ == "__main__":
    main()
