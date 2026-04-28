#!/usr/bin/env python3
"""CLI for aluminum fundamental data and walk-forward backtests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from commodities.aluminum.backtest import (
    build_and_cache_features,
    fetch_and_cache_sources,
    plot_results,
    print_backtest_summary,
    run_aluminum_backtest,
)
from commodities.aluminum.config import (
    DEFAULT_FORECAST_THRESHOLD,
    DEFAULT_MIN_TRAIN_MONTHS,
    DEFAULT_MODEL_TYPE,
    DEFAULT_TRANSACTION_COST_BPS,
    LME_XML_DIR,
    AluminumBacktestConfig,
)


def _add_common_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", default=None, help="First forecast month, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Last forecast month, YYYY-MM-DD")
    parser.add_argument("--min-train-months", type=int, default=DEFAULT_MIN_TRAIN_MONTHS)
    parser.add_argument("--model", default=DEFAULT_MODEL_TYPE, choices=["zero", "ridge", "random_forest"])
    parser.add_argument("--forecast-threshold", type=float, default=DEFAULT_FORECAST_THRESHOLD)
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS)
    parser.add_argument("--refresh", action="store_true", help="Refresh downloaded/cacheable source data")
    parser.add_argument("--lme-xml-dir", default=str(LME_XML_DIR), help="Local licensed LME XML directory")


def _config(args: argparse.Namespace) -> AluminumBacktestConfig:
    return AluminumBacktestConfig(
        start_date=getattr(args, "start_date", None),
        end_date=getattr(args, "end_date", None),
        min_train_months=getattr(args, "min_train_months", DEFAULT_MIN_TRAIN_MONTHS),
        model_type=getattr(args, "model", DEFAULT_MODEL_TYPE),
        forecast_threshold=getattr(args, "forecast_threshold", DEFAULT_FORECAST_THRESHOLD),
        transaction_cost_bps=getattr(args, "transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS),
        refresh=getattr(args, "refresh", False),
        lme_xml_dir=Path(getattr(args, "lme_xml_dir", str(LME_XML_DIR))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aluminum fundamental data and backtest pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch/cache normalized source data")
    fetch_parser.add_argument("--refresh", action="store_true")
    fetch_parser.add_argument("--lme-xml-dir", default=str(LME_XML_DIR))

    build_parser = subparsers.add_parser("build-features", help="Build monthly aluminum features")
    build_parser.add_argument("--refresh", action="store_true")
    build_parser.add_argument("--lme-xml-dir", default=str(LME_XML_DIR))

    run_parser = subparsers.add_parser("run-backtest", help="Run walk-forward backtest")
    _add_common_backtest_args(run_parser)

    subparsers.add_parser("plot-results", help="Plot existing equity curve and drawdown outputs")

    all_parser = subparsers.add_parser("all", help="Fetch data, build features, run backtest, and plot")
    _add_common_backtest_args(all_parser)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")

    if args.command == "fetch-data":
        fetch_and_cache_sources(refresh=args.refresh, lme_xml_dir=args.lme_xml_dir)
        print("Fetched aluminum source data.")
    elif args.command == "build-features":
        features = build_and_cache_features(refresh=args.refresh, lme_xml_dir=args.lme_xml_dir)
        print(f"Built {len(features)} monthly feature rows.")
    elif args.command == "plot-results":
        plot_results()
        print("Plotted aluminum backtest results.")
    elif args.command in {"run-backtest", "all"}:
        result = run_aluminum_backtest(_config(args))
        print_backtest_summary(result["metrics"], result["factor_diagnostics"])
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
