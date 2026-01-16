#!/usr/bin/env python3
"""
Run all market technicals analysis scripts.

Usage:
  python3 market_technicals.py
"""

from market_breadth import main as run_market_breadth
from top50_breadth import main as run_top50_breadth
from price_volume_signals import main as run_price_volume_signals


def main():
    print("=" * 60)
    print("MARKET BREADTH ANALYSIS")
    print("=" * 60)
    run_market_breadth()

    print("\n" + "=" * 60)
    print("TOP 50 BREADTH ANALYSIS")
    print("=" * 60)
    run_top50_breadth()

    print("\n" + "=" * 60)
    print("PRICE/VOLUME SIGNALS")
    print("=" * 60)
    run_price_volume_signals()


if __name__ == "__main__":
    main()
