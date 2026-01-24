#!/usr/bin/env python3
"""
Run all market technicals analysis scripts.

Usage:
  python3 market_technicals.py
"""

from market_breadth import main as run_market_breadth
from top50_breadth import main as run_top50_breadth
from price_volume_signals import main as run_price_volume_signals
from vix_term_structure import main as run_vix_term_structure

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
except ImportError:
    Console = None

CONSOLE = Console() if Console else None


def print_header() -> None:
    if CONSOLE:
        title = Text("Market Technicals", style="bold cyan")
        subtitle = Text("Breadth | Top 50 | Price/Volume | VIX Term", style="dim")
        body = Text.assemble(title, "\n", subtitle)
        CONSOLE.print(Panel.fit(body, box=box.ASCII, padding=(1, 4), style="cyan"))
        return
    print("=" * 60)
    print("MARKET TECHNICALS")
    print("=" * 60)


def print_section(title: str) -> None:
    if CONSOLE:
        CONSOLE.print()
        CONSOLE.rule(f"[bold]{title}[/bold]", characters="-", style="cyan")
        return
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def main():
    print_header()
    print_section("Market Breadth Analysis")
    run_market_breadth()

    print_section("Top 50 Breadth Analysis")
    run_top50_breadth()

    print_section("Price/Volume Signals")
    run_price_volume_signals()

    print_section("VIX Term Structure")
    run_vix_term_structure()


if __name__ == "__main__":
    main()
