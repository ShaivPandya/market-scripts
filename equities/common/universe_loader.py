"""Shared universe/ticker loading utilities."""
from pathlib import Path
from typing import List

import pandas as pd

UNIVERSES_DIR = Path(__file__).parent.parent / "universes"


def clean_ticker(tk: str) -> str:
    """Normalize ticker to Yahoo Finance format.

    Preserves dots for international exchange suffixes (e.g., METSO.HE).
    Only converts dots to dashes for US share classes (e.g., BRK.B -> BRK-B).
    """
    tk = tk.strip().upper()
    # Common international exchange suffixes that use dots
    intl_suffixes = (".HE", ".L", ".TO", ".AX", ".PA", ".DE", ".MI", ".AS", ".SW", ".MC", ".SI", ".HK", ".T", ".NS", ".BO")
    if any(tk.endswith(suffix) for suffix in intl_suffixes):
        return tk
    return tk.replace(".", "-")


def list_universes() -> List[str]:
    """List available universe files in the universes/ folder."""
    if not UNIVERSES_DIR.exists():
        return []
    return sorted([
        f.stem for f in UNIVERSES_DIR.iterdir()
        if f.suffix.lower() in (".csv", ".txt")
    ])


def resolve_universe_path(path_or_name: str) -> Path:
    """Resolve a universe name or path to an actual file path."""
    p = Path(path_or_name)

    # If it's already a valid path, use it
    if p.exists():
        return p

    # Try as a name in universes/ folder (with .csv then .txt)
    for ext in (".csv", ".txt"):
        candidate = UNIVERSES_DIR / f"{path_or_name}{ext}"
        if candidate.exists():
            return candidate

    # Try the exact name in universes/
    candidate = UNIVERSES_DIR / path_or_name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Universe '{path_or_name}' not found. "
        f"Available: {', '.join(list_universes()) or '(none)'}"
    )


def load_universe(path_or_name: str) -> List[str]:
    """
    Load tickers from a file path or universe name.

    Args:
        path_or_name: Either a file path or a universe name (e.g., "consumer_discretionary")

    Returns:
        List of normalized ticker symbols
    """
    path = resolve_universe_path(path_or_name)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}
        if "ticker" in cols_lower:
            tickers = df[cols_lower["ticker"]].astype(str).tolist()
        else:
            tickers = df.iloc[:, 0].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f
                      if line.strip() and not line.strip().startswith("#")]

    # Normalize and deduplicate (preserve order)
    return list(dict.fromkeys(clean_ticker(t) for t in tickers if t.strip()))


def get_sp500_universe() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read()

    tables = pd.read_html(html)
    return sorted({clean_ticker(x) for x in tables[0]["Symbol"].astype(str)})
