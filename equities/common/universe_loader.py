"""Shared universe/ticker loading utilities."""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

UNIVERSES_DIR = Path(__file__).parent.parent / "universes"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ETF_HOLDINGS_CACHE_DIR = _REPO_ROOT / "data_cache" / "etf_holdings"
_ETF_HOLDINGS_CACHE_TTL_SECS = 24 * 60 * 60  # 24 hours
_SP500_CACHE_PATH = _REPO_ROOT / "data_cache" / "universes" / "sp500.txt"
_SP500_CACHE_TTL_SECS = 24 * 60 * 60  # 24 hours

# SPDR Select Sector ETFs (State Street / SSGA)
_SPDR_SECTOR_ETFS = {
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
}

# Some callers pass lower-case shortcuts (xlk, xly, ...)
_SECTOR_SHORTCUTS = {t.lower(): t for t in _SPDR_SECTOR_ETFS}


def clean_ticker(tk: str) -> str:
    """Normalize ticker to Yahoo Finance format.

    Preserves dots for international exchange suffixes (e.g., METSO.HE).
    Only converts dots to dashes for US share classes (e.g., BRK.B -> BRK-B).
    """
    tk = tk.strip().upper()
    # Common international exchange suffixes that use dots
    intl_suffixes = (
        ".HE", ".L", ".TO", ".AX", ".PA", ".DE", ".MI", ".AS", ".SW", ".MC",
        ".SI", ".HK", ".T", ".NS", ".BO",
        ".KS", ".KQ",  # South Korea (KSE / KOSDAQ)
        ".TW", ".TWO",  # Taiwan (TWSE / Taipei Exchange)
    )
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

    try:
        if _SP500_CACHE_PATH.exists():
            age = time.time() - _SP500_CACHE_PATH.stat().st_mtime
            if age <= _SP500_CACHE_TTL_SECS:
                tickers = [
                    clean_ticker(line)
                    for line in _SP500_CACHE_PATH.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                tickers = [t for t in tickers if t]
                if tickers:
                    return sorted(set(tickers))
    except Exception:
        pass

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read()

    tables = pd.read_html(html)
    tickers = sorted({clean_ticker(x) for x in tables[0]["Symbol"].astype(str)})

    try:
        _SP500_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SP500_CACHE_PATH.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    except Exception:
        pass

    return tickers


def _read_cached_etf_holdings(etf_ticker: str) -> Optional[List[str]]:
    etf = clean_ticker(etf_ticker)
    path = _ETF_HOLDINGS_CACHE_DIR / f"{etf}.txt"
    try:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > _ETF_HOLDINGS_CACHE_TTL_SECS:
            return None
        tickers = [
            clean_ticker(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        tickers = [t for t in tickers if t]
        return list(dict.fromkeys(tickers))
    except Exception:
        return None


def _write_cached_etf_holdings(etf_ticker: str, tickers: List[str]) -> None:
    etf = clean_ticker(etf_ticker)
    try:
        _ETF_HOLDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _ETF_HOLDINGS_CACHE_DIR / f"{etf}.txt"
        normalized = [clean_ticker(t) for t in tickers]
        normalized = [t for t in normalized if t]
        normalized = list(dict.fromkeys(normalized))
        path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    except Exception:
        # Cache is best-effort; ignore failures.
        return


def _extract_tickers_from_table(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    for candidate in (
        "ticker",
        "tickers",
        "symbol",
        "symbols",
        "ticker symbol",
        "ticker_symbol",
        "holding ticker",
        "holding_ticker",
        "security ticker",
        "security_ticker",
    ):
        if candidate in cols:
            raw = df[cols[candidate]].astype(str).tolist()
            tickers = [clean_ticker(x) for x in raw]
            tickers = [t for t in tickers if t]
            return list(dict.fromkeys(tickers))

    # Fallback: some exports put tickers in the first column.
    raw = df.iloc[:, 0].astype(str).tolist()
    tickers = [clean_ticker(x) for x in raw]
    tickers = [t for t in tickers if t]
    # If the first column looks like names (has lots of spaces), try to find a better candidate column.
    if tickers and sum(1 for x in raw[:200] if isinstance(x, str) and " " in x.strip()) / max(1, len(raw[:200])) > 0.2:
        best = _best_ticker_column(df)
        if best is not None:
            cand = df[best].astype(str).tolist()
            tickers = [clean_ticker(x) for x in cand]
            tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,15}$")


def _ticker_score(values: List[str]) -> float:
    if not values:
        return 0.0
    sample = values[:250]
    ok = 0
    total = 0
    for v in sample:
        s = str(v).strip().upper()
        if not s:
            continue
        total += 1
        # Reject obvious non-tickers quickly
        if " " in s or s in {"SYMBOL", "TICKER", "NAME"} or s.endswith(":"):
            continue
        s = clean_ticker(s)
        if _TICKER_RE.match(s):
            ok += 1
    if total == 0:
        return 0.0
    return ok / total


def _best_ticker_column(df: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_score = 0.0
    for col in df.columns[:25]:
        try:
            values = df[col].astype(str).tolist()
        except Exception:
            continue
        score = _ticker_score(values)
        if score > best_score:
            best_score = score
            best_col = str(col)
    if best_col is None or best_score < 0.5:
        return None
    return best_col


def fetch_etf_holdings_ssga(etf_ticker: str) -> List[str]:
    """
    Fetch ETF holdings from State Street Global Advisors (SSGA) export files.

    This is intended primarily for SPDR Select Sector ETFs (XLB..XLY).
    Returns an empty list on failure.
    """
    import requests

    etf = clean_ticker(etf_ticker)
    lower = etf.lower()

    # SSGA commonly hosts daily holdings exports at predictable paths.
    # We try a few URL templates to be resilient to site section changes.
    url_templates = [
        "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{lower}.xlsx",
        "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{lower}.csv",
        "https://www.ssga.com/us/en/institutional/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{lower}.xlsx",
        "https://www.ssga.com/us/en/institutional/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{lower}.csv",
    ]
    urls = [t.format(lower=lower) for t in url_templates]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200 or not resp.content:
                continue

            content = resp.content
            if url.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                tickers = _extract_tickers_from_table(df)
            else:
                # Excel exports sometimes contain a cover sheet + a holdings sheet;
                # read the first sheet that yields a plausible ticker column.
                tickers = []
                xls = pd.ExcelFile(io.BytesIO(content))
                for sheet in xls.sheet_names[:5]:
                    df = xls.parse(sheet)
                    tickers = _extract_tickers_from_table(df)
                    if len(tickers) >= 20:
                        break

            tickers = [t for t in tickers if t and t != etf]
            if tickers:
                return tickers
        except Exception:
            continue

    return []


def fetch_etf_holdings_yfinance(etf_ticker: str) -> List[str]:
    """
    Fetch ETF holdings from yfinance.

    Note: Yahoo's fund holdings endpoint is often limited (commonly top 10 holdings).
    Returns an empty list on failure.
    """
    try:
        import yfinance as yf
    except Exception:
        return []

    try:
        t = yf.Ticker(etf_ticker)
        df = t.funds_data.top_holdings
    except Exception:
        return []

    if df is None or df.empty:
        return []

    tickers = [clean_ticker(x) for x in df.index.astype(str).tolist()]
    tickers = [t for t in tickers if t]
    return list(dict.fromkeys(tickers))


def get_etf_holdings(etf_ticker: str, *, use_cache: bool = True) -> List[str]:
    """
    Return (best-effort) full holdings tickers for an ETF.

    Strategy:
      1) Persistent cache (24h TTL)
      2) Provider export for known SPDR sector ETFs (SSGA)
      3) yfinance fallback (may be truncated)
    """
    etf = clean_ticker(etf_ticker)

    if use_cache:
        cached = _read_cached_etf_holdings(etf)
        if cached:
            return cached

    tickers: List[str] = []
    if etf in _SPDR_SECTOR_ETFS:
        tickers = fetch_etf_holdings_ssga(etf)

    if not tickers:
        tickers = fetch_etf_holdings_yfinance(etf)

    if tickers and use_cache:
        _write_cached_etf_holdings(etf, tickers)

    return tickers


def get_universe_tickers(key: str) -> List[str]:
    """
    Resolve a universe key to a ticker list.

    Supported inputs:
      - "sp500": S&P 500 constituents (Wikipedia)
      - Named universe files in `equities/universes/` (e.g., "russell2000")
      - A file path to a .csv/.txt universe
      - SPDR Select Sector ETFs: "XLY" or shortcut "xly" (ETF holdings)

    Returns:
      List of normalized tickers (deduped, order preserved when sourced from files/holdings).
    """
    if not key or not str(key).strip():
        return []

    raw = str(key).strip()
    k = raw.lower()

    if k in ("sp500", "s&p500", "s&p 500", "sp-500"):
        return get_sp500_universe()

    # Common index universes stored as files
    if k in ("russell2000", "sp400", "sp600", "nasdaq100", "dow30"):
        try:
            return load_universe(k)
        except Exception:
            # Fall through to other resolvers
            pass

    # Sector ETF shortcuts (xlk/xly/...)
    if k in _SECTOR_SHORTCUTS:
        return get_etf_holdings(_SECTOR_SHORTCUTS[k])

    # Explicit ETF ticker (e.g., XLY)
    if raw.upper() in _SPDR_SECTOR_ETFS:
        return get_etf_holdings(raw.upper())

    # File path or named universe file
    try:
        return load_universe(raw)
    except Exception:
        return []
