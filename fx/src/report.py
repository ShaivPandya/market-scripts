"""Report generation for FX models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_csv(df: pd.DataFrame, path: Path):
    """Save DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def save_json(obj: dict, path: Path):
    """Save dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_spot_vs_reference(
    ref: pd.DataFrame,
    path: Path,
    pair_name: str = "USDCAD",
    spot_label: str = "CAD per USD",
):
    """Plot spot rate vs valuation reference points."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(ref.index, ref["spot"], label=f"{pair_name} spot")
    plt.plot(ref.index, ref["spot_implied_median"], label="Implied spot (median RER)")
    plt.plot(ref.index, ref["spot_implied_p25"], label="Implied spot (RER p25)")
    plt.plot(ref.index, ref["spot_implied_p75"], label="Implied spot (RER p75)")
    plt.legend()
    plt.title(f"{pair_name} spot vs valuation reference points")
    plt.xlabel("Date")
    plt.ylabel(f"{pair_name} ({spot_label})")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_valuation_zscore(
    df: pd.DataFrame,
    path: Path,
    pair_name: str = "USDCAD",
):
    """Plot real exchange rate z-score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df.index, df["rer_z"])
    plt.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    plt.title(f"{pair_name} real exchange rate z-score (rolling)")
    plt.xlabel("Date")
    plt.ylabel("z-score")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_forecast_distribution(
    draws: np.ndarray,
    point: float,
    spot_now: float,
    path: Path,
    horizon: int,
    pair_name: str = "USDCAD",
    spot_label: str = "CAD per USD",
):
    """Plot forecast distribution histogram."""
    path.parent.mkdir(parents=True, exist_ok=True)
    levels = spot_now * np.exp(draws)
    plt.figure()
    plt.hist(levels, bins=40, alpha=0.7, edgecolor="black")
    plt.axvline(spot_now * np.exp(point), color="red", linestyle="--", label="Point forecast")
    plt.axvline(spot_now, color="blue", linestyle=":", label="Current spot")
    plt.legend()
    plt.title(f"Forecast distribution: {pair_name} level in {horizon} months")
    plt.xlabel(f"{pair_name} level ({spot_label})")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def summarize_distribution(draws: np.ndarray) -> dict:
    """Summarize a distribution with quantiles and moments."""
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    out = {f"q{int(q*100):02d}": float(np.quantile(draws, q)) for q in qs}
    out["mean"] = float(np.mean(draws))
    out["std"] = float(np.std(draws))
    return out
