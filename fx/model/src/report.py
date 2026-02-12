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


# ---------------------------------------------------------------------------
# Driver explanation
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "rer_z": "Valuation (RER z-score)",
    "carry": "Interest Rate Carry",
    "oil_mom12": "Oil Price Momentum",
    "mom12": "FX Momentum",
    "ca_diff": "Current Account Diff.",
    "carry_to_vol": "Risk-Adjusted Carry",
    "vix": "VIX (Risk Appetite)",
}


def _describe_driver(name: str, coeff: float, value: float, contribution: float,
                     base_ccy: str, quote_ccy: str) -> str:
    """Return a one-line interpretation for a single driver."""
    pos = contribution > 0  # positive contribution = base ccy strengthens

    if name == "rer_z":
        if abs(value) > 2.0:
            degree = "significantly "
        elif abs(value) > 1.0:
            degree = ""
        else:
            return "Valuation near fair value"
        if value > 0:
            return f"{base_ccy} {degree}overvalued vs {quote_ccy}, mean-reversion pressures {'upward' if pos else 'downward'}"
        return f"{base_ccy} {degree}undervalued vs {quote_ccy}, mean-reversion pressures {'upward' if pos else 'downward'}"

    if name == "carry":
        if abs(value) < 0.002:
            return "Carry near zero — negligible impact"
        if value > 0:
            return f"Positive carry {'supports' if pos else 'drags on'} {base_ccy} vs {quote_ccy}"
        return f"Negative carry {'supports' if pos else 'drags on'} {base_ccy} vs {quote_ccy}"

    if name == "mom12":
        trend = "strengthening" if value > 0 else "weakening"
        return f"Recent {base_ccy} {trend} {'extends' if pos else 'fades'} momentum"

    if name == "oil_mom12":
        oil_dir = "rising" if value > 0 else "falling"
        return f"Oil momentum ({oil_dir}) {'supports' if pos else 'weighs on'} {base_ccy}"

    if name == "ca_diff":
        surplus = quote_ccy if value > 0 else base_ccy
        return f"{surplus} CA surplus {'supports' if pos else 'weighs on'} {base_ccy}"

    if name == "carry_to_vol":
        return f"Risk-adjusted carry {'supports' if pos else 'weighs on'} {base_ccy}"

    if name == "vix":
        level = "elevated" if value > 20 else "subdued" if value < 15 else "moderate"
        return f"{level.capitalize()} VIX {'supports' if pos else 'weighs on'} {base_ccy}"

    return f"{'Supports' if pos else 'Weighs on'} {base_ccy}"


def _generate_narrative(drivers: list, base_ccy: str, quote_ccy: str,
                        horizon: int, spot_now: float, point_level: float) -> str:
    """Build a 2-4 sentence conclusion paragraph."""
    pct = ((point_level / spot_now) - 1) * 100 if spot_now else 0
    direction = "strengthening" if pct > 0.1 else "weakening" if pct < -0.1 else "roughly flat"

    sentences = [
        f"The model sees {base_ccy} {direction} against {quote_ccy} "
        f"over the next {horizon} months ({pct:+.1f}%)."
    ]

    if not drivers:
        return sentences[0]

    top = drivers[0]
    sentences.append(
        f"The primary driver is {top['label'].lower()} — {top['description'].lower()}."
    )

    if len(drivers) >= 2:
        second = drivers[1]
        verb = "also supports" if second["contribution"] * (1 if pct > 0 else -1) > 0 else "partially offsets, as"
        sentences.append(
            f"{second['label']} {verb} {second['description'][0].lower()}{second['description'][1:]}."
        )

    # Valuation callout if not already the top driver
    rer_drivers = [d for d in drivers if d["name"] == "rer_z"]
    if rer_drivers and rer_drivers[0] is not top and abs(rer_drivers[0]["value"]) > 1.5:
        z = rer_drivers[0]["value"]
        sentences.append(
            f"Notably, {base_ccy} appears {'overvalued' if z > 0 else 'undervalued'} "
            f"(RER z-score: {z:+.2f})."
        )

    return " ".join(sentences)


def build_driver_explanation(
    params: dict,
    feature_values: dict,
    pair_name: str,
    base_ccy: str,
    quote_ccy: str,
    horizon: int,
    spot_now: float,
    point_level: float,
    r2: float,
    nobs: int,
) -> dict:
    """Build a structured driver explanation for a single horizon.

    Returns dict with keys:
        text       – formatted string for terminal output
        drivers    – list of driver dicts sorted by |contribution|
        conclusion – narrative paragraph
    """
    drivers = []
    for feat, coeff in params.items():
        if feat == "const":
            continue
        val = feature_values.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        contrib = coeff * val
        desc = _describe_driver(feat, coeff, val, contrib, base_ccy, quote_ccy)
        drivers.append({
            "name": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "coefficient": float(coeff),
            "value": float(val),
            "contribution": float(contrib),
            "description": desc,
        })

    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    conclusion = _generate_narrative(drivers, base_ccy, quote_ccy, horizon, spot_now, point_level)

    # Format terminal text
    pct = ((point_level / spot_now) - 1) * 100 if spot_now else 0
    lines = [
        f"=== {pair_name} {horizon}-Month Forecast Drivers ===",
        f"Forecast: {point_level:.4f} (current: {spot_now:.4f}, expected move: {pct:+.1f}%)",
        f"Model R²: {r2:.3f} | Observations: {nobs}",
        "",
        "Top drivers (coefficient × current value = contribution to log return):",
    ]
    for i, d in enumerate(drivers, 1):
        lines.append(
            f"  {i}. {d['label']:30s}  coeff={d['coefficient']:+.4f}, "
            f"value={d['value']:+.4f}, contribution={d['contribution']:+.5f}"
        )
        lines.append(f"     → {d['description']}")

    lines.append("")
    lines.append(f"Conclusion: {conclusion}")

    return {
        "text": "\n".join(lines),
        "drivers": drivers,
        "conclusion": conclusion,
    }
