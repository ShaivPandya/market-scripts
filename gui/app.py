#!/usr/bin/env python3
"""
Market Analysis Dashboard - Streamlit GUI

Provides a navigatable interface for:
- Market Technicals (breadth, top 50, price/volume signals)
- Market Dashboard (commodities, equities, currencies performance)
- Liquidity Dashboard (Fed/ECB/BoJ liquidity metrics)

Run:
  streamlit run gui/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "equities" / "market_technicals"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "market_dashboard"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "liquidity"))

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Market Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "📈 Market Technicals"

# Sidebar: Settings Section
st.sidebar.title("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 60, 600, 300)
    st.sidebar.info(f"Will refresh every {refresh_interval}s")

# Visual separator
st.sidebar.divider()

# Sidebar: Navigation Section
st.sidebar.markdown("### Navigation")

# Clickable text navigation
if st.sidebar.button("📈 Market Technicals", use_container_width=True,
                      type="primary" if st.session_state.current_page == "📈 Market Technicals" else "secondary"):
    st.session_state.current_page = "📈 Market Technicals"
    st.rerun()

if st.sidebar.button("📊 Market Dashboard", use_container_width=True,
                      type="primary" if st.session_state.current_page == "📊 Market Dashboard" else "secondary"):
    st.session_state.current_page = "📊 Market Dashboard"
    st.rerun()

if st.sidebar.button("💧 Liquidity", use_container_width=True,
                      type="primary" if st.session_state.current_page == "💧 Liquidity" else "secondary"):
    st.session_state.current_page = "💧 Liquidity"
    st.rerun()

# Visual separator
st.sidebar.divider()

# Page-specific sidebar controls
if st.session_state.current_page == "💧 Liquidity":
    st.sidebar.title("Liquidity Options")
    skip_ecb = st.sidebar.checkbox("Skip ECB data", value=False, help="Skip ECB SDMX fetch if it's slow")
else:
    skip_ecb = False  # Default for non-Liquidity pages

def color_positive_negative(val):
    """Color positive values green, negative red."""
    if pd.isna(val):
        return "color: gray"
    try:
        # Handle string values like "+1.5%" or "-2.3%"
        if isinstance(val, str):
            val = val.replace("%", "").replace("+", "").strip()
            if val == "N/A" or val == "":
                return "color: gray"
            val = float(val)
        if val > 0:
            return "color: #00c853; font-weight: bold"  # Green
        elif val < 0:
            return "color: #ff1744; font-weight: bold"  # Red
        return ""
    except (ValueError, TypeError):
        return ""


def color_signal_flag(val):
    """Color YES green, no gray."""
    if val == "YES":
        return "color: #00c853; font-weight: bold"
    return "color: gray"


def color_return_vs_benchmark(val, is_outperforming=None):
    """Color based on benchmark comparison indicator."""
    if pd.isna(val) or val == "N/A":
        return "color: gray"
    if isinstance(val, str):
        if "(+)" in val:
            return "color: #00c853; font-weight: bold"  # Green - outperforming
        elif "(-)" in val:
            return "color: #ff1744; font-weight: bold"  # Red - underperforming
        # For benchmark rows or simple returns, color by sign
        try:
            num_val = float(val.replace("%", "").replace("+", "").split()[0])
            if num_val > 0:
                return "color: #00c853"
            elif num_val < 0:
                return "color: #ff1744"
        except (ValueError, TypeError):
            pass
    return ""


def color_zscore(val):
    """Color z-scores: green if positive (supportive), red if negative (tightening)."""
    if pd.isna(val) or val == "N/A":
        return "color: gray"
    try:
        if isinstance(val, str):
            val = float(val.replace("+", ""))
        if val >= 1:
            return "color: #00c853; font-weight: bold"  # Strong positive
        elif val > 0:
            return "color: #00c853"  # Positive
        elif val <= -1:
            return "color: #ff1744; font-weight: bold"  # Strong negative
        elif val < 0:
            return "color: #ff1744"  # Negative
        return "color: #ffc107"  # Yellow for near-zero
    except (ValueError, TypeError):
        return "color: gray"


# =============================================================================
# PAGE: Market Technicals
# =============================================================================
if st.session_state.current_page == "📈 Market Technicals":
    st.header("Market Technicals")

    if st.button("Refresh Data", key="refresh_technicals"):
        st.cache_data.clear()

    tech_tab1, tech_tab2, tech_tab3 = st.tabs([
        "Market Breadth",
        "Top 50 Breadth",
        "Price/Volume Signals",
    ])

    # Market Breadth
    with tech_tab1:
        st.subheader("S&P 500 Market Breadth")

        @st.cache_data(ttl=300)
        def fetch_market_breadth():
            try:
                from market_breadth import get_data
                return get_data()
            except Exception as e:
                return {"error": str(e)}

        with st.spinner("Fetching market breadth data..."):
            breadth_data = fetch_market_breadth()

        if "error" in breadth_data:
            st.error(f"Error: {breadth_data['error']}")
        else:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                pct = breadth_data.get("pct_above_200dma", 0)
                highlight = pct > 80 or pct < 15
                st.metric(
                    "Above 200-DMA",
                    f"{pct:.1f}%",
                    f"{breadth_data.get('above_200dma', 0)} / {breadth_data.get('total_analyzed', 0)}",
                )
                if highlight:
                    st.success("Signal Active")

            with col2:
                pct = breadth_data.get("pct_above_20dma", 0)
                highlight = pct > 80 or pct < 20
                st.metric(
                    "Above 20-DMA",
                    f"{pct:.1f}%",
                    f"{breadth_data.get('above_20dma', 0)} / {breadth_data.get('total_analyzed', 0)}",
                )
                if highlight:
                    st.success("Signal Active")

            with col3:
                pct = breadth_data.get("pct_at_20day_high", 0)
                highlight = pct > 50
                st.metric(
                    "At 20-Day Highs",
                    f"{pct:.1f}%",
                    f"{breadth_data.get('at_20day_high', 0)} / {breadth_data.get('total_analyzed', 0)}",
                )
                if highlight:
                    st.success("Signal Active")

            with col4:
                pct = breadth_data.get("pct_at_20day_low", 0)
                highlight = pct > 50
                st.metric(
                    "At 20-Day Lows",
                    f"{pct:.1f}%",
                    f"{breadth_data.get('at_20day_low', 0)} / {breadth_data.get('total_analyzed', 0)}",
                )
                if highlight:
                    st.warning("Capitulation Signal")

    # Top 50 Breadth
    with tech_tab2:
        st.subheader("Top 50 S&P 500 Performers - Breadth")

        @st.cache_data(ttl=300)
        def fetch_top50_breadth():
            try:
                from top50_breadth import get_data
                return get_data()
            except Exception as e:
                return {"error": str(e)}

        with st.spinner("Fetching top 50 breadth data..."):
            top50_data = fetch_top50_breadth()

        if "error" in top50_data:
            st.error(f"Error: {top50_data['error']}")
        elif top50_data.get("universe_size", 0) == 0:
            st.warning("No tickers with sufficient data")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                pct = top50_data.get("pct_below_50dma")
                st.metric("% Below 50-DMA", f"{pct:.1f}%" if pct else "N/A")
                tickers = top50_data.get("tickers_below_50dma", [])
                if tickers:
                    st.caption(f"Tickers: {', '.join(tickers)}")

            with col2:
                pct = top50_data.get("pct_3plus_dist")
                st.metric("% with 3+ Distribution Days", f"{pct:.1f}%" if pct else "N/A")
                tickers = top50_data.get("tickers_3plus_dist", [])
                if tickers:
                    st.caption(f"Tickers: {', '.join(tickers)}")

            with col3:
                pct = top50_data.get("pct_broke_20low")
                st.metric("% Broke 20-Day Low", f"{pct:.1f}%" if pct else "N/A")
                tickers = top50_data.get("tickers_broke_20low", [])
                if tickers:
                    st.caption(f"Tickers: {', '.join(tickers)}")

            st.info(f"Universe: {top50_data.get('universe_size', 0)} stocks with sufficient data")

    # Price/Volume Signals
    with tech_tab3:
        st.subheader("Price/Volume Signals")

        @st.cache_data(ttl=300)
        def fetch_price_volume():
            try:
                from price_volume_signals import get_data
                return get_data()
            except Exception as e:
                return {"error": str(e)}

        with st.spinner("Fetching price/volume signals..."):
            pv_data = fetch_price_volume()

        if "error" in pv_data:
            st.error(f"Error: {pv_data['error']}")
        else:
            latest_df = pv_data.get("latest_df")
            if latest_df is not None and not latest_df.empty:
                st.write("**Latest Signals**")

                display_df = latest_df[["Market", "Date", "DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn", "Close", "RetPct"]].copy()
                display_df["DownsideRecordVol"] = display_df["DownsideRecordVol"].map({True: "YES", False: "no"})
                display_df["NewHigh_LowVol"] = display_df["NewHigh_LowVol"].map({True: "YES", False: "no"})
                display_df["HiVol_Churn"] = display_df["HiVol_Churn"].map({True: "YES", False: "no"})
                display_df["RetPctNum"] = display_df["RetPct"]  # Keep numeric for styling
                display_df["RetPct"] = display_df["RetPct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                display_df["Close"] = display_df["Close"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

                # Apply styling
                styled_df = display_df[["Market", "Date", "DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn", "Close", "RetPct"]].style.applymap(
                    color_signal_flag, subset=["DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn"]
                ).applymap(
                    color_positive_negative, subset=["RetPct"]
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

            hits_df = pv_data.get("hits_df")
            if hits_df is not None and not hits_df.empty:
                st.write("**Recent Signal History**")
                for market in hits_df["MarketName"].unique():
                    with st.expander(f"{market}"):
                        market_hits = hits_df[hits_df["MarketName"] == market].head(10).copy()
                        market_hits["DownsideRecordVol"] = market_hits["DownsideRecordVol"].map({True: "YES", False: "no"})
                        market_hits["NewHigh_LowVol"] = market_hits["NewHigh_LowVol"].map({True: "YES", False: "no"})
                        market_hits["HiVol_Churn"] = market_hits["HiVol_Churn"].map({True: "YES", False: "no"})
                        market_hits["RetPct"] = market_hits["RetPct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

                        display_cols = ["Date", "Close", "RetPct", "DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn"]
                        styled_hits = market_hits[display_cols].style.applymap(
                            color_signal_flag, subset=["DownsideRecordVol", "NewHigh_LowVol", "HiVol_Churn"]
                        ).applymap(
                            color_positive_negative, subset=["RetPct"]
                        )
                        st.dataframe(styled_hits, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: Market Dashboard
# =============================================================================
elif st.session_state.current_page == "📊 Market Dashboard":
    st.header("Market Performance Dashboard")

    if st.button("Refresh Data", key="refresh_dashboard"):
        st.cache_data.clear()

    @st.cache_data(ttl=300)
    def fetch_market_dashboard():
        try:
            from market_dashboard import get_data
            return get_data()
        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Fetching market data from Yahoo Finance..."):
        dashboard_data = fetch_market_dashboard()

    if "error" in dashboard_data:
        st.error(f"Error: {dashboard_data['error']}")
    else:
        st.caption(f"Data as of: {dashboard_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")

        # Commodities
        st.subheader("Commodities")
        commodities = dashboard_data.get("commodities", {})
        if commodities:
            periods = dashboard_data.get("equity_periods", ["1-mo", "3-mo", "6-mo", "1-yr"])
            rows = []
            for name, returns in commodities.items():
                row = {"Name": name}
                for period in periods:
                    val = returns.get(period)
                    row[period] = f"{val:+.1f}%" if val is not None else "N/A"
                rows.append(row)
            df = pd.DataFrame(rows)
            styled_df = df.style.applymap(color_positive_negative, subset=periods)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Equities
        st.subheader("Equities (vs Benchmark)")
        equities = dashboard_data.get("equities", {})
        if equities:
            periods = dashboard_data.get("equity_periods", ["1-mo", "3-mo", "6-mo", "1-yr"])
            sp500_returns = equities.get("S&P 500", {})
            stoxx_returns = equities.get("STOXX 600", {})

            rows = []
            for name, returns in equities.items():
                row = {"Name": name}
                is_benchmark = name in ["S&P 500", "STOXX 600"]
                benchmark = stoxx_returns if name == "Europe Banks" else sp500_returns

                for period in periods:
                    val = returns.get(period)
                    bench_val = benchmark.get(period)
                    if val is None:
                        row[period] = "N/A"
                    elif is_benchmark:
                        row[period] = f"{val:+.1f}%"
                    elif bench_val is not None:
                        indicator = "(+)" if val > bench_val else "(-)" if val < bench_val else "(=)"
                        row[period] = f"{val:+.1f}% {indicator}"
                    else:
                        row[period] = f"{val:+.1f}%"
                rows.append(row)

            df = pd.DataFrame(rows)
            styled_df = df.style.applymap(color_return_vs_benchmark, subset=periods)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.caption("(+) = outperforming benchmark, (-) = underperforming benchmark")

        # Currencies
        st.subheader("Currencies")
        currencies = dashboard_data.get("currencies", {})
        if currencies:
            periods = dashboard_data.get("currency_periods", ["1-mo", "3-mo", "6-mo"])
            rows = []
            for name, returns in currencies.items():
                row = {"Pair": name}
                for period in periods:
                    val = returns.get(period)
                    row[period] = f"{val:+.1f}%" if val is not None else "N/A"
                rows.append(row)
            df = pd.DataFrame(rows)
            styled_df = df.style.applymap(color_positive_negative, subset=periods)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: Liquidity
# =============================================================================
elif st.session_state.current_page == "💧 Liquidity":
    st.header("Liquidity Dashboard")

    if st.button("Refresh Data", key="refresh_liquidity"):
        st.cache_data.clear()

    # Note: skip_ecb is now defined in sidebar section above

    @st.cache_data(ttl=300)
    def fetch_liquidity(skip_ecb: bool):
        try:
            from liquidity import get_snapshot
            return get_snapshot(skip_ecb=skip_ecb)
        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Fetching liquidity data from FRED..."):
        liquidity_data = fetch_liquidity(skip_ecb)

    if "error" in liquidity_data:
        st.error(f"Error: {liquidity_data['error']}")
    elif liquidity_data.get("composite_score") is None:
        st.warning("Insufficient data to compute liquidity score")
    else:
        # Headline metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            score = liquidity_data.get("composite_score", 0)
            regime = liquidity_data.get("regime", "unknown")
            color_map = {"green": "normal", "cyan": "off", "yellow": "inverse", "red": "inverse"}
            st.metric("Composite Score", f"{score:+.2f}")

        with col2:
            regime = liquidity_data.get("regime", "unknown").upper()
            color = liquidity_data.get("regime_color", "gray")
            if color == "green":
                st.success(f"Regime: {regime}")
            elif color == "cyan":
                st.info(f"Regime: {regime}")
            elif color == "yellow":
                st.warning(f"Regime: {regime}")
            else:
                st.error(f"Regime: {regime}")

        with col3:
            st.metric("As of", str(liquidity_data.get("latest_date", "N/A")))

        # Regional scores
        st.subheader("Regional Liquidity Scores")
        regional = liquidity_data.get("regional_scores", {})
        if regional:
            cols = st.columns(len(regional))
            region_labels = {"us": "United States", "europe": "Europe", "japan": "Japan"}
            for i, (region_key, data) in enumerate(regional.items()):
                with cols[i]:
                    score = data.get("score", 0)
                    regime = data.get("regime", "unknown").upper()
                    st.metric(region_labels.get(region_key, region_key), f"{score:+.2f}")
                    color = data.get("color", "gray")
                    if color == "green":
                        st.success(regime)
                    elif color == "cyan":
                        st.info(regime)
                    elif color == "yellow":
                        st.warning(regime)
                    else:
                        st.error(regime)

        # Components table
        st.subheader("Components")
        components = liquidity_data.get("components", [])
        if components:
            rows = []
            for comp in components:
                value = comp.get("value")
                z = comp.get("z_score")
                contrib = comp.get("contribution")
                kind = comp.get("value_kind", "")

                if kind == "billions" and value is not None:
                    value_str = f"${value/1000:.2f}B"
                elif kind == "percent" and value is not None:
                    value_str = f"{value:.2f}%"
                elif kind == "ratio" and value is not None:
                    value_str = f"{value:.3f}"
                elif value is not None:
                    value_str = f"{value:.2f}"
                else:
                    value_str = "N/A"

                # Determine signal text based on z-score
                if z is not None and not pd.isna(z):
                    signal = "supportive" if z >= 0 else "tightening"
                else:
                    signal = "N/A"

                rows.append({
                    "Region": comp.get("region", ""),
                    "Component": comp.get("label", ""),
                    "Value": value_str,
                    "Z-Score": f"{z:+.2f}" if z is not None and not pd.isna(z) else "N/A",
                    "Weight": f"{comp.get('weight', 0)*100:.0f}%",
                    "Contribution": f"{contrib:+.2f}" if contrib is not None and not pd.isna(contrib) else "N/A",
                    "Signal": signal,
                })

            df = pd.DataFrame(rows)
            styled_df = df.style.applymap(
                color_zscore, subset=["Z-Score", "Contribution"]
            ).applymap(
                lambda x: "color: #00c853; font-weight: bold" if x == "supportive" else ("color: #ff1744; font-weight: bold" if x == "tightening" else "color: gray"),
                subset=["Signal"]
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Changes table
        st.subheader("Historical Changes")
        changes = liquidity_data.get("changes", {})
        if changes:
            rows = []
            polarity_map = {}
            for label, data in changes.items():
                row = {"Series": label}
                polarity = data.get("polarity", 1)
                polarity_map[label] = polarity
                for period in ["1w", "1m", "3m"]:
                    val = data.get(period)
                    kind = data.get("value_kind", "")
                    if val is None or pd.isna(val):
                        row[period] = "N/A"
                    elif kind == "billions":
                        row[period] = f"{val/1000:+.2f}B"
                    elif kind == "percent":
                        row[period] = f"{val:+.2f}%"
                    elif kind == "score":
                        row[period] = f"{val:+.2f}"
                    else:
                        row[period] = f"{val:+.2f}"
                rows.append(row)

            df = pd.DataFrame(rows)

            # Custom styling that accounts for polarity (inverted for OAS, NFCI)
            def style_with_polarity(row):
                """Style a row considering its polarity."""
                series_name = row["Series"]
                polarity = polarity_map.get(series_name, 1)
                styles = [""] * len(row)

                for i, (col_name, val) in enumerate(row.items()):
                    if col_name not in ["1w", "1m", "3m"]:
                        continue

                    if pd.isna(val) or val == "N/A":
                        styles[i] = "color: gray"
                        continue

                    try:
                        # Extract numeric value
                        num_str = val.replace("B", "").replace("%", "").replace("+", "").strip()
                        num_val = float(num_str)
                        # Apply polarity: positive polarity means increase is good
                        # negative polarity means decrease is good (e.g., OAS, NFCI)
                        effective_change = num_val * polarity
                        if effective_change > 0:
                            styles[i] = "color: #00c853; font-weight: bold"
                        elif effective_change < 0:
                            styles[i] = "color: #ff1744; font-weight: bold"
                    except (ValueError, TypeError):
                        pass

                return styles

            styled_df = df.style.apply(style_with_polarity, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.caption("Green indicates liquidity-supportive changes, red indicates liquidity-tightening changes")


# Auto-refresh logic
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
