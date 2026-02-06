#!/usr/bin/env python3
"""
Market Analysis Dashboard - Streamlit GUI

Provides a navigatable interface for:
- Market Dashboard (commodities, equities, currencies performance)
- Market Technicals (breadth, top 50, price/volume signals)
- Liquidity Dashboard (Fed/ECB/BoJ liquidity metrics)
- CFTC Positioning (leveraged funds futures positioning)

Run:
  streamlit run gui/app.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "equities" / "market_technicals"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "market_dashboard"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "liquidity"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "breakout"))
sys.path.insert(0, str(PROJECT_ROOT / "macro" / "positioning"))
sys.path.insert(0, str(PROJECT_ROOT / "equities" / "portfolio"))
sys.path.insert(0, str(PROJECT_ROOT / "equities" / "momentum" / "price_momentum"))
sys.path.insert(0, str(PROJECT_ROOT / "fx"))

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
    st.session_state.current_page = "📊 Market Dashboard"

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
if st.sidebar.button("📊 Market Dashboard", width='stretch',
                      type="primary" if st.session_state.current_page == "📊 Market Dashboard" else "secondary"):
    st.session_state.current_page = "📊 Market Dashboard"
    st.rerun()

if st.sidebar.button("📈 Market Technicals", width='stretch',
                      type="primary" if st.session_state.current_page == "📈 Market Technicals" else "secondary"):
    st.session_state.current_page = "📈 Market Technicals"
    st.rerun()

if st.sidebar.button("💧 Liquidity", width='stretch',
                      type="primary" if st.session_state.current_page == "💧 Liquidity" else "secondary"):
    st.session_state.current_page = "💧 Liquidity"
    st.rerun()

if st.sidebar.button("📌 Positioning", width='stretch',
                      type="primary" if st.session_state.current_page == "📌 Positioning" else "secondary"):
    st.session_state.current_page = "📌 Positioning"
    st.rerun()

if st.sidebar.button("🔔 Breakout", width='stretch',
                      type="primary" if st.session_state.current_page == "🔔 Breakout" else "secondary"):
    st.session_state.current_page = "🔔 Breakout"
    st.rerun()

if st.sidebar.button("📈 Portfolio Optimizer", width='stretch',
                      type="primary" if st.session_state.current_page == "📈 Portfolio Optimizer" else "secondary"):
    st.session_state.current_page = "📈 Portfolio Optimizer"
    st.rerun()

if st.sidebar.button("🚀 Momentum", width='stretch',
                      type="primary" if st.session_state.current_page == "🚀 Momentum" else "secondary"):
    st.session_state.current_page = "🚀 Momentum"
    st.rerun()

if st.sidebar.button("💱 FX Model", width='stretch',
                      type="primary" if st.session_state.current_page == "💱 FX Model" else "secondary"):
    st.session_state.current_page = "💱 FX Model"
    st.rerun()

# Visual separator
st.sidebar.divider()

# Page-specific sidebar controls
if st.session_state.current_page == "💧 Liquidity":
    st.sidebar.title("Liquidity Options")
    skip_ecb = st.sidebar.checkbox("Skip ECB data", value=False, help="Skip ECB SDMX fetch if it's slow")
else:
    skip_ecb = False  # Default for non-Liquidity pages

# FX Model sidebar controls
fx_selected_pair = None
fx_bootstrap = 1000
fx_skip_bis = False
fx_horizons_str = "12,24"
fx_run_clicked = False

if st.session_state.current_page == "💱 FX Model":
    st.sidebar.title("FX Model Settings")
    try:
        from src.currency_config import list_pairs
        fx_available_pairs = list_pairs()
    except Exception:
        fx_available_pairs = ["USDCAD", "GBPUSD", "AUDUSD", "USDJPY"]

    fx_selected_pair = st.sidebar.selectbox("Currency Pair", options=fx_available_pairs)
    fx_bootstrap = st.sidebar.number_input("Bootstrap Draws", min_value=100, max_value=5000, value=1000, step=100)
    fx_skip_bis = st.sidebar.checkbox("Skip BIS data", value=False, help="Skip BIS downloads if slow")
    fx_horizons_str = st.sidebar.text_input("Horizons (months)", value="12,24")
    fx_run_clicked = st.sidebar.button("Run Model", type="primary", use_container_width=True)

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

def color_forced_flow(val):
    """Color forced-flow labels from positioning.py."""
    if pd.isna(val) or val == "" or val == "N/A":
        return "color: gray"
    if val == "short_covering":
        return "color: #00c853; font-weight: bold"
    if val == "long_liquidation":
        return "color: #ff1744; font-weight: bold"
    return ""


def color_vix_signal(val):
    """Color VIX term-structure signals."""
    if val == "Fear":
        return "color: #ff1744; font-weight: bold"
    if val == "Complacency":
        return "color: #ffc107; font-weight: bold"
    if val == "Neutral":
        return "color: gray"
    return "color: gray"


# =============================================================================
# PAGE: Market Technicals
# =============================================================================
if st.session_state.current_page == "📈 Market Technicals":
    st.header("Market Technicals")

    if st.button("Refresh Data", key="refresh_technicals"):
        st.cache_data.clear()

    # Use a selector so we only fetch data for the active view, styled to match tabs.
    st.markdown(
        """
        <style>
        .market-tech-tabs [data-testid="stRadio"] > div {
            flex-direction: row;
            gap: 0.25rem;
        }
        .market-tech-tabs [data-testid="stRadio"] label {
            background: #f6f7f9;
            border: 1px solid transparent;
            border-bottom: 2px solid transparent;
            border-radius: 6px 6px 0 0;
            cursor: pointer;
            margin: 0;
            padding: 0.35rem 0.9rem;
        }
        .market-tech-tabs [data-testid="stRadio"] label:has(input:checked) {
            background: #ffffff;
            border: 1px solid #e1e4e8;
            border-bottom-color: #ffffff;
            font-weight: 600;
        }
        .market-tech-tabs [data-testid="stRadio"] input {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_labels = [
        "VIX Term Structure",
        "Market Breadth",
        "Top 50 Breadth",
        "Price/Volume Signals",
    ]
    st.markdown('<div class="market-tech-tabs">', unsafe_allow_html=True)
    selected_tab = st.radio(
        "Technicals View",
        tab_labels,
        horizontal=True,
        key="market_technicals_tab",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Market Breadth
    if selected_tab == "Market Breadth":
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
    elif selected_tab == "Top 50 Breadth":
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
    elif selected_tab == "Price/Volume Signals":
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
                st.dataframe(styled_df, width='stretch', hide_index=True)

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
                        st.dataframe(styled_hits, width='stretch', hide_index=True)

    # VIX Term Structure
    elif selected_tab == "VIX Term Structure":
        st.subheader("VIX Term Structure (3M / 1M)")
        st.caption("High ratio (>= 1.25): later volatility concerns. Low ratio (< 1.0): near-term fear.")

        @st.cache_data(ttl=300)
        def fetch_vix_term_structure():
            try:
                from vix_term_structure import get_data
                return get_data(tail=252, signals_count=20)
            except Exception as e:
                return {"error": str(e)}

        with st.spinner("Fetching VIX term structure data..."):
            vix_data = fetch_vix_term_structure()

        if "error" in vix_data:
            st.error(f"Error: {vix_data['error']}")
        else:
            latest_df = vix_data.get("latest_df")
            if latest_df is not None and not latest_df.empty:
                latest = latest_df.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("3M / 1M Ratio", f"{latest['Ratio']:.2f}")
                with col2:
                    st.metric("VIX", f"{latest['VIX']:.2f}")
                with col3:
                    st.metric(f"3M VIX ({latest['UsedTicker']})", f"{latest['VIX3M']:.2f}")
                with col4:
                    st.metric("Date", str(latest["Date"]))

                signal = latest.get("Signal", "Neutral")
                if signal == "Fear":
                    st.warning("Signal: Fear (near-term volatility elevated)")
                elif signal == "Complacency":
                    st.info("Signal: Complacency (longer-term volatility elevated)")
                else:
                    st.caption("Signal: Neutral")

            recent_df = vix_data.get("recent_df")
            if recent_df is not None and not recent_df.empty:
                st.write("**Ratio Over Time (last 12 months)**")
                chart_df = recent_df[["Date", "Ratio"]].copy()
                chart_df["Date"] = pd.to_datetime(chart_df["Date"])
                chart_df = chart_df.sort_values("Date").set_index("Date")
                st.line_chart(chart_df["Ratio"], height=200)

                st.write("**Recent Ratios**")
                display_recent = recent_df.tail(10)[["Date", "VIX", "VIX3M", "Ratio", "Signal"]].copy()
                display_recent["VIX"] = display_recent["VIX"].apply(lambda x: f"{x:.2f}")
                display_recent["VIX3M"] = display_recent["VIX3M"].apply(lambda x: f"{x:.2f}")
                display_recent["Ratio"] = display_recent["Ratio"].apply(lambda x: f"{x:.2f}")
                styled_recent = display_recent.style.applymap(
                    color_vix_signal, subset=["Signal"]
                )
                st.dataframe(styled_recent, width='stretch', hide_index=True)

            hits_df = vix_data.get("hits_df")
            if hits_df is not None and not hits_df.empty:
                st.write("**Recent Signal Hits**")
                display_hits = hits_df[["Date", "VIX", "VIX3M", "Ratio", "Signal"]].copy()
                display_hits["VIX"] = display_hits["VIX"].apply(lambda x: f"{x:.2f}")
                display_hits["VIX3M"] = display_hits["VIX3M"].apply(lambda x: f"{x:.2f}")
                display_hits["Ratio"] = display_hits["Ratio"].apply(lambda x: f"{x:.2f}")
                styled_hits = display_hits.style.applymap(
                    color_vix_signal, subset=["Signal"]
                )
                st.dataframe(styled_hits, width='stretch', hide_index=True)


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
            st.dataframe(styled_df, width="stretch", hide_index=True)

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
            st.dataframe(styled_df, width="stretch", hide_index=True)
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
            st.dataframe(styled_df, width="stretch", hide_index=True)


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
            st.dataframe(styled_df, width="stretch", hide_index=True)

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
            st.dataframe(styled_df, width="stretch", hide_index=True)
            st.caption("Green indicates liquidity-supportive changes, red indicates liquidity-tightening changes")


# =============================================================================
# PAGE: Positioning
# =============================================================================
elif st.session_state.current_page == "📌 Positioning":
    st.header("CFTC Positioning")
    st.caption("COT participant positioning + simple forced-flow proxies (deleveraging / short-covering) via the CFTC PRE/Socrata API")

    with st.expander("Metric Definitions", expanded=False):
        st.markdown("""
**Net Position** — Long contracts minus short contracts held by the participant group (e.g., leveraged funds).

**Net % Open Interest** — Net position expressed as a percentage of total open interest. Normalizes across different contract sizes and markets.

**Position Z-Score** — How extreme the current positioning is relative to history. Values above +2 or below -2 indicate unusually crowded long or short positioning.

**Deleveraging Z-Score** — Measures how quickly participants are reducing exposure (moving toward flat). A high positive value means positions are being unwound faster than usual.

**Forced Flow** — Flags unusually aggressive position reductions:
- **Long Liquidation**: Longs being closed rapidly (often due to margin calls or stop-losses)
- **Short Covering**: Shorts being closed rapidly (often a squeeze or risk-off unwind)
        """)

    if st.button("Refresh Data", key="refresh_positioning"):
        st.cache_data.clear()

    try:
        from positioning import (
            DATASETS,
            DEFAULT_DOMAIN,
            INSTRUMENTS,
            fetch_market_timeseries,
            fetch_multiple_instruments,
        )
        try:
            from positioning import CANONICAL_GROUPS, GROUP_LABEL, GROUP_PREFIX
        except Exception:
            CANONICAL_GROUPS = ["lev_money", "dealer", "asset_mgr", "other_rept", "nonrept"]
            GROUP_PREFIX = {
                "lev_money": "lf",
                "dealer": "dealer",
                "asset_mgr": "asset_mgr",
                "other_rept": "other_rept",
                "nonrept": "nonrept",
            }
            GROUP_LABEL = {
                "lev_money": "Leveraged Funds",
                "dealer": "Dealer",
                "asset_mgr": "Asset Manager",
                "other_rept": "Other Reportables",
                "nonrept": "Non-Reportables",
            }
    except Exception as e:
        st.error(f"Failed to load positioning module: {e}")
        st.stop()

    @st.cache_data(ttl=3600)
    def cached_fetch_timeseries(
        domain,
        dataset_id,
        app_token,
        market_exact,
        start,
        end,
        groups,
        z_window,
        force_threshold,
    ):
        return fetch_market_timeseries(
            domain=domain,
            dataset_id=dataset_id,
            app_token=app_token,
            market_exact=market_exact,
            start=start,
            end=end,
            groups=groups,
            z_window=z_window,
            force_threshold=force_threshold,
        )

    @st.cache_data(ttl=3600)
    def cached_fetch_summary(
        domain,
        dataset_id,
        app_token,
        instruments,
        start,
        end,
        groups,
        z_window,
        force_threshold,
    ):
        return fetch_multiple_instruments(
            domain=domain,
            dataset_id=dataset_id,
            app_token=app_token,
            instruments=instruments,
            start=start,
            end=end,
            groups=groups,
            z_window=z_window,
            force_threshold=force_threshold,
        )

    st.subheader("Data Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        domain = st.text_input("Socrata domain", value=DEFAULT_DOMAIN)
    with col2:
        dataset_key = st.selectbox("Dataset", options=list(DATASETS.keys()), index=0)
    with col3:
        default_token = os.getenv("SODA_APP_TOKEN", "")
        app_token = st.text_input("Socrata App Token (optional)", value=default_token, type="password")

    dataset_id = DATASETS.get(dataset_key, dataset_key)

    st.subheader("Analytics Settings")
    gcol1, gcol2, gcol3 = st.columns(3)
    extra_group_options = ["dealer", "asset_mgr", "other_rept", "nonrept", "all"]
    with gcol1:
        selected_groups = st.multiselect(
            "Include participant groups",
            options=extra_group_options,
            default=[],
            help="Leveraged Funds are always included; add other groups or choose 'all'.",
            format_func=lambda g: "All groups" if g == "all" else GROUP_LABEL.get(g, g),
        )
    with gcol2:
        z_window = st.number_input(
            "Z-score window (weeks)",
            min_value=0,
            max_value=520,
            value=0,
            step=26,
            help="0 uses the full sample; otherwise uses a rolling window.",
        )
    with gcol3:
        force_threshold = st.slider(
            "Forced-flow threshold (z)",
            min_value=0.5,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help="Flags unusually large moves toward flat positioning (proxy for forced flows).",
        )

    if "all" in selected_groups:
        groups_value = "all"
    elif selected_groups:
        groups_value = ",".join(selected_groups)
    else:
        groups_value = None

    date_cols = st.columns(3)
    with date_cols[0]:
        start_date = st.date_input("Start date", value=datetime(2015, 1, 1).date())
    with date_cols[1]:
        use_end = st.checkbox("Use end date", value=False)
    with date_cols[2]:
        end_date = st.date_input("End date", value=datetime.now().date()) if use_end else None

    if use_end and end_date and start_date and end_date < start_date:
        st.error("End date must be on or after start date.")
        st.stop()

    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if use_end and end_date else None
    token_value = app_token or None

    st.divider()
    view = st.radio("View", ["Instrument summary", "Single market"], horizontal=True)

    if view == "Instrument summary":
        sorted_aliases = sorted(INSTRUMENTS.keys())
        default_aliases = [alias for alias in ["SP500", "NASDAQ", "US10Y", "EUR"] if alias in INSTRUMENTS]
        selected = st.multiselect("Instrument aliases", options=sorted_aliases, default=default_aliases)
        if not selected:
            st.info("Select at least one instrument alias to fetch positioning.")
        else:
            with st.spinner("Fetching positioning summary..."):
                try:
                    results = cached_fetch_summary(
                        domain,
                        dataset_id,
                        token_value,
                        selected,
                        start_str,
                        end_str,
                        groups_value,
                        z_window,
                        force_threshold,
                    )
                except Exception as e:
                    st.error(f"Error fetching summary: {e}")
                    results = []

            if results:
                df = pd.DataFrame(results)
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date

                for c in ["lf_net", "lf_net_pct_oi", "lf_z", "lf_deleveraging_z"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                df["abs_z"] = df["lf_z"].abs() if "lf_z" in df.columns else 0
                df = df.sort_values("abs_z", ascending=False).drop(columns=["abs_z"])

                base_cols = [
                    "instrument",
                    "report_date",
                    "lf_net",
                    "lf_net_pct_oi",
                    "lf_z",
                    "lf_deleveraging_z",
                    "lf_forced",
                ]
                base_cols = [c for c in base_cols if c in df.columns]
                base = df[base_cols].copy()

                base_format = {
                    "lf_net": "{:,.0f}",
                    "lf_net_pct_oi": "{:+.1f}%",
                    "lf_z": "{:+.2f}",
                    "lf_deleveraging_z": "{:+.2f}",
                }
                base_format = {k: v for k, v in base_format.items() if k in base.columns}

                # Rename columns for display
                display_rename = {
                    "instrument": "Instrument",
                    "report_date": "Report Date",
                    "lf_net": "Net Position",
                    "lf_net_pct_oi": "Net % Open Int",
                    "lf_z": "Position Z",
                    "lf_deleveraging_z": "Delev Z",
                    "lf_forced": "Forced Flow",
                }
                base = base.rename(columns=display_rename)
                base_format = {display_rename.get(k, k): v for k, v in base_format.items()}
                pct_col = "Net % Open Int" if "Net % Open Int" in base.columns else None
                z_cols = [c for c in ["Position Z", "Delev Z"] if c in base.columns]
                forced_col = "Forced Flow" if "Forced Flow" in base.columns else None

                styled_base = (
                    base.style.format(base_format)
                    .applymap(color_positive_negative, subset=[pct_col] if pct_col else [])
                    .applymap(color_zscore, subset=z_cols)
                    .applymap(color_forced_flow, subset=[forced_col] if forced_col else [])
                )
                st.dataframe(styled_base, width="stretch", hide_index=True)

                extra_groups = selected_groups
                if "all" in extra_groups:
                    extra_groups = [g for g in CANONICAL_GROUPS if g != "lev_money"]
                if extra_groups:
                    with st.expander("Other participant groups", expanded=False):
                        extra_prefixes = [GROUP_PREFIX.get(g, g) for g in extra_groups if g != "all"]
                        extra_cols = ["instrument", "report_date"]
                        for prefix in extra_prefixes:
                            extra_cols.extend(
                                [
                                    f"{prefix}_net_pct_oi",
                                    f"{prefix}_z",
                                    f"{prefix}_deleveraging_z",
                                    f"{prefix}_forced",
                                ]
                            )
                        extra_cols = [c for c in extra_cols if c in df.columns]
                        extra = df[extra_cols].copy()

                        for c in extra.columns:
                            if c.endswith("_net_pct_oi") or c.endswith("_z") or c.endswith("_deleveraging_z"):
                                extra[c] = pd.to_numeric(extra[c], errors="coerce")

                        # Create friendly column names for extra groups
                        prefix_labels = {
                            "dealer": "Dealer",
                            "asset_mgr": "Asset Mgr",
                            "other_rept": "Other Rept",
                            "nonrept": "Non-Rept",
                        }
                        extra_rename = {"instrument": "Instrument", "report_date": "Report Date"}
                        for prefix in extra_prefixes:
                            label = prefix_labels.get(prefix, prefix.title())
                            extra_rename[f"{prefix}_net_pct_oi"] = f"{label} Net % OI"
                            extra_rename[f"{prefix}_z"] = f"{label} Pos Z"
                            extra_rename[f"{prefix}_deleveraging_z"] = f"{label} Delev Z"
                            extra_rename[f"{prefix}_forced"] = f"{label} Forced"
                        extra = extra.rename(columns=extra_rename)

                        format_map = {}
                        pct_cols = [c for c in extra.columns if "Net % OI" in c]
                        z_cols = [c for c in extra.columns if "Pos Z" in c or "Delev Z" in c]
                        forced_cols = [c for c in extra.columns if "Forced" in c]
                        for c in pct_cols:
                            format_map[c] = "{:+.1f}%"
                        for c in z_cols:
                            format_map[c] = "{:+.2f}"

                        styled_extra = (
                            extra.style.format(format_map)
                            .applymap(color_positive_negative, subset=pct_cols)
                            .applymap(color_zscore, subset=z_cols)
                            .applymap(color_forced_flow, subset=forced_cols)
                        )
                        st.dataframe(styled_extra, width="stretch", hide_index=True)
            else:
                st.warning("No results returned.")

    else:
        st.write("Select a predefined instrument alias.")
        alias = st.selectbox("Instrument alias", options=sorted(INSTRUMENTS.keys()))
        market_name = INSTRUMENTS.get(alias)
        if market_name:
            st.caption(market_name)

        if market_name:
            group_view_map = {"Leveraged Funds": "lf"}
            view_groups = selected_groups
            if "all" in view_groups:
                view_groups = [g for g in CANONICAL_GROUPS if g != "lev_money"]
            for g in view_groups:
                if g in ("all", "lev_money"):
                    continue
                group_view_map[GROUP_LABEL.get(g, g)] = GROUP_PREFIX.get(g, g)

            view_group_label = st.selectbox("View group", options=list(group_view_map.keys()), index=0)
            prefix = group_view_map[view_group_label]

            with st.spinner("Fetching positioning time series..."):
                try:
                    df = cached_fetch_timeseries(
                        domain,
                        dataset_id,
                        token_value,
                        market_name,
                        start_str,
                        end_str,
                        groups_value,
                        z_window,
                        force_threshold,
                    )
                except Exception as e:
                    st.error(f"Error fetching time series: {e}")
                    df = None

            if df is not None and not df.empty:
                df = df.copy()
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
                for c in df.columns:
                    if c.endswith(("_net", "_net_pct_oi", "_z", "_d_net", "_d_net_pct_oi", "_d_z", "_deleveraging", "_deleveraging_z")):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.sort_values("report_date")

                valid_dates = df.dropna(subset=["report_date"])
                if valid_dates.empty:
                    st.warning("No valid report dates returned for this market.")
                    st.stop()

                latest = valid_dates.iloc[-1]
                st.subheader("Latest Reading")
                row1 = st.columns(3)
                row2 = st.columns(3)
                with row1[0]:
                    report_date = latest.get("report_date")
                    st.metric("Report Date", report_date.date().isoformat() if pd.notna(report_date) else "N/A")
                with row1[1]:
                    net = latest.get(f"{prefix}_net")
                    st.metric("Net Position", f"{net:,.0f}" if pd.notna(net) else "N/A")
                with row1[2]:
                    pct = latest.get(f"{prefix}_net_pct_oi")
                    st.metric("Net % Open Interest", f"{pct:+.1f}%" if pd.notna(pct) else "N/A")
                with row2[0]:
                    z = latest.get(f"{prefix}_z")
                    st.metric("Z-Score", f"{z:+.2f}" if pd.notna(z) else "N/A")
                with row2[1]:
                    dz = latest.get(f"{prefix}_deleveraging_z")
                    st.metric("Deleveraging Z", f"{dz:+.2f}" if pd.notna(dz) else "N/A")
                with row2[2]:
                    forced = latest.get(f"{prefix}_forced")
                    forced_str = str(forced) if forced is not None and not pd.isna(forced) else "N/A"
                    st.metric("Forced Flow", forced_str)

                st.divider()
                chart_df = df.set_index("report_date")
                st.write("**Net Position Over Time**")
                if f"{prefix}_net" in chart_df.columns:
                    st.line_chart(chart_df[f"{prefix}_net"], height=200)

                if f"{prefix}_net_pct_oi" in chart_df.columns:
                    st.write("**Net % Open Interest**")
                    st.line_chart(chart_df[f"{prefix}_net_pct_oi"], height=200)

                st.write("**Z-Score**")
                if f"{prefix}_z" in chart_df.columns:
                    st.line_chart(chart_df[f"{prefix}_z"], height=200)

                if f"{prefix}_deleveraging_z" in chart_df.columns:
                    st.write("**Deleveraging Z (proxy for forced flows)**")
                    st.line_chart(chart_df[f"{prefix}_deleveraging_z"], height=200)

                st.subheader("Recent History")
                display_cols = [
                    "report_date",
                    f"{prefix}_net",
                    f"{prefix}_net_pct_oi",
                    f"{prefix}_z",
                    f"{prefix}_deleveraging_z",
                    f"{prefix}_forced",
                ]
                if "open_interest" in df.columns:
                    display_cols.insert(2, "open_interest")
                display_cols = [c for c in display_cols if c in df.columns]
                recent = df.tail(20)[display_cols].copy()
                recent["report_date"] = recent["report_date"].dt.date

                format_map = {}
                if f"{prefix}_net" in recent.columns:
                    format_map[f"{prefix}_net"] = "{:,.0f}"
                if f"{prefix}_net_pct_oi" in recent.columns:
                    format_map[f"{prefix}_net_pct_oi"] = "{:+.1f}%"
                if f"{prefix}_z" in recent.columns:
                    format_map[f"{prefix}_z"] = "{:+.2f}"
                if f"{prefix}_deleveraging_z" in recent.columns:
                    format_map[f"{prefix}_deleveraging_z"] = "{:+.2f}"
                if "open_interest" in recent.columns:
                    format_map["open_interest"] = "{:,.0f}"

                # Rename columns for display
                history_rename = {
                    "report_date": "Date",
                    f"{prefix}_net": "Net Position",
                    f"{prefix}_net_pct_oi": "Net % Open Int",
                    f"{prefix}_z": "Position Z",
                    f"{prefix}_deleveraging_z": "Delev Z",
                    f"{prefix}_forced": "Forced Flow",
                    "open_interest": "Open Interest",
                }
                recent = recent.rename(columns=history_rename)
                format_map = {history_rename.get(k, k): v for k, v in format_map.items()}

                pct_cols = [c for c in recent.columns if "Net % Open Int" in c]
                z_cols = [c for c in recent.columns if "Position Z" in c or "Delev Z" in c]
                forced_cols = [c for c in recent.columns if "Forced" in c]
                styled_recent = (
                    recent.style.format(format_map)
                    .applymap(color_positive_negative, subset=pct_cols)
                    .applymap(color_zscore, subset=z_cols)
                    .applymap(color_forced_flow, subset=forced_cols)
                )
                st.dataframe(styled_recent, width="stretch", hide_index=True)


# =============================================================================
# PAGE: Breakout
# =============================================================================
elif st.session_state.current_page == "🔔 Breakout":
    st.header("Breakout Detector")
    st.caption("FX & Commodities: Tight congestion box breakouts with volume confirmation")

    if st.button("Refresh Data", key="refresh_breakout"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(ttl=300)
    def fetch_breakout():
        try:
            from breakout import get_data
            result = get_data()
            return result
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"}

    def render_position_bar(close, box_lower, box_upper, buffer):
        """Render an HTML bar showing price position within the box."""
        if box_lower is None or box_upper is None or box_lower >= box_upper:
            return ""

        # Calculate breakout levels
        up_breakout = box_upper + buffer
        down_breakout = box_lower - buffer

        # Full range includes buffer zones
        full_range = up_breakout - down_breakout
        if full_range <= 0:
            return ""

        # Calculate positions as percentages
        box_start_pct = ((box_lower - down_breakout) / full_range) * 100
        box_end_pct = ((box_upper - down_breakout) / full_range) * 100
        price_pct = ((close - down_breakout) / full_range) * 100
        price_pct = max(0, min(100, price_pct))

        return f'''
        <div style="position: relative; height: 24px; background: linear-gradient(to right,
            #ff1744 0%, #ff1744 {box_start_pct}%,
            #2d2d2d {box_start_pct}%, #2d2d2d {box_end_pct}%,
            #00c853 {box_end_pct}%, #00c853 100%);
            border-radius: 4px; margin: 4px 0;">
            <div style="position: absolute; left: {price_pct}%; top: 0; bottom: 0;
                width: 3px; background: white; border-radius: 2px;
                box-shadow: 0 0 4px rgba(255,255,255,0.8);"></div>
            <div style="position: absolute; left: {box_start_pct}%; top: 0; bottom: 0;
                width: 1px; background: rgba(255,255,255,0.3);"></div>
            <div style="position: absolute; left: {box_end_pct}%; top: 0; bottom: 0;
                width: 1px; background: rgba(255,255,255,0.3);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-top: 2px;">
            <span>▼ {down_breakout:.4f}</span>
            <span style="color: #666;">Box: {box_lower:.4f} - {box_upper:.4f}</span>
            <span>▲ {up_breakout:.4f}</span>
        </div>
        '''

    def render_progress_bar(value, max_value, color="#00c853", label=""):
        """Render a progress bar with percentage."""
        if max_value <= 0:
            pct = 0
        else:
            pct = min(100, (value / max_value) * 100)
        return f'''
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="flex: 1; height: 8px; background: #2d2d2d; border-radius: 4px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: {color}; border-radius: 4px;"></div>
            </div>
            <span style="font-size: 12px; color: #888; min-width: 60px;">{label}</span>
        </div>
        '''

    def render_distance_indicator(dist_up, dist_down):
        """Render visual distance indicators for breakout levels."""
        up_color = "#00c853" if dist_up is not None and abs(dist_up) < 1 else "#4a4a4a"
        down_color = "#ff1744" if dist_down is not None and abs(dist_down) < 1 else "#4a4a4a"

        up_str = f"{dist_up:+.2f}%" if dist_up is not None else "N/A"
        down_str = f"{dist_down:+.2f}%" if dist_down is not None else "N/A"

        # Highlight if close to breakout
        up_highlight = "font-weight: bold; text-shadow: 0 0 8px #00c853;" if dist_up is not None and abs(dist_up) < 1 else ""
        down_highlight = "font-weight: bold; text-shadow: 0 0 8px #ff1744;" if dist_down is not None and abs(dist_down) < 1 else ""

        return f'''
        <div style="display: flex; justify-content: space-around; padding: 4px 0;">
            <div style="text-align: center;">
                <div style="font-size: 18px; color: {up_color}; {up_highlight}">▲</div>
                <div style="font-size: 14px; color: {up_color}; {up_highlight}">{up_str}</div>
                <div style="font-size: 10px; color: #666;">to breakout</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 18px; color: {down_color}; {down_highlight}">▼</div>
                <div style="font-size: 14px; color: {down_color}; {down_highlight}">{down_str}</div>
                <div style="font-size: 10px; color: #666;">to breakout</div>
            </div>
        </div>
        '''

    def render_volume_gauge(vol_ratio, threshold):
        """Render a volume readiness gauge."""
        if vol_ratio is None:
            return '<div style="color: #666; font-size: 12px;">Volume: N/A</div>'

        pct = min(100, (vol_ratio / threshold) * 100)
        color = "#00c853" if vol_ratio >= threshold else "#ffc107" if pct >= 80 else "#666"
        status = "Ready" if vol_ratio >= threshold else "Building" if pct >= 80 else "Low"

        return f'''
        <div style="text-align: center;">
            <div style="font-size: 11px; color: #888; margin-bottom: 2px;">Volume</div>
            <div style="width: 60px; height: 60px; border-radius: 50%; border: 3px solid {color};
                display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                <div>
                    <div style="font-size: 14px; font-weight: bold; color: {color};">{vol_ratio:.2f}x</div>
                    <div style="font-size: 9px; color: #666;">/ {threshold:.2f}x</div>
                </div>
            </div>
            <div style="font-size: 10px; color: {color}; margin-top: 4px;">{status}</div>
        </div>
        '''

    with st.spinner("Fetching breakout data from Yahoo Finance..."):
        breakout_data = fetch_breakout()

    if "error" in breakout_data:
        st.error(f"Error: {breakout_data['error']}")
    else:
        signals = breakout_data.get("signals", [])
        in_box = breakout_data.get("in_box", [])
        near_misses = breakout_data.get("near_misses", [])
        not_in_box = breakout_data.get("not_in_box", [])

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confirmed Breakouts", len(signals))
        with col2:
            st.metric("In Consolidation", len(in_box))
        with col3:
            st.metric("Near Misses", len(near_misses))
        with col4:
            st.metric("Not in Box", len(not_in_box))

        st.divider()

        # Confirmed Breakouts
        if signals:
            st.subheader("🚀 Confirmed Breakouts")
            for sig in signals:
                dir_color = "#00c853" if sig["direction"] == "UP" else "#ff1744"
                dir_symbol = "▲" if sig["direction"] == "UP" else "▼"
                bg_color = "rgba(0, 200, 83, 0.1)" if sig["direction"] == "UP" else "rgba(255, 23, 68, 0.1)"

                st.markdown(f'''
                <div style="background: {bg_color}; border: 1px solid {dir_color}; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 20px; font-weight: bold; color: white;">{sig['name']}</span>
                            <span style="color: #888; margin-left: 8px;">({sig['market']})</span>
                        </div>
                        <div style="font-size: 24px; color: {dir_color}; font-weight: bold;">
                            {dir_symbol} {sig['direction']} BREAKOUT
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 12px; color: #ccc;">
                        <div><strong>Close:</strong> {sig['close']:.4f}</div>
                        <div><strong>Box:</strong> {sig['box_lower']:.4f} - {sig['box_upper']:.4f}</div>
                        <div><strong>Volume:</strong> {sig['vol_ratio']:.2f}x</div>
                        <div><strong>Date:</strong> {sig['date'][:10]}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            st.divider()

        # Assets in Consolidation Boxes - Visual Cards
        st.subheader("📦 Consolidation Boxes")
        if in_box:
            st.info(f"{len(in_box)} asset(s) in consolidation - watching for breakouts")

            # Create two columns for cards
            cols = st.columns(2)
            for idx, d in enumerate(in_box):
                with cols[idx % 2]:
                    # Determine if close to breakout
                    close_to_up = d['dist_to_up_breakout_pct'] is not None and abs(d['dist_to_up_breakout_pct']) < 1
                    close_to_down = d['dist_to_down_breakout_pct'] is not None and abs(d['dist_to_down_breakout_pct']) < 1
                    border_color = "#00c853" if close_to_up else "#ff1744" if close_to_down else "#444"

                    st.markdown(f'''
                    <div style="background: #1e1e1e; border: 2px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 16px; font-weight: bold; color: white;">{d['name']}</span>
                            <span style="color: #888; font-size: 12px;">{d['market']}</span>
                        </div>
                        <div style="color: #ccc; margin-bottom: 8px;">
                            <strong>Close:</strong> {d['close']:.4f}
                        </div>
                        {render_position_bar(d['close'], d['box_lower'], d['box_upper'], d['buffer'])}
                        {render_distance_indicator(d['dist_to_up_breakout_pct'], d['dist_to_down_breakout_pct'])}
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                            <div style="flex: 1;">
                                {render_progress_bar(d['days_in_box'] or 0, 60, "#ffc107", f"Day {d['days_in_box'] or 0}/60")}
                            </div>
                            <div style="margin-left: 12px;">
                                {render_volume_gauge(d['vol_ratio'], d['vol_threshold'])}
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.caption("No assets currently in consolidation boxes")

        # Near Misses
        st.subheader("⚠️ Near Misses")
        if near_misses:
            for d in near_misses:
                direction = "UP" if "up" in d["status"] else "DOWN"
                dir_color = "#00c853" if direction == "UP" else "#ff1744"
                dir_symbol = "▲" if direction == "UP" else "▼"

                st.markdown(f'''
                <div style="background: rgba(255, 193, 7, 0.1); border: 1px solid #ffc107; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: bold; color: white;">{d['name']}</span>
                            <span style="color: #888; margin-left: 8px;">({d['market']})</span>
                        </div>
                        <span style="color: {dir_color};">{dir_symbol} {direction}</span>
                    </div>
                    <div style="color: #ffc107; font-size: 13px; margin-top: 8px;">
                        ⚠️ {d['near_miss_reason']}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.caption("No near misses")

        # Assets Not in Consolidation - Compact View
        with st.expander(f"📊 Not in Consolidation ({len(not_in_box)})", expanded=False):
            if not_in_box:
                for d in not_in_box:
                    tight = d['tight_count'] if d['tight_count'] is not None else 0
                    threshold = d['tight_threshold']
                    pct = (tight / threshold) * 100 if threshold > 0 else 0

                    # Color based on how close to forming a box
                    progress_color = "#00c853" if pct >= 100 else "#ffc107" if pct >= 60 else "#666"

                    chg_5d = d['pct_change_5d']
                    chg_20d = d['pct_change_20d']
                    chg_5d_color = "#00c853" if chg_5d and chg_5d > 0 else "#ff1744" if chg_5d and chg_5d < 0 else "#666"
                    chg_20d_color = "#00c853" if chg_20d and chg_20d > 0 else "#ff1744" if chg_20d and chg_20d < 0 else "#666"
                    chg_5d_str = f"{chg_5d:+.2f}%" if chg_5d is not None else "N/A"
                    chg_20d_str = f"{chg_20d:+.2f}%" if chg_20d is not None else "N/A"

                    bb_pctl = d['bb_width_percentile']
                    vol_status = "Very Tight" if bb_pctl and bb_pctl <= 25 else "Tight" if bb_pctl and bb_pctl <= 40 else "Normal" if bb_pctl and bb_pctl <= 60 else "Elevated" if bb_pctl and bb_pctl <= 80 else "High" if bb_pctl else "N/A"
                    vol_color = "#00c853" if bb_pctl and bb_pctl <= 40 else "#ffc107" if bb_pctl and bb_pctl <= 60 else "#666"

                    st.markdown(f'''
                    <div style="background: #1a1a1a; border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: bold; color: white;">{d['name']}</span>
                                <span style="color: #666; margin-left: 8px; font-size: 12px;">{d['market']}</span>
                            </div>
                            <div style="font-size: 13px; color: #888;">
                                Close: <span style="color: white;">{d['close']:.4f}</span>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                            <div style="flex: 1;">
                                <div style="font-size: 11px; color: #666; margin-bottom: 2px;">Congestion Progress</div>
                                <div style="height: 6px; background: #2d2d2d; border-radius: 3px; overflow: hidden;">
                                    <div style="width: {pct}%; height: 100%; background: {progress_color};"></div>
                                </div>
                                <div style="font-size: 10px; color: {progress_color}; margin-top: 2px;">{tight}/{threshold} days</div>
                            </div>
                            <div style="text-align: center; margin: 0 16px;">
                                <div style="font-size: 10px; color: #666;">Volatility</div>
                                <div style="font-size: 12px; color: {vol_color};">{vol_status}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 11px;">
                                    <span style="color: #666;">5d:</span> <span style="color: {chg_5d_color};">{chg_5d_str}</span>
                                </div>
                                <div style="font-size: 11px;">
                                    <span style="color: #666;">20d:</span> <span style="color: {chg_20d_color};">{chg_20d_str}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                st.caption("Congestion = tight days count / threshold needed for box formation")
            else:
                st.caption("All assets are in consolidation boxes")


# =============================================================================
# PAGE: Portfolio Optimizer
# =============================================================================
elif st.session_state.current_page == "📈 Portfolio Optimizer":
    st.header("Portfolio Optimizer")
    st.caption("Beta-neutral portfolio construction with volatility targeting")

    # Initialize session state for optimization results
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None

    # Sidebar controls for Portfolio Optimizer
    st.sidebar.title("Optimization Settings")

    book_size = st.sidebar.number_input(
        "Book Size ($)",
        min_value=1000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
        format="%d",
        help="Total portfolio value in USD"
    )

    target_leverage = st.sidebar.slider(
        "Target Gross Leverage",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Target gross exposure as multiple of NAV (max 4.0x per script limits)"
    )

    # Show constraint limits
    with st.sidebar.expander("Constraint Limits"):
        st.caption("**Gross Limits:**")
        st.caption("- Total: 4.0x")
        st.caption("- FX: 2.0x")
        st.caption("- Commodities: 1.0x")
        st.caption("- Bonds: 3.0x (10yr equiv)")
        st.caption("")
        st.caption("**Equity Net:**")
        st.caption("- Min: -50%")
        st.caption("- Max: +100%")
        st.caption("")
        st.caption("**Position Limits:**")
        st.caption("- Long max: +20%")
        st.caption("- Short max: -10%")

    optimize_clicked = st.sidebar.button("Optimize Portfolio", type="primary", width="stretch")

    if optimize_clicked:
        with st.spinner("Downloading price data and running optimization..."):
            try:
                from portfolio_optimizer import get_data as get_portfolio_data
                result = get_portfolio_data(book=book_size, target_leverage=target_leverage)
                st.session_state.optimization_result = result
            except Exception as e:
                import traceback
                st.session_state.optimization_result = {"error": str(e), "traceback": traceback.format_exc()}

    # Display results
    data = st.session_state.optimization_result

    if data is None:
        st.info("Configure settings in the sidebar and click 'Optimize Portfolio' to run the optimizer.")
    elif "error" in data and data["error"]:
        st.error(f"Optimization failed: {data['error']}")
        if "traceback" in data:
            with st.expander("Error details"):
                st.code(data["traceback"])
    else:
        # Status check
        status = data.get("status", "unknown")
        if status != "optimal":
            st.warning(f"Optimization status: {status} - results may not be optimal")

        # Header metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            vol_daily = data.get("vol_daily", 0)
            vol_target = data.get("vol_target", 0.015)
            st.metric(
                "Daily Volatility",
                f"{vol_daily*100:.2f}%",
                delta=f"Target: {vol_target*100:.2f}%"
            )

        with col2:
            gross_lev = data.get("gross_leverage", 0)
            gross_max = data.get("gross_max", 4.0)
            st.metric(
                "Gross Leverage",
                f"{gross_lev:.2f}x",
                delta=f"Max: {gross_max:.1f}x"
            )

        with col3:
            exp = data.get("exposures", {})
            eq_net = exp.get("equity_net", 0)
            color = "normal" if -0.5 <= eq_net <= 1.0 else "inverse"
            st.metric("Equity Net", f"{eq_net*100:+.1f}%")

        with col4:
            if status == "optimal":
                st.success(f"Status: OPTIMAL")
            else:
                st.warning(f"Status: {status.upper()}")

        # Beta hedging info
        col1, col2 = st.columns(2)
        with col1:
            beta_long = data.get("beta_long_spy", 0)
            hedge_spy = data.get("hedge_spy_weight", 0)
            st.metric("Long Beta to SPY", f"{beta_long:+.4f}", delta=f"Hedge: {hedge_spy:+.4f} SPY")
        with col2:
            beta_short = data.get("beta_short_iwm", 0)
            hedge_iwm = data.get("hedge_iwm_weight", 0)
            st.metric("Short Beta to IWM", f"{beta_short:+.4f}", delta=f"Hedge: {hedge_iwm:+.4f} IWM")

        st.divider()

        # Tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs(["Weights", "Exposures", "Constraints", "Max Scaled"])

        # Weights Tab
        with tab1:
            st.subheader("Portfolio Weights")

            weights_df = data.get("weights_df")
            if weights_df is not None and not weights_df.empty:
                display_df = weights_df.copy()

                # Format weight column
                display_df["Weight %"] = display_df["weight"].apply(lambda x: f"{x*100:+.2f}%")

                # Format dollar weight if present
                if "dollar_weight" in display_df.columns:
                    display_df["Dollar"] = display_df["dollar_weight"].apply(lambda x: f"${x:+,.0f}")

                # Format other columns
                display_df["Signal"] = display_df["signal"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
                display_df["Beta SPY"] = display_df["beta_spy"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                display_df["Beta IWM"] = display_df["beta_iwm"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                display_df["Vol"] = display_df["realized_vol"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

                # Select columns to display
                display_cols = ["ticker", "asset", "direction", "Signal", "Beta SPY", "Beta IWM", "Vol", "Weight %"]
                if "Dollar" in display_df.columns:
                    display_cols.append("Dollar")

                styled_df = display_df[display_cols].style.applymap(
                    color_positive_negative, subset=["Weight %"] + (["Dollar"] if "Dollar" in display_df.columns else [])
                )
                st.dataframe(styled_df, width="stretch", hide_index=True)

            # Hedge positions
            st.subheader("Hedge Positions")
            hedges_df = data.get("hedges_df")
            if hedges_df is not None and not hedges_df.empty:
                hedge_display = hedges_df.copy()
                hedge_display["Weight %"] = hedge_display["weight"].apply(lambda x: f"{x*100:+.2f}%")
                if "dollar_weight" in hedge_display.columns:
                    hedge_display["Dollar"] = hedge_display["dollar_weight"].apply(lambda x: f"${x:+,.0f}")

                display_cols = ["ticker", "type", "direction", "Weight %"]
                if "Dollar" in hedge_display.columns:
                    display_cols.append("Dollar")

                styled_hedges = hedge_display[display_cols].style.applymap(
                    color_positive_negative, subset=["Weight %"] + (["Dollar"] if "Dollar" in hedge_display.columns else [])
                )
                st.dataframe(styled_hedges, width="stretch", hide_index=True)

        # Exposures Tab
        with tab2:
            st.subheader("Asset Class Exposures")

            exp = data.get("exposures", {})

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Gross Exposures**")
                for asset_class in ["equity", "fx", "commodity", "bond"]:
                    gross_key = f"{asset_class}_gross"
                    gross = exp.get(gross_key, 0)
                    if gross > 0:
                        # Determine max for progress bar
                        max_val = {"equity": 4.0, "fx": 2.0, "commodity": 1.0, "bond": 3.0}.get(asset_class, 4.0)
                        pct = min(1.0, gross / max_val)
                        st.progress(pct, text=f"{asset_class.title()}: {gross*100:.1f}% (max {max_val*100:.0f}%)")
                    else:
                        st.progress(0.0, text=f"{asset_class.title()}: 0%")

                st.write("")
                st.metric("Total Gross", f"{exp.get('total_gross', 0)*100:.1f}%")

            with col2:
                st.write("**Net Exposures**")
                for asset_class in ["equity", "fx", "commodity", "bond"]:
                    net_key = f"{asset_class}_net"
                    net = exp.get(net_key, 0)
                    if net_key in exp:
                        color = "#00c853" if net >= 0 else "#ff1744"
                        st.metric(asset_class.title(), f"{net*100:+.1f}%")

                st.write("")
                st.metric("Total Net", f"{exp.get('total_net', 0)*100:+.1f}%")

        # Constraints Tab
        with tab3:
            st.subheader("Constraint Utilization")

            constraints = data.get("constraints", {})

            for name, constraint in constraints.items():
                utilization = constraint.get("utilization", 0)
                current = constraint.get("current", 0)
                limit = constraint.get("limit", 1)

                # Color based on how close to limit
                if utilization > 0.9:
                    status = "Near Limit"
                    color = "#ff1744"
                elif utilization > 0.7:
                    status = "Moderate"
                    color = "#ffc107"
                else:
                    status = "Healthy"
                    color = "#00c853"

                # Progress bar with status
                st.markdown(f"**{name}**")
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.progress(min(1.0, utilization), text=f"Current: {current*100:.1f}% / Limit: {limit*100:.0f}%")
                with col_b:
                    if utilization > 0.9:
                        st.error(status)
                    elif utilization > 0.7:
                        st.warning(status)
                    else:
                        st.success(status)

        # Max Scaled Tab
        with tab4:
            st.subheader("Max Scaled Portfolio")
            st.caption("Portfolio scaled to maximum leverage while respecting all constraints")

            max_scaled = data.get("max_scaled", {})

            if max_scaled:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scale Factor", f"{max_scaled.get('scale_factor', 1):.4f}x")
                with col2:
                    st.metric("Daily Volatility", f"{max_scaled.get('vol_daily', 0)*100:.2f}%")
                with col3:
                    binding = max_scaled.get("binding_constraint", "Unknown")
                    st.warning(f"Binding: {binding}")

                # Max scaled exposures
                st.write("**Max Scaled Exposures:**")
                max_exp = max_scaled.get("exposures", {})
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Total Gross", f"{max_exp.get('total_gross', 0)*100:.1f}%")
                with cols[1]:
                    st.metric("Equity Net", f"{max_exp.get('equity_net', 0)*100:+.1f}%")
                with cols[2]:
                    st.metric("FX Gross", f"{max_exp.get('fx_gross', 0)*100:.1f}%")
                with cols[3]:
                    st.metric("Commodity Gross", f"{max_exp.get('commodity_gross', 0)*100:.1f}%")

                # Max scaled weights
                st.write("**Max Scaled Weights:**")
                max_weights_df = max_scaled.get("weights_df")
                if max_weights_df is not None and not max_weights_df.empty:
                    max_display = max_weights_df.copy()
                    max_display["Weight %"] = max_display["weight"].apply(lambda x: f"{x*100:+.2f}%")
                    if "dollar_weight" in max_display.columns:
                        max_display["Dollar"] = max_display["dollar_weight"].apply(lambda x: f"${x:+,.0f}")

                    display_cols = ["ticker", "asset", "direction", "Weight %"]
                    if "Dollar" in max_display.columns:
                        display_cols.append("Dollar")

                    styled_max = max_display[display_cols].style.applymap(
                        color_positive_negative, subset=["Weight %"] + (["Dollar"] if "Dollar" in max_display.columns else [])
                    )
                    st.dataframe(styled_max, width="stretch", hide_index=True)
            else:
                st.info("No max scaled data available")


# =============================================================================
# PAGE: Momentum
# =============================================================================
elif st.session_state.current_page == "🚀 Momentum":
    st.header("Momentum Analysis")
    st.caption("ROC-based momentum metrics for portfolio tickers")

    if st.button("Refresh Data", key="refresh_momentum"):
        st.cache_data.clear()

    @st.cache_data(ttl=300)
    def fetch_momentum():
        try:
            from momentum import get_data
            return get_data()
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"}

    def color_momentum_threshold(val, threshold: float = 1.5):
        """Color momentum values: green if >= threshold, red if < threshold."""
        if pd.isna(val) or val == "N/A":
            return "color: gray"
        try:
            if isinstance(val, str):
                val = float(val.replace("%", "").replace("+", "").strip())
            if val >= threshold:
                return "color: #00c853; font-weight: bold"
            else:
                return "color: #ff1744; font-weight: bold"
        except (ValueError, TypeError):
            return "color: gray"

    with st.spinner("Fetching momentum data from Yahoo Finance..."):
        momentum_data = fetch_momentum()

    if "error" in momentum_data:
        st.error(f"Error: {momentum_data['error']}")
    else:
        results = momentum_data.get("results", [])
        count = momentum_data.get("count", 0)
        data_date = momentum_data.get("date")

        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tickers Analyzed", count)
        with col2:
            st.metric("As of", str(data_date) if data_date else "N/A")

        st.divider()

        if results:
            # Build DataFrame for display
            rows = []
            for r in results:
                rows.append({
                    "Ticker": r["ticker"],
                    "Benchmark": r["benchmark"],
                    "Close": f"{r['close']:.2f}",
                    "20d Avg 63d ROC (%)": f"{r['avg20_roc63']:+.2f}",
                    "42d Rel ROC (%)": f"{r['rel_roc42']:+.2f}",
                    "10d Avg Rel ROC (%)": f"{r['avg10_rel_roc']:+.2f}",
                })

            df = pd.DataFrame(rows)

            # Apply styling
            styled_df = df.style.applymap(
                lambda x: color_momentum_threshold(x, threshold=1.5),
                subset=["20d Avg 63d ROC (%)"]
            ).applymap(
                color_positive_negative,
                subset=["42d Rel ROC (%)", "10d Avg Rel ROC (%)"]
            )

            st.dataframe(styled_df, width="stretch", hide_index=True)

            # Legend
            st.caption("**Color coding:**")
            st.caption("- 20d Avg 63d ROC: Green if >= 1.5%, Red if < 1.5%")
            st.caption("- Relative ROC metrics: Green if positive (outperforming benchmark), Red if negative")
        else:
            st.warning("No momentum data available")


# =============================================================================
# PAGE: FX Model
# =============================================================================
elif st.session_state.current_page == "💱 FX Model":
    st.header("FX Macro Model")
    st.caption("Multi-factor model for currency pair forecasting using FRED, IMF, and BIS data")

    # Initialize session state for FX results
    if "fx_results" not in st.session_state:
        st.session_state.fx_results = None
        st.session_state.fx_pair = None

    # Import FX modules
    try:
        from src.currency_config import get_config, list_pairs
        from src.pipeline import run_pipeline
        fx_module_loaded = True
    except Exception as e:
        st.error(f"Failed to load FX module: {e}")
        fx_module_loaded = False

    if fx_module_loaded and fx_run_clicked and fx_selected_pair:
        # Parse horizons
        try:
            horizons = [int(x.strip()) for x in fx_horizons_str.split(",") if x.strip()]
        except ValueError:
            horizons = [12, 24]

        with st.spinner(f"Running FX model for {fx_selected_pair}... This may take a moment."):
            try:
                config = get_config(fx_selected_pair)
                cache_dir = PROJECT_ROOT / "fx" / "data_cache"
                outdir = PROJECT_ROOT / "fx" / "outputs" / fx_selected_pair.lower()
                cache_dir.mkdir(parents=True, exist_ok=True)
                outdir.mkdir(parents=True, exist_ok=True)

                results = run_pipeline(
                    config=config,
                    start="1970-01-01",
                    outdir=outdir,
                    cache_dir=cache_dir,
                    refresh=False,
                    use_bis=not fx_skip_bis,
                    bootstrap_draws=fx_bootstrap,
                    horizons=horizons,
                )
                st.session_state.fx_results = results
                st.session_state.fx_pair = fx_selected_pair
            except Exception as e:
                import traceback
                st.error(f"Model failed: {e}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                st.session_state.fx_results = None

    # Display results
    if fx_module_loaded:
        results = st.session_state.fx_results
        pair = st.session_state.fx_pair

        if results is None:
            st.info("Select a currency pair and click 'Run Model' to generate forecasts.")
        else:
            st.success(f"Model results for {pair}")

            # Get data from results
            latest_forecast = results.get("latest_forecast", {})
            panel = results.get("panel")
            reference_points = results.get("reference_points")
            models = results.get("models", {})

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Valuation", "Model Details", "Time Series"])

            # Tab 1: Forecast Summary
            with tab1:
                st.subheader("Forecast Summary")

                # Current spot
                first_horizon = list(latest_forecast.keys())[0] if latest_forecast else None
                if first_horizon:
                    spot_now = latest_forecast[first_horizon].get("spot_now", 0)
                    rer_z = latest_forecast[first_horizon].get("valuation_rer_z", 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Spot", f"{spot_now:.4f}")
                    with col2:
                        z_color = "inverse" if abs(rer_z) > 1.5 else "normal"
                        valuation = "Overvalued" if rer_z > 1.5 else "Undervalued" if rer_z < -1.5 else "Fair"
                        st.metric("RER Z-Score", f"{rer_z:+.2f}", delta=valuation, delta_color=z_color)

                st.divider()

                # Forecasts for each horizon
                for horizon, forecast in latest_forecast.items():
                    st.write(f"**{horizon}-Month Forecast**")

                    col1, col2, col3, col4 = st.columns(4)

                    point = forecast.get("point_level", 0)
                    levels = forecast.get("level_q05_q50_q95", {})
                    q05 = levels.get("q05", 0)
                    q50 = levels.get("q50", 0)
                    q95 = levels.get("q95", 0)

                    # Calculate expected return
                    if spot_now > 0:
                        exp_return = ((point / spot_now) - 1) * 100
                    else:
                        exp_return = 0

                    with col1:
                        st.metric("Point Forecast", f"{point:.4f}", delta=f"{exp_return:+.1f}%")
                    with col2:
                        st.metric("5th Percentile", f"{q05:.4f}")
                    with col3:
                        st.metric("Median", f"{q50:.4f}")
                    with col4:
                        st.metric("95th Percentile", f"{q95:.4f}")

                    # Visual range bar
                    if q05 > 0 and q95 > 0:
                        range_pct = ((spot_now - q05) / (q95 - q05)) * 100 if q95 != q05 else 50
                        range_pct = max(0, min(100, range_pct))
                        st.markdown(f'''
                        <div style="position: relative; height: 24px; background: linear-gradient(to right, #ff1744, #ffc107, #00c853); border-radius: 4px; margin: 8px 0;">
                            <div style="position: absolute; left: {range_pct}%; top: -2px; bottom: -2px; width: 4px; background: white; border-radius: 2px; box-shadow: 0 0 4px rgba(255,255,255,0.8);"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #888;">
                            <span>{q05:.4f}</span>
                            <span style="color: white;">Current: {spot_now:.4f}</span>
                            <span>{q95:.4f}</span>
                        </div>
                        ''', unsafe_allow_html=True)

                    st.write("")

            # Tab 2: Valuation
            with tab2:
                st.subheader("Valuation Analysis")

                if reference_points is not None and not reference_points.empty:
                    # Latest reference points
                    latest_ref = reference_points.iloc[-1]

                    st.write("**Spot vs Implied Levels (RER Reversion)**")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Current Spot", f"{latest_ref.get('spot', 0):.4f}")
                    with col2:
                        implied_med = latest_ref.get("spot_implied_median", 0)
                        if latest_ref.get("spot", 0) > 0:
                            diff_pct = ((implied_med / latest_ref.get("spot", 1)) - 1) * 100
                        else:
                            diff_pct = 0
                        st.metric("Implied (Median)", f"{implied_med:.4f}", delta=f"{diff_pct:+.1f}%")
                    with col3:
                        st.metric("Implied (25th pctl)", f"{latest_ref.get('spot_implied_p25', 0):.4f}")
                    with col4:
                        st.metric("Implied (75th pctl)", f"{latest_ref.get('spot_implied_p75', 0):.4f}")

                    st.divider()

                    # RER Z-score chart
                    if panel is not None and "rer_z" in panel.columns:
                        st.write("**Real Exchange Rate Z-Score (120-month rolling)**")
                        rer_chart = panel[["rer_z"]].dropna().tail(240)
                        if not rer_chart.empty:
                            st.line_chart(rer_chart, height=250)
                            st.caption("Positive = base currency overvalued, Negative = base currency undervalued")

                    # Reference points table
                    st.write("**Recent Reference Points**")
                    display_ref = reference_points.tail(12).copy()
                    for col in ["spot", "spot_implied_median", "spot_implied_p25", "spot_implied_p75"]:
                        if col in display_ref.columns:
                            display_ref[col] = display_ref[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    st.dataframe(display_ref, width="stretch", hide_index=False)
                else:
                    st.warning("No reference point data available")

            # Tab 3: Model Details
            with tab3:
                st.subheader("Model Coefficients & Performance")

                for horizon, model_data in models.items():
                    with st.expander(f"**{horizon}-Month Model**", expanded=(horizon == list(models.keys())[0])):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("R-squared", f"{model_data.get('r2', 0):.4f}")
                        with col2:
                            st.metric("Observations", f"{model_data.get('nobs', 0):,}")

                        # Coefficients table
                        params = model_data.get("params", {})
                        if params:
                            st.write("**Regression Coefficients**")
                            coef_rows = []
                            for name, value in params.items():
                                coef_rows.append({"Feature": name, "Coefficient": f"{value:+.4f}"})
                            coef_df = pd.DataFrame(coef_rows)
                            st.dataframe(coef_df, width="stretch", hide_index=True)

                        # Distribution stats
                        dist_log = model_data.get("dist_log_return", {})
                        if dist_log:
                            st.write("**Forecast Distribution (Log Returns)**")
                            dist_cols = st.columns(5)
                            with dist_cols[0]:
                                st.metric("5th pctl", f"{dist_log.get('q05', 0)*100:+.1f}%")
                            with dist_cols[1]:
                                st.metric("25th pctl", f"{dist_log.get('q25', 0)*100:+.1f}%")
                            with dist_cols[2]:
                                st.metric("Median", f"{dist_log.get('q50', 0)*100:+.1f}%")
                            with dist_cols[3]:
                                st.metric("75th pctl", f"{dist_log.get('q75', 0)*100:+.1f}%")
                            with dist_cols[4]:
                                st.metric("95th pctl", f"{dist_log.get('q95', 0)*100:+.1f}%")

            # Tab 4: Time Series
            with tab4:
                st.subheader("Historical Data")

                if panel is not None and not panel.empty:
                    # Spot rate
                    if "spot" in panel.columns:
                        st.write("**Spot Rate**")
                        st.line_chart(panel[["spot"]].dropna().tail(240), height=200)

                    # Carry
                    if "carry" in panel.columns:
                        st.write("**Carry (Interest Rate Differential)**")
                        st.line_chart(panel[["carry"]].dropna().tail(240), height=200)

                    # Oil momentum
                    if "oil_mom12" in panel.columns:
                        st.write("**Oil 12-Month Momentum**")
                        st.line_chart(panel[["oil_mom12"]].dropna().tail(240), height=200)

                    # Momentum
                    if "mom12" in panel.columns:
                        st.write("**Spot 12-Month Momentum**")
                        st.line_chart(panel[["mom12"]].dropna().tail(240), height=200)

                    # Data table
                    st.write("**Recent Data**")
                    display_cols = ["spot", "carry", "rer_z", "oil_mom12", "mom12", "vix"]
                    display_cols = [c for c in display_cols if c in panel.columns]
                    recent_data = panel[display_cols].tail(12).copy()
                    for col in recent_data.columns:
                        recent_data[col] = recent_data[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    st.dataframe(recent_data, width="stretch", hide_index=False)
                else:
                    st.warning("No time series data available")


# Auto-refresh logic
if auto_refresh:
    import time
    time.sleep(refresh_interval)
    st.rerun()
