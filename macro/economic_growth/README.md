# Economic Growth Dashboard

## Overview

The Economic Growth Dashboard is a Python-based tool that tracks key economic indicators across multiple time periods (1-month, 3-month, 6-month, and 1-year). It fetches live data from Yahoo Finance and displays performance metrics color-coded relative to benchmarks, helping investors identify economic trends and relative strength across asset classes.

## How It Works

### Data Collection

1. **Live Market Data**: Uses the `yfinance` library to fetch historical price data for equities, commodities, and currency pairs
2. **CRB Spot Index**: Reads from a local Excel file (`.xlsx` format) containing Moody's Analytics CRB Spot Index data
3. **Time Periods**: Calculates returns over 1-month (30 days), 3-month (91 days), 6-month (182 days), and 1-year (365 days) periods

### Performance Calculation

- For each asset, the dashboard calculates percentage returns by comparing current prices to historical prices at the specified lookback periods
- Returns are calculated as: `((current_price - past_price) / past_price) × 100`

### Benchmark Comparison

**Equities** are color-coded relative to their benchmarks:
- **US Equities** (Russell 2000, S&P 600, Consumer Staples, Transports, Banks, Korea): Compared to **S&P 500**
- **Europe Banks**: Compared to **STOXX 600**
- **Green** = Outperforming benchmark for that period
- **Red** = Underperforming benchmark for that period

**Commodities and Currencies** are colored simply by positive (green) or negative (red) performance.

### Display

Uses the `rich` library to create formatted tables with:
- Real-time progress indicators during data fetching
- Color-coded performance metrics
- Professional terminal-based visualization

## Key Indicators and Their Economic Importance

### 1. **Copper (HG=F)**
**Why It Matters**: Known as "Dr. Copper" for its PhD in economics, copper prices are highly sensitive to global economic activity due to widespread use in construction, electrical equipment, and manufacturing. Rising copper prices often signal economic expansion, while falling prices suggest contraction.

### 2. **CRB Spot Index**
**Why It Matters**: A broad index of non-traded industrial commodities including metals (tin, zinc), textiles (cotton, burlap), and agricultural inputs (tallow, hides). Unlike futures-based indices, the CRB Spot Index is less influenced by investor speculation and more directly reflects real industrial demand, making it a purer measure of production activity.

### 3. **Goldman Sachs Commodity Index (GSG)**
**Why It Matters**: Provides broad exposure to commodity markets including energy, metals, and agriculture. Helps identify inflationary pressures and global demand trends.

### 4. **Russell 2000 (IWM) & S&P 600 (IJR) - Small-Cap Stocks**
**Why They Matter**: Small-cap companies are more economically sensitive than large-caps because they:
- Have less diversified revenue streams
- Are more dependent on domestic economic conditions
- Have less pricing power during slowdowns
- Outperformance vs. S&P 500 suggests risk-on sentiment and economic optimism

### 5. **KBW Bank Index (KBWB) & Europe Banks (EXV1.DE)**
**Why They Matter**: Financial stocks are highly cyclical and economically sensitive because:
- Bank profitability depends on loan demand (which rises with economic growth)
- Net interest margins improve with steeper yield curves
- Credit quality deteriorates during recessions
- Strong bank performance signals confidence in economic growth and credit conditions

### 6. **DJ Transportation Index (IYT)**
**Why It Matters**: Trucking and transportation companies are direct beneficiaries of goods movement in the economy. The Dow Theory suggests transports should confirm trends in industrials - if goods are being produced (industrials rising), they must be shipped (transports rising). Divergence can signal economic weakness ahead.

### 7. **MSCI Korea (EWY)**
**Why It Matters**: South Korean equities are highly cyclical and export-dependent, making them sensitive to:
- Global manufacturing activity (semiconductors, electronics, autos)
- China's economic health (major trading partner)
- Global trade volumes
- Risk appetite in emerging markets
Historically used by macro investors as a proxy for global economic optimism.

### 8. **Consumer Staples (XLP)**
**Why It Matters**: Defensive sector that typically underperforms during economic expansions (relative to cyclicals) and outperforms during slowdowns. Strong relative performance may signal defensive positioning and economic concerns.

### 9. **STOXX 600 (^STOXX)**
**Why It Matters**: European equity benchmark used to assess relative strength of European markets and banks. Important for gauging European economic health and investor sentiment toward the region.

### 10. **AUD/JPY & CAD/JPY Currency Pairs**
**Why They Matter**:
- **Risk-On/Risk-Off Indicators**: Japanese Yen is a traditional safe-haven currency, while AUD and CAD are commodity currencies
- Rising AUD/JPY and CAD/JPY suggest risk appetite and commodity demand (economic growth)
- Falling pairs suggest risk aversion and economic uncertainty
- Both pairs tend to correlate with global risk sentiment and commodity cycles

## Usage

```bash
# Install dependencies
pip install yfinance rich pandas openpyxl

# Run with default CRB file (crb.xlsx)
python economic_growth.py

# Run with custom CRB file path
python economic_growth.py --crb-file path/to/your/crb.xlsx
```

## Interpreting the Dashboard

**Bullish Economic Signals:**
- Small-caps outperforming S&P 500 (green)
- Banks outperforming benchmarks (green)
- Copper and CRB Spot rising
- Transports strong
- Korea outperforming
- AUD/JPY and CAD/JPY rising

**Bearish/Defensive Signals:**
- Small-caps underperforming (red)
- Banks underperforming (red)
- Commodities falling
- Consumer Staples outperforming S&P 500
- Korea underperforming
- Currency pairs falling (Yen strength)
