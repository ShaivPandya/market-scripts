# Government Bonds (`government_bonds/`)

This package has two fixed-income surfaces:

- Web/API yield-curve and bond-dashboard data for the Talisman app.
- A standalone terminal tracker in `government_bond_yields.py` that prints and optionally exports 2-year, 10-year, and US 30-year yield changes.

For the overall repo architecture (API + UI), see the repo root `README.md`.

## Where this is used

- FastAPI:
  - `GET /api/yield-curve`
  - `GET /api/bond-dashboard`
- React UI:
  - “Yield Curve” page
  - “Bond Dashboard” page
- Standalone CLI:
  - `python3 government_bonds/government_bond_yields.py`

## Features

- Yield-curve API returns current and historical points for 3M, 6M, 1Y, 2Y, 5Y, 10Y, and 30Y where the source supports them.
- Bond-dashboard API returns past-year 2Y, 10Y, and 30Y series for US, UK, Germany, and Japan.
- Standalone tracker calculates 1-month, 3-month, 6-month, and 1-year changes.
- Standalone tracker supports optional CSV export.
- US data comes from FRED.
- Germany data comes from Deutsche Bundesbank.
- Web dashboards fetch UK data from the Bank of England and Japan data from Japan MOF.
- The standalone tracker still uses local CSV files for UK and Japan.

## Installation

1. Install the required dependencies:

```bash
# From the repo root:
pip install -r requirements.txt
```

2. Get a free FRED API key:
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for a free account
   - Request an API key (instant approval)

3. Set your FRED API key as an environment variable:

```bash
export FRED_API_KEY='your_api_key_here'
```

This repo also supports `.env` files (recommended for local dev):

```bash
cp .env.example .env
# edit .env and set FRED_API_KEY=...
```

Or add it to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo "export FRED_API_KEY='your_api_key_here'" >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Display bond yields in terminal:

```bash
python3 government_bonds/government_bond_yields.py
```

### Export data to CSV:

```bash
python3 government_bonds/government_bond_yields.py --export
```

This will create a `government_bond_yields.csv` file with all the data.

## Data Sources

- **United States**: FRED Treasury constant-maturity series
- **Germany**: Deutsche Bundesbank term-structure API
- **United Kingdom**:
  - Web dashboards: Bank of England GLC nominal yield curve ZIP
  - Standalone tracker: local CSV files in `data/`
- **Japan**:
  - Web dashboards: Ministry of Finance JGB historical CSV
  - Standalone tracker: local CSV files in `data/`

## Data File Requirements

For the standalone tracker’s UK and Japan paths, place CSV files in the `data/`
directory with the following naming convention:

```
Download Data - BOND_BX_XTUP_TMBMK{COUNTRY_CODE}-{MATURITY}.csv
```

Where:
- **Country Codes**: GB (United Kingdom), JP (Japan)
- **Maturities**: 02Y (2-year), 10Y (10-year)

Example filenames:
- `Download Data - BOND_BX_XTUP_TMBMKGB-02Y.csv` (UK 2-year)
- `Download Data - BOND_BX_XTUP_TMBMKJP-02Y.csv` (Japan 2-year)

### CSV File Format

The CSV files should have the following columns:
- `Date` (format: MM/DD/YYYY)
- `Open` (with % symbol, e.g., "2.105%")
- `High` (with % symbol)
- `Low` (with % symbol)
- `Close` (with % symbol)

See `government_bonds/data/README.md` for the expected filenames and an example.

## Notes

- Yields are expressed as percentages
- Changes are expressed in basis points (bps)
- US Treasury data from FRED is highly reliable and updated daily
- Germany data is fetched live from Deutsche Bundesbank
- Without a FRED API key, US Treasury data will not be available
- Make sure the UK/JP CSV files are present in the `data/` directory before running `government_bond_yields.py`; the web dashboards do not use those files.

## Standalone Tracker Series

### United States (FRED):
- **2-Year**: DGS2 (2-Year Treasury Constant Maturity Rate)
- **10-Year**: DGS10 (10-Year Treasury Constant Maturity Rate)
- **30-Year**: DGS30 (30-Year Treasury Constant Maturity Rate)

### Germany (Bundesbank):
- **2-Year**: `BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A`
- **10-Year**: `BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A`

### UK/Japan Local CSV Files:
- **United Kingdom**: 2-Year and 10-Year
- **Japan**: 2-Year and 10-Year
