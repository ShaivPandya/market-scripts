# Government Bond Yields Tracker (`government_bonds/`)

This script fetches and displays government bond yields for major countries, including historical changes over various time periods.

For the overall repo architecture (API + UI), see the repo root `README.md`.

## Features

- Fetches current yields for 2-year and 10-year government bonds (30-year for US only)
- Tracks yields for: United States, United Kingdom, Germany, and Japan
- Calculates yield changes over: 1 month, 3 months, 6 months, and 1 year
- Optional CSV export functionality
- Uses FRED API for reliable US Treasury data
- Uses Deutsche Bundesbank API for Germany bond data
- Uses local CSV files for UK and Japan bond data

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

- **United States**: FRED (Federal Reserve Economic Data) - Most reliable and accurate source for US Treasury yields
- **Germany**: Deutsche Bundesbank API (term-structure series)
- **United Kingdom / Japan**: Local CSV files in the `data/` directory

## Data File Requirements

For UK and Japan, place CSV files in the `data/` directory with the following naming convention:

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
- Make sure the UK/JP CSV files are present in the `data/` directory before running

## Data Series Used

### United States (FRED):
- **2-Year**: DGS2 (2-Year Treasury Constant Maturity Rate)
- **10-Year**: DGS10 (10-Year Treasury Constant Maturity Rate)
- **30-Year**: DGS30 (30-Year Treasury Constant Maturity Rate)

### Germany (Bundesbank):
- **2-Year**: `BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A`
- **10-Year**: `BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A`

### Other Countries (Local CSV Files):
- **United Kingdom**: 2-Year and 10-Year
- **Japan**: 2-Year and 10-Year
