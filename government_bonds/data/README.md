# Bond Data Files

This directory contains CSV files with historical government bond yield data for
UK and Japan.

These files are read only by the standalone terminal tracker
`government_bonds/government_bond_yields.py`. The app-facing yield-curve and
bond-dashboard APIs fetch UK data from the Bank of England and Japan data from
Japan MOF instead.

## Required Files

Place the following CSV files in this directory:

### United Kingdom
- `Download Data - BOND_BX_XTUP_TMBMKGB-02Y.csv` (2-year bond)
- `Download Data - BOND_BX_XTUP_TMBMKGB-10Y.csv` (10-year bond)

### Japan
- `Download Data - BOND_BX_XTUP_TMBMKJP-02Y.csv` (2-year bond)
- `Download Data - BOND_BX_XTUP_TMBMKJP-10Y.csv` (10-year bond)

## File Format

Each CSV file should have the following structure:

```csv
Date,Open,High,Low,Close
01/09/2026,"2.105%","2.132%","2.101%","2.116%"
01/08/2026,"2.095%","2.118%","2.095%","2.105%"
...
```

### Requirements:
- **Date**: MM/DD/YYYY format
- **Close**: Bond yield percentage with % symbol (e.g., "2.116%")
- The script will automatically parse the percentage values

## Data Sources

These files typically come from a MarketWatch “Download Data” export.
Germany is fetched live from Deutsche Bundesbank and does not require local CSV files.

## Notes

- The script requires at least 1 year of historical data for accurate change calculations
- Missing files will be reported as errors when running the standalone tracker
- The most recent date in each file will be used as the current yield

If you want to use different filenames, update `BOND_DATA_FILES` in `government_bonds/government_bond_yields.py`.
