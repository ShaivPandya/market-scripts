# Quick Setup Guide

## Step 1: Install Dependencies

```bash
cd bonds/government_bonds
pip install -r requirements.txt
```

## Step 2: Get FRED API Key (Required for US Treasury Data)

1. Go to: https://fred.stlouisfed.org/
2. Click "My Account" → "API Keys" (or go directly to https://fred.stlouisfed.org/docs/api/api_key.html)
3. Sign up for a free account if you don't have one
4. Request an API key (instant approval, no credit card required)
5. Copy your API key

## Step 3: Set Your API Key

**Option A: Temporary (current terminal session only)**
```bash
export FRED_API_KEY='your_api_key_here'
```

**Option B: Permanent (recommended)**

For zsh (macOS default):
```bash
echo "export FRED_API_KEY='your_api_key_here'" >> ~/.zshrc
source ~/.zshrc
```

For bash:
```bash
echo "export FRED_API_KEY='your_api_key_here'" >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Run the Script

```bash
python3 government_bond_yields.py
```

Or export to CSV:
```bash
python3 government_bond_yields.py --export
```

## Verify Your Setup

Check if your FRED API key is set:
```bash
echo $FRED_API_KEY
```

If it prints your key, you're all set!

## Troubleshooting

**Error: "FRED_API_KEY not set"**
- Make sure you've exported the environment variable
- Try closing and reopening your terminal after adding to ~/.zshrc or ~/.bashrc

**Error: "Package not installed"**
- Run: `pip install -r requirements.txt`
- If using Python 3 specifically: `pip3 install -r requirements.txt`

**Error: "401 Unauthorized" from FRED**
- Check that your API key is correct
- Verify your FRED account is active
- Request a new API key if needed
