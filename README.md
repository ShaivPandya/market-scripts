# Market Scripts Setup Guide

This guide will help you set up the project dependencies and environment variables.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- fredapi - FRED API client
- matplotlib - Data visualization
- numpy - Numerical computing
- pandas - Data manipulation
- requests - HTTP library
- rich - Terminal formatting
- yfinance - Yahoo Finance data

### 2. Environment Setup

This project uses environment variables to securely manage API keys.

#### Steps

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to the `.env` file:**
   ```bash
   # Open in your preferred editor
   nano .env
   # or
   code .env
   ```

3. **Get your FRED API key:**
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for a free account
   - Copy your API key into the `.env` file

### 3. Verify Installation

Test that everything is set up correctly:

```bash
python load_env.py
```

This will confirm that your environment variables are loaded properly.

## GUI (Streamlit) Usage

The project includes a Streamlit-based dashboard in `gui/app.py`.

### 1. Install GUI Dependencies

```bash
pip install -r gui/requirements.txt
```

### 2. Run the GUI

From the project root:

```bash
streamlit run gui/app.py
```

Streamlit will start a local server (by default at `http://localhost:8501`).
Make sure your `.env` file is set up (see Environment Setup) so the GUI can access API keys.

## Usage in Scripts

Add this to the top of your Python scripts:

```python
from load_env import load_env

# Load environment variables from .env file
load_env()

# Now you can use os.environ as usual
import os
fred_key = os.environ.get('FRED_API_KEY')
```

## Security Notes

- The `.env` file is ignored by git (see `.gitignore`)
- Never commit your actual API keys to version control
- The `.env.example` file is safe to commit as it contains no secrets
- If you accidentally commit a key, rotate it immediately

## Troubleshooting

If you encounter issues:

1. **Module not found errors**: Make sure you've installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**: Verify your `.env` file exists and contains valid API keys:
   ```bash
   python load_env.py
   ```

3. **Permission errors**: You may need to use `pip install --user -r requirements.txt` on some systems.
