"""
Simple .env file loader for market-scripts.

Usage:
    from load_env import load_env
    load_env()  # Call this at the start of your script
"""

import os
from pathlib import Path


def load_env(env_path=None):
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. If None, searches for .env in current directory
                  and parent directories up to the project root.
    """
    if env_path is None:
        # Start from current file's directory and search up
        current = Path(__file__).parent
        while current != current.parent:
            env_file = current / '.env'
            if env_file.exists():
                env_path = env_file
                break
            current = current.parent

        # If still not found, check current working directory
        if env_path is None:
            cwd_env = Path.cwd() / '.env'
            if cwd_env.exists():
                env_path = cwd_env

    if env_path is None:
        # No .env file found, skip silently
        return

    env_path = Path(env_path)
    if not env_path.exists():
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value


if __name__ == "__main__":
    load_env()
    print("Environment variables loaded from .env file")

    # Show which keys are set (without exposing values)
    keys = ['FRED_API_KEY']
    for key in keys:
        if os.environ.get(key):
            print(f"✓ {key} is set")
        else:
            print(f"✗ {key} is not set")
