"""Root conftest — shared fixtures for all tests."""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _load_env():
    """Load .env file once for the entire test session."""
    from dotenv import load_dotenv

    load_dotenv()
