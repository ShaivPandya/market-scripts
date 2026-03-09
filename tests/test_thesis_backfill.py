"""Tests for portfolio/thesis_backfill.py -- markdown parsing into structured entities."""

from __future__ import annotations

from pathlib import Path

import pytest

import portfolio.core_db as core_db
from portfolio.thesis_backfill import _extract_label_and_description, _parse_bullets, backfill_from_markdown


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Point core_db at a temporary database for every test."""
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "test_core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield
    if core_db._conn:
        try:
            core_db._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(core_db, "_conn", None)


SAMPLE_THESIS = """# MU
## Thesis
- Micron is the sole U.S.-based advanced memory manufacturer
- AI-driven data center buildout is generating structurally elevated demand

## Key Catalysts
- **HBM ramp:** HBM3 fully sold out; HBM4 entering production
- **Beat-and-raise earnings:** Management has established a pattern of conservative guidance
- **AI capex cycle:** Hyperscalers treating AI infrastructure spend as existential
- **CHIPS Act tailwinds:** Primary beneficiary of domestic semiconductor subsidies

## Risk Factors
- **Cyclicality:** Memory is acutely cyclical — prior downturn erased ~50% of revenue
- **Customer concentration:** ~50% of revenue from approximately 10 customers
- **AI spending deceleration:** Pullback in hyperscaler capex would reduce HBM demand
"""


def test_parse_bullets_key_catalysts():
    bullets = _parse_bullets(SAMPLE_THESIS, "Key Catalysts")
    assert len(bullets) == 4
    assert "HBM ramp" in bullets[0]


def test_parse_bullets_risk_factors():
    bullets = _parse_bullets(SAMPLE_THESIS, "Risk Factors")
    assert len(bullets) == 3
    assert "Cyclicality" in bullets[0]


def test_parse_bullets_thesis():
    bullets = _parse_bullets(SAMPLE_THESIS, "Thesis")
    assert len(bullets) == 2


def test_parse_bullets_missing_section():
    bullets = _parse_bullets(SAMPLE_THESIS, "Nonexistent Section")
    assert bullets == []


def test_extract_label_and_description():
    label, desc = _extract_label_and_description("**HBM ramp:** HBM3 fully sold out; HBM4 entering production")
    assert label == "HBM ramp"
    assert "HBM3" in desc

    label2, desc2 = _extract_label_and_description("Plain bullet without bold")
    assert len(label2) > 0
    assert desc2 == "Plain bullet without bold"


def test_backfill_creates_catalysts_and_kill_conditions(tmp_path):
    theses_dir = tmp_path / "investment_theses"
    theses_dir.mkdir()
    (theses_dir / "MU.md").write_text(SAMPLE_THESIS)

    result = backfill_from_markdown(theses_dir)
    assert "MU" in result
    assert result["MU"]["catalysts"] == 4
    assert result["MU"]["kill_conditions"] == 3

    catalysts = core_db.get_catalysts("MU")
    assert len(catalysts) == 4
    assert all(c["created_by"] == "backfill" for c in catalysts)

    kcs = core_db.get_kill_conditions("MU")
    assert len(kcs) == 3
    assert all(kc["created_by"] == "backfill" for kc in kcs)


def test_backfill_skips_already_backfilled(tmp_path):
    theses_dir = tmp_path / "investment_theses"
    theses_dir.mkdir()
    (theses_dir / "MU.md").write_text(SAMPLE_THESIS)

    result1 = backfill_from_markdown(theses_dir)
    assert result1["MU"]["catalysts"] == 4

    result2 = backfill_from_markdown(theses_dir)
    assert "MU" not in result2  # Skipped because already backfilled


def test_backfill_skips_empty_files(tmp_path):
    theses_dir = tmp_path / "investment_theses"
    theses_dir.mkdir()
    (theses_dir / "EMPTY.md").write_text("")

    result = backfill_from_markdown(theses_dir)
    assert "EMPTY" not in result


def test_backfill_skips_tbd_bullets(tmp_path):
    theses_dir = tmp_path / "investment_theses"
    theses_dir.mkdir()
    (theses_dir / "STUB.md").write_text(
        "# STUB\n## Thesis\n- TBD\n\n## Key Catalysts\n- TBD\n\n## Risk Factors\n- TBD\n"
    )

    result = backfill_from_markdown(theses_dir)
    assert result.get("STUB", {}).get("catalysts", 0) == 0
