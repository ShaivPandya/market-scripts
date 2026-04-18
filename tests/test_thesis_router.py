from __future__ import annotations

import sys
import types

import api.routers.thesis as thesis_router
from llm_utils import MODEL_SONNET


def test_thesis_status(auth_client, monkeypatch, tmp_path):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    (thesis_dir / "AAA.md").write_text("# AAA\n\n## Thesis\n- good", encoding="utf-8")
    (thesis_dir / "BBB.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    import portfolio.portfolio_db as portfolio_db

    monkeypatch.setattr(
        portfolio_db,
        "get_positions",
        lambda: [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}],
    )

    resp = auth_client.get("/api/v1/thesis/status")
    assert resp.status_code == 200
    assert resp.json() == {"AAA": "populated", "BBB": "empty", "CCC": "missing"}


def test_get_thesis_not_found(auth_client, monkeypatch, tmp_path):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    resp = auth_client.get("/api/v1/thesis/AAA")
    assert resp.status_code == 404


def test_generate_thesis_from_pdf(auth_client, monkeypatch, tmp_path):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["model"] == MODEL_SONNET
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "# MU\n\n"
                            "## Thesis\n- Memory cycle improving\n\n"
                            "## Key Catalysts\n- HBM demand\n\n"
                            "## Risk Factors\n- Pricing pressure"
                        ),
                    }
                ],
                "stop_reason": "end_turn",
            }

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic))

    resp = auth_client.post(
        "/api/v1/thesis/generate",
        data={"ticker": "mu"},
        files={"file": ("deck.pdf", b"%PDF-1.4\nfake content\n", "application/pdf")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["ticker"] == "MU"
    assert "## Thesis" in payload["content"]
    assert (thesis_dir / "MU.md").exists()
