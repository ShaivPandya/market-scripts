from __future__ import annotations

import sys
import types

import api.routers.overview as overview_router
import api.routers.thesis as thesis_router
from llm_utils import MODEL_MID, model_for_tier


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


def test_thesis_meta_includes_position_context_and_latest_evaluation(monkeypatch):
    import portfolio.portfolio_db as portfolio_db
    import portfolio.thesis_db as thesis_db

    monkeypatch.setattr(
        portfolio_db,
        "get_positions",
        lambda: [
            {"ticker": "nvda", "asset": "equity", "direction": "long", "conviction": 5},
            {"ticker": "tsm", "asset": "equity", "direction": "short", "conviction": 3},
        ],
    )
    monkeypatch.setattr(
        thesis_db,
        "get_all_thesis_meta",
        lambda: [
            {
                "ticker": "NVDA",
                "status": "active",
                "created_at": "2026-05-01T00:00:00+00:00",
                "updated_at": "2026-05-01T00:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(
        thesis_db,
        "get_latest_evaluations",
        lambda: [
            {
                "id": 1,
                "ticker": "nvda",
                "evaluated_at": "2026-05-02T12:00:00+00:00",
                "thesis_status": "strengthen",
                "technical_read": "supportive",
                "fundamental_read": "supportive",
                "action": "hold",
                "confidence": "high",
                "key_developments": [],
                "earnings_note": None,
                "risk_flag": None,
            }
        ],
    )

    payload = thesis_router.get_thesis_meta_all()

    assert payload[0]["ticker"] == "NVDA"
    assert payload[0]["direction"] == "long"
    assert payload[0]["conviction"] == 5
    assert payload[0]["last_evaluated"] == "2026-05-02T12:00:00+00:00"
    assert payload[0]["latest_evaluation"]["ticker"] == "NVDA"
    assert payload[1]["ticker"] == "TSM"
    assert payload[1]["status"] == "missing"
    assert payload[1]["direction"] == "short"
    assert payload[1]["latest_evaluation"] is None


def test_get_thesis_not_found(auth_client, monkeypatch, tmp_path):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    resp = auth_client.get("/api/v1/thesis/AAA")
    assert resp.status_code == 404


def test_generate_thesis_from_pdf(auth_client, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["model"] == model_for_tier(MODEL_MID, "anthropic")
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


def test_generate_thesis_from_markdown(auth_client, monkeypatch, tmp_path):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    def fail_pdf_call(*args, **kwargs):
        raise AssertionError("PDF generation should not run for markdown uploads")

    monkeypatch.setattr(thesis_router, "_call_llm_pdf", fail_pdf_call)

    resp = auth_client.post(
        "/api/v1/thesis/generate",
        data={"ticker": "mu"},
        files={
            "file": (
                "thesis.md",
                b"# Old Title\n\n## Thesis\n- Memory cycle improving\n",
                "text/markdown",
            )
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["ticker"] == "MU"
    assert payload["content"].startswith("# MU")
    assert "## Key Catalysts" in payload["content"]
    assert (thesis_dir / "MU.md").read_text(encoding="utf-8") == payload["content"]


def test_generate_overview_from_markdown(auth_client, monkeypatch, tmp_path):
    overview_dir = tmp_path / "investment_overviews"
    overview_dir.mkdir()
    monkeypatch.setattr(overview_router, "OVERVIEWS_DIR", overview_dir)

    def fail_pdf_call(*args, **kwargs):
        raise AssertionError("PDF generation should not run for markdown uploads")

    def fake_markdown_call(*, ticker: str, markdown: str):
        assert ticker == "MU"
        assert "Revenue growth: improving" in markdown
        return (
            "# MU Overview\n\n"
            "## Financials\n"
            "- **3-Year Avg. YoY Revenue Growth**: improving\n"
            "- **3-Year Avg. YoY EPS Growth**: Data not available in source document\n"
            "- **Debt**: Data not available in source document\n"
            "- **Reinvestment Costs**: Data not available in source document\n\n"
            "## Sensitivity to Extrinsic Factors\n\n"
            "| Factor | Sensitivity | Capacity to Deal |\n"
            "|--------|-------------|------------------|\n"
            "| Interest rates | Low | High |\n\n"
            "## Industry\n\n"
            "### Porter's Five Forces\n"
            "- **Competitive Rivalry — Medium**: Fragmented market\n\n"
            "### Supply Outlook\n"
            "- Supply is stable\n\n"
            "### Demand Outlook\n"
            "- Demand is improving\n"
        )

    monkeypatch.setattr(overview_router, "_call_llm_overview_pdf", fail_pdf_call)
    monkeypatch.setattr(overview_router, "_call_llm_overview_markdown", fake_markdown_call)

    resp = auth_client.post(
        "/api/v1/overview/generate",
        data={"ticker": "mu"},
        files={
            "file": (
                "overview.md",
                b"# Old Title\n\n## Financials\n- Revenue growth: improving\n",
                "text/markdown",
            )
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["ticker"] == "MU"
    assert payload["content"].startswith("# MU Overview")
    assert "### Porter's Five Forces" in payload["content"]
    assert (overview_dir / "MU.md").read_text(encoding="utf-8") == payload["content"]


def test_thesis_and_overview_uploads_reject_endpoint_oversized_files(auth_client, monkeypatch):
    monkeypatch.setattr(thesis_router, "MAX_UPLOAD_SIZE_BYTES", 4)
    monkeypatch.setattr(overview_router, "MAX_UPLOAD_SIZE_BYTES", 4)

    thesis = auth_client.post(
        "/api/v1/thesis/generate",
        data={"ticker": "mu"},
        files={"file": ("thesis.md", b"12345", "text/markdown")},
    )
    overview = auth_client.post(
        "/api/v1/overview/generate",
        data={"ticker": "mu"},
        files={"file": ("overview.md", b"12345", "text/markdown")},
    )

    assert thesis.status_code == 413
    assert overview.status_code == 413


def test_direct_markdown_save_models_have_pydantic_size_limits():
    thesis_schema = (
        thesis_router.SaveThesisRequest.model_json_schema()
        if hasattr(thesis_router.SaveThesisRequest, "model_json_schema")
        else thesis_router.SaveThesisRequest.schema()
    )
    overview_schema = (
        overview_router.SaveOverviewRequest.model_json_schema()
        if hasattr(overview_router.SaveOverviewRequest, "model_json_schema")
        else overview_router.SaveOverviewRequest.schema()
    )

    assert thesis_schema["properties"]["content"]["maxLength"] == thesis_router.MAX_UPLOAD_SIZE_BYTES
    assert overview_schema["properties"]["content"]["maxLength"] == overview_router.MAX_UPLOAD_SIZE_BYTES
