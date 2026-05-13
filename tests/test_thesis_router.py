from __future__ import annotations

import sys
import types

import pytest

import api.routers.management_quality as management_quality_router
import api.routers.overview as overview_router
import api.routers.thesis as thesis_router
from llm_utils import MODEL_MID, model_for_tier


@pytest.fixture
def temp_core_db(tmp_path, monkeypatch):
    import portfolio.core_db as core_db

    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "DB_PATH", tmp_path / "core.db")
    monkeypatch.setattr(core_db, "_conn", None)
    yield core_db
    if core_db._conn:
        core_db._conn.close()
    monkeypatch.setattr(core_db, "_conn", None)


def _use_document_generation_warm_worker(monkeypatch):
    monkeypatch.setenv("ASYNC_DISPATCH_BACKEND_DOCUMENT_GENERATION", "warm_worker")


def _finish_document_generation_job(auth_client, job_id: str) -> dict:
    from api.async_job_runner import perform_job

    perform_job(job_id)
    polled = auth_client.get(f"/api/v1/document-generation/async/{job_id}")
    assert polled.status_code == 200
    payload = polled.json()
    assert payload["status"] == "done"
    return payload["result"]


def _document_generation_upload_path(job_id: str):
    from api.document_generation_jobs import _local_path_for_storage_key
    from api.job_queue import get_job

    row = get_job(job_id)
    assert row is not None
    payload = row["payload_json"]
    return _local_path_for_storage_key(payload["storage_key"])


def test_parse_overview_supply_chain_tables_and_placeholders():
    parsed = overview_router.parse_overview_markdown(
        """# MU Overview

## Industry

### Supply Chain

#### Key Suppliers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| ASML | EUV lithography equipment | Material capex supplier |
| Data not available | | | |

#### Key Customers
| Entity | Relationship | Exposure | Notes |
|--------|--------------|----------|-------|
| Nvidia | HBM customer | Significant | AI accelerator demand. |
| AMD | HBM customer | | GPU ramp. |
"""
    )

    assert parsed is not None
    assert parsed["supply_chain"] == {
        "suppliers": [
            {
                "name": "ASML",
                "relationship": "EUV lithography equipment",
                "exposure": "Material capex supplier",
                "notes": None,
            }
        ],
        "customers": [
            {
                "name": "Nvidia",
                "relationship": "HBM customer",
                "exposure": "Significant",
                "notes": "AI accelerator demand.",
            },
            {"name": "AMD", "relationship": "HBM customer", "exposure": None, "notes": "GPU ramp."},
        ],
    }


def test_parse_overview_omits_empty_supply_chain():
    parsed = overview_router.parse_overview_markdown(
        """# MU Overview

## Industry

### Supply Chain
Data not available in source document.
"""
    )

    assert parsed is None


def test_parse_overview_sensitivity_supports_capacity_rationale_column():
    parsed = overview_router.parse_overview_markdown(
        """# MU Overview

## Sensitivity to Extrinsic Factors
| Factor | Sensitivity | Capacity to Deal | Rationale |
|--------|-------------|------------------|-----------|
| Memory pricing | High | Medium-high | Contracted supply helps, but spot pricing still moves margins. |
"""
    )

    assert parsed is not None
    assert parsed["sensitivity"] == [
        {
            "factor": "Memory pricing",
            "sensitivity": "High",
            "capacity": "Medium-high",
            "rationale": "Contracted supply helps, but spot pricing still moves margins.",
        }
    ]


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


def test_generate_thesis_from_pdf(auth_client, monkeypatch, tmp_path, temp_core_db):
    _use_document_generation_warm_worker(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    import llm_utils

    monkeypatch.setattr(llm_utils, "_stored_provider", lambda: None)
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)
    import portfolio.thesis_sync as thesis_sync

    monkeypatch.setattr(
        thesis_sync,
        "_thesis_paths",
        lambda ticker: (thesis_dir / f"{ticker}.md", f"live/theses/{ticker}.md"),
    )
    indexed: list[dict] = []
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))
    llm_calls = []

    class FakeMessages:
        def create(self, **kwargs):
            llm_calls.append(kwargs)
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
    assert resp.status_code == 202
    assert resp.headers["location"].startswith("/api/v1/document-generation/async/")
    queued = resp.json()
    assert queued["status"] == "queued"
    assert queued["timeout_s"] == 1200
    assert llm_calls == []

    payload = _finish_document_generation_job(auth_client, queued["job_id"])
    assert payload["status"] == "applied"
    assert payload["ticker"] == "MU"
    assert "## Thesis" in payload["proposed_change"]["content"]
    assert len(llm_calls) == 1
    assert (thesis_dir / "MU.md").exists()
    assert indexed[0]["doc_type"] == "thesis"
    assert indexed[0]["doc_id"] == "thesis-MU"
    assert len(temp_core_db.get_catalysts("MU")) == 1
    assert len(temp_core_db.get_kill_conditions("MU")) == 1


def test_generate_thesis_from_markdown(auth_client, monkeypatch, tmp_path, temp_core_db):
    _use_document_generation_warm_worker(monkeypatch)
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)
    import portfolio.thesis_sync as thesis_sync

    monkeypatch.setattr(
        thesis_sync,
        "_thesis_paths",
        lambda ticker: (thesis_dir / f"{ticker}.md", f"live/theses/{ticker}.md"),
    )
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: None)

    def fail_pdf_call(*args, **kwargs):
        raise AssertionError("PDF generation should not run for markdown uploads")

    monkeypatch.setattr(thesis_router, "_call_llm_pdf", fail_pdf_call)

    resp = auth_client.post(
        "/api/v1/thesis/generate",
        data={"ticker": "mu"},
        files={
            "file": (
                "thesis.md",
                (
                    b"# Old Title\n\n"
                    b"## Thesis\n- Memory cycle improving\n\n"
                    b"## Key Catalysts\n- **HBM ramp:** HBM3 fully sold out\n\n"
                    b"## Risk Factors\n- **Pricing pressure:** DRAM pricing rolls over\n"
                ),
                "text/markdown",
            )
        },
    )
    assert resp.status_code == 202
    payload = _finish_document_generation_job(auth_client, resp.json()["job_id"])
    assert payload["status"] == "applied"
    assert payload["ticker"] == "MU"
    assert payload["proposed_change"]["content"].startswith("# MU")
    assert "## Key Catalysts" in payload["proposed_change"]["content"]
    assert (thesis_dir / "MU.md").read_text(encoding="utf-8") == payload["proposed_change"]["content"]
    assert len(temp_core_db.get_catalysts("MU")) == 1
    assert len(temp_core_db.get_kill_conditions("MU")) == 1


def test_save_thesis_syncs_catalysts_kill_conditions_and_claims(auth_client, monkeypatch, tmp_path, temp_core_db):
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    monkeypatch.setattr(thesis_router, "THESES_DIR", thesis_dir)

    import portfolio.thesis_sync as thesis_sync

    monkeypatch.setattr(
        thesis_sync,
        "_thesis_paths",
        lambda ticker: (thesis_dir / f"{ticker}.md", f"live/theses/{ticker}.md"),
    )

    resp = auth_client.put(
        "/api/v1/thesis/MU",
        json={
            "content": (
                "# MU\n\n"
                "## Thesis\n- Memory demand improves\n\n"
                "## Key Catalysts\n- **HBM ramp:** HBM3 fully sold out\n\n"
                "## Risk Factors\n- **AI spending deceleration:** Capex slows\n"
            ),
            "apply": True,
            "approval_note": "Apply thesis",
        },
    )

    assert resp.status_code == 200
    assert len(temp_core_db.get_catalysts("MU")) == 1
    assert len(temp_core_db.get_kill_conditions("MU")) == 1
    claims = temp_core_db.get_thesis_claims("MU")
    assert len(claims) == 1
    assert claims[0]["claim"] == "Memory demand improves"


def test_generate_overview_from_markdown(auth_client, monkeypatch, tmp_path):
    _use_document_generation_warm_worker(monkeypatch)
    overview_dir = tmp_path / "investment_overviews"
    overview_dir.mkdir()
    monkeypatch.setattr(overview_router, "OVERVIEWS_DIR", overview_dir)

    def fail_pdf_call(*args, **kwargs):
        raise AssertionError("PDF generation should not run for markdown uploads")

    markdown_calls = []

    def fake_markdown_call(*, ticker: str, markdown: str):
        markdown_calls.append(markdown)
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
            "### Supply Chain\n\n"
            "#### Key Suppliers\n"
            "| Entity | Relationship | Exposure | Notes |\n"
            "|--------|--------------|----------|-------|\n"
            "| ASML | Lithography equipment | Material capex supplier | EUV systems. |\n\n"
            "#### Key Customers\n"
            "| Entity | Relationship | Exposure | Notes |\n"
            "|--------|--------------|----------|-------|\n"
            "| Nvidia | HBM customer | Significant | AI accelerator demand. |\n"
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
    assert resp.status_code == 202
    assert markdown_calls == []
    job_id = resp.json()["job_id"]
    upload_path = _document_generation_upload_path(job_id)
    assert upload_path.exists()
    payload = _finish_document_generation_job(auth_client, job_id)
    assert payload["status"] == "ok"
    assert payload["ticker"] == "MU"
    assert payload["content"].startswith("# MU Overview")
    assert "### Porter's Five Forces" in payload["content"]
    parsed = overview_router.parse_overview_markdown(payload["content"])
    assert parsed is not None
    assert parsed["supply_chain"]["suppliers"][0]["name"] == "ASML"
    assert parsed["supply_chain"]["customers"][0]["name"] == "Nvidia"
    assert (overview_dir / "MU.md").read_text(encoding="utf-8") == payload["content"]
    assert len(markdown_calls) == 1
    assert not upload_path.exists()


def test_parse_management_quality_filters_placeholders_and_splits_responses():
    parsed = management_quality_router.parse_management_quality_markdown(
        "# NVDA Management Quality\n\n"
        "## Executive Summary\n"
        "- **Overall Rating**: **Strong**\n"
        "- **Owner Mindset**: **Strong** -- Management used buybacks well.\n"
        "- **Business Value Understanding**: **Mixed** — Some gaps remain.\n\n"
        "## Management Scorecard\n"
        "| Question | Rating | Evidence |\n"
        "|----------|--------|----------|\n"
        "| Do managers think and act like owners? | **Strong** | Buybacks. |\n\n"
        "## Most Impressive Accomplishments\n"
        "- **AI demand ramp (2024)**: Delivered accelerated revenue growth.\n"
        "- --\n\n"
        "## Biggest Setbacks and Responses\n"
        "- **Gaming correction (2023)**: Demand fell below guidance. **Response**: **Handled well** -- Reset guidance and reduced channel inventory.\n"
        "- --\n"
    )

    assert parsed is not None
    assert parsed["summary"] == {
        "overall_rating": "Strong",
        "owner_mindset": {"rating": "Strong", "text": "Management used buybacks well."},
        "business_value_understanding": {"rating": "Mixed", "text": "Some gaps remain."},
    }
    assert parsed["scorecard"] == [
        {"question": "Do managers think and act like owners?", "rating": "Strong", "evidence": "Buybacks."}
    ]
    assert parsed["accomplishments"] == [
        {
            "title": "AI demand ramp (2024)",
            "text": "Delivered accelerated revenue growth.",
        }
    ]
    assert parsed["setbacks"] == [
        {
            "title": "Gaming correction (2023)",
            "text": "Demand fell below guidance.",
            "response_rating": "Handled well",
            "response_text": "Reset guidance and reduced channel inventory.",
        }
    ]


def test_generate_management_quality_from_markdown_stages_and_indexes(auth_client, monkeypatch, tmp_path, temp_core_db):
    _use_document_generation_warm_worker(monkeypatch)
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()

    import portfolio.management_quality_content as management_quality_content
    import portfolio.portfolio_db as portfolio_db

    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr(
        portfolio_db,
        "get_positions",
        lambda: [{"ticker": "MU", "asset": "equity", "instrument_type": "security"}],
    )

    indexed: list[dict] = []
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: indexed.append(kwargs))

    def fail_pdf_call(*args, **kwargs):
        raise AssertionError("PDF generation should not run for markdown uploads")

    markdown_calls = []

    def fake_markdown_call(*, ticker: str, markdown: str):
        markdown_calls.append(markdown)
        assert ticker == "MU"
        assert "Owner mindset evidence" in markdown
        return (
            "# MU Management Quality\n\n"
            "## Executive Summary\n"
            "- **Overall Rating**: Strong\n"
            "- **Bottom Line**: Management allocates capital with discipline.\n"
            "- **Owner Mindset**: Strong - Buybacks were disciplined.\n"
            "- **Business Value Understanding**: Strong - Focused on HBM mix.\n"
            "- **Follow-through / Character**: Mixed - Some targets slipped.\n\n"
            "## Management Scorecard\n"
            "| Question | Rating | Evidence |\n"
            "|----------|--------|----------|\n"
            "| Do managers think and act like owners? | Strong | Capital returns were disciplined. |\n\n"
            "## Most Impressive Accomplishments\n"
            "- **HBM ramp (2025)**: Improved mix and margins. source-1\n\n"
            "## Biggest Setbacks and Responses\n"
            "- **Inventory correction (2023)**: Demand fell. **Response**: Mixed - Reset guidance.\n\n"
            "## Chronology / Detail\n"
            "### 2025\n"
            "- **Said**: Improve mix.\n"
            "- **Did**: Grew HBM.\n"
            "- **Assessment**: Good follow-through.\n\n"
            "## Evidence Notes\n"
            "- source-1\n"
        )

    monkeypatch.setattr(management_quality_router, "_call_llm_management_quality_pdf", fail_pdf_call)
    monkeypatch.setattr(management_quality_router, "_call_llm_management_quality_markdown", fake_markdown_call)

    resp = auth_client.post(
        "/api/v1/management-quality/generate",
        data={"ticker": "mu"},
        files={
            "file": (
                "management.md",
                b"# Notes\n\nOwner mindset evidence\n",
                "text/markdown",
            )
        },
    )
    assert resp.status_code == 202
    assert markdown_calls == []
    payload = _finish_document_generation_job(auth_client, resp.json()["job_id"])
    assert payload["status"] == "pending_approval_created"
    assert payload["ticker"] == "MU"
    assert payload["proposed_change"]["content"].startswith("# MU Management Quality")
    assert not (mgmt_dir / "MU.md").exists()
    assert len(markdown_calls) == 1

    approved = auth_client.post(
        f"/api/v1/approvals/{payload['approval_id']}/approve", json={"note": "Apply assessment"}
    )
    assert approved.status_code == 200
    assert (mgmt_dir / "MU.md").read_text(encoding="utf-8") == payload["proposed_change"]["content"]
    assert indexed[0]["doc_type"] == "management_quality"
    assert indexed[0]["doc_id"] == "management_quality-MU"

    dossier = auth_client.get("/api/v1/dossier/MU")
    assert dossier.status_code == 200
    dossier_payload = dossier.json()
    assert dossier_payload["management_quality"]["content"].startswith("# MU Management Quality")
    assert dossier_payload["management_quality"]["parsed"]["summary"]["overall_rating"] == "Strong"


def test_save_management_quality_can_apply_immediately(auth_client, monkeypatch, tmp_path, temp_core_db):
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()

    import portfolio.management_quality_content as management_quality_content

    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr("api.retrieval.index_document", lambda **kwargs: None)

    resp = auth_client.put(
        "/api/v1/management-quality/MU",
        json={
            "content": "# Old\n\n## Executive Summary\n- **Overall Rating**: Mixed",
            "apply": True,
            "approval_note": "Apply management assessment",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "applied"
    content = (mgmt_dir / "MU.md").read_text(encoding="utf-8")
    assert content.startswith("# MU Management Quality")
    assert "## Management Scorecard" in content


def test_thesis_overview_and_management_uploads_reject_endpoint_oversized_files(auth_client, monkeypatch):
    monkeypatch.setattr(thesis_router, "MAX_UPLOAD_SIZE_BYTES", 4)
    monkeypatch.setattr(overview_router, "MAX_UPLOAD_SIZE_BYTES", 4)
    monkeypatch.setattr(management_quality_router, "MAX_UPLOAD_SIZE_BYTES", 4)

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
    management_quality = auth_client.post(
        "/api/v1/management-quality/generate",
        data={"ticker": "mu"},
        files={"file": ("management.md", b"12345", "text/markdown")},
    )

    assert thesis.status_code == 413
    assert overview.status_code == 413
    assert management_quality.status_code == 413


@pytest.mark.parametrize(
    "path",
    ["/api/v1/thesis/generate", "/api/v1/overview/generate", "/api/v1/management-quality/generate"],
)
@pytest.mark.parametrize(
    ("data", "file_tuple", "expected_status"),
    [
        ({"ticker": "bad-!"}, ("upload.md", b"# Notes", "text/markdown"), 422),
        ({"ticker": "mu"}, ("upload.md", b"", "text/markdown"), 422),
        ({"ticker": "mu"}, ("upload.pdf", b"not a pdf", "application/pdf"), 422),
        ({"ticker": "mu"}, ("upload.txt", b"notes", "text/plain"), 422),
    ],
)
def test_document_uploads_reject_invalid_inputs_synchronously(auth_client, path, data, file_tuple, expected_status):
    resp = auth_client.post(path, data=data, files={"file": file_tuple})

    assert resp.status_code == expected_status
    assert "job_id" not in resp.text


def test_failed_management_quality_generation_marks_job_error_and_cleans_upload(
    auth_client,
    monkeypatch,
    tmp_path,
    temp_core_db,
):
    _use_document_generation_warm_worker(monkeypatch)
    mgmt_dir = tmp_path / "investment_management_quality"
    mgmt_dir.mkdir()

    import portfolio.management_quality_content as management_quality_content

    monkeypatch.setattr(management_quality_content, "MANAGEMENT_QUALITY_DIR", mgmt_dir)
    monkeypatch.setattr(
        management_quality_router,
        "_call_llm_management_quality_markdown",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    started = auth_client.post(
        "/api/v1/management-quality/generate",
        data={"ticker": "mu"},
        files={"file": ("management.md", b"# Notes\n\nOwner mindset evidence\n", "text/markdown")},
    )
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    upload_path = _document_generation_upload_path(job_id)
    assert upload_path.exists()

    from api.async_job_runner import perform_job

    with pytest.raises(RuntimeError, match="Failed to generate management quality: model unavailable"):
        perform_job(job_id)

    polled = auth_client.get(f"/api/v1/document-generation/async/{job_id}")
    assert polled.status_code == 200
    payload = polled.json()
    assert payload["status"] == "error"
    assert "Failed to generate management quality: model unavailable" in payload["error"]
    assert not upload_path.exists()
    assert temp_core_db.get_pending_approvals(ticker="MU") == []
    assert not (mgmt_dir / "MU.md").exists()


def test_failed_overview_generation_does_not_write_content(auth_client, monkeypatch, tmp_path):
    _use_document_generation_warm_worker(monkeypatch)
    overview_dir = tmp_path / "investment_overviews"
    overview_dir.mkdir()
    monkeypatch.setattr(overview_router, "OVERVIEWS_DIR", overview_dir)
    monkeypatch.setattr(
        overview_router,
        "_call_llm_overview_markdown",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    started = auth_client.post(
        "/api/v1/overview/generate",
        data={"ticker": "mu"},
        files={"file": ("overview.md", b"# Notes\n\nRevenue growth\n", "text/markdown")},
    )
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    upload_path = _document_generation_upload_path(job_id)

    from api.async_job_runner import perform_job

    with pytest.raises(RuntimeError, match="Failed to generate overview: model unavailable"):
        perform_job(job_id)

    assert not upload_path.exists()
    assert not (overview_dir / "MU.md").exists()


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
    management_quality_schema = (
        management_quality_router.SaveManagementQualityRequest.model_json_schema()
        if hasattr(management_quality_router.SaveManagementQualityRequest, "model_json_schema")
        else management_quality_router.SaveManagementQualityRequest.schema()
    )

    assert thesis_schema["properties"]["content"]["maxLength"] == thesis_router.MAX_UPLOAD_SIZE_BYTES
    assert overview_schema["properties"]["content"]["maxLength"] == overview_router.MAX_UPLOAD_SIZE_BYTES
    assert (
        management_quality_schema["properties"]["content"]["maxLength"]
        == management_quality_router.MAX_UPLOAD_SIZE_BYTES
    )


def test_thesis_claim_api_accepts_typed_sources_and_writes_markdown(auth_client, monkeypatch, tmp_path, temp_core_db):
    core_db = temp_core_db
    thesis_dir = tmp_path / "investment_theses"
    thesis_dir.mkdir()
    thesis_file = thesis_dir / "MU.md"
    thesis_file.write_text(
        "# MU\n\n## Thesis\n- Base thesis\n\n## Key Catalysts\n- TBD\n\n## Risk Factors\n- TBD\n",
        encoding="utf-8",
    )

    import portfolio.thesis_sync as thesis_sync

    monkeypatch.setattr(
        thesis_sync,
        "_thesis_paths",
        lambda ticker: (thesis_dir / f"{ticker}.md", f"live/theses/{ticker}.md"),
    )

    catalyst = core_db.create_catalyst("MU", "HBM ramp: HBM3 sold out", created_by="backfill")
    kill_condition = core_db.create_kill_condition(
        "MU",
        "AI spending deceleration: Hyperscaler capex pulls back",
        created_by="backfill",
    )

    resp = auth_client.post(
        "/api/v1/thesis-claims",
        json={
            "ticker": "MU",
            "claim": "HBM supply stays tight: Premium pricing can persist.",
            "expected_evidence": "HBM sell-out commentary.",
            "disconfirming_evidence": "ASP declines.",
            "source_requirements": [
                {
                    "type": "earnings_transcript",
                    "description": "latest earnings call",
                    "required": True,
                    "freshness_days": 45,
                }
            ],
            "cadence": "weekly",
            "confidence": 0.72,
            "linked_catalyst_ids": [catalyst["id"]],
            "linked_kill_condition_ids": [kill_condition["id"]],
            "apply": True,
            "approval_note": "Apply claim",
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "applied"
    claim_row = core_db.get_thesis_claims(ticker="MU")[0]
    assert claim_row["source_requirements"][0]["type"] == "earnings_transcript"
    content = thesis_file.read_text(encoding="utf-8")
    assert "## Thesis Claims" in content
    assert "type=earnings_transcript" in content
    assert "Catalysts: HBM ramp" in content
    assert "Kill conditions: AI spending deceleration" in content

    update = auth_client.put(
        f"/api/v1/thesis-claims/{claim_row['id']}",
        json={
            "status": "supported",
            "source_requirements": ["earnings"],
            "confidence": 0.8,
            "apply": True,
            "approval_note": "Apply claim update",
        },
    )

    assert update.status_code == 200
    assert update.json()["status"] == "applied"
    updated = core_db.get_thesis_claims(ticker="MU")[0]
    assert updated["status"] == "supported"
    assert updated["source_requirements"][0] == {
        "type": "custom",
        "description": "earnings",
        "required": True,
        "freshness_days": None,
    }
    assert "Status: supported" in thesis_file.read_text(encoding="utf-8")


def test_thesis_claim_api_rejects_invalid_confidence(auth_client):
    resp = auth_client.post(
        "/api/v1/thesis-claims",
        json={"ticker": "MU", "claim": "Invalid confidence", "confidence": 1.5},
    )

    assert resp.status_code == 422
