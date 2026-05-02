from __future__ import annotations

import os

import portfolio.news_digests as digests

SAMPLE_DIGEST = """# Newsletter Digest - May 1, 2026

*Generated: 2026-05-01*

## liquidity_path

- [MULTI-SIGNAL] Japan launches FX intervention for the first time since 2024 - (Bloomberg) - [body content]
  - *MOF/BOJ-coordinated dollar-selling intervention with explicit final warning rhetoric.*
- Fitch warns US Debt Burden Far Above Other AA Rated Nations - (Bloomberg) - [body content]

## central_bank_pivot

- FOMC dissents the most since 1992 - (Axios Macro / FT) - [body content]
  - *Three dissenters explicitly say next move could be either a cut or a hike.*
"""


def _isolate_digest_store(monkeypatch, tmp_path):
    base = tmp_path / "news_digests"
    monkeypatch.setattr(digests, "DIGESTS_DIR", base)
    monkeypatch.setattr(digests, "MANIFEST_PATH", base / "manifest.json")
    monkeypatch.setattr(digests, "FILES_DIR", base / "files")
    monkeypatch.setattr(digests, "DIGESTS_GCS_PREFIX", "test/news_digests")
    monkeypatch.setattr(digests, "MANIFEST_GCS_KEY", "test/news_digests/manifest.json")
    monkeypatch.setattr(digests, "FILES_GCS_PREFIX", "test/news_digests/files")
    os.environ["STATE_STORAGE_BACKEND"] = "local"


def test_parse_digest_extracts_title_date_sections_stories_and_notes():
    parsed = digests.parse_digest_markdown(SAMPLE_DIGEST, filename="05012026_digest.md")

    assert parsed["title"] == "Newsletter Digest - May 1, 2026"
    assert parsed["generated_date"] == "2026-05-01"
    assert parsed["slug"] == "newsletter-digest-may-1-2026"
    assert parsed["story_count"] == 3
    assert parsed["section_count"] == 2
    assert parsed["sections"][0]["name"] == "liquidity_path"
    assert parsed["sections"][0]["stories"][0]["headline"].startswith("[MULTI-SIGNAL] Japan launches")
    assert parsed["sections"][0]["stories"][0]["notes"] == [
        "*MOF/BOJ-coordinated dollar-selling intervention with explicit final warning rhetoric.*"
    ]


def test_parse_digest_falls_back_to_filename_date():
    markdown = "# Filename Dated Digest\n\n## section\n\n- Story one\n"

    parsed = digests.parse_digest_markdown(markdown, filename="05012026_digest.md")

    assert parsed["generated_date"] == "2026-05-01"


def test_save_digest_replaces_same_date_and_title(monkeypatch, tmp_path):
    _isolate_digest_store(monkeypatch, tmp_path)

    first = digests.save_digest(SAMPLE_DIGEST, filename="05012026_digest.md")
    second = digests.save_digest(
        SAMPLE_DIGEST + "\n## bubble_watch\n\n- New story\n",
        filename="renamed.md",
    )

    listed = digests.list_digests()
    detail = digests.get_digest(first["id"])

    assert second["id"] == first["id"]
    assert listed["counts"]["digests"] == 1
    assert listed["counts"]["stories"] == 4
    assert detail["filename"] == "renamed.md"
    assert "New story" in detail["content"]


def test_delete_digest_removes_manifest_and_file(monkeypatch, tmp_path):
    _isolate_digest_store(monkeypatch, tmp_path)
    saved = digests.save_digest(SAMPLE_DIGEST, filename="05012026_digest.md")

    assert digests.delete_digest(saved["id"]) is True

    assert digests.list_digests()["counts"] == {"digests": 0, "stories": 0}
    try:
        digests.get_digest(saved["id"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("deleted digest should not be readable")


def test_report_context_uses_recent_window_with_latest_fallback(monkeypatch, tmp_path):
    _isolate_digest_store(monkeypatch, tmp_path)
    old = "# Old Digest\n\n*Generated: 2026-04-01*\n\n## macro\n\n- Old story\n"
    recent = "# Recent Digest\n\n*Generated: 2026-05-01*\n\n## macro\n\n- Recent story\n"
    digests.save_digest(old, filename="old.md")
    digests.save_digest(recent, filename="recent.md")

    context = digests.get_report_context(days=8, now=digests.datetime(2026, 5, 2, tzinfo=digests.UTC))
    fallback = digests.get_report_context(days=2, now=digests.datetime(2026, 6, 1, tzinfo=digests.UTC))

    assert context["fallback_used"] is False
    assert context["digests"][0]["title"] == "Recent Digest"
    assert context["digests"][0]["sections"][0]["stories"][0]["headline"] == "Recent story"
    assert fallback["fallback_used"] is True
    assert fallback["digests"][0]["title"] == "Recent Digest"


def test_router_upload_list_detail_delete(auth_client, monkeypatch, tmp_path):
    _isolate_digest_store(monkeypatch, tmp_path)

    from api.routers import portfolio_news as router

    monkeypatch.setattr(router, "_index_digest_best_effort", lambda _detail: None)
    monkeypatch.setattr(router, "_delete_digest_index_best_effort", lambda _digest_id: None)

    upload = auth_client.post(
        "/api/v1/portfolio-news",
        files={"file": ("05012026_digest.md", SAMPLE_DIGEST, "text/markdown")},
    )
    assert upload.status_code == 200
    digest_id = upload.json()["digest"]["id"]

    listed = auth_client.get("/api/v1/portfolio-news")
    assert listed.status_code == 200
    assert listed.json()["counts"]["digests"] == 1

    detail = auth_client.get(f"/api/v1/portfolio-news/{digest_id}")
    assert detail.status_code == 200
    assert detail.json()["content"] == SAMPLE_DIGEST

    deleted = auth_client.delete(f"/api/v1/portfolio-news/{digest_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
