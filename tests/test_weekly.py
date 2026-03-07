"""Smoke test for weekly report generation."""

from paths import setup_paths

setup_paths()

from api.cache import long_cache, set_cached
from api.routers.weekly_report import (
    _append_sources_section,
    _extract_openai_citations,
    get_weekly_report,
)


def test_weekly_report_returns_dict():
    """Weekly report should return cached data when available."""
    set_cached(long_cache, "weekly_report_generated", {"report": "ok"})
    res = get_weekly_report(cached_only=True)
    assert isinstance(res, dict)
    assert "report" in res


def test_extract_openai_citations_collects_annotations_and_search_sources():
    response = {
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {"title": "Reuters story", "url": "https://www.reuters.com/example"},
                        {"title": "Reuters story", "url": "https://www.reuters.com/example"},
                    ]
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "annotations": [
                            {"title": "Fed release", "url": "https://www.federalreserve.gov/example"},
                        ]
                    }
                ],
            },
        ]
    }

    citations = _extract_openai_citations(response)

    assert citations == [
        ("Reuters story", "https://www.reuters.com/example"),
        ("Fed release", "https://www.federalreserve.gov/example"),
    ]


def test_append_sources_section_adds_markdown_links():
    report = "# Weekly Report"

    updated = _append_sources_section(
        report,
        [("Reuters story", "https://www.reuters.com/example")],
    )

    assert "## Sources" in updated
    assert "- [Reuters story](https://www.reuters.com/example)" in updated
