import sqlite3
import sys
import types
from datetime import UTC, datetime

import httpx
import pytest

from api.exceptions import DataFetchError
from macro.central_banks.central_bank import (
    Item,
    _content_is_current,
    init_db,
    resolve_item_url,
    set_content,
    upsert_item,
)


def test_resolve_fed_minutes_url_prefers_html_document():
    press_release_url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260218a.htm"
    minutes_url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == press_release_url
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
                <html><body>
                  <a href="/monetarypolicy/fomcminutes20260128.pdf">PDF</a>
                  <a href="/monetarypolicy/fomcminutes20260128.htm">HTML</a>
                </body></html>
            """,
        )

    item = Item(
        source="FED",
        title="Minutes of the Federal Open Market Committee, January 27-28, 2026",
        url=press_release_url,
        published_at=datetime(2026, 2, 18, tzinfo=UTC),
        guid="fed-minutes-1",
        kind="FOMC minutes",
    )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        resolved = resolve_item_url(client, item)

    assert resolved.url == minutes_url
    assert resolved.kind == "FOMC minutes"


def test_upsert_item_updates_existing_metadata():
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        init_db(conn)

        published_at = datetime(2026, 2, 18, tzinfo=UTC)
        original = Item(
            source="FED",
            title="Minutes of the Federal Open Market Committee, January 27-28, 2026",
            url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20260218a.htm",
            published_at=published_at,
            guid="fed-minutes-2",
            kind="FOMC minutes (press release)",
        )
        updated = Item(
            source="FED",
            title=original.title,
            url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
            published_at=published_at,
            guid=original.guid,
            kind="FOMC minutes",
        )

        upsert_item(conn, original)
        upsert_item(conn, updated)

        row = conn.execute("SELECT kind, url FROM items WHERE guid=?", (original.guid,)).fetchone()
        assert row == ("FOMC minutes", "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm")
    finally:
        if conn is not None:
            conn.close()


def test_content_is_current_requires_matching_content_url():
    assert not _content_is_current(
        "announcement text", None, "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm"
    )
    assert not _content_is_current(
        "announcement text",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260218a.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
    )
    assert _content_is_current(
        "actual minutes",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
    )


def test_set_content_persists_content_url():
    conn = sqlite3.connect(":memory:")
    try:
        init_db(conn)
        item = Item(
            source="FED",
            title="Minutes of the Federal Open Market Committee, January 27-28, 2026",
            url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm",
            published_at=datetime(2026, 2, 18, tzinfo=UTC),
            guid="fed-minutes-3",
            kind="FOMC minutes",
        )
        upsert_item(conn, item)
        set_content(conn, item.guid, item.url, "minutes text")

        row = conn.execute("SELECT content_url, content_text FROM items WHERE guid=?", (item.guid,)).fetchone()
        assert row == (item.url, "minutes text")
    finally:
        conn.close()


def _stub_central_bank_module(get_data):
    module = types.ModuleType("macro.central_banks.central_bank")
    module.get_data = get_data
    return module


def test_central_bank_route_raises_on_error_payload(monkeypatch):
    from api.routers import central_banks as router

    monkeypatch.setitem(
        sys.modules,
        "macro.central_banks.central_bank",
        _stub_central_bank_module(lambda refresh=False: {"error": "database unavailable"}),
    )
    monkeypatch.setattr(router, "get_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda *_args, **_kwargs: None)

    with pytest.raises(DataFetchError) as exc:
        router.get_central_banks(refresh=True)

    assert exc.value.source == "central_banks"
    assert exc.value.detail == "database unavailable"


def test_central_bank_route_refresh_updates_default_cache(monkeypatch):
    from api.routers import central_banks as router

    payload = {
        "items": [],
        "by_source": {},
        "counts": {"total": 0},
        "last_updated": "2026-04-30T00:00:00+00:00",
    }
    cache_sets: list[tuple[str, dict]] = []

    monkeypatch.setitem(
        sys.modules,
        "macro.central_banks.central_bank",
        _stub_central_bank_module(lambda refresh=False: payload),
    )
    monkeypatch.setattr(router, "get_cached", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(router, "set_cached", lambda _cache, key, value: cache_sets.append((key, value)))

    result = router.get_central_banks(refresh=True)

    assert result["counts"]["total"] == 0
    assert [key for key, _value in cache_sets] == ["central_banks"]
