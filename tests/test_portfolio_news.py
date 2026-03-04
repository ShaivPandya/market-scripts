from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse, unquote
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import portfolio.portfolio_news as news


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _build_rss(items: list[dict[str, str]]) -> bytes:
    parts = ["<?xml version='1.0' encoding='UTF-8'?><rss><channel>"]
    for item in items:
        parts.append(
            "<item>"
            f"<title>{escape(item.get('title', ''))}</title>"
            f"<link>{escape(item.get('link', 'https://example.com/article'))}</link>"
            f"<description>{escape(item.get('description', ''))}</description>"
            f"<source>{escape(item.get('source', 'Reuters'))}</source>"
            f"<pubDate>{escape(item.get('pubDate', 'Wed, 04 Mar 2026 02:00:00 GMT'))}</pubDate>"
            "</item>"
        )
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


def _install_urlopen(monkeypatch, xml_payload: bytes, captured_urls: list[str]) -> None:
    def _fake_urlopen(req, timeout=10):
        url = getattr(req, "full_url", str(req))
        captured_urls.append(url)
        return _FakeResponse(xml_payload)

    monkeypatch.setattr(news.urllib.request, "urlopen", _fake_urlopen)


def test_strict_query_for_fly_uses_aliases_only(monkeypatch):
    monkeypatch.setattr(news, "NEWS_STRICT_TICKERS", {"FLY"})

    query, strict_mode, aliases = news._build_google_rss_query("FLY", "Firefly Aerospace Inc.")

    assert strict_mode is True
    assert "firefly aerospace" in aliases
    assert re.search(r"\bfly\b", query, flags=re.IGNORECASE) is None
    assert '"firefly aerospace"' in query


def test_strict_filter_drops_unrelated_articles(monkeypatch):
    monkeypatch.setattr(news, "NEWS_STRICT_TICKERS", {"FLY"})

    xml_payload = _build_rss(
        [
            {"title": "Persian Gulf crisis boosts Chinese airlines' routes to Europe amid turmoil"},
            {"title": "What's going on with Kylian Mbappe's knee injury at Real Madrid?"},
            {"title": "Chimpanzees Are Really Into Crystals"},
            {"title": "Firefly Aerospace secures NASA launch contract"},
        ]
    )
    captured_urls: list[str] = []
    _install_urlopen(monkeypatch, xml_payload, captured_urls)

    out = news._query_google_rss("FLY", "Firefly Aerospace Inc.")

    assert len(captured_urls) == 1
    assert len(out) == 1
    assert out[0]["title"] == "Firefly Aerospace secures NASA launch contract"


def test_strict_filter_keeps_article_when_alias_in_description(monkeypatch):
    monkeypatch.setattr(news, "NEWS_STRICT_TICKERS", {"FLY"})

    xml_payload = _build_rss(
        [
            {
                "title": "Space startup prepares launch vehicle update",
                "description": "Firefly Aerospace said it completed final hot-fire testing.",
            }
        ]
    )
    captured_urls: list[str] = []
    _install_urlopen(monkeypatch, xml_payload, captured_urls)

    out = news._query_google_rss("FLY", "Firefly Aerospace Inc.")

    assert len(captured_urls) == 1
    assert len(out) == 1
    assert out[0]["ticker"] == "FLY"


def test_non_strict_ticker_behavior_is_unchanged(monkeypatch):
    monkeypatch.setattr(news, "NEWS_STRICT_TICKERS", {"FLY"})

    xml_payload = _build_rss(
        [
            {
                "title": "Semiconductor stocks rally on AI optimism",
                "description": "Chipmakers gained in early trading.",
            }
        ]
    )
    captured_urls: list[str] = []
    _install_urlopen(monkeypatch, xml_payload, captured_urls)

    out = news._query_google_rss("MU", "Micron Technology, Inc.")

    assert len(out) == 1
    assert len(captured_urls) == 1

    query_param = parse_qs(urlparse(captured_urls[0]).query).get("q", [""])[0]
    decoded_query = unquote(query_param)
    assert '(MU OR "Micron Technology, Inc.")' in decoded_query
