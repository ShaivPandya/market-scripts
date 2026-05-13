from __future__ import annotations

from auto_report import fetch_state


class _FakeResponse:
    def __init__(self, payload=None):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse({"detail": "ok"})

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/api/portfolio-settings"):
            return _FakeResponse({"book_size": 125000})
        return _FakeResponse(
            {
                "positions": [
                    {"ticker": "MU", "role": "position"},
                    {"ticker": "SH", "role": "hedge"},
                ]
            }
        )


def test_fetch_state_logs_in_before_fetch_when_password_present(monkeypatch, tmp_path):
    import api.portfolio_settings as portfolio_settings

    session = _FakeSession()
    github_env = tmp_path / "github.env"
    saved_book_size: list[float] = []

    monkeypatch.setenv("TALISMAN_API_URL", "https://example.test")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setenv("TALISMAN_API_PASSWORD", "report-password")
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    monkeypatch.setattr(fetch_state.requests, "Session", lambda: session)
    monkeypatch.setattr(portfolio_settings, "set_portfolio_book_size", lambda value: saved_book_size.append(value))

    assert fetch_state.fetch_and_seed() == 0

    assert [call[:2] for call in session.calls] == [
        ("POST", "https://example.test/api/auth/login"),
        ("GET", "https://example.test/api/portfolio-positions"),
        ("GET", "https://example.test/api/portfolio-settings"),
    ]
    assert session.calls[0][2]["json"] == {"password": "report-password"}
    assert session.calls[0][2]["headers"] == {
        "X-Api-Proxy-Secret": "proxy-secret",
        "X-Request-Schema-Name": "post:/api/auth/login",
        "X-Request-Schema-Version": "1",
    }
    assert session.calls[1][2]["params"] == {"include_hedges": "true"}
    assert "TALISMAN_BOOK_SIZE=125000.00\n" in github_env.read_text(encoding="utf-8")
    assert saved_book_size == [125000.0]
