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
        return _FakeResponse(
            {
                "positions": [
                    {"ticker": "MU", "role": "position"},
                    {"ticker": "SH", "role": "hedge"},
                ]
            }
        )


def test_fetch_state_logs_in_before_fetch_when_password_present(monkeypatch):
    session = _FakeSession()
    saved: list[tuple[list[dict], str]] = []

    monkeypatch.setenv("TALISMAN_API_URL", "https://example.test")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setenv("TALISMAN_API_PASSWORD", "report-password")
    monkeypatch.setattr(fetch_state.requests, "Session", lambda: session)
    monkeypatch.setattr(
        "portfolio.portfolio_db.save_positions",
        lambda positions, role="position": saved.append((positions, role)),
    )

    assert fetch_state.fetch_and_seed() == 0

    assert [call[:2] for call in session.calls] == [
        ("POST", "https://example.test/api/v1/auth/login"),
        ("GET", "https://example.test/api/v1/portfolio-positions"),
    ]
    assert session.calls[0][2]["json"] == {"password": "report-password"}
    assert session.calls[0][2]["headers"] == {
        "X-Api-Proxy-Secret": "proxy-secret",
        "X-Request-Schema-Name": "post:/api/v1/auth/login",
        "X-Request-Schema-Version": "1",
    }
    assert session.calls[1][2]["params"] == {"include_hedges": "true"}
    assert saved == [
        ([{"ticker": "MU", "role": "position"}], "position"),
        ([{"ticker": "SH", "role": "hedge"}], "hedge"),
    ]
