from __future__ import annotations

import json

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
    session = _FakeSession()
    github_env = tmp_path / "github.env"
    state_path = tmp_path / "portfolio_state.json"

    monkeypatch.setenv("TALISMAN_API_URL", "https://example.test")
    monkeypatch.setenv("API_PROXY_SECRET", "proxy-secret")
    monkeypatch.setenv("TALISMAN_API_PASSWORD", "report-password")
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    monkeypatch.setenv("AUTO_REPORT_PORTFOLIO_STATE_PATH", str(state_path))
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(fetch_state.requests, "Session", lambda: session)

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
    github_env_text = github_env.read_text(encoding="utf-8")
    assert "TALISMAN_BOOK_SIZE=125000.00\n" in github_env_text
    assert f"AUTO_REPORT_PORTFOLIO_STATE_PATH={state_path}\n" in github_env_text
    assert state_path.exists()
    assert '"ticker": "MU"' in state_path.read_text(encoding="utf-8")


def test_daily_report_loads_cached_portfolio_state_without_database(monkeypatch, tmp_path):
    from auto_report import auto_daily_report

    state_path = tmp_path / "portfolio_state.json"
    state_path.write_text(
        json.dumps(
            {
                "positions": [
                    {"ticker": " mu ", "role": "position", "direction": "LONG", "conviction": None},
                    {"ticker": "SH", "role": "hedge", "direction": "short", "conviction": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AUTO_REPORT_PORTFOLIO_STATE_PATH", str(state_path))
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    df = auto_daily_report.load_portfolio()

    assert list(df["ticker"]) == ["MU"]
    assert list(df["direction"]) == ["long"]
    assert list(df["conviction"]) == [3]


def test_recommendation_persistence_can_be_skipped_for_report_sync(monkeypatch):
    from auto_report.recommendations import persist_recommendations

    monkeypatch.setenv("AUTO_REPORT_SKIP_LOCAL_PERSISTENCE", "1")
    monkeypatch.setenv("STATE_DB_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = persist_recommendations(
        {
            "report_type": "daily",
            "as_of": "2026-05-15",
            "stance": "Neutral / Watchful",
            "recommendation_status": "clear",
            "critical_data_quality": "ok",
            "recommended_actions": [{"action": "watch", "instrument": "portfolio"}],
        },
        source_report_path="auto_report/outputs/daily/recommendations.md",
        source_json_path="auto_report/outputs/daily/recommendations.json",
    )

    assert result == []
