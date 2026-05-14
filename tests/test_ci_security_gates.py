"""Tests for CI workflow security gates — SHA-35.

These are text-level assertions that the security scanning jobs exist
in the CI workflow and don't require production secrets to run.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def _load_ci() -> dict:
    return yaml.safe_load(CI_PATH.read_text())


class TestCISecurityGatesPresent:
    """All required security scanning jobs must exist in ci.yml."""

    def test_pip_audit_job_exists(self) -> None:
        ci = _load_ci()
        assert "security-pip-audit" in ci["jobs"]

    def test_npm_audit_job_exists(self) -> None:
        ci = _load_ci()
        assert "security-npm-audit" in ci["jobs"]

    def test_secret_scan_job_exists(self) -> None:
        ci = _load_ci()
        assert "security-secret-scan" in ci["jobs"]

    def test_python_sast_job_exists(self) -> None:
        ci = _load_ci()
        assert "security-python-sast" in ci["jobs"]

    def test_trivy_job_exists(self) -> None:
        ci = _load_ci()
        assert "security-trivy" in ci["jobs"]


class TestCISecurityGatesNoSecrets:
    """Security jobs must not reference production secrets."""

    def _job_text(self, job_name: str) -> str:
        """Return the raw YAML text of a specific job for text-level checks."""
        ci_text = CI_PATH.read_text()
        ci = yaml.safe_load(ci_text)
        # Serialize the job back to text for pattern matching
        import json

        return json.dumps(ci["jobs"][job_name])

    def test_pip_audit_no_secrets(self) -> None:
        text = self._job_text("security-pip-audit")
        assert "secrets." not in text
        assert "DATABASE_URL" not in text

    def test_npm_audit_no_secrets(self) -> None:
        text = self._job_text("security-npm-audit")
        assert "secrets." not in text

    def test_secret_scan_no_production_secrets(self) -> None:
        text = self._job_text("security-secret-scan")
        assert "secrets." not in text

    def test_python_sast_no_secrets(self) -> None:
        text = self._job_text("security-python-sast")
        assert "secrets." not in text

    def test_trivy_no_secrets(self) -> None:
        text = self._job_text("security-trivy")
        assert "secrets." not in text


class TestCISecurityGatesSeverity:
    """Security gates should block on High/Critical severity."""

    def test_npm_audit_level(self) -> None:
        ci_text = CI_PATH.read_text()
        assert "audit-level=high" in ci_text

    def test_trivy_severity(self) -> None:
        ci_text = CI_PATH.read_text()
        assert "HIGH,CRITICAL" in ci_text

    def test_trivy_exit_code(self) -> None:
        ci_text = CI_PATH.read_text()
        assert "--exit-code 1" in ci_text

    def test_bandit_severity_flags(self) -> None:
        ci_text = CI_PATH.read_text()
        # -ll = medium+ severity, -ii = medium+ confidence
        assert "-ll" in ci_text
        assert "-ii" in ci_text
