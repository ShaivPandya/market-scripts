"""Tests for api/industry_pdf_backfill.py — idempotent local→GCS PDF upload."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from api import industry_pdf_backfill as backfill
from api import state_storage
from macro.industry import industry_monitor as im


def _make_pdf(path: Path, body: bytes = b"%PDF-FAKE-BODY") -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def test_enumerate_items_covers_every_sector_company_with_filename_map():
    items = backfill._enumerate_items()
    expected = sum(len(cfg["companies"]) for cfg in im.SECTORS.values())
    assert len(items) == expected

    by_ticker = {(it.sector, it.ticker): it for it in items}
    odfl = by_ticker[("Trucking", "ODFL")]
    assert odfl.gcs_key.endswith("/trucking/ODL.pdf")
    assert odfl.local_path.name == "ODL.pdf"


def test_md5_b64_matches_python_md5(tmp_path):
    pdf = tmp_path / "x.pdf"
    body = b"%PDF-DATA-12345"
    pdf.write_bytes(body)
    expected = base64.b64encode(hashlib.md5(body).digest()).decode("ascii")
    assert backfill._md5_b64(pdf) == expected


class _FakeBlob:
    def __init__(self, *, exists: bool, md5: str | None = None):
        self._exists = exists
        self.md5_hash = md5
        self.metadata: dict[str, str] | None = None
        self.uploaded_path: str | None = None

    def exists(self) -> bool:
        return self._exists

    def reload(self) -> None:
        pass

    def upload_from_filename(self, path: str, content_type: str | None = None) -> None:
        self.uploaded_path = path


class _FakeBucket:
    def __init__(self, blobs: dict[str, _FakeBlob]):
        self._blobs = blobs
        self.requested: list[str] = []

    def blob(self, key: str) -> _FakeBlob:
        self.requested.append(key)
        return self._blobs.setdefault(key, _FakeBlob(exists=False))


@pytest.fixture
def gcs_env(monkeypatch):
    monkeypatch.setenv("STATE_STORAGE_BACKEND", "gcs")
    monkeypatch.setenv("GCS_STATE_BUCKET", "fake-bucket")


def test_upload_refuses_when_backend_not_gcs(monkeypatch, capsys):
    monkeypatch.delenv("STATE_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    rc = backfill.upload(dry_run=True)
    assert rc == 2
    captured = capsys.readouterr()
    assert "STATE_STORAGE_BACKEND" in captured.err


def test_upload_dry_run_skips_md5_match_and_does_not_upload(gcs_env, monkeypatch, tmp_path, capsys):
    # Force every locator to point at a file we control.
    body = b"%PDF-MATCHED"
    md5 = base64.b64encode(hashlib.md5(body).digest()).decode("ascii")

    fake_local = tmp_path / "fake.pdf"
    fake_local.write_bytes(body)

    monkeypatch.setattr(
        backfill, "_get_pdf_locator", lambda sector, ticker: (fake_local, f"prefix/{sector}/{ticker}.pdf")
    )
    # Reset SECTORS-driven enumeration to a tiny fixture for clarity.
    monkeypatch.setattr(im, "SECTORS", {"Demo": {"type": "leading", "companies": [("AAA", "Alpha", "X", "BMO")]}})

    blobs: dict[str, _FakeBlob] = {"prefix/Demo/AAA.pdf": _FakeBlob(exists=True, md5=md5)}
    bucket = _FakeBucket(blobs)
    monkeypatch.setattr(state_storage, "_bucket", lambda: bucket)

    rc = backfill.upload(dry_run=True)
    assert rc == 0

    out = capsys.readouterr().out
    assert "SKIP (md5 match)" in out
    assert "WOULD UPLOAD" not in out


def test_upload_uploads_when_blob_missing(gcs_env, monkeypatch, tmp_path, capsys):
    body = b"%PDF-NEW"
    fake_local = tmp_path / "new.pdf"
    fake_local.write_bytes(body)

    monkeypatch.setattr(
        backfill, "_get_pdf_locator", lambda sector, ticker: (fake_local, f"prefix/{sector}/{ticker}.pdf")
    )
    monkeypatch.setattr(im, "SECTORS", {"Demo": {"type": "leading", "companies": [("AAA", "Alpha", "X", "BMO")]}})

    blobs: dict[str, _FakeBlob] = {}
    bucket = _FakeBucket(blobs)
    monkeypatch.setattr(state_storage, "_bucket", lambda: bucket)

    rc = backfill.upload(dry_run=False)
    assert rc == 0

    out = capsys.readouterr().out
    assert "UPLOADED" in out
    uploaded_blob = blobs["prefix/Demo/AAA.pdf"]
    assert uploaded_blob.uploaded_path == str(fake_local)
    assert uploaded_blob.metadata == {"source": "backfill", "ticker": "AAA", "sector": "Demo"}


def test_upload_skips_missing_local_files(gcs_env, monkeypatch, tmp_path, capsys):
    missing_local = tmp_path / "absent.pdf"  # not created
    monkeypatch.setattr(
        backfill, "_get_pdf_locator", lambda sector, ticker: (missing_local, f"prefix/{sector}/{ticker}.pdf")
    )
    monkeypatch.setattr(im, "SECTORS", {"Demo": {"type": "leading", "companies": [("AAA", "Alpha", "X", "BMO")]}})

    bucket = _FakeBucket({})
    monkeypatch.setattr(state_storage, "_bucket", lambda: bucket)

    rc = backfill.upload(dry_run=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "SKIP (no local file)" in out
    assert "UPLOADED" not in out
