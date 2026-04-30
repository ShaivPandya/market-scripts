"""One-shot backfill of Industry Monitor PDFs from local disk to GCS.

Operational flow:

1. From a workstation with the local PDF set under ``macro/industry/files/`` and
   credentials for ``$GCS_STATE_BUCKET``:

       STATE_STORAGE_BACKEND=gcs GCS_STATE_BUCKET=talisman-dev-state \
           python -m api.industry_pdf_backfill upload --dry-run

2. Re-run without ``--dry-run`` to perform the upload. The command is idempotent:
   blobs whose md5 already matches the local file are skipped.

The bucket key shape mirrors what ``industry_monitor._get_pdf_locator`` reads at
runtime, including the ``_TICKER_FILENAME_MAP`` rename (``ODFL`` -> ``ODL.pdf``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from api import state_storage
from macro.industry import industry_monitor as im


@dataclass
class BackfillItem:
    sector: str
    ticker: str
    local_path: Path
    gcs_key: str


def _enumerate_items() -> list[BackfillItem]:
    items: list[BackfillItem] = []
    for sector, cfg in im.SECTORS.items():
        for ticker, _name, _sub, _report_time in cfg["companies"]:
            local_path, gcs_key = _get_pdf_locator(sector, ticker)
            items.append(BackfillItem(sector=sector, ticker=ticker, local_path=local_path, gcs_key=gcs_key))
    return items


def _get_pdf_locator(sector: str, ticker: str) -> tuple[Path, str]:
    return im._get_pdf_locator(sector, ticker)


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _blob_md5_matches(gcs_key: str, local_md5_b64: str) -> bool:
    """Return True if the GCS blob exists with a matching md5."""
    bucket = state_storage._bucket()  # noqa: SLF001  reuse adapter's client + bucket name
    blob = bucket.blob(gcs_key)
    if not blob.exists():
        return False
    blob.reload()
    return (blob.md5_hash or "") == local_md5_b64


def upload(*, dry_run: bool) -> int:
    if not state_storage.use_gcs_state():
        print(
            "STATE_STORAGE_BACKEND is not 'gcs' (and ENVIRONMENT != 'production'). "
            "Refusing to run — set STATE_STORAGE_BACKEND=gcs and GCS_STATE_BUCKET first.",
            file=sys.stderr,
        )
        return 2

    items = _enumerate_items()
    uploaded = 0
    skipped_match = 0
    missing_local = 0
    print(f"Backfill plan: {len(items)} ticker entries against prefix '{im.INDUSTRY_TRANSCRIPTS_PREFIX}/'")

    for item in items:
        if not item.local_path.is_file():
            missing_local += 1
            print(f"  SKIP (no local file)  {item.sector:>14} / {item.ticker:>5}  {item.local_path}")
            continue

        local_md5 = _md5_b64(item.local_path)
        try:
            already_matches = _blob_md5_matches(item.gcs_key, local_md5)
        except Exception as ex:
            print(f"  ERROR  {item.sector:>14} / {item.ticker:>5}  {item.gcs_key}: {ex}", file=sys.stderr)
            return 1

        if already_matches:
            skipped_match += 1
            print(f"  SKIP (md5 match)      {item.sector:>14} / {item.ticker:>5}  gs://.../{item.gcs_key}")
            continue

        if dry_run:
            print(f"  WOULD UPLOAD          {item.sector:>14} / {item.ticker:>5}  {item.local_path} -> {item.gcs_key}")
            continue

        state_storage.upload_file(
            item.local_path,
            item.gcs_key,
            content_type="application/pdf",
            metadata={"source": "backfill", "ticker": item.ticker, "sector": item.sector},
        )
        uploaded += 1
        print(f"  UPLOADED              {item.sector:>14} / {item.ticker:>5}  -> gs://.../{item.gcs_key}")

    print(
        f"\nDone. uploaded={uploaded} skipped_md5_match={skipped_match} missing_local={missing_local} dry_run={dry_run}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload", help="Upload local PDFs to the configured GCS bucket")
    up.add_argument("--dry-run", action="store_true", help="Print planned actions without uploading")
    args = parser.parse_args(argv)

    if args.cmd == "upload":
        return upload(dry_run=bool(args.dry_run))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
