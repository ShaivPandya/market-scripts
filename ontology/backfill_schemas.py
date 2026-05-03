from __future__ import annotations

import argparse
import json
from pathlib import Path

from ontology.repository import OntologyRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill ontology rows to typed schema_version=1 payloads.")
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite ontology database path.")
    parser.add_argument("--write", action="store_true", help="Rewrite rows. Default is dry-run mode.")
    args = parser.parse_args()

    repo = OntologyRepository(db_path=args.db_path)
    report = repo.backfill_schema_versions(dry_run=not args.write)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
