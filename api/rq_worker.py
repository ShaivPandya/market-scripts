"""Deprecated RQ worker entrypoint.

Async work now runs on demand through ``python -m api.async_job_runner run`` in
the generic Cloud Run Job.
"""

from __future__ import annotations

import sys


def main(_argv: list[str] | None = None) -> int:
    print(
        "api.rq_worker is deprecated. Use `python -m api.async_job_runner run [job_id]` instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
