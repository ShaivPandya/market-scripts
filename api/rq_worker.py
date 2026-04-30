"""RQ worker entrypoint for Cloud Run worker pools.

Run:
    python -m api.rq_worker default screens reports
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main(argv: list[str] | None = None) -> int:
    try:
        from redis import Redis
        from rq import Queue, Worker
    except ImportError as exc:
        raise RuntimeError("rq and redis are required to run async workers.") from exc

    args = list(argv if argv is not None else sys.argv[1:])
    queue_names = args or [
        q.strip() for q in (os.getenv("ASYNC_WORKER_QUEUES") or "default,screens,reports").split(",") if q.strip()
    ]
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is required to run async workers.")

    connection = Redis.from_url(redis_url)
    queues = [Queue(name, connection=connection) for name in queue_names]
    worker = Worker(queues, connection=connection)
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
