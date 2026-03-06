"""
Centralized logging configuration for the API server.

Call ``configure_logging()`` once at startup (in api/main.py).
In production (``json_format=True``), logs are emitted as single-line JSON
for easy ingestion by log aggregators.
"""

import json
import logging
import uuid
from contextvars import ContextVar

# Per-request correlation ID, set by the request-id middleware in main.py.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(""),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(*, json_format: bool = False, level: int = logging.INFO) -> None:
    """Set up root logger with either human-readable or JSON format."""
    root = logging.getLogger()

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.setLevel(level)


def generate_request_id() -> str:
    """Return a short, unique request ID."""
    return uuid.uuid4().hex[:12]
